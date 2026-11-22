"""Training loop for the byte-level networks.

bf16 autocast on CUDA, AdamW with warmup + cosine LR, early stopping on val_a accuracy,
a full checkpoint every epoch (`last.pt`) so an interrupted run resumes where it stopped.
Plain cross-entropy without label smoothing: smoothing changes calibration, which is what we measure.
"""

from __future__ import annotations

import csv
import logging
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .batches import BlockBatches

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    patience: int = 5
    grad_clip: float = 1.0
    amp: bool = True
    compile: bool = False
    resident: Any = "auto"
    eval_batch_size: int = 4096
    log_every: int = 500

    @classmethod
    def from_dict(cls, options: dict[str, Any]) -> TrainConfig:
        known = {f.name: f for f in fields(cls)}
        unknown = set(options) - set(known)
        if unknown:
            raise ValueError(f"unknown train options: {sorted(unknown)}")
        casts = {"int": int, "float": float}
        # YAML 1.1 reads `2e-3` (no dot) as a string
        return cls(**{k: casts[known[k].type](v) if known[k].type in casts else v for k, v in options.items()})


@dataclass
class TrainState:
    epoch: int = 0
    best_epoch: int = 0
    best_val_acc: float = -1.0
    bad_epochs: int = 0
    train_seconds: float = 0.0


def _optimizer(model: nn.Module, cfg: TrainConfig, device: torch.device) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or name in ("cls", "pos") else decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, fused=device.type == "cuda")


def _warmup_cosine(total_steps: int, warmup_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def _save_checkpoint(path, model, opt, sched, train: BlockBatches, state: TrainState) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "sampler": train.generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "state": asdict(state),
        },
        tmp,
    )
    tmp.replace(path)


def _load_checkpoint(path, model, opt, sched, train: BlockBatches) -> TrainState:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    sched.load_state_dict(ckpt["scheduler"])
    train.generator.set_state(ckpt["sampler"])
    torch.set_rng_state(ckpt["torch_rng"])
    if ckpt["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    return TrainState(**ckpt["state"])


def _train_epoch(net, batches: BlockBatches, opt, sched, cfg: TrainConfig, amp: bool, epoch: int) -> tuple[float, float]:
    net.train()
    device = batches.device
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    seen = 0
    t0 = time.perf_counter()
    for step, (xb, yb) in enumerate(batches, 1):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = net(xb)
        loss = F.cross_entropy(logits.float(), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        loss_sum += loss.detach() * len(yb)
        correct += (logits.argmax(dim=1) == yb).sum()
        seen += len(yb)
        if step % cfg.log_every == 0:
            rate = seen / (time.perf_counter() - t0)
            log.info(
                "epoch %d step %d/%d | loss %.4f | acc %.4f | lr %.2e | %.0f blocks/s | epoch eta %.1f min",
                epoch, step, len(batches), loss_sum.item() / seen, correct.item() / seen,
                sched.get_last_lr()[0], rate, (len(batches) - step) * batches.batch_size / rate / 60,
            )
    return loss_sum.item() / seen, correct.item() / seen


@torch.no_grad()
def evaluate(net: nn.Module, batches: BlockBatches) -> tuple[float, float]:
    """Mean cross-entropy and accuracy in fp32."""
    net.eval()
    loss_sum, correct, seen = 0.0, 0, 0
    for xb, yb in batches:
        logits = net(xb).float()
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(dim=1) == yb).sum().item()
        seen += len(yb)
    return loss_sum / seen, correct / seen


@torch.no_grad()
def predict_logits(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    """fp32 logits for every row of `x` (memory-mapped arrays are read sequentially)."""
    model.eval()
    parts = []
    for a in range(0, len(x), batch_size):
        xb = torch.from_numpy(np.array(x[a : a + batch_size])).to(device)
        parts.append(model(xb).float().cpu().numpy())
    return np.concatenate(parts)


def fit(
    model: nn.Module, train: BlockBatches, val: BlockBatches, cfg: TrainConfig, device: torch.device, out_dir: Path
) -> dict[str, Any]:
    """Train until `cfg.epochs` or early stop; best weights in best.pt, per-epoch log in history.csv."""
    net = torch.compile(model) if cfg.compile else model
    opt = _optimizer(model, cfg, device)
    steps = len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _warmup_cosine(cfg.epochs * steps, max(1, round(cfg.warmup_epochs * steps)))
    )
    amp = cfg.amp and device.type == "cuda"
    checkpoint = out_dir / "last.pt"
    state = TrainState()
    if checkpoint.exists():
        state = _load_checkpoint(checkpoint, model, opt, sched, train)
        log.info("resumed after epoch %d (best val_a acc %.4f at epoch %d)", state.epoch, state.best_val_acc, state.best_epoch)

    while state.epoch < cfg.epochs and state.bad_epochs < cfg.patience:
        epoch = state.epoch + 1
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        train_loss, train_acc = _train_epoch(net, train, opt, sched, cfg, amp, epoch)
        train_time = time.perf_counter() - t0
        val_loss, val_acc = evaluate(net, val)
        state.epoch = epoch
        state.train_seconds += time.perf_counter() - t0
        improved = val_acc > state.best_val_acc
        if improved:
            state.best_val_acc, state.best_epoch, state.bad_epochs = val_acc, epoch, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            state.bad_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_a_loss": round(val_loss, 6),
            "val_a_acc": round(val_acc, 6),
            "lr": sched.get_last_lr()[0],
            "epoch_seconds": round(time.perf_counter() - t0, 1),
            "train_blocks_per_s": round(steps * train.batch_size / train_time),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if device.type == "cuda" else None,
        }
        _append_csv(out_dir / "history.csv", row)
        log.info(
            "epoch %d/%d | train loss %.4f acc %.4f | val_a loss %.4f acc %.4f%s | %.0fs",
            epoch, cfg.epochs, train_loss, train_acc, val_loss, val_acc, " (best)" if improved else "", row["epoch_seconds"],
        )
        _save_checkpoint(checkpoint, model, opt, sched, train, state)

    reason = "early stop" if state.bad_epochs >= cfg.patience else "max epochs"
    log.info("training ended (%s) after %d epochs; best val_a acc %.4f at epoch %d", reason, state.epoch, state.best_val_acc, state.best_epoch)
    return asdict(state)
