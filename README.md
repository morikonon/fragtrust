# fragtrust

Experiments for a study of trustworthy file fragment classification (ICECC 2027, Special Session 7):
how honest the confidence of FFT-75 classifiers is, what accuracy an examiner can be guaranteed when
the model may abstain, and what the models do with encrypted or random data they never saw.

| ID | Model | Size |
|---|---|---|
| M1 | 256-bin byte histogram + 6 block statistics → LightGBM | – |
| M2 | 1D CNN over byte embeddings, three widths (`cnn_s`, `cnn_m`, `cnn_l`) | 0.10M / 0.58M / 3.1M params |
| M3 | Byte transformer, 16-byte patches, 4 layers, d=128 | 0.91M params |

The trust analysis runs on saved logits (CPU only): temperature scaling; ECE, NLL, Brier;
risk–coverage, AURC and accuracy-targeted abstention thresholds with a Learn-then-Test guarantee;
conformal prediction sets (LAC, APS, RAPS); out-of-distribution probes (AES-CTR ciphertext and random
bytes); accuracy by block entropy; exact-duplicate blocks between the official train and test splits.

## Layout

```
configs/data/      dataset build configs (FFT-75 subset, synthetic smoke data)
configs/models/    paper models
configs/smoke/     tiny models for the smoke run
scripts/           server setup, download, full run, smoke run, sync
src/fragtrust/
  data/            streaming .npz reader, subset builder, duplicate check, OOD probes, synthetic data
  features.py      hand-crafted block features (LightGBM)
  models/          CNN, transformer, LightGBM
  training/        GPU batching, training loop with resumable checkpoints
  trust/           calibration, selective prediction, conformal sets, OOD scores, per-run analysis
  experiment.py    tune / train / ensemble
  summarize.py     tables over all runs
  benchmark.py     CPU throughput
tests/             pytest (unit tests + end-to-end pipeline + resume check)
NOTES.md           methodological decisions and their reasons
```

## 1. Check the code (laptop, a few minutes, synthetic data)

The scripts put `src/` on `PYTHONPATH`, so nothing has to be installed if the dependencies in
`pyproject.toml` are present.

```bash
bash scripts/smoke.sh
```

```bash
python -m pytest
```

`smoke.sh` runs every step on a 7-class synthetic dataset and writes to `results_smoke/`.

## 2. Full run on the GPU server

Needs Linux, one GPU with ≥ 16 GB, ≥ 16 CPU cores, ≥ 64 GB RAM, about 100 GB of disk,
Python ≥ 3.10 and a CUDA build of PyTorch (a RunPod PyTorch image has one).

Set up the environment, run the tests and record the hardware:

```bash
bash scripts/setup_server.sh
```

Run the smoke pipeline once on the GPU (exercises bf16 training and GPU-resident data):

```bash
bash scripts/smoke.sh
```

Download FFT-75 scenario 1, 4096-byte blocks. Log in at IEEE DataPort, copy the signed link of
`4k_1.tar.gz` and pass it in quotes. The link expires; if the transfer stops, rerun with a fresh link
and `wget` resumes.

```bash
bash scripts/download_fft75.sh "PASTE_SIGNED_URL_HERE"
```

Start the base plan inside tmux so it survives a dropped SSH session:

```bash
tmux new -s fft
```

```bash
bash scripts/run_base_plan.sh
```

The script also writes its output to `results/logs/base_plan.log`. Detach with `Ctrl-b d`, reattach
with `tmux attach -t fft`. Every step skips finished runs and
resumes unfinished ones from their last epoch, so after a crash or a pod restart run the same
script again.

### Watching progress

```bash
python -m fragtrust.cli status
```

```bash
tail -f results/runs/cnn_m_s0/train.log
```

```bash
tail -f results/logs/lgbm.out
```

```bash
nvidia-smi
```

### Copying results to Google Drive

Configure an rclone remote named `gdrive` once (`rclone config`), then:

```bash
bash scripts/sync_results.sh
```

## Commands

All commands accept `--results DIR` (default `results`). Run them from the repository root, either as
`fragtrust ...` after `pip install -e .` or as `python -m fragtrust.cli ...`.

| Command | What it does |
|---|---|
| `env` | hardware and package versions → `results/hardware.json`, `results/environment.txt` |
| `prepare --config C` | stratified subset of the official files, duplicate check, entropy, OOD probes |
| `synth --config C` | synthetic dataset in the same format |
| `tune --config C` | LightGBM grid on a training subsample, selected by val_a log-loss |
| `train --config C --seed 0 1 2` | one run per seed: weights, logits, metrics, logs |
| `ensemble --name N RUN...` | deep ensemble of finished runs |
| `analyze [RUN...]` | trust analysis (default: every finished run not yet analysed) |
| `summarize` | `results/summary/runs.csv`, `by_model.csv`, `by_model.md` (mean ± std over seeds) |
| `status` | state, epoch and accuracy of every run |
| `benchmark RUN... --threads 1 8` | CPU throughput, minutes per 32 GB / 256 GB drive |

## Outputs

```
results/
  hardware.json, environment.txt
  data/<dataset>/        prepare.log, manifest.json, class_balance.csv, leakage.json, leakage_per_class.csv
  tuning/lgbm/           tune.log, grid.csv, best.json
  runs/<model>_s<seed>/
    config.yaml          exact config of the run
    meta.json            seed, git commit, hardware, timings, parameter count, model size
    train.log            full log; history.csv: one row per epoch
    best.pt | model.txt  weights
    logits_<split>.npy   val_a, val_b, test, ood_aes_ctr, ood_random
    metrics.json, per_class_test.csv, confusion_test.npy
    analysis/            trust.json, reliability_*.csv, risk_coverage.csv,
                         conformal_class_coverage.csv, entropy_bins.csv, analyze.log
    DONE                 marks a finished run
  summary/               runs.csv, by_model.csv, by_model.md
  benchmarks/<host>/     <run>.json
```

## Evaluation protocol

| Split | Used for |
|---|---|
| train (20,000 blocks per class) | fitting |
| val_a (2,000 per class) | early stopping, LightGBM grid, temperature scaling |
| val_b (2,000 per class) | abstention thresholds, conformal quantiles |
| test (official, all blocks) | every reported number, touched once per run |
| probes | AES-CTR and random blocks, never seen in training |

## Disk and time

The 4096-byte scenario-1 archive is 24.18 GB (DataPort listing); archive plus extracted files take
about 50 GB, the processed subset about 10.5 GB, cached LightGBM features about 2.5 GB, and each run's
logits about 0.25 GB. After `prepare` has finished, the archive can be deleted (its sha256 is recorded).
Time per epoch is logged after the first epoch; use it to update the GPU-hour budget.
