# fragtrust

Experiments for a study of trustworthy file fragment classification (ICECC 2027, Special Session 7):
how honest the confidence of FFT-75 classifiers is, what accuracy an examiner can be guaranteed when
the model may abstain, what the models do with encrypted or random data they never saw, and whether
the guarantees survive on a realistic exhibit (a quick-formatted FAT32 USB image).

| ID | Model | Size |
|---|---|---|
| M1 | 256-bin byte histogram + 6 block statistics → LightGBM | – |
| M2 | 1D CNN over byte embeddings, three widths (`cnn_s`, `cnn_m`, `cnn_l`) | 0.10M / 0.58M / 3.1M params |
| M3 | Byte transformer, 16-byte patches, 4 layers, d=128 | 0.91M params |

The trust analysis runs on saved logits (CPU only): temperature scaling; ECE, NLL, Brier;
risk–coverage, AURC and accuracy-targeted abstention thresholds with a Learn-then-Test guarantee;
conformal prediction sets (LAC, APS, RAPS); out-of-distribution probes (AES-CTR ciphertext and random
bytes); accuracy by block entropy; exact-duplicate blocks between the official train and test splits.
On top of that: TreeSHAP attributions of the LightGBM model, CPU throughput with ONNX Runtime, and the
case study (every cluster of a formatted FAT32 image classified, against a header-signature carver,
with and without recalibration on part of the image).

## Layout

```
configs/data/        dataset build configs (FFT-75 subset, synthetic smoke data, class groups)
configs/models/      paper models
configs/smoke/       tiny models and paper config for the smoke run
configs/case_study/  FAT32 image build config
configs/paper.yaml   which runs go into which figure and table
scripts/             server setup, download, base plan, case study, smoke run, sync
src/fragtrust/
  data/              streaming .npz reader, subset builder, duplicate check, OOD probes, synthetic data
  features.py        hand-crafted block features (LightGBM)
  models/            CNN, transformer, LightGBM
  training/          GPU batching, training loop with resumable checkpoints
  trust/             calibration, selective prediction, conformal sets, OOD scores, per-run analysis
  case_study/        FAT32 reader, image builder, ground truth, signature carver, scan, evaluation
  report/            paper figures and tables, number tags for the draft
  experiment.py      tune / train / ensemble
  explain.py         TreeSHAP for LightGBM
  benchmark.py       CPU throughput
  summarize.py       tables over all runs
data/case_study/     evidence files and images (README: what to collect and how to build)
tests/               pytest: unit tests, end-to-end pipeline, resume, case study, report
NOTES.md             methodological decisions and their reasons
```

## 1. Check the code (laptop, synthetic data)

The scripts put `src/` on `PYTHONPATH`, so nothing has to be installed if the dependencies in
`pyproject.toml` are present.

```bash
bash scripts/smoke.sh
```

`smoke.sh` runs every step on a 7-class synthetic dataset and writes to `results_smoke/`: training,
trust analysis, SHAP, the case study on a simulated image, the CPU benchmark (ONNX Runtime if
installed, otherwise PyTorch) and all paper figures and tables in `results_smoke/paper/`. Rerunning
skips finished steps; delete `results_smoke/` for a clean run.

```bash
python -m pytest -rs
```

On macOS two tests are skipped: building a FAT32 image needs dosfstools and mtools (Linux), and the
ONNX export test needs the `bench` extra (`pip install -e ".[bench]"`).

## 2. Full run on the GPU server

Needs Linux, one GPU with ≥ 16 GB, ≥ 16 CPU cores, ≥ 64 GB RAM, about 100 GB of disk,
Python ≥ 3.10 and a CUDA build of PyTorch (a RunPod PyTorch image has one).

Set up the environment: system packages (dosfstools, mtools, tmux, wget, as root), a `.venv` that
reuses the image's PyTorch, the package with the `dev` and `bench` extras, the tests (none should be
skipped here) and the hardware record. Safe to rerun.

```bash
bash scripts/setup_server.sh
```

Run the smoke pipeline once on the GPU (exercises bf16 training, GPU-resident data and GPU scanning):

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

The base plan checks the GPU, the extras and the free disk first, then builds the subset, trains
LightGBM (5 seeds, on the CPUs in the background) and the networks (CNN-M ×3, CNN-S, CNN-L,
transformer ×3, on the GPU), builds the CNN-M ensemble, runs the trust analysis, SHAP and the CPU
benchmark, and writes the paper figures and tables. Its output also goes to
`results/logs/base_plan.log`. Detach with `Ctrl-b d`, reattach with `tmux attach -t fft`. Every step
skips finished work and resumes unfinished runs from their last epoch, so after a crash or a pod
restart run the same script again.

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

## 3. Case study

Collect the evidence files as described in [data/case_study/README.md](data/case_study/README.md)
(any OS), copy them to `data/case_study/files/` on the server, and after the base plan run:

```bash
bash scripts/run_case_study.sh
```

It builds the formatted FAT32 image if it does not exist yet (Linux only; the build verifies every
file by sha256 and that the format changed no evidence cluster), scans every cluster with LightGBM,
CNN-M and the transformer, evaluates them and rebuilds the paper assets. Output goes to
`results/case_study/fat32_1g/`, the log to `results/logs/case_study.log`.

## 4. Paper assets and numbers

```bash
python -m fragtrust.cli report
```

reads `configs/paper.yaml` and writes every figure whose inputs exist to `paper/figures/` (SVG and
600-dpi PNG, 122 mm wide for LNCS), every table to `paper/tables/` (Markdown and CSV), and
`paper/assets.json` with the result files each asset was built from. Missing inputs skip an asset
with a log line; nothing else fails. If benchmarks from several hosts exist, set `benchmark_host`
in `configs/paper.yaml`.

Numbers in the draft are written as tags and resolved from the result files:

```
accuracy {{results/summary/by_model.csv:cnn_m/accuracy|pct1}}%,
ECE {{results/runs/cnn_m_s0/analysis/trust.json:calibration/temperature_scaled/ece|.3f}}
```

```bash
python -m fragtrust.cli numbers paper/draft.md
```

writes `paper/draft.rendered.md`, fails on any tag that does not resolve and lists numbers typed by
hand outside tags. `.Nf` and `pctN` round half up, exactly like the tables.

## Commands

All commands accept `--results DIR` (default `results`). Run them from the repository root, either as
`fragtrust ...` after `pip install -e .` or as `python -m fragtrust.cli ...`. Commands that write a
result skip it if it exists; `--force` redoes it.

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
| `explain RUN...` | TreeSHAP of a LightGBM run on stratified test blocks → `analysis/shap_*.csv`, `shap.json` |
| `benchmark RUN... [--threads 1 16] [--backend onnx\|torch]` | CPU throughput, minutes per 32 GB / 256 GB drive; default 1 thread and every available CPU |
| `case-build --config C` | formatted FAT32 evidence image with block-level ground truth (Linux) |
| `case-simulate --data D --out O` | simulated image from a processed dataset, for smoke tests only |
| `case-scan RUN... --image I` | logits for every cluster of the image |
| `case-eval RUN... --image I` | transferred guarantees, signature baseline, recalibration on 20 % of the files |
| `report [--config C]` | all paper figures and tables (default `configs/paper.yaml`) |
| `numbers DRAFT` | resolve number tags in the draft, list untagged numbers |

## Outputs

```
results/
  hardware.json, environment.txt
  logs/                  base_plan.log, lgbm.out, case_study.log
  data/<dataset>/        prepare.log, manifest.json, class_balance.csv, leakage.json, leakage_per_class.csv
  tuning/lgbm/           tune.log, grid.csv, best.json
  runs/<model>_s<seed>/
    config.yaml          exact config of the run
    meta.json            seed, git commit, hardware, timings, parameter count, model size
    train.log            full log; history.csv: one row per epoch
    best.pt | model.txt  weights (model.onnx after an ONNX benchmark)
    logits_<split>.npy   val_a, val_b, test, ood_aes_ctr, ood_random
    metrics.json, per_class_test.csv, confusion_test.npy
    analysis/            trust.json, reliability_*.csv, risk_coverage.csv,
                         conformal_class_coverage.csv, entropy_bins.csv, analyze.log;
                         LightGBM: shap_global.csv, shap_per_class.csv, shap.json
    DONE                 marks a finished run
  summary/               runs.csv, by_model.csv, by_model.md
  benchmarks/<host>/     <run>.<backend>.json, benchmark.log
  case_study/<image>/
    scans/<run>.npy      logits for every cluster
    signatures.npz       header matches and propagated carver labels
    <run>/               case_study.json, per_type.csv, recalibration_splits.csv, disk_map.npz
paper/                   figures/, tables/, assets.json
```

## Evaluation protocol

| Split | Used for |
|---|---|
| train (20,000 blocks per class) | fitting |
| val_a (2,000 per class) | early stopping, LightGBM grid, temperature scaling |
| val_b (2,000 per class) | abstention thresholds, conformal quantiles |
| test (official, all blocks) | every reported number, touched once per run |
| probes | AES-CTR and random blocks, never seen in training |
| case-study image | transferred thresholds as they are; recalibration on 20 % of the image's files, evaluated on the other 80 % |

## CPU benchmark

The deployment question is whether an examiner can scan a drive on an ordinary computer. Networks
are exported to ONNX and run with ONNX Runtime, the same runtime on a laptop and a server; the
export is checked against PyTorch (maximum absolute logit difference, stored in the result).
LightGBM runs with its own library, feature extraction included. Each model is measured in a fresh
process, so the peak memory is its own and LightGBM's and PyTorch's OpenMP runtimes never meet. The
default thread counts are 1 and every CPU the process may use; inside a container that is the cgroup
CPU limit, not the host's core count.

## Disk and time

The 4096-byte scenario-1 archive is 24.18 GB (DataPort listing); archive plus extracted files take
about 50 GB, the processed subset about 10.5 GB, cached LightGBM features about 2.5 GB, and each run's
logits about 0.25 GB. After `prepare` has finished, the archive can be deleted (its sha256 is recorded).
The case-study image and its pre-format copy take 2 GB. Time per epoch is logged after the first
epoch; use it to update the GPU-hour budget.
