# Decisions

Running log of methodological choices; the Methodology section is written from it.

## Data
- FFT-75 scenario 1 (75 classes), 4096-byte blocks.
- Training subset: 20,000 blocks per class (1.5M), stratified, seed 2026, so that all models see the
  same data and fit the time budget. The official test split is used in full.
- Validation: 4,000 blocks per class, split per class into val_a and val_b (2,000 each). val_a is for
  model selection and temperature; val_b is a separate calibration set for thresholds and conformal
  quantiles, so the guarantees are not fitted on the data used to select the model.
- Exact-duplicate check: BLAKE2b-64 hash of every block of the official train and test files;
  reported for the official split and for our subset. FiFTy splits at block level, so duplicates
  across splits are possible.
- Effect of the duplicates (N5): `prepare` saves a boolean mask per test block,
  `test_duplicates.npz` with `official_train` (block also in the official train split) and
  `subset_train` (also in our training subset, i.e. the leakage our models actually saw; always a
  subset of the first). `analyze` reports accuracy, macro-F1 and ECE after temperature scaling on all
  test blocks, on the duplicated blocks and on the test set without them, for both masks. Reason: a
  2.1 % overlap can only inflate the headline accuracy if the model does much better on the
  duplicates, which this measures directly. Macro-F1 of a group is averaged over the classes present
  in that group. The ECE of the duplicated group (≈ 16k blocks, unevenly spread over the 15 bins) is
  noisier than on the full test set; read the accuracy difference first. Datasets prepared before the
  masks existed get them from `fragtrust duplicates` (one hashing pass over the official train file;
  the counts must reproduce `leakage.json`, otherwise it stops); without the file `analyze` logs that
  and skips the breakdown (synthetic data).

## Models and training
- No label smoothing, mixup or other regularisers that change confidence: the study measures calibration.
- Networks: AdamW, warmup + cosine LR, bf16 autocast on CUDA, fp32 for all saved logits.
  Early stopping on val_a accuracy (patience 5, max 30 epochs).
- LightGBM: 262 features (normalised byte histogram, entropy, mean, std, longest run, bigram entropy,
  printable share). Grid {num_leaves 63/127} × {learning rate 0.05/0.1} on 5,000 blocks per class,
  selected by val_a log-loss; early stopping on val_a log-loss. Seeds act through feature and row
  bagging (0.8). LightGBM log-probabilities serve as logits.
- Seeds: 5 for LightGBM, 3 for CNN-M and the transformer, 1 per capacity point (CNN-S, CNN-L).
- Deep ensemble = mean of the probabilities of the three CNN-M seeds; no extra training.

## Trust analysis
- Temperature fitted on val_a by minimising NLL.
- ECE with 15 equal-width bins and 15 equal-mass bins; NLL; multiclass Brier score.
- Abstention: confidence = maximum softmax probability after temperature scaling (predictive
  entropy also reported). Thresholds for 95 % and 99 % target accuracy fitted on val_b in two ways:
  empirical (largest coverage meeting the target) and Learn-then-Test fixed-sequence binomial
  testing with δ = 0.05, which bounds the selective error with probability ≥ 95 %. Accepted-set sizes
  smaller than the minimum that can pass the test even with zero errors are skipped; that minimum,
  m_min = ⌈log δ / log(1 − (1 − target))⌉ (59 at 95 %, 299 at 99 % for δ = 0.05), depends only on
  target and δ, and no threshold exists if val_b has fewer than m_min blocks. `trust.json` records
  m_min per target and the calibration size n_cal.
- A target counts as met only if the threshold answers at least one test block and the accuracy of
  the answers reaches the target. Reason: abstaining on everything trivially has no errors; counting
  it as a pass made "meets the target" true exactly when the guarantee is useless. The number of
  answered blocks is recorded with every threshold, and tables print "no answers" for such cases.
- Conformal sets on val_b with α ∈ {0.10, 0.05}: LAC, randomised APS, RAPS with λ = 0.01 and
  k_reg = 2 fixed in advance (not tuned). Reported: coverage, set size, per-class coverage.
- OOD probes (never in training): 20,000 AES-256-CTR encryptions of random test blocks (one key, a
  fresh nonce per block) and 20,000 uniformly random blocks. AUROC against the test split using MSP,
  entropy and (networks only) energy; energy is undefined for log-probabilities.
- Model-free baseline for the probes: AUROC of the byte entropy of the block itself (bits per byte,
  higher = more likely a probe). Reason: AES-CTR and random blocks have near-maximal entropy
  (≈ 7.955 bits), which a plain threshold can detect without any model; a model score adds
  information only where it beats this baseline.
- Duplicate breakdown in `trust.json`: `duplicates/<side>/<group>/{n,accuracy,macro_f1,ece}` with
  side `official_train` or `subset_train` and group `all`, `duplicated` or `without`; plus
  `duplicates/<side>/all_minus_without/{accuracy,macro_f1,ece}` (positive accuracy = duplicates
  inflate it). Empty groups have `n` 0 and null metrics. Table `duplicates` in the report.
- Entropy bins up to 8 bits with fine bins near the top, because uniformly random 4096-byte blocks
  have an expected empirical entropy of about 8 − 255/(2·4096·ln 2) ≈ 7.955 bits.

## Explanations
- TreeSHAP for LightGBM only, exact for trees (LightGBM `pred_contrib`), on a stratified test sample
  of 20 blocks per class. Global importance = mean |SHAP| over blocks and classes; per class, the top 5
  features by mean SHAP toward the true class. Spearman correlation with gain importance is reported
  as a consistency check, not as evidence.

## Case study
- Exhibit: 1 GB FAT32 image with 4096-byte clusters, equal to the model block size, so one cluster is
  one block. Built with dosfstools + mtools, without mounting: zero-filled fillers of 32–512 KB fill the
  volume except a 64 MB tail, every second filler (in allocation order) is deleted, the evidence files
  are copied in a fixed random order (seed 0), so files larger than a hole are fragmented. Then quick
  format with `mkfs.fat` on the same image.
- Ground truth comes from the FAT cluster chains before the format and is verified by reassembling
  every file to its sha256; the build also checks that the format changed no evidence cluster and
  leaves no visible file. The scan reads the formatted image with the geometry from the build record.
- Labels: file extension mapped to the FFT-75 class (aliases such as jpeg→jpg), overridden by
  `labels.csv`. Types outside FFT-75 (`unknown`, `encrypted`) are not scored for accuracy; for them and
  for free space we report mean confidence, the share abstained at each target and the conformal set
  size.
- Metrics on clusters of known type, for all clusters and by group (fragmented/contiguous,
  header/middle/tail of the file): accuracy, ECE, coverage and accuracy of answers at the 95 % and 99 %
  targets, conformal coverage and set size. Thresholds, temperature and conformal quantiles are those
  fitted on FFT-75 (transferred), unchanged.
- Carver baseline: header signatures at the start of a cluster (table in `case_study/signatures.py`),
  checked in order with specific signatures before generic ones; container formats (ZIP, OLE2, TIFF,
  ISO-BMFF) give a candidate set. Two variants: header clusters only, and propagated, where a header's
  type extends to following clusters until the next header or an all-zero cluster (the contiguity
  assumption of a simple carver, broken by fragmentation). Reported: coverage, accuracy of answers
  (true type in the candidate set), mean candidate-set size.
- Recalibration: 20 % of the image's labelled files (grouped by file, so clusters of one file never
  fall on both sides) refit temperature, Learn-then-Test thresholds and conformal quantiles; both the
  transferred and the refitted parameters are evaluated on the other 80 %. 20 random splits, mean ± std,
  the share of splits whose answers meet the 99 % target (at least one answer, see Trust analysis) and,
  separately, the share of splits where the threshold answers nothing (`no_answers@t`; `answered@t`
  is the count per split and group).
- `case-simulate` builds an image from test blocks of a processed dataset for smoke tests only; its
  "files" are not real files, so it is never used for the paper.

## CPU benchmark
- Decision: networks run with ONNX Runtime (opset 17, dynamic batch, batch 256, CPU execution
  provider, intra-op threads = the thread count, one inter-op thread). Reason: the PyTorch build on the
  M2 laptop has no oneDNN and its `conv1d` is about 10× slower than im2col + BLAS on the same machine,
  so PyTorch eager numbers would depend on how PyTorch was built rather than on the model. ONNX
  Runtime is the same runtime on a laptop and a server. The export is checked against PyTorch on one
  batch (maximum absolute logit difference, stored with the result); `--backend torch` remains for
  comparison.
- 20,000 official test blocks, one warm-up batch, wall-clock time of one pass; LightGBM time includes
  feature extraction. Each model is measured in a fresh process (own peak memory; LightGBM's and
  PyTorch's OpenMP runtimes in one process deadlocked on macOS). Thread counts: 1 and every CPU the
  process may use, which inside a container is the cgroup CPU limit, not the host's core count.
- Results are stored per host. The laptop needs `pip install -e ".[bench]"`; without it only the
  server CPU is reported.

## Reporting
- Every figure and table is generated by `fragtrust report` from result files; `paper/assets.json`
  lists the sources of each asset and the git commit.
- Every number in the draft is a tag resolved from a result file (`fragtrust numbers`); numbers typed
  by hand are listed for checking.
- Rounding: `.Nf` and `pctN` in the draft and all table cells round half up on the decimal value as
  written in the result file (0.8765 at one decimal in percent is 87.7; binary floating point gives
  87.6), so the text and the tables agree in the last digit.
- Figures: LNCS text width 122 mm, SVG + 600-dpi PNG, DejaVu Sans. Series colours come from a palette
  checked for colour-vision deficiency, and every series also has its own marker and line style, so
  the figures read in greyscale.

## Engineering
- Default thread counts follow the CPUs actually available (affinity mask and cgroup quota): on a
  RunPod pod `os.cpu_count()` reports the host's cores, and LightGBM with that many threads on a
  16-vCPU limit runs slower.
- Parallel feature extraction uses spawned workers: forking a process that already runs LightGBM's
  OpenMP threads is unsafe.
- The CUDA code path (bf16, GPU-resident data, GPU scanning) is covered by `scripts/smoke.sh` on the
  server; locally only CPU and MPS were tested.
