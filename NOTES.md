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
  smaller than the minimum that can pass the test even with zero errors are skipped; that minimum
  depends only on n, target and δ.
- Conformal sets on val_b with α ∈ {0.10, 0.05}: LAC, randomised APS, RAPS with λ = 0.01 and
  k_reg = 2 fixed in advance (not tuned). Reported: coverage, set size, per-class coverage.
- OOD probes (never in training): 20,000 AES-256-CTR encryptions of random test blocks (one key, a
  fresh nonce per block) and 20,000 uniformly random blocks. AUROC against the test split using MSP,
  entropy and (networks only) energy; energy is undefined for log-probabilities.
- Entropy bins up to 8 bits with fine bins near the top, because uniformly random 4096-byte blocks
  have an expected empirical entropy of about 8 − 255/(2·4096·ln 2) ≈ 7.955 bits.

## Open points
- CPU benchmark on the M2 laptop: the installed PyTorch build has no oneDNN, and its `conv1d` is
  about 10× slower than im2col + BLAS on the same machine. A benchmark from this stack would
  understate what a CNN can do on a laptop. Options: the official PyTorch wheel, ONNX Runtime, or
  reporting the server CPU. Decide before running E1 throughput.
- The CUDA code path (bf16, GPU-resident data) is covered by `scripts/smoke.sh` on the server;
  locally only CPU and MPS were tested.
