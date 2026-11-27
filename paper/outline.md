# Paper outline (W1, 2026-11-27)

Target: 11–12 LNCS pages including references (limit 12; each extra page costs 70 USD).
Double-blind. English. Abstract 150–250 words.

Status of inputs: no real results yet except CNN-M seed 0 training (about 77 % val accuracy after
5 epochs; not usable in the paper until analysed). Research notes (`team/research/`) are not
written yet; every claim that depends on the novelty check is marked [NOVELTY]; every missing
reference is `[CITE: ...]`.

## W2 update (2026-11-27): what the first full draft changed

The draft is `draft.md` (built from tags; `draft.rendered.md` is the output of `fragtrust numbers`).
Changes against the W1 plan below:
- Contributions now use the safe wording of `team/research/novelty.md` §3 (five items: calibration
  "to our knowledge the first"; LTT + conformal "with guarantees", citing Magika/MimeLens/ByteRCNN;
  ciphertext probes citing MimeLens; controlled FAT32 image; duplicate count + its small effect).
  C6 (protocol) became one sentence: "Code ... will be released upon acceptance".
- Structure: image construction moved from Sect. 5 into Sect. 4.6 (Methods); Sect. 5 is hardware,
  versions and the benchmark protocol only; new Sect. 6.5 "Duplicated test blocks" (C3 numbers).
- Supported by CNN-M s0 (tagged): ECE already low (0.010 → 0.005), so C1 may end as "well
  calibrated; scaling changes little"; empirical thresholds miss 95/99 % on test, LTT meets both;
  per-type conformal coverage far below target for some types; probes abstained ~100 % but byte
  entropy AUROC nearly equals MSP; duplicate effect +0.10 pp (small; said so).
- Provisional numbering: Table 1 dataset, 2 main_results, 3 calibration, 4 selective, 5 conformal,
  6 ood, 7 case_study, 8 recalibration; Fig. 1 per_class_f1, 2 reliability, 3 risk_coverage,
  4 entropy, 5 disk_map. Cut already: table `leakage` (in text), table `duplicates` (text only),
  figure `capacity` and `shap` (one PENDING sentence each). Still at risk for 12 pages: 8 tables +
  5 figures; next cuts per the priority list below (per_class_f1, then merge calibration+selective).
- References: 31 entries, numbered by first citation, F/A level only (R2) plus R1 sources; open
  markers: `[CITE: Sceadan]` (level M, not cited as a reference), MoDiCo venue, `[CHECK]` for
  pages/volume of Guzhov & Wirth, EnCoD, Sadinle.
- W1 open questions 1–3 are resolved by C2 (`m_min`, `meets` needs ≥ 1 answer, byte-entropy
  baseline). Question 4 (case-study `build.json` under `data/`) is still open.
- New open point: the LTT implementation indexes hypotheses by accepted-set size on val_b, not by a
  fixed threshold grid; flagged in a comment in Sect. 4.3 for the reviewer/coder.

## Working title

Main option:
**Can an Examiner Trust a File Fragment Classifier? Calibration, Guaranteed Abstention and
Conformal Sets from Benchmark to a Formatted FAT32 Image**

Shorter fallback: *Trustworthy File Fragment Classification: Calibration and Coverage Guarantees
under Transfer to Real Media*

Keywords: file fragment classification, file carving, calibration, selective prediction,
conformal prediction, digital forensics.

## Research questions (from CONTEXT.md §10.2, as implemented)

- RQ1 Accuracy and cost: LightGBM on 262 byte features vs. byte CNN (S/M/L) vs. byte transformer;
  accuracy, macro-F1, parameters, size, CPU throughput (ONNX Runtime; LightGBM incl. features).
- RQ2 Calibration and guarantees on FFT-75: ECE/NLL/Brier before and after temperature scaling;
  risk–coverage, AURC, coverage at 95 % / 99 % target accuracy (empirical vs. Learn-then-Test,
  δ = 0.05); conformal sets LAC/APS/RAPS at α = 0.10 / 0.05.
- RQ3 High-entropy and unseen data: accuracy vs. byte entropy; AES-CTR and random probes:
  AUROC (MSP, predictive entropy, energy), share abstained, conformal set size, predicted class.
- RQ4 Transfer: quick-formatted FAT32 image with ground truth; accuracy, ECE, whether the
  transferred guarantees hold; fragmented vs. contiguous, header/middle/tail; vs. header-signature
  carver; recalibration on 20 % of the image's files.

## Page budget

| # | Section | Pages | RQ | Assets |
|---|---|---|---|---|
| – | Title, abstract, keywords | 0.5 | – | – |
| 1 | Introduction | 1.2 | all | – |
| 2 | Related work | 1.0 | – | – |
| 3 | Data and leakage check | 0.6 | (data) | table `dataset`; leakage in text (table `leakage` only if space) |
| 4 | Methods | 1.7 | RQ1–4 | – (optional small pipeline sketch: not produced by `fragtrust report`, skip unless space) |
| 5 | Experimental setup | 0.4 | – | – |
| 6 | Results | 3.9 | RQ1–4 | see below |
| 7 | Discussion and limitations | 0.8 | – | – |
| 8 | Conclusion | 0.3 | – | – |
| – | References (20–25) | 1.1 | – | – |
| | **Total** | **11.5** | | |

Float budget inside Results (LNCS text width 122 mm; one figure ≈ 0.3–0.4 page, one table
≈ 0.2–0.35 page): 4 figures + 6 tables ≈ 2.6 pages of the 3.9. Text must stay short.

Priority of assets (cut from the bottom if over 12 pages):
1. `main_results` (table), `calibration` (table), `reliability` (figure), `selective` (table),
   `risk_coverage` (figure), `conformal` (table), `case_study` (table), `recalibration` (table),
   `ood` (table), `entropy` (figure), `disk_map` (figure), `dataset` (table)
2. `per_class_f1` (figure) — keep if space; otherwise name the weakest classes in text with tags
3. `capacity` (figure) — replace by one sentence with tags for CNN-S/M/L if short of space
4. `leakage` (table) — the two numbers fit in one sentence of Sect. 3
5. `shap` (figure `shap_top_features`) — cut first (plan §10.1 E6); at most one sentence

---

## Abstract (draft, ~215 words; [RESULT] = number tag to be filled)

File carving after deletion or a quick format leaves an examiner with raw 4096-byte clusters whose
file type is unknown. Machine-learning fragment classifiers are usually compared by top-1 accuracy,
but a forensic report needs to know when a label can be relied on. We study this question on the
FFT-75 benchmark (75 types) and on a quick-formatted FAT32 image with ground truth. We compare a
gradient-boosted model on byte statistics, byte-level convolutional networks of three sizes and a
small byte transformer. Before temperature scaling, the expected calibration error is [RESULT];
after it, [RESULT]. With Learn-then-Test thresholds (δ = 0.05), the best model labels [RESULT] of
test blocks at a target accuracy of 99 % and passes the rest to a human; conformal sets reach
[RESULT] coverage at a 95 % target with a mean of [RESULT] types per block. Encrypted and random
blocks never seen in training are [RESULT] at the 99 % target. On the formatted image, parameters
fitted on FFT-75 [RESULT: hold / do not hold] the target (accuracy of answers [RESULT]);
refitting them on 20 % of the image's files [RESULT]. A header-signature carver labels [RESULT]
of evidence clusters. We also find that [RESULT] of the official FFT-75 test blocks occur verbatim
in the training split. Code and protocol are released for reproduction.

Notes: δ = 0.05, 99 %, 95 %, 20 % and 4096 are design constants, not results (allowed as typed
numbers; `fragtrust numbers` will list them as warnings — answer them in the check). Replace the
last sentence by an anonymised statement ("released upon acceptance") — no repository link.

## Contributions (conditional; keep only those the results support)

- C1 Calibration. If temperature-scaled ECE differs clearly from uncalibrated ECE for at least one
  model family: "first measurement of calibration of FFT-75 classifiers, showing that confidence is
  [over/under]-confident by X and that temperature scaling reduces ECE to Y" [NOVELTY: drop "first"
  if Tesserae/MoDiCo/ByteRCNN report calibration]. If ECE is already low without scaling: claim
  only "FFT-75 classifiers are [well] calibrated; scaling changes little" — still useful, weaker.
- C2 Guaranteed outputs. If LTT thresholds meet the target on test for all models and coverage at
  99 % is non-trivial: "examiner-facing operating points with a finite-sample guarantee: X % of
  blocks at ≥ 99 % accuracy". If empirical thresholds miss the target on test while LTT meets it:
  add "empirical thresholds are not enough". If conformal per-class coverage is far below target
  for some classes (`classes_below_target` > 0): state that marginal coverage hides class-level
  failure. [NOVELTY]
- C3 Transfer to real media (main claim if C1/C2 are not new). If transferred thresholds miss the
  target on the image: "benchmark guarantees do not carry over to a formatted FAT32 image; a
  20 % file-level recalibration restores them at coverage X" (only if the recalibrated
  `meets@0.99` share is high **and** recalibrated coverage is > 0, see the vacuity warning in
  Sect. 6.4). If they hold: "guarantees transfer on this image" + limitation "one image, one
  file collection".
- C4 Unknown data. If AES/random probes are mostly abstained at 99 %: "abstention catches unseen
  high-entropy data without an extra class". If not (high confidence on probes): report as a
  negative result — the model assigns them confidently to a known high-entropy class.
- C5 Leakage. If the official test-in-train share is non-negligible: report it as a benchmark
  issue (one sentence + number), not as a main contribution.
- C6 Reproducible protocol: code, configs, number tags, image builder. Always claimable, but phrase
  without links (double-blind).

Do not claim: the task, new architectures, lightweight models, explainability, an "unknown" output
by itself, state of the art.

---

## 1 Introduction (1.2 p, all RQs)

- Forensic setting: deleted or quick-formatted media; FAT metadata gone; examiner sees clusters.
  Carvers use headers and contiguity; fragmentation breaks them [CITE: carving / fragmentation
  survey, e.g. Garfinkel 2007 or Poisel & Tjoa]. Fragment classifiers fill the gap.
- Accuracy plateau on FFT-75: FiFTy 77.5 % (2019) → about 82–84 % (2023–2026) with much larger
  models [CITE: FiFTy; 2023–2026 works from related_work.md]. These are literature numbers, typed
  with their citation (not result tags).
- Gap: a label that goes into a report needs a known error rate; top-1 accuracy does not say which
  blocks are safe. A 2026 carving system uses uncalibrated scores [CITE: Scalpel3] [NOVELTY].
- What we do: measure calibration, give operating points with guarantees (LTT abstention,
  conformal sets), probe unseen data, test transfer on a formatted FAT32 image. State RQ1–RQ4.
- Contributions list C1–C6 (only the surviving ones), each one sentence, each pointing to a
  results subsection.

## 2 Related work (1.0 p)

- Fragment classification: FiFTy/FFT-75 [CITE], byte CNNs and transformers 2023–2026 [CITE from
  related_work.md], ByteRCNN incl. its open-set study [CITE] [NOVELTY], Tesserae / MoDiCo
  (NeurIPS 2026) [CITE] [NOVELTY]. One sentence each: what they report (accuracy only?).
- Real-media evaluation: FiFTy on DFRWS-2006 (accuracy only; number to re-check before use)
  [CITE]; carvers with signatures [CITE: Scalpel / Foremost / PhotoRec, whichever is read].
- Calibration and temperature scaling [CITE: Guo et al. 2017]; selective classification
  [CITE: Geifman & El-Yaniv 2017]; Learn-then-Test [CITE: Angelopoulos et al. LTT];
  conformal sets LAC/APS/RAPS [CITE: Sadinle et al. 2019; Romano et al. 2020; Angelopoulos et al.
  2021]; OOD baselines MSP / energy [CITE: Hendrycks & Gimpel 2017; Liu et al. 2020].
- Positioning paragraph: no prior work found that combines calibrated scores, finite-sample
  guarantees and a transfer test on a formatted image for this task [NOVELTY — rewrite after R1].

## 3 Data and leakage check (0.6 p, data for all RQs)

- FFT-75 scenario 1: 75 types, 4096-byte blocks, official split 6.144M / 768k / 768k [CITE: FiFTy;
  DataPort]. Licence CC BY.
- Our subset (seed 2026): 20,000 train blocks per class; val 4,000 per class split into val_a
  (early stopping, grid, temperature) and val_b (thresholds, conformal quantiles); full official
  test. Probes: 20,000 AES-256-CTR encryptions of random test blocks (one key, fresh nonce) and
  20,000 uniform random blocks. → table `dataset`.
- Why two validation halves: guarantees must not be fitted on data used for model selection.
- Leakage: BLAKE2b-64 hash of every block; test blocks also in train (official split and our
  subset), how many with the same label, how many are constant blocks. One sentence with tags;
  table `leakage` only if space. Consequence stated plainly: test accuracy is slightly optimistic;
  we did not re-evaluate on a deduplicated test set (not implemented — do not claim it).

Tags:
- `results/data/fft75_4096_s1_v1/leakage.json:official_train_vs_test/test_in_train|,d`
- `...:official_train_vs_test/test_in_train_frac|pct1`
- `...:official_train_vs_test/test_in_train_same_label|,d`,
  `.../test_in_train_constant_blocks|,d`, `subset_train_vs_test/test_in_train_frac|pct2`
- Split sizes come from `results/data/fft75_4096_s1_v1/manifest.json:splits/<split>/rows|,d`
  (check the key once the manifest exists).

## 4 Methods (1.7 p)

4.1 Models (RQ1, 0.4 p)
- LightGBM on 262 features (normalised 256-bin histogram, Shannon entropy, mean, std, longest run,
  bigram entropy, printable share); grid {num_leaves 63/127} × {lr 0.05/0.1} on 5,000 blocks per
  class, selected by val_a log-loss; 5 seeds (bagging 0.8); log-probabilities used as logits.
- Byte CNN: embedding + Conv1d blocks + global pooling; widths S/M/L = 0.10M / 0.58M / 3.1M params
  (take exact counts from `meta.json` via tags). Byte transformer: 16-byte patches → 256 tokens,
  0.91M params. AdamW, warmup + cosine, early stop on val_a accuracy (patience 5, max 30 epochs);
  no label smoothing or mixup (they change confidence). CNN-M and transformer 3 seeds; S/L 1 seed.
- Deep ensemble = mean probability of the three CNN-M seeds.

4.2 Calibration (RQ2, 0.25 p)
- Temperature on val_a by NLL. ECE with 15 equal-width and 15 equal-mass bins, NLL, multiclass
  Brier. Formulas in one line each.

4.3 Selective prediction with guarantees (RQ2, 0.45 p)
- Confidence = MSP after scaling (predictive entropy as a second score). Risk–coverage curve, AURC.
- Thresholds for 95 % and 99 % target accuracy on val_b: empirical (largest coverage meeting the
  target on val_b) and Learn-then-Test fixed-sequence binomial test, δ = 0.05: with probability
  ≥ 0.95 over the calibration draw, the selective error is ≤ 1 − target. Sizes below the minimum
  testable size m_min = ⌈log δ / log(1 − ε)⌉ are skipped; m_min depends only on n, ε, δ, so the
  sequence stays data-independent.
- Assumption stated: exchangeability of val_b and test blocks.

4.4 Conformal prediction sets (RQ2, 0.3 p)
- Split conformal on val_b, α ∈ {0.10, 0.05}: LAC, randomised APS, RAPS (λ = 0.01, k_reg = 2,
  fixed in advance). Marginal coverage guarantee; per-class coverage reported because the
  guarantee is not class-conditional. Main setting for later sections: APS at 95 %.

4.5 Unseen data and entropy (RQ3, 0.15 p)
- AUROC of probes vs. test using MSP, predictive entropy, energy (networks only; undefined for
  LightGBM log-probabilities). Share abstained at each LTT threshold; APS-95 set size; top
  predicted classes. Byte-entropy bins with fine bins near 8 bits (random 4096-byte blocks have
  expected empirical entropy ≈ 7.955 bits; derived constant, give the formula).
- NOT implemented: a byte-entropy threshold baseline for detecting encrypted/random blocks
  (planned in CONTEXT §10.1). Either the coder adds it before freeze, or the paper does not
  mention it.

4.6 Case study (RQ4, 0.15 p here, details in Sect. 5)
- See Sect. 5 for the image; here only the evaluation logic: transferred parameters (temperature,
  LTT thresholds, conformal quantile) unchanged vs. recalibrated on 20 % of the image's labelled
  files (grouped by file), evaluated on the other 80 %, 20 random splits.

## 5 Experimental setup (0.4 p)

- Hardware: training GPU and CPU type from `meta.json:hardware/...` (no server or host names in
  the paper — describe as "one NVIDIA <model> GPU"; check what the tag returns before using it).
- CPU throughput: ONNX Runtime 1 thread and all available threads, batch 256, 20,000 test blocks,
  one warm-up batch; LightGBM time includes feature extraction; each model in a fresh process.
  Why ONNX Runtime and not PyTorch eager (build-dependent conv1d speed) — one sentence.
- Case-study image: 1 GB FAT32, 4096-byte clusters (= block size); zero fillers 32–512 KB except a
  64 MB tail, every second filler deleted, evidence copied in fixed random order (seed 0) so large
  files fragment; quick format with `mkfs.fat`. Ground truth from FAT chains before the format,
  verified by sha256 reassembly; the build checks that the format changed no evidence cluster.
  Evidence: own collected files (types, count, size from `build.json`; sources documented
  separately; nothing personal).
- Labels by extension (aliases), overrides in `labels.csv`; types outside FFT-75 and free space not
  scored for accuracy.
- Carver baseline: header signatures at cluster start, specific before generic, container formats
  give candidate sets; variants "header only" and "propagated" (until next header or zero cluster).
- Seeds, results freeze date, all numbers from result files (one sentence).

Tags (case-study image):
- `data/case_study/images/fat32_1g/build.json:evidence/files|,d`, `.../evidence/clusters|,d`,
  `.../evidence/fragmented_files|,d`, `.../evidence/fragments_max|d`,
  `.../verification/evidence_clusters_changed_by_format|d`
  (open question: this file lives under `data/`, not `results/`; it must be synced from the server
  and the tag path must resolve from `fragtrust/`).

## 6 Results (3.9 p; one subsection per RQ, answer first, then evidence)

6.1 RQ1 Accuracy and cost (0.8 p)
- Answer sentence first: which family is most accurate, by how much, at what CPU cost.
- Table `main_results`: accuracy, macro-F1 (mean ± std over seeds), params, size, train time,
  CPU MB/s at 1 and all threads.
- Capacity: CNN-S/M/L accuracy vs. params → figure `capacity` (or one sentence). Expected
  (hypothesis, do not state as a result): saturation.
- Per-class F1 → figure `per_class_f1`; name the weakest classes / groups (8 coarse groups from
  `configs/data/fft75_groups.yaml`).
- Context: compare with literature accuracy only with citations, no "state of the art" wording.
- SHAP (only if space): Spearman between SHAP and gain importance as a consistency check.

Tags:
- `results/summary/by_model.csv:<model>/accuracy|pct1`, `.../macro_f1|pct1`,
  `.../accuracy_std|pct1`, `.../params|,.0f`, `.../size_mb|.1f`, `.../train_min|.0f`
  with `<model>` ∈ {lgbm, cnn_m, transformer, cnn_m_ens, cnn_s, cnn_l} (row names = model names in
  `by_model.csv`; check once real summary exists)
- `results/runs/<run>/metrics.json:accuracy|pct1`, `...:macro_f1|pct1`
- `results/benchmarks/<host>/<run>.onnx.json:results/0/mb_per_s|.1f` (1 thread; index 1 = all
  threads, order checked on smoke output; LightGBM file is `<run>.lightgbm.json`) — host name
  appears only in the tag path, never in text
- `results/runs/lgbm_s0/analysis/shap.json:spearman_shap_vs_gain|.2f`

6.2 RQ2 Calibration and guarantees on FFT-75 (1.2 p)
- Calibration: ECE / adaptive ECE / NLL / Brier before → after scaling, fitted temperature
  → table `calibration`, figure `reliability`. Direction of miscalibration (over/under-confident)
  from mean confidence vs. accuracy.
- Selective: AURC, coverage at 95/99 % for empirical vs. LTT; does test accuracy of answers meet
  the target? → table `selective`, figure `risk_coverage`. MSP vs. predictive entropy in one
  sentence.
- Conformal: LAC/APS/RAPS at 90/95 %: coverage, mean set size, singleton share, worst-class
  coverage, classes below target → table `conformal`. Which classes fall below target.
- Point to make (if supported): marginal guarantees hold, class-level ones do not.

Tags (per run; ensemble = `cnn_m_ens`):
- `results/runs/<run>/analysis/trust.json:calibration/uncalibrated/ece|.3f`,
  `.../calibration/temperature_scaled/ece|.3f`, `.../ece_adaptive|.3f`, `.../nll|.3f`,
  `.../brier|.3f`, `.../mean_confidence|pct1`, `temperature|.2f`
- `...:selective/msp/aurc|.4f`, `...:selective/msp/targets/0.99/ltt/coverage|pct1`,
  `.../targets/0.99/ltt/selective_accuracy|pct2`, `.../targets/0.99/empirical/meets_target`,
  same for `0.95`, and `selective/neg_entropy/...`
- `...:conformal/aps@0.95/coverage|pct1`, `.../mean_size|.2f`, `.../singleton_frac|pct1`,
  `.../worst_class_coverage|pct1`, `.../classes_below_target|d`; same for `lac@`, `raps@`, `@0.90`
- Seed means: `results/summary/by_model.csv:<model>/ece|.3f`, `/ece_ts`, `/nll_ts`, `/aurc`,
  `/coverage_at_95`, `/coverage_at_99`, `/aps95_coverage`, `/aps95_set_size` (+ `_std`)
- Missing in results (ask coder): `m_min` per target is not written to `trust.json`; needed if
  the text gives the minimum testable set size as a number.

6.3 RQ3 High-entropy and unseen data (0.7 p)
- Accuracy, confidence and coverage at 99 % by byte-entropy bin → figure `entropy`; where accuracy
  drops (top bins) and whether the abstention follows it.
- Probes: AUROC (MSP / entropy / energy), share abstained at 95/99 %, APS-95 set size, which known
  class they are assigned to → table `ood`.
- Honest reading: FFT-75 contains compressed/high-entropy types, so random/AES data resemble
  in-distribution blocks; low AUROC would be expected and is not a model failure by itself.

Tags:
- `results/runs/<run>/analysis/trust.json:ood/ood_aes_ctr/auroc/msp|.3f`, `.../auroc/energy|.3f`,
  `ood/ood_aes_ctr/abstained/0.99|pct1`, `ood/ood_aes_ctr/aps@0.95_sets/mean_size|.2f`,
  `ood/ood_aes_ctr/mean_confidence|pct1`, `ood/ood_aes_ctr/top_predicted/<class>|pct1`;
  same for `ood_random`
- `results/summary/by_model.csv:<model>/auroc_aes|.3f`, `/auroc_random|.3f`
- `results/runs/<run>/analysis/entropy_bins.csv:<bin>/accuracy|pct1` (row key = bin label as
  written, e.g. `[7.95, 8.001)`; awkward in a tag — prefer the figure and cite only 1–2 bins)

6.4 RQ4 Transfer to a formatted FAT32 image (1.2 p)
- Answer first: do transferred parameters meet 95/99 % on the image (accuracy of answers,
  coverage), and how ECE and conformal coverage change vs. FFT-75 test.
- Table `case_study`: all / fragmented / contiguous / header / middle / tail for each model, and
  the carver variants (coverage, accuracy of answers, mean candidate-set size).
- Figure `disk_map`: cluster map of the image (truth vs. prediction vs. abstained).
- Unseen types (e.g. `encrypted`, `unknown`) and free space: confidence, share abstained, set size.
  Free space is all-zero after quick format; how it is labelled (likely a sparse/empty class) must
  be stated, not hidden.
- Table `recalibration`: transferred vs. recalibrated (20 % of files, 20 splits): accuracy, ECE,
  coverage and selective accuracy at 95/99 %, share of splits meeting the target, conformal
  coverage and set size; mean ± std.
- **Vacuity warning**: `meets@t` is true when coverage is 0 (no answers). On smoke data the
  recalibrated 99 % threshold had coverage 0 and `meets@0.99` = 1.0 — small calibration sets cannot
  certify 99 % (m_min). Always report coverage next to "meets"; never write "recalibration
  restores the guarantee" if recalibrated coverage is near 0. Flag for coder: consider reporting
  the share of splits with coverage > 0.

Tags (run = `cnn_m_s0`, `lgbm_s0`, `transformer_s0`):
- `results/case_study/fat32_1g/<run>/case_study.json:transferred/all/accuracy|pct1`,
  `.../all/ece|.3f`, `.../all/coverage@0.99|pct1`, `.../all/selective_accuracy@0.99|pct2`,
  `.../all/meets@0.99`, `.../all/aps@0.95_coverage|pct1`, `.../all/aps@0.95_size|.2f`;
  same with `fragmented`, `contiguous`, `header`, `middle`, `tail`
- `...:signatures/header/all/coverage|pct1`, `signatures/propagated/all/accuracy_answered|pct1`,
  `.../mean_set_size_answered|.2f`, `.../correct_overall|pct1`; per group as above
- `...:unseen_types/encrypted/abstained@0.99|pct1`, `.../mean_confidence|pct1`,
  `not_evidence/abstained@0.99|pct1`, `not_evidence/top_predicted/<class>|pct1`
- `...:recalibration/recalibrated/coverage@0.99/mean|pct1`, `.../std|pct1`,
  `recalibration/recalibrated/meets@0.99/mean|pct0`, `recalibration/transferred/...` same keys,
  `recalibration/recalibrated/cal_clusters/mean|,.0f`, `recalibration/files|d`
- `...:evidence_clusters_known_type|,d`, `clusters|,d`, `macro_f1|pct1`

## 7 Discussion and limitations (0.8 p)

- Examiner guidance in 3–4 sentences: which operating point to use, what a conformal set means
  in a report, when to recalibrate (only if supported by 6.2–6.4).
- Limitations, stated plainly:
  - one image, one file system (FAT32), one collection of evidence files chosen by us; 4096-byte
    clusters equal to the block size (other cluster sizes not tested);
  - clusters of one file are correlated, so exchangeability (needed by LTT and conformal) is
    violated on the image; recalibration splits by file to limit this, but the guarantee is then
    empirical, not formal;
  - LTT guarantee is with probability 1 − δ over the calibration draw; one test pass that meets
    the target is consistent with it but does not prove it;
  - conformal coverage is marginal, not per class;
  - leakage: test blocks duplicated in train were not removed from the test set;
  - training subset (20,000 per class) instead of the full 81,920; few seeds for S/L;
  - no adversarial / anti-forensic data; probes are synthetic.
- Threats to validity: label by extension; FFT-75 class list taken from FiFTy's labels file.

## 8 Conclusion (0.3 p)

- Two or three sentences answering RQ1–RQ4 with the main tags repeated sparingly.
- Future work (only things not done): other file systems and cluster sizes, class-conditional
  conformal, larger real-media corpora, deduplicated FFT-75 test.

## References (1.1 p, 20–25)

Only from `team/research/` and `literature/INDEX.md` (both empty on 2026-11-27). All references
above are `[CITE: ...]` placeholders until R2/H1 are done.

---

## Open questions / requests

1. Coder: `m_min` per target is not in `trust.json`; add it if the text should give it as a number.
2. Coder: `meets_target` / `meets@t` is true at zero coverage; add an explicit coverage-> 0 flag or
   the share of recalibration splits with coverage > 0.
3. Coder: byte-entropy threshold baseline for probes (CONTEXT §10.1) is not implemented; add or
   drop from the plan.
4. Case-study `build.json` is under `data/case_study/images/fat32_1g/`, not `results/`; sync it and
   confirm the tag path resolves.
5. Researcher (R1): result of the novelty check decides whether C1/C2 keep "first" and whether
   C3 becomes the lead contribution.
