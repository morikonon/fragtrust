# FFT-75 dataset

Checked 2026-11-22. "Code" = read in the FiFTy source, not in a published specification.

| Item | Value | Source |
|---|---|---|
| Name, DOI | File Fragment Type (FFT) – 75 Dataset, 10.21227/kfxw-8084 | IEEE DataPort page |
| Authors | Govind Mittal, Pawel Korus, Nasir Memon | IEEE DataPort page |
| Licence | Creative Commons Attribution (link to CC BY 4.0) | IEEE DataPort page |
| Access | files behind a (free) IEEE account login; the page also says "requires an IEEE DataPort Subscription": inconsistent, confirm when logging in | IEEE DataPort page |
| Files | `4k_1.tar.gz` … `4k_6.tar.gz`, `512_1.tar.gz` … `512_6.tar.gz`; scenario 1, 4096 B: 24.18 GB | IEEE DataPort page |
| Split | 80 % train / 10 % validation / 10 % test, at block level | DataPort abstract; code |
| Inside the archive | `4k_1/train.npz`, `4k_1/val.npz`, `4k_1/test.npz` | FiFTy README |
| Arrays | `x`: uint8 (n, block_size); `y`: integer label = index into the scenario's class list | code (`train.py`, `framework.py`) |
| Compression | written with `np.savez_compressed` | code |
| Class order, scenario 1 | `configs/data/fft75_s1_classes.json` | `fifty/utilities/labels.json`, key `"1"` |
| Code licence | Apache-2.0 | github.com/mittalgovind/fifty |

Sources:
- https://ieee-dataport.org/open-access/file-fragment-type-fft-75-dataset
- https://github.com/mittalgovind/fifty (README, `fifty/utilities/labels.json`, `fifty/commands/train.py`, `fifty/utilities/framework.py`)

## To verify after the download

1. `prepare.log` prints the array keys, row counts and number of classes of each file. Expected for
   scenario 1 at 4096 B: 75 classes, 102,400 blocks per class in total (verify the split sizes).
2. Compare `configs/data/fft75_s1_classes.json` with the "Human-readable labels" file on DataPort.
3. Record the licence text shown on the download page.

## Processed subset `fft75_4096_s1_v1`

Built by `fragtrust prepare --config configs/data/fft75_4096_s1_v1.yaml` (seed 2026): 20,000 training
blocks per class, 4,000 validation blocks per class split into two halves (val_a, val_b), and the full
official test split. Row indices into the official files are saved in `processed/.../indices/`, and
`manifest.json` holds the sha256 of every input and output file. `test_duplicates.npz` flags every test
block that also occurs in the official train split (`official_train`) and in our 20,000-per-class
training subset (`subset_train`); for a subset prepared before this file existed,
`fragtrust duplicates --config configs/data/fft75_4096_s1_v1.yaml` adds it from the official
`train.npz` without rebuilding anything.
