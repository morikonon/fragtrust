# Case-study evidence files

The case study asks what an examiner gets from the models on a realistic exhibit: a 1 GB USB stick
(FAT32, 4096-byte clusters) holding ordinary files, quick-formatted before seizure. Nothing is
visible in the file system any more, but the data clusters are intact. Every cluster is classified
without file-system metadata and compared with ground truth recorded before the format.

This folder holds the evidence files (`files/`, never committed) and the built images (`images/`,
never committed). Only this README is in git.

## 1. Collect the files (any OS)

Put the files under `data/case_study/files/`, in any subfolders and with any names (they are renamed
to `CASE/F0001.ext` inside the image). Hidden files (names starting with `.`) are skipped.

**Amount.** 300–450 MB in total. The 1 GB image has about 500 MB of free space after the filler
pattern; the build stops with a clear message if the files do not fit.

**Variety.** The models know the 75 FFT-75 types in `configs/data/fft75_s1_classes.json`. Aim for
at least 25 of them, roughly what a personal USB stick holds, and keep any single type below a
quarter of the bytes (videos fill space fast: two or three short clips at most).

| Group | Useful types (extension = label) |
|---|---|
| image | jpg, heic, png, gif, bmp, tiff; camera raw if available: cr2, nef, arw, dng, orf, rw2, raf; psd, ai, eps |
| video | mp4, mov, 3gp, avi, mkv, webm |
| audio | mp3, m4a, wav, flac, ogg, wma, aiff |
| document | pdf, docx, doc, xlsx, xls, pptx, ppt, epub, mobi, djvu |
| text | txt, md, csv, json, xml, html, log, rtf, tex |
| archive and software | zip, 7z, gz, bz2, xz, rar, jar, apk, deb, rpm, exe, dll, elf binaries |
| other | ttf, sqlite, pcap, dwg |

**Sizes.** Mostly 10 KB – 20 MB per file. Files larger than about 0.5 MB get split into fragments in
the image, which is the point; files under 4 KB fill a single cluster.

**Types the models never saw.** Add two or three encrypted files, for example

```bash
openssl enc -aes-256-cbc -pbkdf2 -in some_archive.zip -out secret.bin
```

and label them `encrypted` in `labels.csv` (below). A file whose extension is not among the 75 types
(say `.odt` or `.iso`) is labelled `unknown` automatically; one or two of those are welcome too.

**Rights and anonymity.** Only files you made yourself or that are openly licensed (public domain,
CC0, CC BY: Wikimedia Commons, Project Gutenberg, public-domain audio, open-source binaries). No
personal data of other people and nothing confidential. The paper is double-blind: remove anything
that names the authors (document properties such as Author, photo EXIF with owner name or GPS).

## 2. Describe them

`files/SOURCES.md`: where each file or folder comes from and under which licence; the paper's data
statement is written from it.

```markdown
| Path | Source | Licence | Notes |
|---|---|---|---|
| photos/ | own phone photos | own | EXIF location removed |
| books/*.epub | Project Gutenberg | public domain | |
| secret.bin | openssl enc of a zip of photos/ | own | label: encrypted |
```

`files/labels.csv` (optional): labels for files whose extension does not give the type. Paths are
relative to `files/`, with forward slashes; separate the two columns with a comma, not a semicolon
(spreadsheet programs with a Russian locale save `;`, so write the file in a text editor).

```
path,label
secret.bin,encrypted
tools/hello,elf
```

A label is one of the 75 class names, `encrypted` or `unknown`. Without an entry the label comes
from the extension, case-insensitive, with `jpeg→jpg`, `tif→tiff`, `htm→html`, `markdown→md`,
`tgz→gz`, `text→txt`; any other extension gives `unknown`. A path in `labels.csv` that does not exist
stops the build.

## 3. Build the image and run the case study (Linux server)

The build needs `mkfs.fat` (dosfstools) and mtools, which `scripts/setup_server.sh` installs; it does
not need root or mounting. Copy the folder to the server, then run the whole case study after the
base plan has finished:

```bash
rsync -a data/case_study/files/ SERVER:fragtrust/data/case_study/files/
```

```bash
bash scripts/run_case_study.sh
```

The script builds the image if `data/case_study/images/fat32_1g/build.json` does not exist, scans
every cluster with each model listed under `case_study.runs` in `configs/paper.yaml`, evaluates them
and rebuilds the paper assets. To build the image alone:

```bash
python -m fragtrust.cli case-build --config configs/case_study/fat32_1g.yaml
```

What the build does (`src/fragtrust/case_study/build.py`): create an empty FAT32 image; fill it with
zero-filled filler files of 32–512 KB except a 64 MB tail; delete every second filler, leaving holes;
copy the evidence files in a fixed random order (seed 0), so large files are split across holes;
read the cluster chains back from the FAT and check that every file reassembles to its sha256; quick
format the image with `mkfs.fat` and check that no evidence cluster changed.

Output in `data/case_study/images/fat32_1g/`:

| File | Content |
|---|---|
| `image.img` | the quick-formatted image the models scan |
| `image_preformat.img` | the same image before the format (kept for verification) |
| `ground_truth.csv` | one row per data cluster: file, label, index in file, header/middle/tail, bytes used, fragmented |
| `files.csv` | one row per evidence file: source path, label, size, sha256, clusters, fragments |
| `build.json` | geometry, fragmentation statistics, label counts, tool versions, verification, image sha256 |
| `build.log` | the build log |

`build.json → verification` must read: every file reassembled, `evidence_clusters_changed_by_format`
0, `files_after_format` 0. To rebuild, delete the image folder first.
