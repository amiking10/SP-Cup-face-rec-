# Downloading two more C3 datasets

We provide a local download helper, not dataset mirrors. **No files are automatically
downloaded by adding this script to GitHub.** Run these commands on your own machine.

## 1. ChokePoint (re-identification, including crowded portal sequences)

Source: https://zenodo.org/records/815657

Inspect the files on the verified Zenodo record:

```powershell
python scripts/download_c3_datasets.py chokepoint --list
```

Download **groundtruth.tar.xz** (~0.5 MB) plus **P2E_S5.tar.xz** (~180 MB, crowded
sequence with possible occlusions):

```powershell
python scripts/download_c3_datasets.py chokepoint
```

For another crowded sequence:

```powershell
python scripts/download_c3_datasets.py chokepoint --files P2L_S5.tar.xz
```

The official dataset documentation states that most ChokePoint sequences contain
only one subject at a time; the S5 crowded sequences are special cases. Do not
assume the crowded sequences have complete per-person face-box annotations.
The script checks the Zenodo MD5 checksum where present.

## 2. VPCD (identity-labeled face tracks across movies/shows)

Source: https://www.robots.ox.ac.uk/~vgg/data/Video_Person_Clustering/

From the official page, click **All Annotations, Features and Labels in VPCD**.
Copy the actual download/archive URL and run:

```powershell
python scripts/download_c3_datasets.py vpcd --vpcd-url "PASTE_OFFICIAL_ARCHIVE_URL_HERE"
```

VPCD labels face tracks with identity, but this is **metadata**, not an
automatically downloadable set of reusable movie videos. The original owners
retain copyright in footage. Verify annotations against footage before
reporting tracking benchmark performance.

## Extraction and validation

Use 7-Zip on Windows to extract tar.xz/tar.gz. Never run or trust files from
unverified mirrors. Keep all downloaded archives, extracted faces, and videos
under `data/` (ignored by Git).

Before scoring, inspect that your video frames align to the annotations,
identify missing face boxes, and independently verify identity after a
disappearance. For AVA, also see [DATASETS.md](DATASETS.md).

## Notes

- VPCD archive URL is intentionally supplied explicitly; the Oxford download
  site can change its file links, and this script does not invent a direct URL.
- ChokePoint files can be very large; do not download all by default.
- MobiFace is **not** automatically downloadable: it requires institutional
  approval via https://mobiface.github.io/.
- This script is network-dependent. It has not been executed against the
  Zenodo/Oxford download endpoints in a live Windows environment.
