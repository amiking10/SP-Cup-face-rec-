# MobiFace80 — verified Google Drive source

**Status: source registered; dataset bytes and original videos are NOT in Git.**

- [Shared Google Drive folder](https://drive.google.com/drive/folders/1LrVrx_InOo3og9RfCynQkMkfaVydn7ra)
- [MobiFace project page](https://mobiface.github.io/)
- [Source inventory](source.json)

The connected Drive folder was inspected on **2026-10-30**. It contains `mobiface-toolkit.zip`, `fg2019_poster.pdf`, `mobiface80.meta.csv`, and the `mobiface80/` subtree with `train.meta.csv`, `test.meta.csv`, `train/`, `test/`, and `benchmark_results/`. Individual `*.annot.csv` files are visible under `train/` and `test/`.

## Verified native columns

`mobiface80.meta.csv`:
```csv
name,youtube_video_id,width,height,frame_rate,first_frame,last_frame
```

Sample `mobiface80/test/7I5t6BAHSGQ_0.annot.csv`:
```csv
frame,left,top,width,height
0,76,89,99,160
1,77,89,99,160
```

The annotation file supplies an annotated **target face box** per listed frame in pixel left/top/width/height coordinates. The dataset is **single-target-annotated**, not necessarily single-person video: other visible people may not have annotations. It does **not** support treating every unannotated face as a false positive or computing exhaustive multi-face MOT metrics. Verify handling of absence and gaps on more sequences before normalizing.

## Local acquisition

1. Open the [shared folder](https://drive.google.com/drive/folders/1LrVrx_InOo3og9RfCynQkMkfaVydn7ra) in your signed-in browser.
2. Download `mobiface80.meta.csv`, `mobiface80/train.meta.csv`, `mobiface80/test.meta.csv`, and the corresponding `*.annot.csv` files or permitted archive.
3. Keep the extracted files under `datasets/data/raw/mobiface/` while preserving their relative paths, for example:

```text
datasets/data/raw/mobiface/
  mobiface80.meta.csv
  mobiface80/
    train.meta.csv
    test.meta.csv
    train/<sequence>.annot.csv
    test/<sequence>.annot.csv
```

4. Inspect the dataset's terms and obtain the **matching source video clips** through an authorized method. The metadata points to YouTube video IDs and original `first_frame`/`last_frame` ranges. The inspected folder has not been verified to contain the actual videos. Do not bypass access restrictions or assume all YouTube videos remain obtainable.
5. Preserve source FPS/frame indices, confirm annotation-to-video alignment, and keep a clear train/test split.

The repository's `datasets/data/.gitignore` excludes local recordings/annotations. No external data is embedded in this PR.

## Next development milestone (not implemented in this source-only PR)

Write a strict MobiFace adapter that translates native `frame,left,top,width,height` to `x1,y1,x2,y2`, validates resolution, timestamps, original frame offsets, annotation gaps and splits, and feeds **raw video only** to the baseline before comparing predictions with GT. Calculate target-box IoU/recall/retention; do not claim bystander rejection, IDF1, HOTA, physical reacquisition, or full detection precision without exhaustive identities/labels.
