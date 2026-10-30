# C3 face-tracking benchmark datasets

**Scope:** IEEE SP Cup 2027 C3 — tracking one designated face among bystanders. This is a dataset acquisition/preparation guide, not evidence of benchmark performance.

For executable video-aligned AVA preparation, leakage-separated inference/evaluation,
MOT/C3 metrics, oracle experiments and Windows commands, see
[BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md). The older preparation script below produces
metadata only; the benchmark uses a stricter common schema with source frame indices.

## Recommended sources and limitations

| Dataset | Official source | Useful annotations | C3 use | Availability / caution |
| --- | --- | --- | --- | --- |
| **AVA ActiveSpeaker v1.0** (start here) | https://sites.research.google/gr/ava/download/ ; https://github.com/cvdfoundation/ava-dataset | Time-indexed normalized face bounding boxes and `entity_id` linking adjacent frames | Multi-face association, short-term ID switches, occlusion within a track | Annotation archives public; video files are hosted separately. `entity_id` is **not** a guaranteed persistent identity across scene changes. Movie rights belong to their respective owners. |
| **VPCD** | https://www.robots.ox.ac.uk/~vgg/data/Video_Person_Clustering/ | >35k face-tracks labeled with character/person identity | Longer-term identity matching, reappearance | Annotation/features can be downloaded; source video availability and rights need review separately; face-track metadata may not yield full per-frame bounding-box ground truth without preprocessing. |
| **MobiFace** | https://mobiface.github.io/ | >95k annotated target-face bounding boxes in 80 mobile recordings | Tracking robustness, motion, pose, mobile capture | **Single-target annotations**; videos can contain multiple faces. Bystanders are not exhaustively labeled. Request access with institutional academic email; do not mirror. |
| **ChokePoint** | http://arma.sourceforge.net/chokepoint/ ; https://zenodo.org/records/815657 | Verified XML person IDs and left/right eye points; no native face boxes | Eye-containment and prediction-ID reuse diagnostics | P1E_S1 camera views tested with the separate eye-point command. Do not infer box IoU, standard IDF1/HOTA, or physical absence from these labels. See [additional dataset validation](ADDITIONAL_DATASET_VALIDATION.md). |
| **Self-annotated permitted videos** | Local, not distributed in Git | Framewise face boxes + physical-person identity | Target-specific C3 error testing | Follow organizer ethics/IRB and informed-consent rules for participant-collected competition data. Do not commit identifiable recordings. |

**Avoid as primary benchmarks:** MOT17/SportsMOT (pedestrian or player tracking, not face tracks); UBFC-rPPG/PURE (usually single-person physiology); MobiFace alone (one annotated target, no exhaustive bystander identities). These can support separate secondary tests.

## Recommended order

1. **AVA ActiveSpeaker**: inspect annotations first, filter sequences with >=2 simultaneous face entities, then get matching video clips. Measure track association only on annotated intervals.
2. **VPCD**: identity persistence and re-entry, subject to footage availability. Review its identity/track formats before implementing a loader.
3. **MobiFace**: request institutional access, evaluate target bbox stability and robustness to mobile motion.
4. **ChokePoint**: inspect sequence annotation completeness and use only validated clips for supervised evaluation.

## Expected local layout

```text
data/
  raw/
    ava_active_speaker/
      annotations/       # optional extracted CSV files
      videos/             # video_id.mp4 matched to annotations
    vpcd/
    mobiface/
    chokepoint/
  processed/
    ava_active_speaker/
      annotations.csv     # normalized common schema
      multi_face_timestamps.csv
```

All datasets/data directories and downloaded archives must remain **untracked**. This branch adds ignore rules; if an existing .gitignore exists after merge, retain them.

## AVA annotation preparation

Obtain the training/validation annotation archives from the **official AVA page** (linked above). They are typically called `ava_activespeaker_train_v1.0.tar.bz2` and `ava_activespeaker_val_v1.0.tar.bz2`. Extract locally or pass their directories to:

```powershell
py -3.11 datasets/scripts/prepare_ava_annotations.py --input "datasets/data/raw/ava_active_speaker/annotations" --output "datasets/data/processed/ava_active_speaker"
```

The script reads all CSVs under the directory, supports official rows with either 5 CSV fields (one encoded box) or 8 fields (four coordinates in separate CSV fields), validates normalized boxes, and writes:

- `annotations.csv`: `video_id,timestamp_s,entity_id,x1,y1,x2,y2,label,source_file`
- `multi_face_timestamps.csv`: `video_id,timestamp_s,num_faces`, restricted to timestamps with >=2 unique face IDs
- `summary.json`: input/output counts and invalid-row counts

**The output is metadata only.** Actual videos must be acquired separately and stored under `datasets/data/raw/ava_active_speaker/videos`. Check that the video filename and timestamps correspond to annotation `video_id` before running the tracker.

The output `entity_id` is a local annotation track identifier. **Do not treat it as a verified biometric identity after a cut/disappearance.** For designated-target scoring after reappearance, use an independently verified identity mapping.

## Suggested evaluation protocol

- Choose a subject only at an annotated time where at least two different face boxes coexist.
- Annotate the selected subject's true identity across the evaluation interval (manually validate reappearance if needed).
- A tracker that produces an ID when no target face is present is **not** automatically correct.
- Evaluate target false acceptance (bystander assigned as target), missed visible target, loss/reacquisition, and ID switches against independently checked annotations.
- Assess spatial localization separately using IoU.
- Compare identical video segments and resolution/fps conditions, including 640x480, 320x240, and 10fps stress conditions from the competition.
- Don't use missing annotated frames as proof of absent faces; annotation gaps may exist.
- For physiological claims, add a separate rPPG dataset and ground-truth HR study. AVA/VPCD have no standard synchronized PPG ground truth.

## Legal and scientific safeguards

- Follow each dataset's license and usage restrictions; a public annotations link does not automatically permit redistribution of movie footage.
- Do not commit or redistribute third-party videos, participant faces, personal identifiers, or pretrained weights without permission.
- The official competition PDF requires IRB approval and informed consent for newly participant-collected competition datasets.
- Record the exact dataset release, selected video IDs, annotation extraction date, excluded intervals, and benchmark splits.
- Avoid selecting/tuning candidate models on the held-out evaluation videos.

## References

- AVA ActiveSpeaker official description and format: https://sites.research.google/gr/ava/download/
- CVDF official download mirror: https://github.com/cvdfoundation/ava-dataset
- VPCD: https://www.robots.ox.ac.uk/~vgg/data/Video_Person_Clustering/
- MobiFace: https://mobiface.github.io/
- ChokePoint: http://arma.sourceforge.net/chokepoint/

## Verified MobiFace Google Drive source (2026-10-30)

We inspected the shared [MobiFace80 Drive folder](https://drive.google.com/drive/folders/1LrVrx_InOo3og9RfCynQkMkfaVydn7ra). It contains metadata, train/test per-sequence target-face box CSVs, a toolkit, and reference results. The verified native columns are `frame,left,top,width,height` and the video metadata gives YouTube source ID, dimensions, FPS, and original frame interval. See [MobiFace source and acquisition guide](mobiface/README.md) and [source inventory](mobiface/source.json).

**No raw dataset files are committed, no matching video footage has been verified, and the benchmark does not yet implement a native MobiFace loader.** Single-target annotations cannot support exhaustive bystander false-positive or identity metrics.
