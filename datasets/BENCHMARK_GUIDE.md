# Reproducible C3 benchmark

This benchmark measures the existing tracker without training or tuning it. Normal
inference receives decoded video frames only. Ground truth enters once for a recorded
target initialization, then is used **after inference** for scoring and visualization.
An optional oracle experiment supplies ground-truth **boxes, never identities**, and
is reported separately. Synthetic tests establish infrastructure behavior, not real
face-tracking performance.

## Install on Windows

From the repository root in PowerShell, using the existing Python 3.11 environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test,benchmark]"
.\.venv\Scripts\python.exe -m c3_tracker.models --download
.\.venv\Scripts\python.exe -m c3_tracker.evaluate --help
```

The benchmark extra pins motmetrics 1.4.0 and pandas 2.2.3. The tracker keeps its
NumPy 1.26.4 pin. For optional reference HOTA, use a separate environment because
TrackEval 1.3.0 requires NumPy 2 on Python 3.11:

```powershell
.\.venv\Scripts\python.exe -m venv .metrics-venv
.\.metrics-venv\Scripts\python.exe -m pip install -r datasets/requirements-hota.txt
```

Pass `--hota-python .metrics-venv\Scripts\python.exe` when running a benchmark.
Without it HOTA is null with an explicit reason. TrackEval receives only post-inference
metric inputs in a local subprocess; its dependencies never replace the tracker stack.

## Dataset support and required files

| Dataset | Official source | Implemented support | Required mapping / limitation |
| --- | --- | --- | --- |
| AVA ActiveSpeaker v1.0 | [Google format/release](https://sites.research.google/gr/ava/download/), [CVDF videos/archives](https://github.com/cvdfoundation/ava-dataset) | Verified 8-field rows and explicitly encoded 5-field boxes; strict video-aware normalization | `entity_id` is a track-local face label. It is not a persistent physical identity across cuts or disappearance. AVA Actions body-box CSVs are rejected. |
| ChokePoint | [Official site](http://arma.sourceforge.net/chokepoint/), [official archive](https://zenodo.org/records/815657) | Verified native XML eye-point diagnostics for P1E_S1; separate command below | Person IDs and eye points do not supply face boxes. No native box-IoU/MOT loader is claimed. Bystanders can be visible without labels. |
| VPCD | [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/Video_Person_Clustering/) | No native loader claimed | Obtain matching footage as well as track/person labels; verify exact time/frame/box metadata and character-to-person semantics. Track-level labels alone do not supply exhaustive per-frame boxes. |
| MobiFace80 | [Provided source and native guide](mobiface/README.md), [official project](https://mobiface.github.io/) | Native metadata/annotation audit, explicit video alignment, common-schema preparation and evaluation | Supplied 64-train/16-test split preserved. One test sequence evaluated; single-target labels do not certify exhaustive bystander coverage or physical-person identities. |
| Verified local common format | Locally permitted footage | Supported through `--dataset common --manifest ...` | Explicit coverage and identity semantics are mandatory. Persistent identities require a recorded provenance. |

No native support is inferred from dataset names. The CLI returns a useful error for
unverified face-box ChokePoint/VPCD loaders. MobiFace uses `audit-mobiface`,
`prepare-mobiface`, then `--dataset mobiface`; its [native guide](mobiface/README.md)
defines the required files and explicit clip/source alignment. ChokePoint eye diagnostics use a
separate metric family and entry point; see [additional dataset validation](ADDITIONAL_DATASET_VALIDATION.md).
Do not download, mirror or commit footage
whose source terms do not permit your intended use. These datasets do not provide
synchronized physiological ground truth for HR/SpO2/blood-pressure claims.

For a verified conversion, a common manifest looks like this (paths are relative to
the manifest; these are placeholder filenames, not a claim about an unverified native release):

```json
{
  "schema_version": 1,
  "dataset": "common",
  "release": "locally-verified-conversion-v1",
  "videos": [{
    "video_id": "clip_01",
    "video_path": "../../raw/my_dataset/clip_01.mp4",
    "annotations_path": "clip_01/annotations.csv",
    "video_start_timestamp_s": 0,
    "cfr": true,
    "coverage": "complete",
    "coverage_provenance": "Record the actual annotation coverage audit here",
    "identity_scope": "persistent",
    "identity_annotations_complete": true,
    "identity_provenance": "Record the independently verified physical-person mapping here",
    "absence_annotations_complete": true,
    "known_empty_frames": []
  }]
}
```

Use `partial`, `track_local` or `none`, and false completeness flags where those
stronger claims are not established. Empty frames must be explicit, not inferred
from missing annotation rows. Then run:

```powershell
.\.venv\Scripts\python.exe -m c3_tracker.evaluate --dataset common --manifest datasets/data/processed/common/manifest.json --output results/common_baseline
```

## Local layout and AVA download

All dataset material stays in ignored `datasets/data/`; benchmark results stay in ignored `results/`.

```text
datasets/data/
  raw/ava_active_speaker/
    ava_activespeaker_val_v1.0.tar.bz2
    annotations/...-activespeaker.csv
    videos/<video_id>.mkv
    manifest.json
  processed/ava_active_speaker/
    manifest.json
    <video_id>/annotations.csv
```

The following commands download the official validation annotations and one matching
video (~165 MB). The source video is a full movie, not a clip starting at timestamp 900.
The archive's internal directory name may contain `test`; the URL/release is recorded as validation.

```powershell
$avaRoot = "datasets/data/raw/ava_active_speaker"
New-Item -ItemType Directory -Force "$avaRoot/annotations", "$avaRoot/videos" | Out-Null
Invoke-WebRequest "https://s3.amazonaws.com/ava-dataset/annotations/ava_activespeaker_val_v1.0.tar.bz2" -OutFile "$avaRoot/ava_activespeaker_val_v1.0.tar.bz2"
tar -xjf "$avaRoot/ava_activespeaker_val_v1.0.tar.bz2" -C "$avaRoot/annotations"
Invoke-WebRequest "https://s3.amazonaws.com/ava-dataset/trainval/053oq2xB3oU.mkv" -OutFile "$avaRoot/videos/053oq2xB3oU.mkv"
Get-ChildItem "$avaRoot/annotations" -Recurse -Filter "053oq2xB3oU-activespeaker.csv"
```

For other videos consult the [official filename list](https://s3.amazonaws.com/ava-dataset/annotations/ava_speech_file_names_v1.txt).
Train annotations use the analogous `ava_activespeaker_train_v1.0.tar.bz2` URL. Obtain
annotations and matching footage separately; the runner does not download data implicitly.

## Prepare AVA annotations without guessing alignment

Create `datasets/data/raw/ava_active_speaker/manifest.json`. Adjust `annotation_path`
to the extracted file reported above. This concrete example evaluates source frames
22500–22749 (900–910 seconds) of the 25 FPS, 480x360 source video:

```json
{
  "dataset": "ava",
  "release": "v1.0",
  "split": "validation",
  "videos": [{
    "video_id": "053oq2xB3oU",
    "video_path": "videos/053oq2xB3oU.mkv",
    "annotation_path": "annotations/ava_activespeaker_test_v1.0/053oq2xB3oU-activespeaker.csv",
    "video_start_timestamp_s": 0.0,
    "cfr": true,
    "annotation_window_s": [900, 910],
    "start_frame": 22500,
    "end_frame_exclusive": 22750,
    "coverage": "partial",
    "identity_annotations_complete": true,
    "coverage_note": "Released annotated entities; exhaustive visible-face coverage not independently certified"
  }]
}
```

`video_start_timestamp_s` is the original annotation timestamp corresponding to local
video frame zero. A separately cut clip starting at original second 900 needs offset
900, not zero. `cfr: true` is an explicit declaration; positive decoder timestamps are
also checked during processing. VFR media requires a PTS-aware conversion and annotation remapping.

The strict default alignment tolerance is 5.1 ms. The actual example annotation clock
has up to 10 ms jitter relative to the verified 25 FPS video. This explicit tolerance
was checked to map 1,020 boxes onto 250 unique source frames, with no entity/frame collisions:

```powershell
.\.venv\Scripts\python.exe -m c3_tracker.evaluate prepare --manifest "$avaRoot/manifest.json" --output "datasets/data/processed/ava_active_speaker" --alignment-tolerance 0.0101
```

The adapter rejects out-of-range timestamps, excessive alignment error, conflicting
or collapsed entity/frame rows, invalid boxes, wrong video IDs, and invalid speech labels.
It never silently drops malformed rows. `annotation_window_s` is the only explicit
annotation-time filter; excluded counts are recorded. Preparation records source and
annotation SHA256 hashes, resolution, FPS, frame count, offset and maximum time error.
Changing a prepared video or annotation file requires preparation again into a fresh directory.

Do not increase tolerance until rows happen to fit. Verify the actual media clock,
bound errors below half a source frame, inspect overlay alignment, and retain the
alignment metadata. If deliberate resampling is needed, this optional video-only helper
records every nearest-source-frame mapping and original/output hashes:

```powershell
.\.venv\Scripts\python.exe -m c3_tracker.evaluate prepare-video --input "$avaRoot/videos/053oq2xB3oU.mkv" --output "$avaRoot/videos/clip_30fps.mp4" --start-seconds 900 --duration 10 --fps 30
```

This does **not** make mismatched annotations magically valid. It changes the inference
video and must be declared in the manifest and rechecked. Native video is used for the
concrete example above. The old `datasets/scripts/prepare_ava_annotations.py` remains a
metadata-inspection utility; its normalized coordinates without frame indices are not
the benchmark's video-aligned common format.

If metadata inspection alone is needed, keep its output separate from prepared video GT:

```powershell
.\.venv\Scripts\python.exe datasets/scripts/prepare_ava_annotations.py --input datasets/data/raw/ava_active_speaker/annotations --output datasets/data/processed/ava_metadata
```

Do not pass `ava_metadata` to the benchmark as video-aligned annotations.

## Run the complete existing baseline

```powershell
.\.venv\Scripts\python.exe -m c3_tracker.evaluate --dataset ava --data-root datasets/data --output results/ava_baseline --annotate-video
```

The default is the current MediaPipe + aligned SFace + DeepSORT stack, including its
YuNet landmark alignment and selected-face safeguards. SFace runs on CPU. The optional
MobileNet backend uses the upstream ImageNet appearance model. Exact options, code hash,
source hashes, model/library versions, hardware, actual device and initialization audit
are saved. No evaluation-set fine tuning takes place. Seed defaults to 0 and OpenCV
threads to 1; GPU/third-party execution is not promised bit-identical across hardware.

Use `--video-id ID` repeatedly to select manifest entries, `--max-frames N` for an explicit
smoke-test truncation, and `--no-target` for an undirected MOT experiment. These choices
are recorded. Outputs must use a fresh/empty directory; previous results are never overwritten.

Run the same registered sequences at each requested operating condition:

```powershell
.\.venv\Scripts\python.exe -m c3_tracker.evaluate --dataset ava --data-root datasets/data --output results/ava_conditions --matrix
```

This creates six experiments: original resolution, 640x480 and 320x240, each at native
FPS and 10 FPS. Resize uses explicit x/y scaling, not an undisclosed letterbox. The
10 FPS scheduler chooses nearest source frame indices on a video-zero-based grid;
timestamps remain the real source-frame times. It never interpolates boxes or scores
an unannotated frame as empty. Temporal gaps break metric continuity. Requested FPS
above source FPS is rejected; use `--resolution` and `--fps` for individual conditions.

For post-inference stage scoring, add `--stage-diagnostics`. This captures actual
baseline detector output, successful descriptors, tentative/confirmed observations,
and a separate low-confidence detector probe. Its explicit confidence-eligible subset
is diagnostic only: the baseline's input is preserved, including API-returned scores
below the configured argument. Instrumented throughput includes probe overhead.
See [AVA diagnostics](AVA_DIAGNOSTICS.md) for the six-condition counts, exact
initialization failure and reproducible commands.

## Ground-truth-box oracle experiment

Oracle and end-to-end results must remain separate. The existing aligned SFace path
uses a face detector for landmarks and may drop unalignable boxes; calling it a pure
detector-bypass experiment would be misleading. The oracle therefore requires the
existing MobileNet crop embedder, which consumes the supplied boxes without a face detector.
Use the **same MobileNet backend for both runs** when isolating the detector's contribution:

```powershell
.\.venv\Scripts\python.exe -m c3_tracker.evaluate --dataset ava --data-root datasets/data --appearance-model mobilenet --output results/ava_mobilenet_end_to_end
.\.venv\Scripts\python.exe -m c3_tracker.evaluate --dataset ava --data-root datasets/data --appearance-model mobilenet --experiment oracle --output results/ava_mobilenet_oracle --annotate-video
```

Oracle inference accepts boxes only (confidence 1.0, constant face class); no annotation
person/entity ID reaches DeepSORT, including through list order: boxes are ordered by
coordinates, independently of annotation labels. It requires exhaustive boxes on **every processed
frame**, including explicitly known empty frames. Partial coverage or unknown-frame gaps
raise an error; select a verified dense interval rather than inventing empty detections.
The illustrative partial-coverage manifest above must first be reviewed and certified
before an oracle run is valid. Comparing SFace end-to-end against MobileNet oracle also
changes the appearance model and is not a clean detector ablation.

## Coverage, identities and target initialization

`coverage: partial` is the conservative default. Recall and matched IoU are measured
against provided faces; unmatched predictions may be unlabeled faces. Therefore precision,
false-positive count, IDSW, IDF1, fragmentation and HOTA are null where exhaustive
coverage/identity validity is not established. `unmatched_predictions` is still reported.

To enable complete-coverage metrics, review the selected interval and explicitly set:

```json
"coverage": "complete",
"coverage_provenance": "Describe release, interval and how every visible face/empty frame was checked",
"identity_annotations_complete": true
```

Completeness is a user-supplied scientific assertion, not something inferred from a
nonempty CSV. AVA still has `identity_scope: track_local`; physical-person reacquisition
metrics remain unavailable. For independently verified persistent identities use a common
manifest with `identity_scope: persistent`, `identity_provenance`, complete coverage and
`absence_annotations_complete: true`. Do not relabel AVA entities as physical people automatically.

Target designation is deterministic: choose the first sampled annotated frame with at
least two visible faces after the first two processed frames; designate its lexicographically
first annotation ID; associate it using one-to-one IoU matching to observed confirmed tracks.
For MobiFace's `single_target` annotations, the first visible designated-target frame
after the same warmup is eligible without a two-face requirement. This is not the
official SOT protocol that initializes with a GT box on the first video frame.
Only that frame's information can initialize the tracker. The exact boxes, IDs, source
index, timestamp, chosen prediction, IoU and outcome are saved. The backend receives only
the chosen **predicted track ID**. Initialization is attempted once. If that designated face
has no eligible prediction, the failure is recorded and target metrics are unavailable;
the benchmark does not use future GT to retry or silently pick an easier person.
After initialization no replacement target is selected, even after expiry. The initialization
frame is excluded from target-metric denominators. Detection/MOT metrics cover all scored frames.

## Common schemas and metric definitions

CSV ground truth:

```text
video_id,frame_index,timestamp_s,person_id,x1,y1,x2,y2,visible
```

CSV predictions (additional frame/state fields are mandatory for round-trip validation):

```text
video_id,frame_index,timestamp_s,track_id,x1,y1,x2,y2,observed,confirmed,processing_ms,selected_track_id,target_visible,target_status,processed_frame_index,width,height
```

`frame_index` is the zero-based **source** index. `timestamp_s = frame_index / source_fps`
is local to the source file, not a new 10 FPS counter. Manifest offset links it to AVA
original timestamps. Boxes are continuous pixel xyxy with right/bottom exclusive, bounded
by [0,width] and [0,height]. Invisible GT rows use empty boxes. A prediction frame with
no tracks has one empty-track sentinel row. Every track row in a frame must agree on
time, dimensions, selection state and latency. Exported predictions are reloaded and
validated before scoring; wrong-frame joins fail explicitly.

Only `observed AND confirmed` boxes enter detection/MOT metrics, matching the existing
usable-ROI policy. Predictions-only Kalman boxes are exported but never treated as detections.
Consequently confirmation startup misses count against the end-to-end baseline. These
are full-pipeline box metrics, not isolated MediaPipe detector AP.

| Metric | Definition / denominator |
| --- | --- |
| Precision | Hungarian IoU matches / observed confirmed predictions, only with exhaustive coverage. |
| Recall | Matches / annotated visible face instances. Partial coverage means recall of provided faces only. |
| IoU | Mean intersection-over-union over matched boxes at the configured gate (default 0.5). No matches yields null. |
| False positives / missed faces | Unmatched predictions with exhaustive coverage / unmatched visible GT faces. |
| IDSW, IDF1, fragmentation | motmetrics 1.4.0 CLEAR/Identity metrics, limited to complete appropriate identities and contiguous scored blocks. Unknown annotation frames reset continuity. IDF1 uses global bipartite matching within each block; it is not physical-person verification. |
| HOTA | Reference TrackEval average over 19 IoU thresholds, isolated environment, same annotation eligibility. |
| Target recall | Correct selected matches / annotated target-visible frames after initialization. |
| Wrong-person assignment rate | Selected boxes matched to another annotated face / evaluable observed selected assignments. |
| False-target assignment rate | (Wrong-person matches + unmatched selected boxes in exhaustive frames) / evaluable observed selected assignments. Unmatched boxes in partial coverage are unjudgeable. |
| Bystander replacement frames | Number of sampled frames where selected box matches a different annotated face. |
| Target loss duration | Incorrect/unassigned target-visible samples times nominal sampling period. Episodes break at unknown coverage and are marked censored without observed recovery. It is sampled exposure, not reconstructed continuous sensor time. |
| Correct target reacquisition rate | Successful physical return episodes / successful plus completed unsuccessful return episodes. Unresolved episode ends are censored. Needs persistent identities and known absence. |
| False reacquisition rate | Wrong/false observed returns after runtime loss / evaluable observed return attempts. Different denominator from physical return recall. |
| Processing FPS | Processed inference frames / inference wall time, including skipped-frame decoding, excluding startup, exports, scoring and overlays. |
| Processing latency | Per selected-frame read + resize + detection/embedding/tracking + selection state, excluding skipped reads and export/evaluation. Mean and p95 reported. |

Rates use [0,1], not percentages. Zero denominators and unsupported identity scenarios
produce null with reasons. Multivideo detection and IDF1 use count aggregation, not
unweighted means. Per-video C3 denominators/scopes are authoritative. No metric is
computed from tracker IDs as ground truth, and sparse coverage never proves physical absence.

## Outputs and visual review

```text
results/ava_baseline/
  predictions.csv
  frame_metrics.csv
  tracking_metrics.json
  performance.json
  experiment_config.json
  per_video/<id>/initialization_diagnostics.json # root copy with stage diagnostics
  report.md
  annotated_video.mp4           # optional; per_video/<id>/ for several videos
  annotated_video.labels.json  # short display label -> original annotation ID
  per_video/<id>/predictions.csv
  hota_input.json              # optional post-inference metric input
  hota_metrics.json            # optional
  diagnostic_stages.csv        # --stage-diagnostics, per-frame counts and matches
  stage_detections.csv         # returned detector/descriptor boxes, no embeddings
  stage_snapshots.jsonl        # inference-only stage telemetry
  stage_metrics.json          # stage denominators and sensitivity analysis
  initialization_failure.json # failed initialization with stage diagnostics
  initialization_frame.png    # post-inference diagnostic illustration
  diagnostic_report.md        # optional stage report and example PNG references
```

Overlays are generated in a separate post-inference pass. Cyan boxes are GT, orange
are predictions, green is the selected prediction, red is a wrong selection, and yellow
marks the missed designated GT box. Headers show source frame/time, target outcome,
ID-switch events and unknown coverage. A 62-pixel visualization-only header preserves
the entire original image; metric coordinates remain unchanged. Short GT labels map
back to full entity IDs in the labels JSON. Per-frame CSV rows expose misses, assignments,
unmatched predictions and evaluability rather than hiding failures.

## Validation and extension

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests datasets/tests
$env:C3_RUN_MODEL_TESTS = "1"
$env:C3_HOTA_PYTHON = (Resolve-Path ".metrics-venv/Scripts/python.exe").Path
.\.venv\Scripts\python.exe -m pytest -q tests datasets/tests
Remove-Item Env:\C3_RUN_MODEL_TESTS, Env:\C3_HOTA_PYTHON
```

CI creates a tiny lossless video with colored rectangles and AVA-format annotations.
It verifies schemas, alignment, resizing/subsampling, IoU, standard ID metrics, target
errors, missing coverage, initialization failures, future-GT isolation and oracle-box
isolation without dataset downloads. The color backend is **not** a face tracker.
Optional tests run the actual pretrained models and reference HOTA on controlled
fixtures; they also do not establish real-world tracking quality.
Set `C3_RUN_DATASET_TESTS=1` as well to audit the actual 80 MobiFace native annotation
files when they are present locally. This opt-in check never downloads data.

The local real AVA smoke run and six-condition matrix validate execution and alignment
on a short interval, not a full AVA benchmark. The conservative partial-coverage run
withholds unsupported identity metrics, and can legitimately report failed target
initialization. Do not tune model thresholds on held-out validation/test recordings.

The locally validated interval is `053oq2xB3oU`, original seconds `[900,910)`, from
the official validation release: 250 source frames at 25 FPS and 1,020 annotated face
instances. All six conditions completed; each 10 FPS run processed 100 source frames.
The original-resolution native-FPS run matched 207 annotated faces at IoU >= 0.5
(annotation-relative recall 0.20294, mean matched IoU 0.56620). Precision, FP counts,
and identity metrics were withheld because exhaustive annotation coverage was not
independently certified. The one target initialization attempt failed, so this run
provides no target-retention or reacquisition accuracy estimate. These are scoped
smoke-test observations, not scores for the AVA split or other datasets. The tracker
and thresholds were left unchanged; no full-split score is claimed. The later
[AVA stage report](AVA_DIAGNOSTICS.md) reproduces this result and explains the overlap
failure. The [MobiFace guide](mobiface/README.md) records its now-supported native
adapter, 80-file annotation audit and one real test-sequence evaluation.

To add ByteTrack, BoT-SORT or another algorithm, implement `TrackerBackend` in
`c3_tracker.benchmark.backend`: `process(frame, timestamp_s, oracle_boxes=None)` returns
`Observation` records, `designate(predicted_track_id)` receives the one initialization,
and `close()` releases resources. A `factory(config)` must not receive annotations.
Use `--backend-factory your_package.your_module:factory` without rewriting scoring,
datasets or visualization. Optionally implement `runtime_info()` with actual device/model
provenance. For oracle mode consume the supplied boxes only and never infer identity from
their list positions or labels. Preserve the common prediction semantics and test the
same synthetic infrastructure before comparing algorithms on identical registered splits.
