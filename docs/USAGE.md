# FaceRec usage and export reference

Start with the [quick start](../README.md). This guide keeps the detailed controls,
ROI settings, recording behavior and diagnostics in one place. Installation pins
and environment compatibility are explained in [Dependencies](DEPENDENCIES.md).

Commands run from the repository root after installing the package. If the virtual
environment is not activated, replace `python` with `.\.venv-shared\Scripts\python.exe`.

## Command-line options

These are all options for the desktop tracker, `python -m c3_tracker`. Options
beginning with `--` are command-line arguments. Combine them after the command;
switches such as `--record-pair` take no value. All switches default to off.

```powershell
python -m c3_tracker --help
python -m c3_tracker --source video --input "C:\videos\base.mp4" --record-pair --output-dir exports
```

### Input and display

| Option | Default | Values and purpose |
| --- | --- | --- |
| `--help`, `-h` | Off | Print the available arguments and exit. |
| `--source` | Required | `webcam` or `video`. |
| `--camera-index` | `0` | Nonnegative camera index; used with `webcam`. |
| `--input` | None | Existing video file path; required with `--source video`. Quote paths containing spaces. |
| `--width` | `640` | Positive integer; requested webcam width in pixels. Does not resize video files. |
| `--height` | `480` | Positive integer; requested webcam height in pixels. Actual camera resolution may differ. |
| `--headless` | Off | Hide the preview; no mouse target selection. Webcam runs also require `--max-frames`. |
| `--debug-overlay` | Off | Show raw detections, tentative tracks and predicted boxes in the preview. |
| `--max-frames` | No limit | Positive integer; stop after this many processed frames. |

### Detection and tracking

| Option | Default | Values and purpose |
| --- | --- | --- |
| `--confidence` | `0.65` | Number in `(0, 1]`; configured MediaPipe confidence argument. It is not an independently guaranteed floor on every returned score. |
| `--model-selection` | `0` | MediaPipe detector: `0` short range, `1` full range. |
| `--appearance-model` | `sface` | `sface` or `mobilenet`; appearance features used to associate detections with tracks. |
| `--model-dir` | `.models` | Directory containing the verified SFace/YuNet weights. |
| `--max-cosine-distance` | SFace `0.45`; MobileNet `0.2` | Number in `(0, 1]`; appearance-distance limit. Lower values are stricter. Selected-target checks add further restrictions. |
| `--track-memory-seconds` | `3` | Finite positive seconds to retain a missing track. Retention does not guarantee reacquisition. |
| `--max-age` | Unset | Positive integer; use unmatched-frame retention instead of timed retention. Overrides `--track-memory-seconds`. |
| `--n-init` | `3` | Positive integer; consecutive observations required to confirm a new track. |
| `--gpu` | Off | Request CUDA for MobileNet. Needs compatible CUDA Torch wheels; SFace stays on CPU. |

### Green capture box

The default box already extends 20% above the detected face. Each value below
adjusts that new default box: positive expands outward, negative trims inward.
Values must be finite fractions in `[-1, 1]`, and together must leave positive
width and height. For example, `--roi-top 0.1` adds another 10% of the new default
height above its top; use `--roi-bottom=-0.1` to trim the bottom by 10%.

| Option | Default | Adjustment |
| --- | --- | --- |
| `--roi-top` | `0` | Move the top edge, using the new default height. |
| `--roi-bottom` | `0` | Move the bottom edge, using the new default height. |
| `--roi-left` | `0` | Move the left edge, using the new default width. |
| `--roi-right` | `0` | Move the right edge, using the new default width. |

### Recording and output

| Option | Default | Values and purpose |
| --- | --- | --- |
| `--record-pair` | Off | Export clean original and face-centered MP4s after selection. Requires a CFR video file and an interactive preview; unsupported for webcam or headless mode. |
| `--pair-loss-policy` | `stop` | `stop`: finalize both videos at first target loss. `mark`: keep the timeline with invalid FACE LOST placeholders; requires `--record-pair`. |
| `--record` | Off | Save `annotated_video.mp4`, including boxes and overlays. |
| `--record-raw` | Off | Also save clean `source_video.mp4`; requires `--record`. |
| `--diagnostics` | Off | Write detailed association/rejection information to `association_diagnostics.jsonl`. |
| `--output-dir` | `results` | Parent output folder. Each run gets a generated subfolder; relative paths use the current working directory. |

Model setup is a separate command: `python -m c3_tracker.models --help`.
The full research checkout also provides annotated-dataset evaluation with its own CLI in the
[benchmark guide](../datasets/BENCHMARK_GUIDE.md). These are separate from the
desktop options above.

## Which output files do I need?

The FaceVitals GUI opens a video directly; it does not need every tracker log.
For the planned paired workflow, keep both MP4s, the validity CSV and pair metadata
together. Small run metadata and summary files are useful for reproducibility.

| File | Created when | Role in the FaceVitals workflow |
| --- | --- | --- |
| `original.mp4` | `--record-pair`, after a valid selection | Clean full-frame input for the original-video comparison. |
| `face_centered.mp4` | `--record-pair`, after a valid selection | Selected-face input for pulse processing. |
| `frame_validity.csv` | Pair recording starts | Required by a future full-timeline consumer to reject FACE LOST frames; the current upstream GUI does not read it. Keep it with the pair. |
| `paired_recording.json` | Pair export enabled | Timing, selected ID, crop configuration, verification and stop reason. Keep it with the pair; not read by the current GUI. |
| `run_metadata.json` | Every initialized run | Configuration, versions and source details. Useful provenance; not a GUI input. |
| `summary.json` | Run finalizes | Processing totals, target losses and errors. Useful status record; not a GUI input. |
| `tracking_log.csv` | Every initialized run | Per-frame tracks and boxes. Used for debugging/evaluation; not required for the manual video handoff. |
| `annotated_video.mp4` | `--record` | Visual review only; use clean videos for pulse processing. |
| `source_video.mp4` | `--record --record-raw` | Clean recording for replay/debugging. Not needed as an additional copy when the required paired original is already available. |
| `association_diagnostics.jsonl` | `--diagnostics` | Detailed debugging; optional. |
| PNG screenshots | S/Space or Save control | Visual review; optional. |

Currently the three normal log/metadata files are automatic; there is no
`--no-logs` or minimal-output mode. Pair metadata can exist without MP4s if no
valid target was selected. No output files need to be committed when integrating
the tool into FaceVitals; recordings, run outputs and local datasets stay ignored.

### Run folders and IDs

Every initialized run creates a new subfolder, for example
`results/20261104T091949_255208Z_80787d/`. The ID combines UTC date/time,
microseconds and six random hexadecimal characters. Microseconds help distinguish
near-simultaneous starts; they do not describe camera or physiological timing
precision. The ID is unrelated to the selected track ID. `--output-dir exports`
changes only the parent directory. Press O to open the current run folder.

## Controls and selected target

Left-click a **confirmed, currently observed** box to select the target. The target
is green; other currently detected, confirmed tracks are orange. Unconfirmed and
missing tracks are hidden in the normal view so prediction boxes do not accumulate.
Use `--debug-overlay` to see raw detections, tentative tracks and labeled predictions.
In overlap, the smallest enclosing selectable box wins (ID breaks equal-area ties).
**R** resets, **S** or **Space** saves an annotated PNG, **Q** quits. Focus the
tracking window for keyboard shortcuts; lowercase keys also work. You can also
click **Save [S/Space]** in the window. A saved-file message appears for six seconds
and the terminal prints the full path. Click **Open folder [O]** (or press O) to
find screenshots and logs in the current run's folder. Screenshot failures are
displayed and printed without stopping tracking.
After `--n-init` consecutive observations (default 3), a track becomes confirmed.
`NOT SELECTED`, `VISIBLE`, `LOST`, and one-frame `REACQUIRED` appear in the overlay.
`REACQUIRED` means an observation matched the **same still-alive ID** after loss,
not independently verified identity or recovery of a deleted track.
Select manually again after deletion.
Selecting a face freezes its clicked-frame anchor and up to eight consistent
recent views. Every confirmed association, including overlap-based matching,
must pass appearance checks **before assignment**, so a rejected selected match
cannot consume another track's detection. A lost selection
needs three consecutive compatible observations, with a margin over other faces,
before turning green again. Until then it stays LOST. R clears the reference;
deliberately selecting another face replaces it. Uncertain matches can remain LOST.
Competitor references come from distinct faces observed at the same time; a stale
ID for the same face is not automatically treated as another person. A verified
selected-face return is considered before duplicate IDs and still needs confirmation.

## Configuration and outputs

```powershell
python -m c3_tracker --source webcam --confidence 0.65 --n-init 3 --max-cosine-distance 0.45
python -m c3_tracker --source webcam --track-memory-seconds 5
python -m c3_tracker --source video --input "C:\videos\test.mp4" --headless --max-frames 100 --record
python -m c3_tracker --source webcam --record --record-raw --diagnostics
python -m c3_tracker --source webcam --record --record-raw --roi-top 0.1 --roi-bottom=-0.1 --roi-left 0.05 --roi-right 0.05
```

The selected face's **green box is the capture region**. By default its top is 20%
of the original face height higher; its other edges retain their original positions.
There is one green outline. This new box is the reference for four command-line
adjustments, all defaulting to **0**:

| Option | Positive value | Negative value | Fraction of |
| --- | --- | --- | --- |
| `--roi-top` | Raise top | Lower top | New default height |
| `--roi-bottom` | Lower bottom | Raise bottom | New default height |
| `--roi-left` | Move left edge left | Move left edge right | New default width |
| `--roi-right` | Move right edge right | Move right edge left | New default width |

Values are fractions: `0.1` means 10%, not 0.1 pixels. `0` keeps the **new default**
edge, including its already-higher top. A 100-pixel-tall original face becomes 120
pixels tall by default; `--roi-top 0.1` raises its top another 12 pixels.
The example above expands the top another 10%, trims the bottom 10%, and expands
each side 5%. Allowed values are [-1,1]; combined adjustments must leave positive
width and height. Bounds are rounded outward to integer pixels and clipped to the
image. Fractions use the new default dimensions **before** clipping at image edges.
The previous temporary `--forehead-margin` option is replaced by these four options.

Tracking, appearance embeddings, click selection and benchmark boxes continue to
use the original detector geometry. Capture requires a selected, confirmed face
with a current observation; LOST tracks supply no capture region. Padding can
include hair/background and is not forehead-only skin segmentation.

`tracking_log.csv` includes `capture_roi_left/top/right/bottom` on eligible target rows;
other rows leave these fields blank. Use these integer bounds to crop the corresponding
**unannotated** frame as `frame[top:bottom, left:right]`. Right/bottom are exclusive.
`--record --record-raw` saves the matching source recording. Screenshots and the
annotated recording include the adjusted green box; separate cropped images are not saved.

### Export a synchronized pair for FaceVitals

Use a clean constant-frame-rate (CFR) base video:

```powershell
.\.venv-shared\Scripts\python.exe -m c3_tracker --source video --input "C:\videos\base.mp4" --record-pair
```

Click a confirmed face. Both exports start on the first processed frame with a valid
selected green box. They end together at the **first target loss**, before that missing
frame. The valid segment is saved immediately and never resumes automatically, even
if the face returns. Pressing **Q**, closing the window, reaching the end of the input,
resetting selection with **R**, selecting a different ID, or `--max-frames` also closes
and saves the pair. After target loss, the preview can continue until you quit.

Each run's results folder contains:

- `original.mp4`: full, clean frames from the selected interval of the base video.
- `face_centered.mp4`: the matching green-box crops, with no boxes, labels or controls.
- `paired_recording.json`: selected ID, ROI margins, source start/end frames and times,
  output dimensions, frame count, duration, stop reason and decoded-output verification.
- The usual `tracking_log.csv` and `run_metadata.json`; output frame zero corresponds
  to the metadata's `first_source_frame` in that log and input video.

Both MP4s use the source FPS and the same `mp4v` codec, with no audio. The face output
size is fixed from the first valid green box, rounded up to even dimensions. Later
crops are resized to fit that canvas with aspect ratio preserved and centered black
padding. An odd-sized full frame receives one padding row/column instead of losing
image pixels. Both outputs are re-encoded; neither contains screen overlays.
Before reporting success, both files are reopened and decoded to check equal frame
counts, FPS and duration. Inconsistent source timestamps or skipped source frames
stop the run and finalize the preceding segment rather than silently changing time.

If you never select a valid face, no pair MP4s are created; metadata records
`not_recorded`. `--record-pair` requires interactive video mode; webcam/headless modes
are rejected. Existing `--record`/`--record-raw` behavior remains available separately.
This exports inputs for FaceVitals; it does not run pulse estimation.

For the **full-timeline comparison**, add `--pair-loss-policy mark`:

```powershell
.\.venv-shared\Scripts\python.exe -m c3_tracker --source video --input "C:\videos\base.mp4" --record-pair --pair-loss-policy mark
```

Recording starts at the first valid selection, then keeps every source frame until
Q, EOF or a selection reset/change. During target loss, `original.mp4` continues
unchanged and `face_centered.mp4` contains a **FACE LOST** placeholder. When the
same selected track becomes valid again, its face crop returns on the original clock.
`frame_validity.csv` contains one row per output frame with output/source indices,
timestamps, selected ID, `face_valid` (0/1) and reason. No missing interval is deleted.

The full-timeline files require a consumer that reads `frame_validity.csv`. The
FaceVitals GUI on upstream `main` does not read that file; its forehead extractor
can reuse a previous ROI after a detection miss. Do not treat a FACE LOST
placeholder, a held image, or a zero BPM as a physiological sample. A compatible
consumer must preserve missing intervals, reset pulse state across gaps, and wait
for a fresh valid analysis window after reacquisition.

`facevitals.compare_pair` and `facevitals.paired_input` have been used on a local
integration branch; they are absent from the upstream `main` snapshot reviewed
for this guide. They are development integration work, not commands supplied by
the current upstream installation. See the [handoff status](../README.md#facevitals-handoff).

Missing confirmed tracks are retained for **3 seconds** by default. Use
`--track-memory-seconds 5` for a five-second absence window. Webcam expiry uses
monotonic elapsed time; video expiry uses video timestamps, regardless of processing FPS.
The optional `--max-age` overrides timed memory with a limit in unmatched **frames**.
An ID may still fail to match within that window if appearance is incompatible or
the return is ambiguous; retention alone is not an identity guarantee.
`--confidence` filters MediaPipe
face detections (default 0.65, increased to reduce background false positives).
`--max-cosine-distance` controls appearance matching (default SFace distance 0.45;
selected reference views use a stricter cap of 0.4, or 0.35 after loss, and a 0.08 margin
over competing references). Bank views must be within distance 0.25 of the clicked
anchor and come from the latest 30 observations. Candidates must also remain within
distance 0.6 of the clicked anchor (or `max-cosine-distance + 0.15`, if smaller).
The bank never learns from later rejected observations. These are conservative
heuristics, not calibrated probabilities.
`--appearance-model mobilenet` retains the older ImageNet backend for comparisons
(default distance 0.2); it is not a face recognition network.
Association uses position-only
Kalman gating so head-size/aspect changes do not independently reject a match. A
guarded fallback recovers unmatched confirmed tracks throughout their lifetime,
requiring both compatible appearance and nearby, similarly sized boxes with an
unambiguous match. The allowed displacement grows with the detection gap, capped
at 2.5 box widths/heights in normalized center distance. It never revives deleted IDs or merges IDs.
Appearance history is bounded to 100 features per track. These changes reduce specific
fragmentation cases; they do not establish real-person identity accuracy. `--model-selection`
0 is the short-range detector, 1 is full-range. `--gpu` applies only to the optional
MobileNet backend and requires a compatible CUDA torch/torchvision installation.
Headless mode has no mouse selection; it measures detection/tracking/I/O only. Headless
webcam runs require `--max-frames`. Videos are processed as fast as possible, without
real-time playback pacing. Width/height request webcam settings only; videos retain native
resolution. A camera may negotiate a different resolution; the metadata records the actual one.

DeepSORT is based on [levan92/deep_sort_realtime](https://github.com/levan92/deep_sort_realtime),
pinned to commit `3341f823ee0ad20aa58c906079a37f3d43b42bbe` in `pyproject.toml`.
That upstream commit clears pending appearance features after fitting them, so missing
frames do not repeatedly refill the gallery with stale features. Its package version
still reports 1.3.2. If upgrading an existing 1.3.2 environment, force the source change:

```powershell
python -m pip install --force-reinstall --no-deps "deep-sort-realtime @ https://github.com/levan92/deep_sort_realtime/archive/3341f823ee0ad20aa58c906079a37f3d43b42bbe.zip"
```

Each run creates `results/<UTC timestamp>_<suffix>/` (override with `--output-dir`):

- `tracking_log.csv`: one row per track per frame, including tentative and predicted
  tracks; an empty-ID row preserves frames with no tracks. Coordinates are pixel
  left/top/right/bottom, clipped to the image. `has_detection=1` identifies a current
  original detection box; otherwise the box is a prediction. `detection_confidence`
  is a detector score, blank for predictions; it is **not association confidence**.
  `selected_target_id` persists across missing frames. `is_target` can be true for a
  predicted track, so valid downstream ROIs must require **is_target AND has_detection
  AND confirmed AND target_visible**. The separate `capture_roi_*` bounds add forehead
  padding; original tracking coordinates remain available. No physiological sample
  extraction is implemented here.
- `run_metadata.json`: configuration, package/Python versions, actual resolution,
  reported source FPS, timestamp basis, OS/CPU details, measurement definitions.
- `summary.json`: frame count, selected-target visible/lost frames, loss episodes,
  mean/p95 processing latency, measured end-to-end FPS, and any runtime error.
- `annotated_video.mp4` with `--record`, plus PNGs requested with S.
- `source_video.mp4` with `--record --record-raw`: the same frames before overlays,
  encoded as MP4. This is unannotated source video, not lossless sensor data. Its
  frame indices match the annotated recording and CSV.
- `association_diagnostics.jsonl` with `--diagnostics`: one entry per frame with
  detected/alignable face counts, selected-track state, actual appearance distances
  and limits, competing-face margin, motion gate, return-confirmation progress, and
  rejection reason. No embedding vectors or frozen references are written.

Use source video plus live diagnostics to investigate false losses. Re-running models
on an annotated recording can change alignment and embeddings, so that replay cannot
reconstruct the original live rejection reliably. Both videos are finalized when you quit
with Q. The source recording and extra logging add I/O work; compare like-for-like settings.

Capture FPS is metadata, **not measured processing FPS**. Processing latency includes
read + detection + tracking + annotation (first frame is pre-read before model loading).
It excludes CSV, video encoding, and GUI waiting. End-to-end FPS includes these loop
costs, excludes model startup, and includes any frame-read blocking. Display FPS is a
cumulative estimate; use the final summary for comparisons. Video timestamps are frame
index / reported FPS (constant-frame-rate assumption); webcam timestamps are monotonic
capture-loop elapsed time. Variable-frame-rate sources should be transcoded to CFR first.
The MP4 uses source FPS (30 if the camera reports none), so it does not reproduce webcam
wall-clock timing when processing falls behind. Review CSV timestamps as well.

## Runtime and research boundary

The desktop runtime consists of detection, appearance embeddings, association,
target selection, ROI geometry, recording, controls, visualization and run logging.
Its integration export includes relevant automated tests, which use synthetic
inputs/mocks by default. These tests are part of maintaining the tool, not local
experiment outputs. Model checks are explicitly opt-in and their assets stay local.

Dataset loaders, benchmark metrics and offline evaluation remain in the full
FaceRec checkout. They are excluded from the
[runtime integration bundle](DEPENDENCIES.md#runtime-integration-export), along with
all datasets, run results, weights and environments. Existing research commands
continue to work in this checkout. No FaceVitals GUI changes are made by exporting.

The exporter has its own `--list-files`, `--output-dir` and `--verify DIRECTORY`
options; these belong after `python tools/export_runtime.py`, not after
`python -m c3_tracker`. Verification checks a pristine bundle's inventory and
hashes before transfer. See the linked integration guide for the full commands.

## Experiments and evaluation

For reproducible annotated-video benchmarking, start with
[datasets/BENCHMARK_GUIDE.md](../datasets/BENCHMARK_GUIDE.md). The new
`python -m c3_tracker.evaluate` CLI runs raw-video inference, then separate AVA/common
ground-truth scoring, C3 evaluation and optional overlays. The existing
`c3_tracker.evaluation` commands below remain available for manually annotated live runs.

Follow [TESTING_GUIDE.md](TESTING_GUIDE.md) for the stationary and crossing
experiments, independent person annotations, resolution comparisons, and evaluation:

```powershell
python -m c3_tracker.evaluation results\RUN\tracking_log.csv
python -m c3_tracker.evaluation results\RUN\tracking_log.csv --annotations annotations.csv
```

The first command reports runtime statistics only. Person-based metrics require independent
annotations; tracker IDs are never treated as ground truth. Record participants only with
consent and retain outputs locally. Results, videos, screenshots inside results, environments,
and model weights are ignored by Git. Never commit participant data elsewhere.

## Troubleshooting

- `No module named c3_tracker`: activate/select the right environment and run `python -m pip install -e .`.
- Missing face model: run `python -m c3_tracker.models --download` from the project root.
- No `mp.solutions`: recreate the Python 3.12 environment with the pinned requirements.
- `pkg_resources` missing: install the pinned setuptools; do not upgrade it past 80.x.
- Torchvision operator/import error: reinstall the matched CPU torch/torchvision pair in a clean venv.
- Camera cannot open: check permissions, close camera users, try `--camera-index 1`.
- No boxes: improve frontal lighting, move closer, lower `--confidence` moderately,
  or try full-range `--model-selection 1`. Face detection is not guaranteed in profile.
- Cannot select: wait for confirmation; click an observed box. Reset before selecting another person.
- GUI errors: run on a local Windows desktop, not a remote/headless shell; remove conflicting
  headless OpenCV packages by rebuilding the venv. Use `--headless` only for batch video tests.
- Invalid video / MP4 writer: check path, permissions and codecs; try a normal CFR H.264/MP4 input.
  OpenCV may report corrupt/truncated video as end-of-file; compare processed frame counts to source.
- Low FPS: try 320×240, better lighting and fewer faces; avoid changing settings between compared runs.
- Upstream deprecation warnings (`pkg_resources`, `torch.load`) can occur with the optional
  MobileNet backend. The default backend uses hash-verified ONNX models.

## Layout and validation boundary

`src/c3_tracker/` contains detection, tracking, target manager, capture/CLI, visualization,
logging and evaluation adapters. `tests/` covers deterministic geometry, state transitions,
mocked track semantics, input errors, CSV schema and annotated evaluation. `docs/` contains
protocols and limitations. Backend replacement should preserve `Observation` semantics.

Validation includes Windows automated tests, desktop control integration, headless video
output, and actual SFace model checks with sample photographs. Distinct-face replacement
regressions check that a second face cannot take the selected ID in those scenarios.
These are controlled pipeline tests; sustained real two-person tracking and crossings
still need independent review on the intended PC. GPU execution is not established.
