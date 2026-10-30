# FaceVitals C3: multi-face tracking baseline

A Python 3.11 / OpenCV desktop test application for IEEE SP Cup 2027 C3 experiments.
MediaPipe detects faces; aligned OpenCV SFace embeddings and `deep-sort-realtime`
associate them within a run. YuNet provides five facial landmarks for alignment.
SFace runs locally on CPU; face references remain in memory and are not written to logs.
No named-person identification, physiological measurements, custom training, or clinical
validation are implemented.

**A track ID is an association hypothesis, not a verified person identity.** The application
never automatically selects a new ID after losing the selected track, but DeepSORT can
still assign a bystander to an existing ID. Human review is required for C3 evaluation.
See [baseline limitations](docs/C3_BASELINE_LIMITATIONS.md).

## Windows + VS Code setup (PowerShell)

1. Install 64-bit **Python 3.11** from python.org (enable the Python launcher), Git,
   VS Code, and Microsoft's Python extension. This package deliberately requires 3.11.
2. Clone and open the project:

   ```powershell
   git clone https://github.com/amiking10/SP-Cup-face-rec-.git
   cd SP-Cup-face-rec-
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -r requirements.txt
   python -m pip install -e .
   python -m c3_tracker.models --download
   python -m pip check
   python -m pytest -q
   python -m c3_tracker --help
   code .
   ```

   If activation is blocked, avoid changing system policy: use `.\.venv\Scripts\python.exe`
   instead of `python` for every command. In VS Code, **Python: Select Interpreter** →
   `.venv\Scripts\python.exe`. Open a new terminal after selecting it.
3. In Windows Settings → Privacy & security → Camera, enable camera access and
   access for desktop apps. Close Teams/Zoom/other programs using the camera.
4. Run from the repository root:

   ```powershell
   python -m c3_tracker --source webcam --camera-index 0 --width 640 --height 480 --record
   python -m c3_tracker --source video --input "C:\videos\crossing.mp4" --record
   ```

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
```

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
pinned to commit `3341f823ee0ad20aa58c906079a37f3d43b42bbe` in both dependency files.
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
  AND confirmed AND target_visible**. No sample extraction is implemented here.
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

## Experiments and evaluation

For reproducible annotated-video benchmarking, start with
[datasets/BENCHMARK_GUIDE.md](datasets/BENCHMARK_GUIDE.md). The new
`python -m c3_tracker.evaluate` CLI runs raw-video inference, then separate AVA/common
ground-truth scoring, C3 evaluation and optional overlays. The existing
`c3_tracker.evaluation` commands below remain available for manually annotated live runs.

Follow [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) for the stationary and crossing
experiments, independent person annotations, resolution comparisons, and evaluation:

```powershell
python -m c3_tracker.evaluation results\RUN\tracking_log.csv
python -m c3_tracker.evaluation results\RUN\tracking_log.csv --annotations annotations.csv
```

The first command reports runtime statistics only. Person-based metrics require independent
annotations; tracker IDs are never treated as ground truth. Record participants only with
consent and retain outputs locally. Results, videos, screenshots inside results, environments,
and model weights are ignored by Git. Never commit participant data elsewhere.

## Dependency choices

Top-level runtime and test dependencies are pinned in `requirements.txt` / `pyproject.toml`:
Python 3.11, MediaPipe **0.10.21**, deep-sort-realtime **1.3.2**, NumPy **1.26.4**,
OpenCV Python/contrib **4.11.0.86**, torch **2.5.1**, torchvision **0.20.1**,
setuptools **80.9.0**, pytest **8.3.5**.

MediaPipe 0.10.21 retains `mp.solutions.face_detection` and bundles its detector models;
newer Tasks-only releases are not drop-in compatible. NumPy 1.26 avoids NumPy 2 ABI
mismatches. Torch/torchvision are a matched release pair; CPU wheel local suffixes
(`+cpu`) satisfy these pins. The default face backend requires the YuNet 2023mar and
SFace 2021dec ONNX weights from [OpenCV Zoo](https://github.com/opencv/opencv_zoo),
pinned by commit and SHA256 in `models.py`. The setup command downloads about 39 MB
into ignored `.models/`; startup verifies the hashes and does not download automatically.
Use `--model-dir PATH` for a different local folder. DeepSORT's bundled MobileNetV2
is an optional comparison backend. Setuptools <81
provides the deprecated `pkg_resources` required by this DeepSORT release. Upstream
DeepSORT requires `opencv-python` and MediaPipe requires `opencv-contrib-python`:
both distributions share `cv2`, so **both are pinned to exactly the same version**.
Do not add headless OpenCV wheels to this desktop environment. When repairing OpenCV,
recreate the venv rather than uninstalling one overlapping distribution in place.
Transitive packages are resolved by pip; these are top-level pins, not a full platform lock.

## Troubleshooting

- `No module named c3_tracker`: activate/select the right environment and run `python -m pip install -e .`.
- Missing face model: run `python -m c3_tracker.models --download` from the project root.
- No `mp.solutions`: recreate the Python 3.11 environment with the pinned requirements.
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
