# FaceVitals C3: multi-face tracking baseline

A Python 3.11 / OpenCV desktop test application for IEEE SP Cup 2027 C3 experiments.
MediaPipe detects faces; `deep-sort-realtime` associates detections using its bundled
pretrained MobileNetV2 appearance embedder and Kalman motion model. CPU is the default.
No physiological measurements, biometric identification, custom training, or clinical
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
   # Until merged, use the implementation branch:
   git switch feat/c3-live-tracking
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -r requirements.txt
   python -m pip install -e .
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
   is green; other/tentative/predicted tracks are orange; raw detections are thin gray.
   In overlap, the smallest enclosing selectable box wins (ID breaks equal-area ties).
   **R** resets, **S** saves an annotated PNG, **Q** quits. Lowercase keys also work.
   After `--n-init` consecutive observations (default 3), a track becomes confirmed.
   `NOT SELECTED`, `VISIBLE`, `LOST`, and one-frame `REACQUIRED` appear in the overlay.
   `REACQUIRED` means a new observation matched the **same still-alive ID**, never
   face recognition or recovery of a deleted track. Select manually again after deletion.

## Configuration and outputs

```powershell
python -m c3_tracker --source webcam --confidence 0.6 --max-age 30 --n-init 3 --max-cosine-distance 0.2
python -m c3_tracker --source video --input "C:\videos\test.mp4" --headless --max-frames 100 --record
```

`--max-age` is unmatched **frames**, not seconds. `--confidence` filters MediaPipe
face detections. `--max-cosine-distance` controls appearance matching. `--model-selection`
0 is the short-range detector, 1 is full-range. `--gpu` is optional and requires a
compatible CUDA torch/torchvision installation; the documented CPU wheels do not use CUDA.
Headless mode has no mouse selection; it measures detection/tracking/I/O only. Headless
webcam runs require `--max-frames`. Videos are processed as fast as possible, without
real-time playback pacing. Width/height request webcam settings only; videos retain native
resolution. A camera may negotiate a different resolution; the metadata records the actual one.

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
(`+cpu`) satisfy these pins. DeepSORT bundles its ImageNet MobileNetV2 weights, so
no external face-recognition weights or runtime download is needed. Setuptools <81
provides the deprecated `pkg_resources` required by this DeepSORT release. Upstream
DeepSORT requires `opencv-python` and MediaPipe requires `opencv-contrib-python`:
both distributions share `cv2`, so **both are pinned to exactly the same version**.
Do not add headless OpenCV wheels to this desktop environment. When repairing OpenCV,
recreate the venv rather than uninstalling one overlapping distribution in place.
Transitive packages are resolved by pip; these are top-level pins, not a full platform lock.

## Troubleshooting

- `No module named c3_tracker`: activate/select the right environment and run `python -m pip install -e .`.
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
- Upstream deprecation warnings (`pkg_resources`, `torch.load`) are expected with these pins.
  Only the package's bundled pretrained weights are loaded; do not replace them with untrusted files.

## Layout and validation boundary

`src/c3_tracker/` contains detection, tracking, target manager, capture/CLI, visualization,
logging and evaluation adapters. `tests/` covers deterministic geometry, state transitions,
mocked track semantics, input errors, CSV schema and annotated evaluation. `docs/` contains
protocols and limitations. Backend replacement should preserve `Observation` semantics.

Cloud validation used Python 3.11 on Linux: automated tests plus a headless annotated-video
smoke run using repeated sample-photo frames and actual pretrained models. It is not a live
identity experiment. Windows GUI, mouse/keyboard interaction, webcam hardware, GPU execution,
and real two-person association/crossing remain to be validated on the intended PC.
