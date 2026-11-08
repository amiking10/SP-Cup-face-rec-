# Tracking core and FaceVitals workflow

## Recorded-video workflow

In the FaceVitals runtime branch, install its tracking extra in Python 3.12,
explicitly prepare the pinned YuNet/SFace models, then launch:

```powershell
python -m facevitals.recorded
```

Open a clean recorded video, select a detected face in its first frame and press
Process. FaceVitals tracks the selected person, extracts its centered 1.5x face
ROI and applies its existing pulse methods. BPM and waveform displays use the
same processing outputs. Missing/unconfirmed frames stay unavailable (NaN).
Paired MP4 export is not needed for this direct path. Original source pixels are
not annotated for pulse extraction. Live capture is outside this integration.
For existing models, pass `--tracker-model-dir PATH` to the recorded interface;
no download is needed when both pinned model files are already present. From the
FaceRec checkout with the shared environment installed against the clean
FaceVitals checkout, use:

```powershell
.\.venv-shared\Scripts\python.exe -m facevitals.recorded --tracker-model-dir .\.models --output-dir .\exports\facevitals
```

## Source package API

`c3_tracker.detection.FaceDetector.detect(frame_bgr)` returns face detections with
pixel xyxy bounds. `c3_tracker.tracking.FaceTracker.update(detections, frame_bgr,
timestamp=seconds)` returns observations: raw ID, bounds, current-detection status
and confirmation. `TargetManager.select(x,y,observations)` selects a confirmed
observed face; `update` preserves the selected raw ID and reports loss rather than
silently selecting another person. Video timestamps come from the source clock.

This low-level core does not perform pulse processing, enlarge ROIs, save videos
or supply a desktop CLI. FaceVitals' public `track_target` adapter owns the
centered 1.5x xywh output, first-frame designation and full-frame validity mask.
Its optional automatic designation remains a separate ROI variant.

## Models and tests

`python -m c3_tracker.models --download --model-dir .models` downloads only the
two pinned OpenCV models after an explicit request and verifies their hashes.
The archived photograph/dataset download helpers are not part of this package.
Run `python -m pytest -q` for synthetic association, selection, embedding, model
integrity and source-export tests. No camera or dataset is required.

See [installation and archives](../README.md) and [limits](C3_BASELINE_LIMITATIONS.md).
