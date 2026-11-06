# FaceRec: selected-face video export for FaceVitals

FaceRec tracks a selected face and exports synchronized original and cropped videos
for processing in FaceVitals. It runs as a Python 3.12 desktop application; pulse
estimation belongs to FaceVitals. Track IDs describe associations within a run.
See the [tracking limits](docs/C3_BASELINE_LIMITATIONS.md).

## Install on Windows

Install 64-bit Python 3.12 and Git, then run these commands in PowerShell:

```powershell
git clone https://github.com/amiking10/SP-Cup-face-rec-.git
cd SP-Cup-face-rec-
py -3.12 -m venv .venv-shared
.\.venv-shared\Scripts\python.exe -m pip install --upgrade pip
.\.venv-shared\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-shared\Scripts\python.exe -m pip install -e .
.\.venv-shared\Scripts\python.exe -m c3_tracker.models --download
```

If the `py` launcher is unavailable, use your installed Python 3.12 executable
for the `-m venv .venv-shared` command.

Run subsequent commands from this repository's root. In VS Code, select
`.venv-shared\Scripts\python.exe` as the Python interpreter. See
[Dependencies and environments](docs/DEPENDENCIES.md) for development installs,
the compatibility requirements and troubleshooting dependency conflicts.

Use this same interpreter for FaceVitals. The [shared setup](docs/DEPENDENCIES.md#shared-facevitals-install)
installs both projects without replacing an existing environment.

## Export a video pair

Start with a clean constant-frame-rate video file. `--input` takes a file, not an
output directory. Any boxes already drawn into the source remain in its pixels.

```powershell
.\.venv-shared\Scripts\python.exe -m c3_tracker --source video --input "C:\videos\base.mp4" --record-pair --output-dir exports
```

1. Wait for a confirmed face box, then click the person to select.
2. The green box shows the crop region. Recording begins on the first valid
   selected frame. Its top extends 20% above the detected face by default.
3. Press **Q** to finish. With the default `stop` loss policy, both videos also
   stop and save at the first target loss.
4. Open the run folder printed in the terminal, or use **O** in the tracker window.

Each run creates its own folder:

```text
exports/<run-id>/
  original.mp4           # clean full frames from the selected interval
  face_centered.mp4      # matching selected-face crops, without drawn boxes
  paired_recording.json  # selected track, timing, crop settings and verification
  frame_validity.csv    # validity and source/output frame mapping
  tracking_log.csv
  run_metadata.json
  summary.json
```

Keep the run folder together. Desktop paired exports use MP4/mp4v at the source
FPS, with no audio. The recorder reopens both files to verify their saved timing
and frame counts. Detailed crop sizing and timestamp rules are in the
[export reference](docs/USAGE.md#export-a-synchronized-pair-for-facevitals).
If no valid face is selected, paired videos are not created; the pair metadata
records `not_recorded`.

## FaceVitals handoff

In FaceVitals' offline GUI, use **Select Video File**, choose
`face_centered.mp4`, select POS or PBV, and press **Start**. Process `original.mp4`
separately for the original view. The GUI opens local files; `exports/` is our
convenient storage location, not an automatically watched upload folder.

| Export mode | Target lost | Current FaceVitals GUI |
| --- | --- | --- |
| `stop` (default) | Both files end before the missing frame | Open the saved uninterrupted clip manually |
| `mark` | Original continues; crop contains a FACE LOST placeholder and `face_valid=0` | Needs an adapter that honors the validity CSV |

For a complete timeline, add `--pair-loss-policy mark` to the export command.
Missing intervals must remain unavailable pulse data. A consumer must reset at
gaps and collect a fresh valid window after the selected face returns.

The GUI's forehead extractor can hold a previous ROI on misses and does not read
`frame_validity.csv`. The paired-input consumers `facevitals.compare_pair` and
`facevitals.paired_input` are local development work and are absent from the
[upstream main snapshot reviewed on 6 November 2026](https://github.com/Technion-SIPL-SPcup27/facevitals/tree/cbbbc3af55f9529fcc8eef058adbd0aca9c0d196).
Full-timeline GUI integration is pending. The tracker export does not itself
calculate BPM.

## Webcam tracking and recording

```powershell
.\.venv-shared\Scripts\python.exe -m c3_tracker --source webcam --camera-index 0 --width 640 --height 480
```

Click a confirmed face to select it. **R** resets selection, **S** or **Space**
saves an annotated screenshot, **O** opens the run folder, and **Q** quits.
Enable Windows camera permissions and close other applications using the camera.

`--record --record-raw` can save annotated and clean webcam recordings. Their
timing limitations are documented in the [usage reference](docs/USAGE.md).
Paired selected-face export currently requires an existing video and an
interactive window; webcam paired export is not implemented.

## Configuration and file locations

The [usage reference](docs/USAGE.md) includes the
[complete command-line option tables](docs/USAGE.md#command-line-options) and
[which output files to keep](docs/USAGE.md#which-output-files-do-i-need), with
detailed recording and tracking behavior. To list the available options directly:

```powershell
.\.venv-shared\Scripts\python.exe -m c3_tracker --help
```

Suggested local locations:

| Location | Purpose |
| --- | --- |
| `recordings/` | Clean input recordings |
| `exports/` | Paired exports made with `--output-dir exports` |
| `results/` | Existing default run directory and benchmark outputs |
| `.models/` | Downloaded, hash-verified model weights |
| `datasets/data/` | Local datasets and derived data |

These data locations are ignored by Git. `--output-dir` and `--model-dir` can
point elsewhere. Existing outputs are retained; this convention changes no
runtime defaults or stored files. Keep recordings and participant metadata local.

## Development and research references

For later FaceVitals integration, generate a
[runtime-only source bundle](docs/DEPENDENCIES.md#runtime-integration-export).
It includes the desktop tool and its regression tests without dataset adapters,
evaluation commands, local data or historical results. The research checkout
remains intact; exporting does not modify FaceVitals or merge any files.

The exporter can also check a pristine bundle with
`python tools/export_runtime.py --verify build/runtime`. It rejects modified,
missing or extra files. The [validation record](docs/VALIDATION.md) separates
current Windows regression checks from earlier model and dataset measurements.
A headless FaceVitals tracker and ROI-to-BPM pipeline are prepared in a separate local checkout;
see [adapter status and test commands](docs/DEPENDENCIES.md#local-facevitals-adapter-work).
Missing intervals remain unavailable and restart the pulse window. A maintained
recorded-video interface now adds target drawing and timeline review; see the
[launch command](docs/DEPENDENCIES.md#local-facevitals-adapter-work).
Live processing, deleted-ID recovery and scored dataset acceptance remain pending;
this work is local and the legacy FaceVitals GUI is unchanged.

- [Agent and integration instructions](AGENTS.md)
- [Dependencies and development setup](docs/DEPENDENCIES.md)
- [Controls, exports and diagnostics](docs/USAGE.md)
- [Dataset availability](datasets/README.md) and [benchmark commands](datasets/BENCHMARK_GUIDE.md)
- [Manual testing and annotations](docs/TESTING_GUIDE.md)
- [Validation history](docs/VALIDATION.md) and [baseline limitations](docs/C3_BASELINE_LIMITATIONS.md)
