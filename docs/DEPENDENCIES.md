# Dependencies and environments

[`pyproject.toml`](../pyproject.toml) defines Python 3.12 and the exact top-level
dependency versions. [`requirements.txt`](../requirements.txt) delegates to the
editable package with its `test` extra. Run commands from the FaceRec root.

## Install profiles

Create `.venv-shared` with Python 3.12 and install the matching CPU Torch wheels
as shown in the [quick start](../README.md#install-on-windows), then choose a profile:

| Profile | Command |
| --- | --- |
| Desktop tracker and paired recording | `python -m pip install -e .` |
| Tracker plus automated tests | `python -m pip install -e ".[test]"` |
| Tracker, tests and benchmark dependencies | `python -m pip install -e ".[test,benchmark]"` |
| Compatibility requirements entry point | `python -m pip install -r requirements.txt` |

Use `.\.venv-shared\Scripts\python.exe` in place of `python` when the environment
is not activated. Select that interpreter in VS Code. Keep an older `.venv` or
`.venv-legacy` intact until the shared setup works; do not upgrade it in place.
HOTA's external evaluator is separate; see the [benchmark guide](../datasets/BENCHMARK_GUIDE.md).

## Runtime integration export

From the FaceRec root, inspect the integration file list or create a new bundle:

```powershell
.\.venv-shared\Scripts\python.exe tools/export_runtime.py --list-files
.\.venv-shared\Scripts\python.exe tools/export_runtime.py --output-dir build/runtime
.\.venv-shared\Scripts\python.exe tools/export_runtime.py --verify build/runtime
```

The exporter uses only the Python 3.12 standard library. `--output-dir` defaults
to `build/runtime`; an existing destination is refused, so choose a new folder for
each additional export. Nothing is removed or overwritten. `build/` is ignored.
`--verify` is read-only: it checks the explicit file inventory and SHA256 hashes,
rejecting modified/missing files and extra recordings, sidecars or build artifacts.
Verify a pristine export before installation/testing; those operations can create
cache and build files in its directory. Do not copy them into the integration.
The manifest detects accidental changes; it is not a digital signature.

[`tools/export_runtime.py`](../tools/export_runtime.py) explicitly lists 14 runtime
modules and 11 test files. The bundle keeps the `c3_tracker` package and launch
command. It excludes `benchmark/`, `evaluate.py`, `evaluation.py`, dataset scripts,
research tests, recordings, downloaded weights, environments and historical reports.
The original research entry points stay available in the full FaceRec checkout.

The generated `pyproject.toml` takes runtime dependencies and the `test` extra
from this repository's canonical configuration; it omits the `benchmark` extra.
The bundle README supplies setup/recording instructions, and its usage reference
reuses the option/output tables in `docs/USAGE.md`. `runtime_manifest.json` records
every exported file's SHA256 hash (except itself), source commit/dirty state when
Git is available, and the original configuration hash. A dirty source tree is
identified as such; a commit ID alone does not describe its uncommitted contents.

Use the existing shared interpreter to install/verify a bundle when you choose to
switch to it. This replaces that interpreter's FaceRec installation target:

```powershell
.\.venv-shared\Scripts\python.exe -m pip install -e "build/runtime[test]"
Push-Location build/runtime
..\..\.venv-shared\Scripts\python.exe -I -m pytest -q
..\..\.venv-shared\Scripts\python.exe -m pip check
Pop-Location
```

Reinstall `-e ".[test,benchmark]"` from the FaceRec root to restore the full research
installation. The exporter itself never installs anything or changes environments.
CI installs the exported package in its own job and runs it with isolated Python
imports. The full-checkout tests also verify that the export cannot import the
excluded evaluators or benchmark package. Default regression tests use no datasets
and download no models; model tests stay opt-in.

This is a reviewable integration artifact. Porting the reusable library and desktop
interface into FaceVitals, updating its packaging, and validating its GUI are later
steps. Do not merge the whole FaceRec checkout or copy a local output directory.

The [validation record](VALIDATION.md#windows-maintenance-checks-6-november-2026)
contains the completed runtime/export checks and retained report paths. The
committed-source bundle `build/runtime-20261106-191a59a/` is a local artifact, not
part of Git. Existing bundles are snapshots: later source/documentation edits do
not update them automatically. Export into a new directory when a newer copy is
needed, then verify it before transfer.

## Local FaceVitals adapter work

The integration is saved in a separate ignored FaceVitals checkout:
`worktrees/facevitals-integration`, branch `amita/c3-tracker-adapter`, local commit
`69f86df` (following adapter commit `089db69`), based on upstream `cbbbc3a`.
Current upstream `83c544e` was subsequently merged; recorded interface commit
`9b4f7bb` adds the Tk workflow described below.
Commit `748a675` subsequently changes designation to clicking detected face boxes;
it preserves the centered 1.5x ROI, NaN-on-loss policy and maintained processing.
Commit `2a61670` adds the waveform/BPM plots and results displays described below,
without backend changes. Its [validation record](VALIDATION.md#recorded-gui-signals-and-results-6-november-2026)
includes full-suite and independent GUI checks.
It is not vendored into FaceRec and has not
been pushed or merged into FaceVitals. The legacy FaceRec snapshot is unchanged.

The adapter provides `facevitals.tracking.track_target`: one centered-1.5x xywh box
per decoded frame, with NaNs for unconfirmed/missing targets. It ports the runtime
association core and adds first-frame designated selection and a separate automatic
first-second policy. Tracked mean, 72-pixel crop and grid ROI variants now feed
FaceVitals pulse methods. Missing samples stay unavailable; each uninterrupted
valid span starts fresh pulse processing on the original video clock. No predicted
box or placeholder is passed to a pulse method. A maintained recorded-video Tk
interface now calls this path with detected first-frame face selection and timeline review.
Deleted-ID recovery, live processing and scored dataset acceptance remain later
steps. The read-only legacy GUI is unchanged. No dataset runs accompanied this step.

Launch the recorded-video interface from the FaceRec root:

```powershell
.\.venv-shared\Scripts\python.exe -m facevitals.recorded --tracker-model-dir .models --output-dir results/facevitals-recorded
```

Open a clean recorded video, click the target's detected face box in the first frame,
choose a tracked method, then press **Process video**. After processing, use the
slider to review boxes and BPM availability. Each run saves three processing CSVs
in a new subfolder; the source video is unchanged. This interface processes the
base video directly, rather than loading the desktop crop/validity pair. See the
[manual check](TESTING_GUIDE.md#recorded-facevitals-interface-check) before using it.
No additional Python dependency or environment is required on this Windows setup.
The selected detection designates the person; processing retains the documented
centered 1.5x tracked ROI and NaN on loss. The legacy GUI backend is not substituted.
The **Signals** tab shows the existing pulse waveform and per-window BPM with
missing data left as gaps; its red cursor follows the shared video slider. The
large BPM/quality display clears during loss or unavailable output. The **Results**
tab lists each window's time, BPM, quality and availability, plus the mean of
accepted window BPMs (a display summary, not a different final HR estimator).
Processing has an activity indicator and elapsed time. These are offline results;
no percentage/ETA, cancellation, SpO2, HRV, respiration or stress backend was added.

The shared editable installation now loads this checkout, with the FaceRec
installation preserved. From the FaceRec root, process a clean base video directly:

```powershell
.\.venv-shared\Scripts\python.exe -m facevitals.process "C:\videos\base.mp4" --pipeline toolbox-pos@track --init-box 100 80 160 180 --tracker-model-dir .models --out results/facevitals-designated
.\.venv-shared\Scripts\python.exe -m facevitals.process "C:\videos\base.mp4" --pipeline toolbox-pos@track-auto --tracker-model-dir .models --out results/facevitals-auto
```

Replace the video path and example first-frame `X Y W H` coordinates with the
intended face. Automatic mode chooses using the first second and takes no
`--init-box`; it is a separate method variant. These commands use existing verified
tracker weights in `.models/`. The headless path tracks the base video itself;
it does not import desktop `frame_validity.csv` or require a paired crop video.
Outputs include BPM windows and the actual sampled ROI bounds; unavailable
windows have no BPM and are marked abstained. This does not change the GUI handoff.

From this FaceRec root, run the new checkout's focused tests with the existing
shared interpreter; pytest uses that checkout's configured `src` import path:

```powershell
Push-Location worktrees/facevitals-integration
..\..\.venv-shared\Scripts\python.exe -m pytest -q tests/test_tracking_traces.py tests/test_tracked_signal.py tests/test_tracking_process.py
..\..\.venv-shared\Scripts\python.exe tools/repo_health.py
Pop-Location
```

The full Python suite passed 534 tests with 76 optional checks skipped. The final
focused run passed 53 tests, including four late input-validation cases added after
the full run. A real-model smoke check passed separately using existing verified
weights. See the [validation record](VALIDATION.md#facevitals-tracked-roi-integration-6-november-2026)
for report paths, optional-test limits and independent review.

After the current-main merge and recorded interface addition, the full suite
passed 551 tests with 76 optional checks skipped. All 13 recorded-interface tests
also passed independently, including real Tk widget checks with synthetic video.
See the [recorded-interface validation](VALIDATION.md#facevitals-recorded-interface-6-november-2026).

Reports and dependency resolution are local under `results/facevitals-integration/`.
The shared editable installation was switched to this tested checkout without
changing dependency versions; `pip check` passed. Source/weight declarations were
audited; redistribution/submission clearance remains pending as documented in the
FaceVitals checkout's integration guide.

## C3 acceptance and licence preparation

FaceVitals local commit `b9922df` introduced a metadata-only preflight, an inactive
`docs/prereg/H-C3-TRACK.md` draft with proposed amendment A1, and a source/model
licence audit at `docs/methods/c3-tracker-licences.md` in the integration checkout.
The user subsequently approved A1 on 6 November 2026 for implementation. This does
not clear licences or bypass missing code, validation and input prerequisites.

The approved amendment uses a common scoring-only warmup/source-gap/target
visibility mask for all arms, while retaining full-timeline coverage and the
original criteria separately. It also freezes proposed target-level capture and
two-way bootstrap definitions. It changes no ROI, NaN-on-loss behavior, filtering,
windowing or HR estimator. A1 approval is recorded before implementation/scoring;
the original criteria have not been overwritten. No scored run has started.

From the FaceRec root, inspect a dataset version without scoring:

```powershell
.\.venv-shared\Scripts\python.exe worktrees/facevitals-integration/tools/c3_acceptance_preflight.py --data-root datasets/data/processed/c3_facevitals --version v0 --out results/facevitals-integration/c3-preflight-next
```

Use a new output folder; existing paths are refused. Exit 2 means incomplete
prerequisites. This does not render/decode video or compute BPM. The saved local
inventory found L0/L1 only, with 41 targets and 122 recipes. L2/L3 data remain
missing from that local version; generation, baseline and scoring code are now
implemented as described below.
The current raw capture table is not an acceptance decision. No scored run was launched.

Licence evidence is not submission clearance. FaceRec rights, the DeepSORT
original-to-fork grant chain, toolbox intended-use terms and final package notices
still require authorized decisions. Model/pipeline clearance allowlists were not
changed. See [validation](VALIDATION.md#c3-acceptance-and-licence-preparation-6-november-2026)
for the completed metadata and regression checks.

## Running approved A1 testing

FaceVitals checkpoint `193f738` implements the A1 scorer, all-detected-faces POS
baseline, L3s/L3l trajectories and matte-derived visibility, and a build/freeze/run
launcher. It reuses existing evaluation and pulse methods. The user approval was
committed first as `f81c2ff`. Tests use synthetic cases; dataset acceptance is NOT RUN.

From the FaceRec root, the laptop launch command is:

```powershell
& .\worktrees\facevitals-integration\tools\run_c3_acceptance.ps1 -Python .\.venv-shared\Scripts\python.exe -SourceRoot "D:\source_datasets" -DataRoot .\datasets\data\processed\c3_facevitals -TrackerModelDir .\.models -Version v2a1
```

Replace `SourceRoot` with the folder containing `UBFC-rPPG/DATASET_2` (the
previously supplied WSL datasets folder is one local option). Source videos stay
read-only; the data root gets a new recipe version. Existing L3 versions and
input folders are refused. The launcher builds all five layouts, freezes hashes
and eligibility, then scores only if the complete input gate passes. It does not
install dependencies or download weights. Long runs belong in the user's terminal
under the FaceVitals evaluation procedure. Runtime and disk use can be substantial.

The current local v0 preparation is **not ready**: missing L2/L3 scenes and 12/81
L1 pairs with fewer than three usable opportunity windows. Do not drop pairs,
relax coverage, or reinterpret generation success as acceptance. A fresh complete
cohort needs the same eligibility checks. Failed outputs are retained for inspection.

After a scored run, review FaceVitals
`results/E047-c3-track-acceptance/acceptance.md` and `acceptance.json`, its standard
evaluation tables and ledger entries. `COMPLETED` describes execution; inspect the
separate arm/layout PASS, FAIL, INVALID or INCOMPLETE verdicts. Original coverage
and A1 coverage remain separate. Technical errors are recorded INVALID. Per-run
render/trace caches stay local and must not be merged or shipped. Licence clearance
and the final independent PR/submission review remain separate requirements.

## Shared FaceVitals install

The shared pins match FaceVitals GitHub `main` at
[`4d92cc099b6644267511be70fb4542fe18748e3f`](https://github.com/Technion-SIPL-SPcup27/facevitals/blob/4d92cc099b6644267511be70fb4542fe18748e3f/pyproject.toml),
reviewed on 4 November 2026. Use a fresh checkout of that snapshot to reproduce
the setup, rather than installing an unverified local research branch:

```powershell
# Run from FaceRec after its quick-start installation. The destination must be new.
git clone --recurse-submodules --branch main https://github.com/Technion-SIPL-SPcup27/facevitals.git ..\facevitals-shared
git -C ..\facevitals-shared checkout 4d92cc099b6644267511be70fb4542fe18748e3f
git -C ..\facevitals-shared submodule update --init --recursive
.\.venv-shared\Scripts\python.exe -m pip install -e "..\facevitals-shared[dev,dl]" -e ".[test,benchmark]"
.\.venv-shared\Scripts\python.exe -m pip check
.\.venv-shared\Scripts\python.exe -m facevitals.methods
```

Private-repository access is required. Both packages use the same interpreter
and environment; there is no second FaceVitals venv. Editable installs keep the
FaceVitals checkout and its legacy/submodule resources available. Keep that
checkout in place. Recheck upstream `main` and resolve changes before choosing
a newer revision; a snapshot match does not guarantee future pin compatibility.

On 6 November 2026, GitHub `main` was reviewed again at
[`cbbbc3a`](https://github.com/Technion-SIPL-SPcup27/facevitals/blob/cbbbc3af55f9529fcc8eef058adbd0aca9c0d196/pyproject.toml).
Its dependency configuration is unchanged from the tested snapshot above.
The baseline setup commands intentionally retain the original tested revision.
The newer local integration checkout and its separate test record are described
above; activating it replaces the editable FaceVitals target, not the shared pins.

This installs FaceVitals' Python package and optional deep-learning dependencies.
The legacy Kivy GUI has additional dependencies outside its `pyproject.toml`;
follow the upstream GUI instructions if you need that interface. Installing the
package alone does not establish GUI readiness or implement the pending
[full-timeline validity adapter](../README.md#facevitals-handoff).

## Why these pins exist

- Python 3.12, MediaPipe 0.10.14, NumPy 1.26.4, SciPy 1.17.1 and setuptools
  80.10.2 match FaceVitals. MediaPipe retains `mp.solutions.face_detection`;
  tasks-only releases are not direct replacements. NumPy stays on 1.x for ABI
  compatibility and FaceVitals' legacy `np.mat` usage.
- Torch 2.14.1 and TorchVision 0.29.1 run together. The CPU wheel suffix `+cpu`
  satisfies these pins. Both remain installed for the optional MobileNet backend;
  the default SFace backend uses OpenCV ONNX models. GPU execution is not validated.
- DeepSORT stays pinned to upstream commit
  `3341f823ee0ad20aa58c906079a37f3d43b42bbe`, which clears pending appearance
  features after fitting them. Its reported version 1.3.2 does not identify this
  source revision.
- Upstream dependencies require both `opencv-python` and `opencv-contrib-python`.
  They share `cv2` and use the same 4.11.0.86 version. Use a fresh environment;
  avoid headless wheels and uninstalling one overlapping wheel in place.
- Setuptools stays below 81 for upstream `pkg_resources` imports. Pytest 9.1.1
  matches the FaceVitals development pin.

These are top-level pins, not a full transitive dependency lock. The environment
alignment changes installation requirements, not tracking thresholds, ROI bounds,
recording timestamps or pulse algorithms.

## Verification history

See [Windows environment and model checks](VALIDATION.md#windows-environment-and-model-checks-4-november-2026)
for the original shared installation, model tests and MATLAB-related FaceVitals
test failures. That history is not a claim that the GUI or the latest upstream
snapshot has passed locally. The isolated FaceVitals checkout under ignored
`results/environment-alignment/` preserves the original validation source. The
current editable install uses `worktrees/facevitals-integration`; keep that checkout
in place or reinstall from another verified checkout.

## Model files

The SFace backend uses YuNet landmarks and aligned SFace embeddings from OpenCV
Zoo. Sources, commit pins and SHA256 values are in
[`models.py`](../src/c3_tracker/models.py). `python -m c3_tracker.models --download`
is an explicit setup step; startup verifies the files without downloading them.
Weights live in ignored `.models/` by default. `--model-dir` changes that location.

Dependency troubleshooting is in the [usage reference](USAGE.md#troubleshooting).
