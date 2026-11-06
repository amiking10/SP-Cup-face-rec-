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

The first adapter step is saved in a separate ignored FaceVitals checkout:
`worktrees/facevitals-integration`, branch `amita/c3-tracker-adapter`, local commit
`089db69`, based on upstream `cbbbc3a`. It is not vendored into FaceRec and has not
been pushed or merged into FaceVitals. The legacy FaceRec snapshot is unchanged.

The adapter provides `facevitals.tracking.track_target`: one centered-1.5x xywh box
per decoded frame, with NaNs for unconfirmed/missing targets. It ports the runtime
association core and adds first-frame designated selection and a separate automatic
first-second policy. Deleted-ID recovery, ROI registration, signal-gap handling and
GUI integration remain later steps. No scored dataset runs accompanied the port.

From this FaceRec root, run the new checkout's focused tests with the existing
shared interpreter; pytest uses that checkout's configured `src` import path:

```powershell
Push-Location worktrees/facevitals-integration
..\..\.venv-shared\Scripts\python.exe -m pytest -q tests/test_tracking.py tests/test_tracking_association.py
..\..\.venv-shared\Scripts\python.exe tools/repo_health.py
Pop-Location
```

The final focused run passed 45 tests with one opt-in real-model test skipped;
the real-model smoke check had also passed separately using existing verified
weights. The full Python suite passed 403 tests with 76 optional checks skipped,
before the final crossing regression was added; that added regression then passed
separately and is included in the focused count. MATLAB was excluded from that
test process's PATH because its installed executable fails standalone startup.
MATLAB submission validation remains unverified. Repository health has no errors
and one existing unverified paper-card warning. An independent precommit reviewer
found no confirmed blocker; this is not formal PR approval.

Reports and dependency resolution are local under `results/facevitals-integration/`.
The shared editable installation was not replaced. Source/weight declarations were
audited; redistribution/submission clearance remains pending as documented in the
FaceVitals checkout's integration guide.

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
This was a source review, not a new installation or full FaceVitals test run;
the reproducible setup commands intentionally retain the tested revision.

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
`results/environment-alignment/` backs the current editable install; keep it in
place or reinstall from another verified checkout.

## Model files

The SFace backend uses YuNet landmarks and aligned SFace embeddings from OpenCV
Zoo. Sources, commit pins and SHA256 values are in
[`models.py`](../src/c3_tracker/models.py). `python -m c3_tracker.models --download`
is an explicit setup step; startup verifies the files without downloading them.
Weights live in ignored `.models/` by default. `--model-dir` changes that location.

Dependency troubleshooting is in the [usage reference](USAGE.md#troubleshooting).
