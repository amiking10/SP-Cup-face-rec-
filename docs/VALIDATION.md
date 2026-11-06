# Implementation validation

This record separates runtime/package regressions from historical model and
dataset measurements. Use [Dependencies](DEPENDENCIES.md) for the current setup,
[the testing guide](TESTING_GUIDE.md) for procedures, and
[dataset reports](../datasets/README.md) for their original protocols and results.
Passing installation or export tests is not evidence of tracking or BPM accuracy.

## C3 acceptance and licence preparation: 6 November 2026

Local FaceVitals commit `b9922df`, against current upstream `83c544e`. This is
engineering/readiness work, not a scored dataset run or completed licence clearance.

- Metadata-only preflight inventories recipes/arrays by SHA256, source existence
  and stats, common source-gap windows, pairing metadata and excluded subjects.
  Its status is always NOT RUN / NOT ACCEPTANCE while audited prerequisites are
  missing; it refuses to overwrite prior reports.
- Existing local `v0`: 122 recipes, L0=41 and L1=81, both spanning 41 targets;
  203 referenced source entries reachable, no per-recipe validation problems and
  no excluded subject. L2/L3s are absent from this local inventory. Evidence:
  `results/facevitals-integration/c3-preflight-v0-20261106/manifest.json` and
  `readiness.json`. Source stats are not a replacement for final media-content hashes.
- No video was decoded or rendered, no reference signal values were loaded and
  no BPM scores were generated. Historical pilots/cohort outputs were retained;
  none were relabelled as acceptance of the current port.
- TDD red-to-green and independent focused review: 25 preflight tests passed.
  Malformed subject/pairing metadata guards and presence checks for every person's
  PPG key were added without reading those signals.
- Full suite: 582 passed, 76 optional checks skipped, 7 upstream warnings. Report:
  `results/facevitals-integration/acceptance-preparation-full-python-tests.xml`.
  MATLAB was excluded from this test process's PATH as in the preceding runs;
  submission validation is not claimed.
- Health: no errors, existing unverified paper-card warning retained. Independent
  review resolved the proposed bootstrap's zero-partner ambiguity. This was
  precommit engineering review, not formal PR approval.

The inactive H-C3-TRACK draft retains original thresholds and proposes A1 for
explicit review: common opportunity-window coverage and defined capture/bootstrap
statistics. The user authorized preparing an amendment, not activating it.
Clean-source 40-second clips expose a 6/7 initial-coverage ceiling; source-gap
exclusions can change that denominator. No processing change was made to hide it.
Missing layouts, baseline and scoring implementation still prevent full acceptance.

The licence audit verified primary-source declarations at exact dependency/model
pins and recorded active model hashes plus unused bundled assets. FaceRec rights,
DeepSORT provenance, toolbox use and final redistribution notices remain open;
no allowlist was relaxed. These commits are local and unpublished.

## Recorded GUI signals and results: 6 November 2026

Local FaceVitals commit `2a61670` adds display-only waveform/BPM plots, current
BPM/quality, a window table and labelled accepted-window mean, plus an activity
indicator and elapsed processing time. Video, Signals and Results use notebook
tabs; the shared slider moves graph cursors without modifying processing results.

- Tests first: new presentation-helper imports failed before implementation.
- Full suite: 556 passed, 77 skipped, 7 upstream warnings. Report:
  `results/facevitals-integration/gui-signals-full-python-tests.xml`.
  One Tk error-path test skipped because its display was unavailable in that
  full run; it passed in the independent focused suite of all 19 GUI tests.
  Other optional-test limits and MATLAB PATH exclusion are unchanged.
- Independent review found and resolved stale BPM/quality/cursors after preview
  decode failure. A synthetic widget regression verifies they clear. No confirmed
  major issue remained; this was engineering review, not formal PR approval.
- Synthetic Signals/Results screenshots were rendered and visually inspected.
  Local screenshots remain under `results/facevitals-integration/`; no participant
  images or dataset recordings were used.
- Health: no errors; the existing unverified paper-card warning remains. Whitespace
  was checked. No dependencies, legacy sources or backend files changed.

Waveform gaps are not interpolated. The summary mean is presentation of accepted
window BPMs, not a new HR estimator. No SpO2, HRV, respiration, stress estimator,
HeartPy finalization or processing cancellation was added. Tracking, 1.5x ROIs,
NaN-on-loss, extraction, filtering and BPM windowing are unchanged. These changes
remain local and unpublished; actual-video manual checks are still required.

## First-frame face selection: 6 November 2026

Local FaceVitals commit `748a675` replaces manual rectangle drawing with detected
first-frame face boxes and explicit click selection. Processing stays disabled
without a selected detection, including no-face and detector-error cases.

- Tests first: updated tests failed before the new selection helpers existed.
- Full suite: 554 passed, 76 optional checks skipped, 7 upstream warnings. Report:
  `results/facevitals-integration/face-selection-full-python-tests.xml`.
- Independent focused review: all 16 recorded-interface checks passed, including
  scaled click mapping, selection switching, blank clicks and failure cases.
  No confirmed major issue remained; this was not formal PR approval.
- Repository health: no errors; the existing unverified paper-card warning remains.
  Changed Markdown file targets and whitespace were checked.

The user explicitly confirmed centered 1.5x tracked face boxes and NaN on loss.
Pulse processing, filtering, windowing, ROI enlargement and missing-frame policy
were unchanged. The maintained processing backend remains active; legacy forehead
extraction, pixel reuse and HeartPy finalization were not substituted. Future
departures from the FaceVitals contract require permission. MATLAB exclusion and
optional-test limits are the same as the recorded-interface run below. No dataset
or personal-video processing was performed. This commit is local and unpublished.

## FaceVitals recorded interface: 6 November 2026

Local FaceVitals commit `9b4f7bb` adds recorded-video target drawing, background
processing and timeline review. Upstream main `83c544e` was fetched, inspected and
merged as `d864972` before this change; its additions were paper/backlog updates.

- Tests were written first and failed because the new module did not yet exist.
- Full Python suite: 551 passed, 76 skipped, 7 upstream warnings. Report:
  `results/facevitals-integration/recorded-full-python-tests.xml`.
- Independent focused run: all 13 recorded-interface tests passed, including real
  Tk widgets driven with synthetic video. No participant video was processed.
- Review found and resolved method-selector mislabeling and an EOF window display
  boundary error, with regressions for both. No confirmed major issue remained.
  This is precommit engineering review, not formal PR approval.
- Repository health: no errors; the pre-existing unverified paper-card warning
  remains. CLI help, changed Markdown file targets and whitespace were checked.

The same shared Python environment loads the new module; no dependency was added.
The legacy GUI, legacy branch and read-only third-party sources were preserved.
MATLAB was excluded only from the test process's PATH, as in the preceding run;
the skipped optional checks and submission validation are not claimed to pass.
No dataset benchmark or new accuracy measurement was run. Actual-video desktop
interaction remains a [manual check](TESTING_GUIDE.md#recorded-facevitals-interface-check).
Live processing, recovery after ID expiry and dataset acceptance remain pending.
These FaceVitals changes are local and have not been pushed or merged upstream.

## FaceVitals tracked ROI integration: 6 November 2026

Local FaceVitals branch `amita/c3-tracker-adapter`, commit `69f86df`, based on
GitHub main `cbbbc3a`. Source is in ignored `worktrees/facevitals-integration`.

- Full Python suite: 534 passed, 76 skipped; report
  `results/facevitals-integration/roi-full-python-tests.xml`.
- Final focused gap, extraction and process checks: 53 passed; report
  `results/facevitals-integration/roi-final-focused-tests.xml`. This includes four
  late invalid-designation cases added after full-suite collection.
- Real-model smoke: 1 passed using existing verified weights, without downloads.
- Synthetic video-to-BPM regression verifies the valid signal, missing intervals,
  fresh windows after gaps and CSV output. Cache regressions cover changed model
  contents, configuration and composite rendering inputs.
- Independent precommit review found a composite cache invalidation bug; it was
  fixed and covered by a regression. No confirmed blocker remained. This is
  engineering review, not formal PR approval.
- Repository health: no errors, one pre-existing unverified paper-card warning.
- Shared editable install now loads this checkout, while FaceRec still loads its
  own `src/`; dependency versions are unchanged and `pip check` passes.

MATLAB was excluded from the test process's PATH because its installed executable
fails standalone startup. Optional checks were skipped; MATLAB submission
validation and GUI readiness remain unverified. No dataset benchmark or new
accuracy measurement was run. Deleted-ID recovery and submission/licence clearance
remain pending. The FaceVitals commits have not been pushed or merged.

## Windows maintenance checks: 6 November 2026

Python 3.12.10 in `.venv-shared`, with the pins in `pyproject.toml`:

- `python -m pytest -q tests datasets/tests`: 190 passed, 8 skipped. Report:
  `results/repo-maintenance-20261106/full-tests.xml`.
- `python -m pip check`: no broken requirements.
- The desktop argument table was checked against all 29 option names, including
  the help alias. Local documentation targets and anchors were checked.
- `python tools/export_runtime.py --verify build/runtime`: the existing 30-file
  bundle passed inventory and SHA256 checks. New regressions reject changed,
  missing and extra files, including extra entries added to the manifest itself.
- A fresh bundle generated from committed source `191a59a` was also verified under
  `build/runtime-20261106-191a59a/`, with `source.dirty=false`.

These checks did not run dataset pilots, modify FaceVitals or validate its GUI.
The GitHub workflow is configured to check both runtime exports and the research
checkout; these local results do not assert that a remote CI run passed.

## Windows environment and model checks: 4 November 2026

Installed FaceRec and FaceVitals snapshot
[`4d92cc0`](https://github.com/Technion-SIPL-SPcup27/facevitals/tree/4d92cc099b6644267511be70fb4542fe18748e3f)
in one Windows Python 3.12.10 environment, including tracker `test,benchmark` and
FaceVitals `dev,dl` extras. Both pinned FaceVitals submodules were initialized.
Existing environments and the neighboring research checkout were preserved.

- Initial tracker regression run: 177 passed, 8 skipped.
- `C3_RUN_MODEL_TESTS=1` on `tests/test_models.py` and
  `tests/test_benchmark_models.py`: 6 passed, covering SFace, MobileNet,
  loss/reacquisition and headless recorded-video artifacts.
- After runtime/test separation: full checkout 184 passed, 8 skipped. A built wheel
  installed into an ignored verification directory passed 87 exported tests, with
  5 opt-in model tests skipped. Isolated imports confirmed the benchmark package
  and both evaluators were absent. The shared editable installation was preserved.
- FaceVitals' full suite initially returned 333 passed, 3 skipped, 2 failures and
  72 setup errors. Both missing-ME-rPPG-model failures passed on recheck after that
  submodule was initialized. All 72 setup errors came from a MATLAB build fixture;
  standalone `matlab -batch "disp(version)"` also exited with status 1. The full
  FaceVitals suite therefore did not pass on this machine.

Install/version records and JUnit reports remain under ignored
`results/environment-alignment/` and `results/runtime-export-check/`. These were
regression checks, not new dataset accuracy evaluations or interactive GUI checks.

## Historical Linux smoke check

Historical environment: Linux x86_64, Python 3.11.17, CPU torch 2.5.1+cpu / torchvision 0.20.1+cpu.
These were the pins at the time of this run. Current setup and shared-environment
verification are in [Dependencies](DEPENDENCIES.md).

Executed:

- `python -m c3_tracker --help`: CLI available.
- `uv pip check --python /workspace/c3-venv/bin/python`: all 44 installed packages compatible.
- `python -m pytest -q`: 27 deterministic checks passed (model checks skipped by default).
- `C3_RUN_MODEL_TESTS=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest -q`:
  29 passed, with three upstream pkg_resources/torch.load deprecation warnings.
- Actual-model CLI smoke: 12 CFR video frames containing two copies of the bundled
  Matplotlib sample photograph, MediaPipe + DeepSORT on CPU, headless with recording.
  CSV contained two raw IDs across all 12 frames and 20 confirmed observed rows
  (confirmation begins on frame 2); no target selected in headless mode.
  Metadata and summary written with no error; output MP4 decoded to exactly 12 frames.
- Runtime evaluator on that log: 12 frames; identity metrics deliberately not calculated.
- `git diff --check`: no whitespace errors.

The short smoke run measured about 24 processing FPS with two CPU threads, excluding
model initialization. This is a one-off pipeline check, not sustained performance evidence.

Not executed in this historical Linux run: Windows installation, webcam capture, desktop click/key interactions,
GPU, consenting two-person stationary/crossing experiments, independent identity
annotations or sustained benchmarks at all three resolutions. Follow TESTING_GUIDE.md
on the intended PC; do not infer C3 success from sample-photo frames or unit tests.
