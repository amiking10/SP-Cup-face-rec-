# Implementation validation

This record separates runtime/package regressions from historical model and
dataset measurements. Use [Dependencies](DEPENDENCIES.md) for the current setup,
[the testing guide](TESTING_GUIDE.md) for procedures, and
[dataset reports](../datasets/README.md) for their original protocols and results.
Passing installation or export tests is not evidence of tracking or BPM accuracy.

## Final licence and merge review: 8 November 2026

**Verdict: BLOCKED.** GitHub upstream remains `83c544e`. Independent review of
FaceVitals `b5712f6` ran in a detached source-first checkout: **692 passed,
78 skipped**, health passed with the pre-existing paper-card warning, no merge
conflicts, and no added participant media/models. Initial Git ownership failures
were reproduced and resolved with a process-only safe-directory setting before
the clean test run; no global Git settings were changed. Reports and synthetic
clock/no-image reproductions remain at
`results/facevitals-integration/final-review-20261108/`.

The identified metadata admission defect is fixed in FaceVitals `1b7767b`:
both prohibited identities are rejected as targets or bystanders; exact frozen
target identities and bystander membership replace unsafe count-only admission.
Tests first reproduced the failures. Post-fix **700 passed, 76 skipped**, with
independent follow-up **50 focused tests passed**. Report:
`results/facevitals-integration/final-review-20261108/root-fixed-gates-regression.xml`.
The detached copy's extra skips include opt-in model checks unavailable there;
optional MATLAB/Tk/end-to-end checks remain outside the tested scope.

FaceVitals `3b189c7` preserves full pinned licence notices and the E048b
ledger-backed report, updates current focus, and records remaining decisions.
The user could not confirm FaceRec ownership; no grant was assigned. DeepSORT
inherited-code and toolbox intended-use approvals remain unresolved; the
[licence audit](../worktrees/facevitals-integration/docs/methods/c3-tracker-licences.md)
contains the decision matrix and draft request. Model/pipeline allowlists are
unchanged. A scoped installed-asset inventory is retained separately at
`results/facevitals-integration/licence-review-20261108/installed-inventory.json`.

The independent review also confirms incompatible baseline/tracked clocks affect
both partial pairing and full acceptance scoring. A common-clock correction
requires explicit approval before implementation; no ROI, signal processing,
windowing or thresholds changed during this review. No merge/push/external request
occurred. The clean disposable checkout was removed after its reports were saved;
all benchmark outputs were retained.

## Interrupted full run and cached partial check: 7 November 2026

The user stopped `E047-c3-track-acceptance` for runtime cost. The verified worker
was terminated after encoder interruption did not stop the evaluator. Retain its
123 completed baseline caches, generated recipes and `manual_stop.json`. There
is no standard acceptance report; full acceptance is NOT ESTABLISHED. Do not
resume against the original input freeze without reviewing changed source and
preregistration hashes.

FaceVitals `c35c63d` preregisters a descriptive one-target, three-scene check
(L0/L1/L3s, same partner), selected by numeric identity and existing-cache presence
before new BPM results. `experiments/e048_c3_partial.py --baseline-only` recovered
E048a in one second without rendering or inference. Its ledger-backed report is
[here](../worktrees/facevitals-integration/results/E048a-c3-cached-baseline/summary.md),
recorded in FaceVitals `b5712f6`: three clips / 21 windows / N=1, descriptive only.
E048b designated-tracking comparison completed, but no finite paired windows
match because of the clock mismatch. Its marginal results are descriptive only.
No pulse, ROI or
window logic changed. The three-scene comparison uses a 12 GB render budget to
retain all its renders. Full 41-target acceptance and licences remain pending.

Validation: **694 passed, 76 skipped**; report
`results/facevitals-integration/e048-partial-runner-final-regression.xml`.
An initial run exposed a flaky substring assertion matching `999` inside a SHA;
the replacement directly guards against reference-pulse reads. Initial failure
report retained. Repository health has the existing unverified paper warning.

## A2 implementation: 7 November 2026

FaceVitals approval `e20829a` precedes implementation `d45fb06`. A2 adds explicit
45-second timestamp-only source selection, native frame/reference alignment,
complete global pairing, protocol-specific freezing and a reference-only planning
command. A1 remains the default. Independent local review required and verified
exact-first-frame face detection and at most two partners per target. This is
engineering review, not merge approval or acceptance.

Initial full regression: **690 passed, 77 skipped**, report
`results/facevitals-integration/a2-regression-20261107.xml`. Optional MATLAB checks
were skipped with MATLAB removed from this test process's PATH because the local
MATLAB executable cannot start; submission/MATLAB compatibility is not validated.
Repository health passes with the existing unverified Bland/Altman paper warning.

The first real metadata check (`v3a2-plan-20261107`) retained INCOMPLETE evidence:
the new validator incorrectly rejected duplicate native timestamps in nine
sources, before reading reference pulses. The correction preserves existing
`frame_index_at` behavior without deduplicating or renumbering frames. Subsequent
planning must use a new version; failed evidence remains under the data root.

Correction committed as FaceVitals `df86ad0` after **692 passed, 76 skipped**
(`results/facevitals-integration/a2-native-clock-regression-20261107.xml`). The new
`v3a2-plan-20261107-nativeclock` reference-only check is **READY_FOR_RENDER**:
all 41 targets, 81 valid pairs, maximum three uses per bystander. Subject starts
were frozen before reference samples were read. No source video was decoded and
no candidate BPM was estimated. Provenance and protocol-file hashes are retained
at `results/facevitals-integration/a2-reference-readiness-20261107/readiness.json`.
This is input readiness, not scored performance or acceptance. Real first-frame
detection, rendered occlusion geometry and candidate scoring remain pending.
Long rendering/evaluation must run in the user's terminal per FaceVitals' skill.

The first user-terminal full build (`v3a2-20261107-full`) stopped before reference
pairing because the launcher passed PowerShell's provider-qualified UNC path to
Python (`Microsoft.PowerShell.Core\FileSystem::...`). The retained result is
INCOMPLETE with zero discovered sources, not an absent dataset or candidate
failure. The launcher now uses `ProviderPath` for native-program arguments and
checks that the source dataset folder is readable before generation. Direct path
verification and an argument-recording launcher smoke test pass; the smoke test
performs no generation or scoring. Retrying requires a fresh version name.
Full regression after this launcher correction: **692 passed, 76 skipped**,
report `results/facevitals-integration/a2-unc-launcher-regression.xml`.

## Dataset-window feasibility audit: 7 November 2026

**Latest user decision:** "Approve A2 and continue", superseding the initial
"Keep 40 seconds; report incomplete." Approval was committed in FaceVitals as
`e20829a` before implementation. The original first-40-second cohort remains
**INCOMPLETE**. A2 authorizes 45-second timestamp-selected intervals with all 41
targets and unchanged ROI, pulse processing and acceptance thresholds. Candidate
scoring remains **NOT RUN** until the new input gates pass.

Read-only metadata audit against FaceVitals source `0fb12c1` and upstream `83c544e`.
Candidate HR scoring remains NOT RUN. Source videos were not decoded or rendered,
no model/ROI/filter/window logic changed and existing recipes were not modified.

- All 41 stored source-gap masks match recalculation from their source timestamps.
  Original ground-truth timestamp lines were separately checked for subjects 30
  and 3 and match the stored arrays. Reference waveforms were not loaded or scored.
- Under first-40-second input and A1's common masks, subject 30 has only two solo
  L1 opportunities. Adding a bystander cannot increase that count. With the
  current centered crossing, 28 of 41 targets have fewer than three solo L3
  opportunities. Re-pairing alone cannot make the current construction complete.
- Existing 12 failed L1 pairs are explained by the union of source-gap exclusions
  plus the common warmup. Each pair's excluded window indices are retained locally.
- All stored sources cover 45 seconds (minimum 46.234 seconds). A diagnostic using
  45-second source intervals and deterministic timestamp-only starts finds a
  source-feasible interval for all 41 and a source-gap-only 82-pair assignment.
  HR separation, harmonic checks, lighting and actual matte geometry are not
  established by that necessary-but-insufficient feasibility graph.
- A2 was proposed in the FaceVitals preregistration, initially declined, then approved. It keeps the
  centered crossing and pulse/ROI/scoring settings but changes clip duration and
  source starts. The timestamp audit below remains evidence about the original
  construction, distinct from new A2 implementation and reference-pairing checks.

Evidence is local under `results/facevitals-integration/a1-window-feasibility-20261107/`:
`summary.json`, `source_timing_check.json`, `invalid_pairs.json`,
`proposed_a2_feasibility.json` and `provenance.json`. The latter records hashes
of the audited recipes/arrays and source revision. Alternative timing/interval
diagnostics were not applied or reported as a model result. No scored ledger run
was fabricated. Processing code is unchanged; no new source-code regression run
was needed for this read-only audit. Documentation health/whitespace were checked.

## A1 implementation and synthetic testing: 6 November 2026

Approval committed in FaceVitals as `f81c2ff` before implementation; source
checkpoint `193f738`. Added the pure scorer, all-faces baseline, L3s/L3l rendering
and frozen-input orchestration. No tracker, selected-face ROI, pulse formula,
filter or existing pipeline's HR clock was changed. The all-faces method is a new
explicitly named evaluation baseline, not a replacement for existing extraction.

- Full suite: 647 passed, 77 skipped, 7 upstream warnings. Report:
  `results/facevitals-integration/a1-implementation-release-tests.xml`.
- A final preflight layout-detection fix was then covered by 65 focused scorer,
  preflight and runner checks, all passed. Report:
  `results/facevitals-integration/a1-final-focused-tests.xml`.
- Planted data exercise PASS, FAIL, INVALID and INCOMPLETE; exact crossed-bootstrap
  omission/redraw rules; source/visibility masks; NaN availability; CP bounds;
  baseline-teeth gates; all 41 expected target IDs; immutable input hashes; and
  preservation/logging of technical failures. An actual synthetic pulse passed
  through the existing evaluator into A1 while original coverage remained failed.
- L3 tests verify pixel-centre rasterization, alpha>=0.5 visibility, prescribed
  short/long durations, deterministic trajectory replay, mismatched-mask rejection
  and refusal to overwrite old L3 outputs. Only synthetic videos were rendered.
- The all-faces baseline tests verify equal face weighting, existing POS reuse,
  actual enlarged sample pixels and full-clock missing-frame behavior.
- PowerShell launcher syntax and Python CLI help checked. Health: no errors,
  existing unverified paper-card warning retained. Initial independent review
  findings were fixed; a complete independent pre-merge review is still pending.

On committed source, preparation checked the existing local v0 metadata and five
model assets. It returned NOT RUN: missing acceptance layouts, inconsistent
cross-layout sets and 12 of 81 L1 pairs with fewer than three opportunity windows.
Evidence: `results/facevitals-integration/a1-existing-cohort-gate/inputs.json` and
`readiness.json`. No real video was decoded and no candidate HR score was computed
by this gate. Full source media hashes are deferred until metadata pass, so this
is not a completed input freeze. No historical pilot was relabelled as a new run.

Long laptop scoring uses the prepared user-terminal launcher under the evaluation
skill; no IGUM connection or long dataset job was started here. Missing data
eligibility, licence decisions, independent review and submission validation
remain separate gates. MATLAB and optional-test limitations are unchanged.

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
