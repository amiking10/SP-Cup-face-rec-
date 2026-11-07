# FaceRec contributor and agent instructions

Start with [README.md](README.md), [usage](docs/USAGE.md) and
[dependencies](docs/DEPENDENCIES.md). This repository supplies tracking and paired
video exports; FaceVitals supplies pulse processing.

## FaceVitals integration guidance

Read GitHub's current `main` when planning FaceVitals changes:

- [AGENTS.md](https://github.com/Technion-SIPL-SPcup27/facevitals/blob/main/AGENTS.md)
- [CLAUDE.md](https://github.com/Technion-SIPL-SPcup27/facevitals/blob/main/CLAUDE.md)
- [CONTRIBUTING.md](https://github.com/Technion-SIPL-SPcup27/facevitals/blob/main/CONTRIBUTING.md)
- [Maintenance procedure](https://github.com/Technion-SIPL-SPcup27/facevitals/blob/main/.claude/skills/spcup27-maintain/SKILL.md)

Use a consistent commit snapshot for each review and report its SHA. A stale local
FaceVitals checkout is not evidence of what upstream `main` currently supports.
Keep shared skills and policies maintained upstream; record tracker-specific
differences here. Once integrated, the FaceVitals root instructions also apply.

## Integration working branch

`legacy/facerec-standalone` preserves the working standalone tool at `ce9ed3a`.
Keep that snapshot unchanged. Continue FaceRec integration preparation on
`amita/facevitals-integration`; the earlier branches remain available.

Use FaceVitals' current upstream procedures rather than copying their contents:

| Work | Upstream procedure |
| --- | --- |
| Start or refresh an actual FaceVitals checkout | [spcup27-sync](https://github.com/Technion-SIPL-SPcup27/facevitals/blob/main/.claude/skills/spcup27-sync/SKILL.md) |
| Port the tracker as an ROI, with tests first | [spcup27-add-method](https://github.com/Technion-SIPL-SPcup27/facevitals/blob/main/.claude/skills/spcup27-add-method/SKILL.md) and the [C3 integration contract](https://github.com/Technion-SIPL-SPcup27/facevitals/blob/main/docs/methods/c3-tracker-integration.md) |
| Review a completed FaceVitals branch/PR independently | [spcup27-pr-reviewer role](https://github.com/Technion-SIPL-SPcup27/facevitals/blob/main/.claude/agents/spcup27-pr-reviewer.md); use a fresh review session with that file as its brief |
| Register a future scored comparison before running it | [spcup27-preregister](https://github.com/Technion-SIPL-SPcup27/facevitals/blob/main/.claude/skills/spcup27-preregister/SKILL.md), then [spcup27-evaluate](https://github.com/Technion-SIPL-SPcup27/facevitals/blob/main/.claude/skills/spcup27-evaluate/SKILL.md) |
| Finish changes to FaceVitals | [spcup27-maintain](https://github.com/Technion-SIPL-SPcup27/facevitals/blob/main/.claude/skills/spcup27-maintain/SKILL.md), its tests and repository health check |

Read the applicable role/skill before using it. The reviewer reports findings and
never merges or posts externally. Benchmark runs remain outside the initial port.
An actual FaceVitals implementation branch must start from its own current `main`;
do not merge FaceRec's repository history into FaceVitals. This FaceRec branch
records preparation, not a completed FaceVitals port or review.

The first actual FaceVitals adapter is in the separate ignored checkout
`worktrees/facevitals-integration`, branch `amita/c3-tracker-adapter`, based on
upstream `cbbbc3a`. Local commits `089db69` and `69f86df` implement the headless box
adapter, tracked ROI variants and gap-aware pulse processing. The shared
environment's editable FaceVitals install now points at this checkout; keep it in
place. Current upstream `83c544e` has also been merged into the FaceVitals branch.
The maintained `facevitals.recorded` Tk interface supports recorded input, target
selection and results review (initial local commit `9b4f7bb`); the legacy GUI remains read-only
and unchanged.
Local commit `2a61670` adds waveform/BPM plots and window-result displays from
existing outputs; it does not add the legacy physiological estimators or alter
processing. Preserve this distinction when describing GUI feature parity.
Local `b9922df` adds acceptance preflight and licence evidence. The user approved
H-C3-TRACK/A1 on 6 November 2026 for implementation; scoring is still NOT RUN while
complete eligible input data remain unavailable.
Metadata inventory is NOT RUN / NOT ACCEPTANCE; neither licence declarations nor
passing tests clear submission. See `docs/DEPENDENCIES.md` for the open gates.
Local `193f738` implements A1 scoring, the all-faces baseline, L3 generation and
frozen-run orchestration. Synthetic checks pass; real acceptance is still NOT RUN.
The retained v0 lacks layouts and has insufficient common windows for some pairs.
The 7 November timestamp audit proves the first-40-second construction cannot
admit all 41 targets under A1. After initially declining, the user explicitly
approved A2 on 7 November; approval is committed as FaceVitals `e20829a`. Implement
45-second timestamp-only source selection as specified, keeping original A1
incompleteness distinct. Candidate scoring remains NOT RUN until gates pass.
Do not drop targets or silently change data admission/thresholds beyond approved A2.
Long laptop jobs follow
FaceVitals' evaluation skill; broader independent review remains required before merge.
See `docs/DEPENDENCIES.md` for commands and remaining scope. Live processing,
deleted-ID recovery and scored dataset acceptance remain pending. These FaceVitals
commits have not been pushed or merged; they are not upstream functionality.

## Tracker-specific boundaries

- Preserve FaceVitals processing logic. Ask the user before departing from its
  integration contract or changing ROI extraction, missing-frame handling,
  filtering, windowing or BPM estimation. The user explicitly confirmed centered
  1.5x tracked face ROIs and NaN on loss; do not add legacy pixel reuse or forehead
  extraction implicitly. The current recorded UI uses the maintained pipeline,
  not the legacy Kivy backend.

- Python 3.12 and `pyproject.toml` define this component's installation. Keep shared
  pins aligned with FaceVitals; use the shared setup in `docs/DEPENDENCIES.md`.
- `tools/export_runtime.py` is the explicit integration file list. Its export
  contains the desktop runtime and relevant tests, not dataset adapters or
  evaluation commands. Keep runtime tests independent of research helpers; verify
  the exported copy as well as the full checkout before proposing integration.
  Run its `--verify DIRECTORY` check before transferring a pristine bundle;
  extra local artifacts must not be included in the integration.
- Keep the existing `src/c3_tracker/`, `tests/`, `docs/` and `datasets/` entry points
  usable. Review imports, documentation and historical run references before
  moving or removing files. `evaluate.py` and `evaluation.py` have different jobs.
- Preserve tracking defaults, selected-target behavior, ROI geometry, pixel crops
  and source/output timing during organization-only changes.
- Clean exports and annotated previews are different artifacts. Predicted boxes
  and FACE LOST placeholders are not valid face samples. A full-timeline consumer
  must honor `frame_validity.csv`; describe pending adapters as pending.
- Recordings, crops, weights, caches and participant sidecars stay outside Git.
  Keep existing outputs and experiment history. Use the ignored locations in the
  README or an external data directory; never delete recordings as a tidy-up.
- Do not claim tracking or BPM accuracy from successful file export. Preserve
  sources, versions and recorded evidence when documenting past runs.
- Keep future FaceVitals adapters outside its read-only `legacy/` and
  `third_party/` trees. GUI and reusable tracking code need separate interfaces.

## Change and verification scope

Work on a topic branch and keep changes reviewable. Source changes use existing
automated tests (`python -m pytest -q`); benchmark extras are documented separately.
Check documentation links and described CLI options when editing guides. Record
what was actually checked. New datasets, pilots and benchmarks require task scope;
a documentation cleanup does not authorize them. Follow FaceVitals' contribution
checks before a FaceVitals PR. Pushes and merges require the user's instruction.

## Documentation maintenance

Maintain the tracked root, `docs/` and `datasets/` Markdown files; generated bundles,
local dataset documentation and other repository checkouts are separate artifacts.
Keep CLI options in `docs/USAGE.md`, installation pins in `pyproject.toml`, validation
history in `docs/VALIDATION.md`, and dataset evidence in its recorded reports.
Distinguish current setup commands from historical measurements: retain recorded
revisions, environments, denominators and limitations. Never relabel old results as
a new run. Check local links and CLI examples after editing; regenerate a bundle
into a new directory if its copied documentation needs updating.
