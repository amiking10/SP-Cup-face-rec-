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

## Tracker-specific boundaries

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
