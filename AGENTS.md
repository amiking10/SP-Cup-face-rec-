# FaceRec core instructions

This is `amita/facevitals-runtime-cleanup`, a core-only source/reference branch.
Read [README.md](README.md), [usage](docs/USAGE.md) and
[limitations](docs/C3_BASELINE_LIMITATIONS.md). The previous source is preserved
on `archive/facerec-research-20261108` at `0ebd778`; keep that snapshot unchanged.

## Scope

Keep only detection, association, SFace embedding, tracking, target selection,
model verification and their fast tests. The seven runtime modules and five
runtime tests are listed explicitly in `tools/export_runtime.py`. Keep the core
independent of datasets, research helpers and the previous desktop/export UI.
Use Python 3.12 and the exact pins in `pyproject.toml`. SFace does not require
Torch; MobileNet dependencies are optional. No automatic model downloads.

FaceVitals is the maintained user-facing home. Its clean checkout is
`worktrees/facevitals-runtime-cleanup`, branch `amita/c3-runtime-integration`.
`worktrees/facevitals-integration` retains the complete research implementation.
Read current FaceVitals main and its AGENTS.md, CLAUDE.md, CONTRIBUTING.md and
applicable `.claude/skills/` before FaceVitals changes. Follow its independent
`.claude/agents/spcup27-pr-reviewer.md` review before proposing a merge.
Do not merge FaceRec history into FaceVitals, modify main, push without user
instruction, or treat a source export as merge/submission approval.

## Preserve behavior and evidence

Do not change association defaults, selected-target safety checks or model pins
as part of cleanup. FaceVitals uses centered 1.5x xywh boxes and NaN on loss.
Ask before changing its ROI, missing-frame, window, filter or BPM behavior.
Clock-policy approval and licence decisions remain open. The user said ownership
was uncertain and used Codex to adapt algorithms from repositories; no original
rights grant or third-party clearance may be inferred.

Datasets, model weights, recordings, results and other checkouts stay outside Git.
Their existing files are retained locally. Never recursively delete data as a
cleanup step. Historical source/tests/docs are in the remote archives. New
benchmarks need explicit task scope; do not start dataset runs for a cleanup.

## Validation

Run `python -m pytest -q`. Build a fresh core export and run its `--verify` check.
Test exported modules with imports forced to the export's own source directory.
Preserve the explicit inventory and hash verification. Keep concise setup/usage
here; FaceVitals policies, pulse algorithms and benchmark machinery stay there.
User-approved cleanup supersedes the old requirement to retain the desktop and
research entry points on this branch; those remain available in the archive.
