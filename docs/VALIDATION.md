# Implementation validation

Environment: Linux x86_64, Python 3.11.17, CPU torch 2.5.1+cpu / torchvision 0.20.1+cpu.
All runtime/test top-level versions match the pins documented in the README.

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

Not executed: Windows installation, webcam capture, desktop click/key interactions,
GPU, consenting two-person stationary/crossing experiments, independent identity
annotations or sustained benchmarks at all three resolutions. Follow TESTING_GUIDE.md
on the intended PC; do not infer C3 success from sample-photo frames or unit tests.
