# FaceRec tracking core for FaceVitals

This branch contains the reusable tracking core and fast tests. FaceVitals owns
recorded-video processing, its GUI, pulse methods and BPM output. FaceRec is a
migration/reference source package; do not maintain a second pulse workflow here.

## Install and test

Use Python 3.12. In an existing shared environment:

```powershell
.\.venv-shared\Scripts\python.exe -m pip install -e ".[test]"
.\.venv-shared\Scripts\python.exe -m pytest -q
```

The default SFace backend needs no Torch/TorchVision installation. The optional
`mobilenet` extra retains the older backend for explicit callers. Model downloads
are explicit and verified:

```powershell
python -m c3_tracker.models --download --model-dir .models
```

Tests use synthetic inputs and fake model files; they do not download datasets,
weights or photographs. See [API and FaceVitals workflow](docs/USAGE.md) and
[tracking limitations](docs/C3_BASELINE_LIMITATIONS.md).
Cleanup verification on 8 November 2026: 53 core/export-tool tests and 43 isolated
exported-core tests passed. The wheel inventory contains exactly seven modules,
no media/weights and no mandatory Torch dependency. These are software/package
checks, not dataset acceptance or licence clearance.

## FaceVitals integration

The maintained destination is
[FaceVitals' runtime branch](https://github.com/Technion-SIPL-SPcup27/facevitals/tree/amita/c3-runtime-integration).
It uses `facevitals.tracking.track_target`, tracked ROIs and the existing pulse
pipeline. There is no standalone `python -m c3_tracker` desktop command on this
branch. Use `python -m facevitals.recorded` in the FaceVitals checkout instead.

A source-only export can be created from this repository:

```powershell
python tools/export_runtime.py --output-dir build/core
python tools/export_runtime.py --verify build/core
```

The export contains the seven core modules, five core test files and short guides;
it does not contain a desktop UI, research scripts, datasets, results or weights.
The curated manifest includes SHA256 hashes. Existing destinations are refused.

## Preserved history and data

The complete previous desktop/export/research source is preserved at
[archive/facerec-research-20261108](https://github.com/amiking10/SP-Cup-face-rec-/tree/archive/facerec-research-20261108)
(commit `0ebd778`). The separate FaceVitals research branch is archived at
[archive/c3-integration-research-20261108](https://github.com/Technion-SIPL-SPcup27/facevitals/tree/archive/c3-integration-research-20261108).
Research tests and historical reports belong to those snapshots. Local datasets,
recordings, results, environments and model files are ignored and retained on disk.

## Clearance and validation boundaries

Repository ownership does not establish rights to all copied/adapted code.
FaceRec ownership/contributor grants, DeepSORT inherited-code permissions and
selected rPPG-Toolbox intended-use approval remain unresolved. Read the
[licence evidence and required decisions](https://github.com/Technion-SIPL-SPcup27/facevitals/blob/amita/c3-runtime-integration/docs/methods/c3-tracker-licences.md).
No MIT licence is assigned to the whole project. Do not redistribute models or
claim competition clearance from passing tests. The benchmark clock mismatch
also remains open; this cleanup does not alter ROI or pulse timing policies.
