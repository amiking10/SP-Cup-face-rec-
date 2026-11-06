"""Export the desktop tracker and its regression tests without research/data files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MODULES = (
    '__init__', '__main__', 'app', 'association', 'controls', 'detection',
    'embedding', 'logging_utils', 'models', 'paired_recording', 'roi',
    'target_manager', 'tracking', 'visualization',
)
RUNTIME_TESTS = (
    'test_association', 'test_detection', 'test_desktop', 'test_inputs',
    'test_logging', 'test_models', 'test_pair_mark', 'test_paired_recording',
    'test_roi', 'test_target_manager', 'test_tracking',
)
COPIED_FILES = (
    *(f'src/c3_tracker/{name}.py' for name in RUNTIME_MODULES),
    *(f'tests/{name}.py' for name in RUNTIME_TESTS),
    '.gitignore',
)
GENERATED_FILES = ('pyproject.toml', 'README.md', 'docs/USAGE.md', 'runtime_manifest.json')


def runtime_project(config):
    """Keep canonical runtime/test pins without the research dependency extra."""
    project = config['project']
    lines = ['[build-system]']
    for key, value in config['build-system'].items():
        lines.append(f'{key} = {json.dumps(value)}')
    lines.extend(['', '[project]'])
    for key in ('name', 'version', 'description', 'requires-python', 'dependencies'):
        lines.append(f'{key} = {json.dumps(project[key])}')
    lines.extend([
        '', '[project.optional-dependencies]',
        f'test = {json.dumps(project["optional-dependencies"]["test"])}',
        '', '[tool.setuptools.packages.find]', 'where = ["src"]',
        '', '[tool.pytest.ini_options]', 'testpaths = ["tests"]', '',
    ])
    return '\n'.join(lines)


def runtime_readme(config):
    dependencies = config['project']['dependencies']
    cpu_pins = ' '.join(next(pin for pin in dependencies if pin.startswith(name + '=='))
                        for name in ('torch', 'torchvision'))
    return rf'''# FaceRec runtime integration bundle

This bundle contains selected-face tracking, the desktop interface, paired video
export and relevant regression tests. It is a reviewable source export for later
FaceVitals integration; generating it does not modify or merge into FaceVitals.
Dataset adapters, evaluation commands and historical results stay in FaceRec.

## Install and run

Use Python 3.12. Run from this bundle's root. For an existing shared Python 3.12
environment, use its interpreter instead of creating another environment below.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install {cpu_pins} --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m c3_tracker.models --download
.\.venv\Scripts\python.exe -m c3_tracker --source webcam
```

If `py` is unavailable, use the installed Python 3.12 executable to create the venv.
Model downloads are explicit and hash-verified. Weights and test photographs are
not included. Track IDs are associations within a run, not verified identities.

## Export a clean video pair

```powershell
.\.venv\Scripts\python.exe -m c3_tracker --source video --input "C:\videos\base.mp4" --record-pair --output-dir exports
```

Use a clean constant-frame-rate file. Click a confirmed face to select it. Recording
starts with the first valid selection. Q quits, R resets, S/Space saves a screenshot,
and O opens the output folder. The green capture box extends 20% above the detected
face by default; the four ROI options adjust it further. It is not a skin mask.

Outputs are `exports/<run-id>/original.mp4`, `face_centered.mp4`,
`paired_recording.json` and `frame_validity.csv`, plus the usual run logs. The MP4s
share the source FPS and interval, have no audio and contain no drawn overlays.
With the default `stop` loss policy both videos stop and save at first target loss.
Webcam paired recording and headless target selection are not supported.

Open each uninterrupted MP4 manually in FaceVitals' existing GUI to process it.
`--pair-loss-policy mark` keeps missing intervals as invalid FACE LOST frames;
it requires a consumer that reads the validity CSV, resets pulse processing at gaps
and waits for a fresh valid window after return. The current GUI does not read
that CSV. This bundle does not calculate BPM or add that GUI integration.

See the [option and output tables](docs/USAGE.md) for every desktop CLI argument.

## Verify the runtime

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Model tests are opt-in with `C3_RUN_MODEL_TESTS=1`. They also need the explicit
`python -m c3_tracker.models --download --test-fixtures` setup. Ordinary tests
use synthetic pixels/mocks and do not download models or datasets. Passing tests
does not establish identity or pulse accuracy. GPU execution is not validated.

`runtime_manifest.json` lists SHA256 hashes, the source Git commit when available,
and whether the source tree had uncommitted changes. This is a selected export,
not the complete research checkout. Keep runtime outputs outside Git.
'''


def runtime_usage(text):
    # Reuse the maintained tables instead of maintaining another CLI reference.
    start = text.index('## Command-line options\n')
    end = text.index('## Controls and selected target\n', start)
    tables = text[start:end].replace(
        '(../datasets/BENCHMARK_GUIDE.md)',
        '(https://github.com/amiking10/SP-Cup-face-rec-/blob/main/datasets/BENCHMARK_GUIDE.md)')
    return '# FaceRec runtime reference\n\nSee [installation and recording](../README.md).\n\n' + tables


def source_revision(root):
    try:
        commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, check=True, timeout=10)
        status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'],
                                capture_output=True, text=True, check=True, timeout=10)
        return {'commit': commit.stdout.strip(), 'dirty': bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'dirty': None}


def export_runtime(destination, *, source_root=ROOT):
    root = Path(source_root).resolve()
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Export destination already exists; choose a new folder: {destination}')
    files = {}
    for relative in COPIED_FILES:
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Export source must be a regular file inside the repository: {relative}')
        files[relative] = path.read_bytes()
    project_bytes = (root / 'pyproject.toml').read_bytes()
    config = tomllib.loads(project_bytes.decode('utf-8'))
    files['pyproject.toml'] = runtime_project(config).encode('utf-8')
    files['README.md'] = runtime_readme(config).encode('utf-8')
    files['docs/USAGE.md'] = runtime_usage((root / 'docs/USAGE.md').read_text(encoding='utf-8')).encode('utf-8')
    manifest = {
        'schema_version': 1, 'purpose': 'FaceRec runtime integration bundle',
        'source': source_revision(root),
        'source_pyproject_sha256': hashlib.sha256(project_bytes).hexdigest(),
        'files': {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
    }
    files['runtime_manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode('utf-8')
    destination.mkdir(parents=True)
    for relative, data in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return manifest


def verify_runtime(directory):
    """Check the curated inventory as well as hashes; never trust extra manifest entries."""
    directory = Path(directory).absolute()
    paths = {}
    for path in directory.rglob('*'):
        if path.is_symlink() or path.is_junction():
            raise ValueError(f'Linked paths are not part of a runtime bundle: {path}')
        if path.is_file():
            paths[path.relative_to(directory).as_posix()] = path
    expected = set(COPIED_FILES) | set(GENERATED_FILES)
    missing = expected - paths.keys()
    unexpected = paths.keys() - expected
    if missing or unexpected:
        raise ValueError(f'Bundle inventory differs: missing={sorted(missing)}, '
                         f'unexpected={sorted(unexpected)}')
    manifest = json.loads(paths['runtime_manifest.json'].read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1:
        raise ValueError('Unsupported runtime manifest schema')
    hashes = manifest.get('files')
    if not isinstance(hashes, dict) or set(hashes) != expected - {'runtime_manifest.json'}:
        raise ValueError('Runtime manifest contains missing or unexpected file entries')
    for relative, digest in hashes.items():
        if hashlib.sha256(paths[relative].read_bytes()).hexdigest() != digest:
            raise ValueError(f'Exported file changed: {relative}')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--output-dir', type=Path, default=Path('build/runtime'),
                        help='new export directory (default: build/runtime); existing folders are refused')
    modes.add_argument('--list-files', action='store_true', help='list bundle files without writing anything')
    modes.add_argument('--verify', type=Path, metavar='DIRECTORY',
                       help='check an existing bundle for changed, missing or unexpected files; no writes')
    args = parser.parse_args(argv)
    if args.list_files:
        print('\n'.join(sorted((*COPIED_FILES, *GENERATED_FILES))))
        return 0
    try:
        manifest = verify_runtime(args.verify) if args.verify is not None else export_runtime(args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Export failed: {exc}\n')
    action = 'verified' if args.verify is not None else 'saved'
    directory = args.verify if args.verify is not None else args.output_dir
    print(f'Runtime bundle {action}: {directory.resolve()} ({len(manifest["files"]) + 1} files)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
