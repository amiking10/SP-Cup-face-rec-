"""Export and verify the small tracking core; never include local research or data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MODULES = ('__init__','association','detection','embedding','models','target_manager','tracking')
RUNTIME_TESTS = ('test_association','test_detection','test_tracking','test_target_manager','test_models')
COPIED_FILES = (
    *(f'src/c3_tracker/{name}.py' for name in RUNTIME_MODULES),
    *(f'tests/{name}.py' for name in RUNTIME_TESTS),
    'pyproject.toml','.gitignore','docs/USAGE.md','docs/C3_BASELINE_LIMITATIONS.md',
)
GENERATED_FILES = ('README.md','runtime_manifest.json')


def runtime_readme():
    return """# FaceRec tracking-core export

Use Python 3.12: `python -m pip install ".[test]"`, then `python -m pytest -q`.
The default SFace tracker does not require Torch. Explicit model preparation:
`python -m c3_tracker.models --download --model-dir .models`.

See [the API and FaceVitals workflow](docs/USAGE.md) and
[tracking limits](docs/C3_BASELINE_LIMITATIONS.md). The maintained FaceVitals
implementation is on its
[runtime branch](https://github.com/Technion-SIPL-SPcup27/facevitals/tree/amita/c3-runtime-integration).
This export contains no desktop UI, dataset adapters, pulse methods, media or
weights. `runtime_manifest.json` records the source revision and file hashes.
Verify it using `tools/export_runtime.py --verify DIRECTORY` from the FaceRec
source checkout; that build tool is not part of the runtime export.

The [source archive](https://github.com/amiking10/SP-Cup-face-rec-/tree/archive/facerec-research-20261108)
retains the previous full tool. Source/third-party redistribution permissions
remain unresolved; this export grants no licence or submission approval.
"""


def export_runtime(destination, source_root=ROOT):
    root=Path(source_root).resolve()
    destination=Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(f'Existing export is retained: {destination}')
    files={relative:(root/relative).read_bytes() for relative in COPIED_FILES}
    files['README.md']=runtime_readme().encode('utf-8')
    manifest={
        'schema_version':1,'purpose':'FaceRec tracking-core source export',
        'source':source_revision(root),
        'source_pyproject_sha256':hashlib.sha256(files['pyproject.toml']).hexdigest(),
        'files':{name:hashlib.sha256(data).hexdigest() for name,data in sorted(files.items())},
    }
    files['runtime_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode('utf-8')
    destination.mkdir(parents=True)
    for relative,data in files.items():
        target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
    return manifest


def source_revision(root):
    try:
        commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, check=True, timeout=10)
        status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'],
                                capture_output=True, text=True, check=True, timeout=10)
        return {'commit': commit.stdout.strip(), 'dirty': bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'dirty': None}


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
