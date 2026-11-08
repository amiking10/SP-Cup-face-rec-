"""The integration bundle must run without access to the research package."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def exporter():
    spec = importlib.util.spec_from_file_location('runtime_export', ROOT / 'tools/export_runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_is_an_explicit_runtime_inventory(tmp_path, exporter):
    destination = tmp_path / 'runtime'
    exporter.export_runtime(destination)
    expected_modules = {
        '__init__', '__main__', 'app', 'association', 'controls', 'detection',
        'embedding', 'logging_utils', 'models', 'paired_recording', 'roi',
        'target_manager', 'tracking', 'visualization',
    }
    actual = {p.stem for p in (destination / 'src/c3_tracker').glob('*.py')}
    assert actual == expected_modules
    assert not (destination / 'src/c3_tracker/benchmark').exists()
    assert not (destination / 'src/c3_tracker/evaluate.py').exists()
    assert not (destination / 'src/c3_tracker/evaluation.py').exists()
    assert {p.name for p in destination.iterdir()} == {
        'src', 'tests', 'docs', 'README.md', 'pyproject.toml', '.gitignore', 'runtime_manifest.json',
    }
    for name in ('test_logging.py', 'test_pair_mark.py', 'test_paired_recording.py', 'test_models.py'):
        assert (destination / 'tests' / name).is_file()
    assert not (destination / 'tests/test_evaluation.py').exists()
    assert not (destination / 'tests/test_runtime_export.py').exists()
    assert not any(p.suffix in {'.mp4', '.csv', '.onnx', '.jpg', '.npz'}
                   for p in destination.rglob('*'))
    manifest = json.loads((destination / 'runtime_manifest.json').read_text())
    files = {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}
    assert set(manifest['files']) == files - {'runtime_manifest.json'}
    for relative, digest in manifest['files'].items():
        data = (destination / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        if relative.startswith(('src/', 'tests/')):
            assert data == (ROOT / relative).read_bytes()


def test_export_keeps_canonical_pins_but_not_benchmark_extras(tmp_path, exporter):
    exporter.export_runtime(tmp_path / 'runtime')
    original = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    exported = tomllib.loads((tmp_path / 'runtime/pyproject.toml').read_text())
    for name in ('name', 'version', 'description', 'requires-python', 'dependencies'):
        assert exported['project'][name] == original['project'][name]
    assert exported['build-system'] == original['build-system']
    assert exported['project']['optional-dependencies'] == {
        'test': original['project']['optional-dependencies']['test'],
    }


def test_export_rejects_existing_destination_without_changing_it(tmp_path, exporter):
    destination = tmp_path / 'existing'
    destination.mkdir()
    sentinel = destination / 'recording.mp4'
    sentinel.write_bytes(b'preserve existing output')
    with pytest.raises(FileExistsError):
        exporter.export_runtime(destination)
    assert sentinel.read_bytes() == b'preserve existing output'
    assert list(destination.iterdir()) == [sentinel]


def test_missing_source_does_not_create_partial_export(tmp_path, exporter):
    with pytest.raises(FileNotFoundError):
        exporter.export_runtime(tmp_path / 'output', source_root=tmp_path / 'missing-source')
    assert not (tmp_path / 'output').exists()


def test_exported_cli_and_imports_cannot_use_original_package(tmp_path, exporter):
    destination = tmp_path / 'runtime'
    exporter.export_runtime(destination)
    script = '''
import importlib, importlib.util, pathlib, sys
source = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source))
import c3_tracker
assert pathlib.Path(c3_tracker.__file__).is_relative_to(source)
for name in ('app', 'association', 'controls', 'detection', 'embedding',
             'logging_utils', 'models', 'paired_recording', 'roi',
             'target_manager', 'tracking', 'visualization'):
    module = importlib.import_module('c3_tracker.' + name)
    assert pathlib.Path(module.__file__).is_relative_to(source)
for name in ('benchmark', 'evaluate', 'evaluation'):
    assert importlib.util.find_spec('c3_tracker.' + name) is None
from c3_tracker.app import parser
options = parser().parse_args(['--source', 'video', '--input', 'base.mp4', '--record-pair'])
assert options.record_pair and options.pair_loss_policy == 'stop'
assert '--record-pair' in parser().format_help()
'''
    result = subprocess.run([sys.executable, '-I', '-c', script, str(destination / 'src')],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_exported_documentation_has_no_missing_local_targets(tmp_path, exporter):
    destination = tmp_path / 'runtime'
    exporter.export_runtime(destination)
    for path in (destination / 'README.md', destination / 'docs/USAGE.md'):
        text = path.read_text(encoding='utf-8')
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            if '://' in link or link.startswith('#'):
                continue
            target = link.split('#')[0]
            assert (path.parent / target).exists(), (path.name, link)
    assert '## Command-line options' in (destination / 'docs/USAGE.md').read_text()
    assert '## Which output files do I need?' in (destination / 'docs/USAGE.md').read_text()


def test_verify_accepts_a_clean_export(tmp_path, exporter):
    destination = tmp_path / 'runtime'
    expected = exporter.export_runtime(destination)
    assert exporter.verify_runtime(destination) == expected


@pytest.mark.parametrize('change', ['modified', 'missing', 'extra', 'extra-in-manifest'])
def test_verify_rejects_changed_or_added_files(tmp_path, exporter, change):
    destination = tmp_path / 'runtime'
    exporter.export_runtime(destination)
    if change == 'modified':
        (destination / 'src/c3_tracker/app.py').write_text('# changed after export\n')
    elif change == 'missing':
        (destination / 'tests/test_roi.py').unlink()
    else:
        extra = destination / 'local_recording.mp4'
        extra.write_bytes(b'synthetic placeholder, not a recording')
        if change == 'extra-in-manifest':
            path = destination / 'runtime_manifest.json'
            manifest = json.loads(path.read_text())
            manifest['files'][extra.name] = hashlib.sha256(extra.read_bytes()).hexdigest()
            path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        exporter.verify_runtime(destination)


def test_verify_cli_reports_failure_without_editing_the_bundle(tmp_path, exporter):
    destination = tmp_path / 'runtime'
    exporter.export_runtime(destination)
    unexpected = destination / 'private_sidecar.json'
    unexpected.write_text('{"synthetic": true}')
    result = subprocess.run([sys.executable, str(ROOT / 'tools/export_runtime.py'),
                             '--verify', str(destination)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    assert 'unexpected' in result.stderr.lower()
    assert unexpected.read_text() == '{"synthetic": true}'
