"""The core export cannot depend on archived desktop or research modules."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def exporter():
    spec=importlib.util.spec_from_file_location('runtime_export',ROOT/'tools/export_runtime.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_is_an_explicit_core_inventory(tmp_path,exporter):
    destination=tmp_path/'core'
    manifest=exporter.export_runtime(destination)
    assert {p.stem for p in (destination/'src/c3_tracker').glob('*.py')}==set(exporter.RUNTIME_MODULES)
    assert not any(p.suffix in {'.mp4','.csv','.onnx','.jpg','.npz'} for p in destination.rglob('*'))
    assert not (destination/'src/c3_tracker/app.py').exists()
    assert not (destination/'src/c3_tracker/benchmark').exists()
    assert not (destination/'src/c3_tracker/__main__.py').exists()
    assert not (destination/'tests/test_runtime_export.py').exists()
    files={p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}
    assert set(manifest['files'])==files-{'runtime_manifest.json'}
    for relative,digest in manifest['files'].items():
        data=(destination/relative).read_bytes()
        assert hashlib.sha256(data).hexdigest()==digest
        if relative in exporter.COPIED_FILES:
            assert data==(ROOT/relative).read_bytes()
    config=tomllib.loads((destination/'pyproject.toml').read_text())
    assert not any(pin.startswith(('torch==','torchvision==')) for pin in config['project']['dependencies'])
    assert 'benchmark' not in config['project']['optional-dependencies']


def test_existing_destination_is_retained(tmp_path,exporter):
    destination=tmp_path/'existing'
    destination.mkdir()
    sentinel=destination/'recording.mp4'
    sentinel.write_bytes(b'preserve')
    with pytest.raises(FileExistsError): exporter.export_runtime(destination)
    assert sentinel.read_bytes()==b'preserve'


def test_missing_source_cannot_create_partial_export(tmp_path,exporter):
    with pytest.raises(FileNotFoundError): exporter.export_runtime(tmp_path/'out',tmp_path/'missing')
    assert not (tmp_path/'out').exists()


def test_export_imports_do_not_fall_back_to_old_checkout(tmp_path,exporter):
    destination=tmp_path/'core'
    exporter.export_runtime(destination)
    script="""
import importlib,importlib.util,pathlib,sys
source=pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0,str(source))
import c3_tracker
assert pathlib.Path(c3_tracker.__file__).is_relative_to(source)
for name in ('association','detection','embedding','models','target_manager','tracking'):
    module=importlib.import_module('c3_tracker.'+name)
    assert pathlib.Path(module.__file__).is_relative_to(source)
for name in ('app','controls','logging_utils','paired_recording','roi','visualization','benchmark','evaluate','evaluation','__main__'):
    assert importlib.util.find_spec('c3_tracker.'+name) is None
"""
    result=subprocess.run([sys.executable,'-I','-c',script,str(destination/'src')],cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr


def test_exported_guides_have_no_missing_local_links(tmp_path,exporter):
    destination=tmp_path/'core'
    exporter.export_runtime(destination)
    for path in [destination/'README.md',*(destination/'docs').glob('*.md')]:
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',path.read_text()):
            if '://' not in link and not link.startswith('#'):
                assert (path.parent/link.split('#')[0]).exists(),(path,link)


def test_verify_accepts_clean_export(tmp_path,exporter):
    destination=tmp_path/'core'
    manifest=exporter.export_runtime(destination)
    assert exporter.verify_runtime(destination)==manifest


@pytest.mark.parametrize('change',['modified','missing','extra','extra-in-manifest'])
def test_verify_rejects_changed_or_extra_files(tmp_path,exporter,change):
    destination=tmp_path/'core'
    exporter.export_runtime(destination)
    if change=='modified': (destination/'src/c3_tracker/tracking.py').write_text('# changed')
    elif change=='missing': (destination/'tests/test_tracking.py').unlink()
    else:
        extra=destination/'local_recording.mp4'
        extra.write_bytes(b'not runtime')
        if change=='extra-in-manifest':
            path=destination/'runtime_manifest.json'
            manifest=json.loads(path.read_text())
            manifest['files']['local_recording.mp4']=hashlib.sha256(extra.read_bytes()).hexdigest()
            path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): exporter.verify_runtime(destination)
