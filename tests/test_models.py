"""Model integrity checks use fake bytes, with no downloads or real photographs."""
import hashlib
import pytest
from c3_tracker import models


def test_missing_or_modified_models_are_rejected(tmp_path,monkeypatch):
    data=b'known model bytes'
    monkeypatch.setattr(models,'MODELS',{'face.onnx':('face',hashlib.sha256(data).hexdigest())})
    with pytest.raises(RuntimeError,match='Missing face model'):
        models.model_path('face.onnx',tmp_path)
    path=tmp_path/'face.onnx'
    path.write_bytes(b'modified')
    with pytest.raises(RuntimeError,match='integrity'):
        models.model_path('face.onnx',tmp_path)
    path.write_bytes(data)
    assert models.model_path('face.onnx',tmp_path)==path


def test_existing_verified_models_do_not_use_the_network(tmp_path,monkeypatch):
    data=b'known model bytes'
    monkeypatch.setattr(models,'MODELS',{'face.onnx':('face',hashlib.sha256(data).hexdigest())})
    (tmp_path/'face.onnx').write_bytes(data)
    monkeypatch.setattr(models,'urlopen',lambda *a,**k:pytest.fail('Existing model should not download'))
    models.download(tmp_path)
