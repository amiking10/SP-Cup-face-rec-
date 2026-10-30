"""Opt-in real model smoke checks on synthetic pixels; no dataset accuracy claim."""
import json
import os
import pytest
from c3_tracker.benchmark.ava import prepare_ava
from c3_tracker.benchmark.inference import BenchmarkConfig
from c3_tracker.benchmark.runner import run_benchmark
from benchmark_helpers import synthetic_ava

pytestmark=pytest.mark.skipif(os.environ.get('C3_RUN_MODEL_TESTS')!='1',reason='opt-in actual pretrained model integration')


def test_actual_oracle_deepsort_never_constructs_a_face_detector(tmp_path,monkeypatch):
    from c3_tracker import detection
    def forbidden_detector(*args,**kwargs):
        raise AssertionError('Face detector must not run in oracle experiment')
    monkeypatch.setattr(detection,'FaceDetector',forbidden_detector)
    raw=synthetic_ava(tmp_path)
    folder=tmp_path/'processed';prepare_ava(raw,folder)
    output=tmp_path/'oracle'
    result=run_benchmark(folder/'manifest.json','ava',output,
                         BenchmarkConfig(experiment='oracle',appearance_model='mobilenet'),
                         hota_python=os.environ.get('C3_HOTA_PYTHON'))
    metrics=json.loads((output/'tracking_metrics.json').read_text())['per_video']['toy']
    assert metrics['detection']['recall']==pytest.approx(.75)  # two confirmation frames are not scored as detections
    assert metrics['detection']['precision']==1
    assert metrics['tracking']['idsw']==0
    assert metrics['target']['target_recall']==1
    runtime=json.loads((output/'performance.json').read_text())['per_video']['toy']['backend']
    assert runtime['actual_device']=='cpu' and runtime['oracle'] is True
    if os.environ.get('C3_HOTA_PYTHON'):
        assert metrics['tracking']['hota']==pytest.approx(.75)
