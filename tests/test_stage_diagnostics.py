from dataclasses import replace
import csv
import json
import pytest

from c3_tracker.benchmark.diagnostics import initialization_diagnostic
from c3_tracker.benchmark.inference import BenchmarkConfig, infer_video, plan_initialization, FrameAlignmentError
from c3_tracker.benchmark.runner import run_benchmark
from c3_tracker.benchmark.ava import prepare_ava
from c3_tracker.benchmark.schema import load_sequence,GroundTruth
from c3_tracker.tracking import Observation
from benchmark_helpers import synthetic_ava,ColorBackend

BOX=(10,20,30,40)
DETECTION={'bbox':BOX,'confidence':.9}


def test_upstream_numpy_coordinates_are_serializable_in_initialization_evidence():
    import numpy as np
    observation=Observation('1',tuple(np.float32(value) for value in BOX),True,True,np.float32(.9))
    evidence=initialization_diagnostic(BOX,[observation],None,BenchmarkConfig())
    assert json.loads(json.dumps(evidence,allow_nan=False))['available_predictions'][0]['bbox']==list(BOX)


def test_low_confidence_probe_is_separate_and_does_not_change_baseline_detector_inputs(monkeypatch):
    import numpy as np
    from c3_tracker import detection,tracking
    from c3_tracker.benchmark.backend import ExistingTrackerBackend
    calls=[]
    class Detector:
        def __init__(self,confidence,model_selection):self.confidence=confidence
        def detect(self,frame):
            calls.append(('detect',self.confidence))
            boxes=[detection.Detection(BOX,.9),detection.Detection((40,20,50,30),.6)]
            return boxes+[detection.Detection((50,20,60,30),.2)] if self.confidence<.2 else boxes
        def close(self):pass
    class Tracker:
        def __init__(self,**kwargs):self.last_stage_snapshot=None
        def update(self,detections,frame,timestamp):
            calls.append(('track',len(detections)))
            values=[{'bbox':list(d.bbox),'confidence':d.confidence} for d in detections]
            self.last_stage_snapshot={'baseline_detector_output':values,'descriptor_success':values}
            return [Observation('1',BOX,True,True)]
    monkeypatch.setattr(detection,'FaceDetector',Detector)
    monkeypatch.setattr(tracking,'FaceTracker',Tracker)
    image=np.zeros((64,100,3),np.uint8)
    diagnostic=ExistingTrackerBackend(BenchmarkConfig(stage_diagnostics=True))
    result=diagnostic.process(image,0)
    assert calls==[('detect',.65),('track',2),('detect',.01)]
    assert len(diagnostic.stage_snapshot()['raw_detector_probe'])==3
    assert len(diagnostic.stage_snapshot()['baseline_detector_output'])==2
    assert len(diagnostic.stage_snapshot()['confidence_accepted'])==1
    calls.clear()
    baseline=ExistingTrackerBackend(BenchmarkConfig())
    assert baseline.process(image,0)==result
    assert calls==[('detect',.65),('track',2)]
    assert baseline.stage_snapshot() is None


def test_multiple_candidates_are_reported_without_inventing_an_ambiguity_rejection():
    evidence=initialization_diagnostic(BOX,[Observation('1',BOX,True,True),Observation('2',BOX,True,True)],None,BenchmarkConfig())
    assert evidence['multiple_plausible_tracks'] is True and evidence['plausible_confirmed_tracks']==2
    assert 'no new ambiguity rejection' in evidence['ambiguity_policy']


@pytest.mark.parametrize('raw,accepted,described,observations,expected',[
    ([],[],[],[],'no_raw_detection_overlapping_designated_target'),
    ([{**DETECTION,'confidence':.3}],[],[],[],'detection_below_confidence_in_probe'),
    ([DETECTION],[],[],[],'raw_probe_baseline_disagreement'),
    ([DETECTION],[DETECTION],[],[],'appearance_extraction_failure'),
    ([DETECTION],[DETECTION],[DETECTION],[Observation('1',BOX,True,False)],'track_still_unconfirmed'),
    ([DETECTION],[DETECTION],[DETECTION],[],'no_matching_predicted_track'),
    ([{**DETECTION,'bbox':(10,20,18,40)}],[{**DETECTION,'bbox':(10,20,18,40)}],
     [{**DETECTION,'bbox':(10,20,18,40)}],[Observation('1',(10,20,18,40),True,True)],
     'raw_detection_below_initialization_iou_gate'),
])
def test_initialization_failure_categories_are_based_on_actual_stage_evidence(raw,accepted,described,observations,expected):
    snapshot={'raw_detector_probe':raw,'baseline_detector_output':accepted,'confidence_accepted':accepted,'descriptor_success':described}
    report=initialization_diagnostic(BOX,observations,snapshot,BenchmarkConfig())
    assert report['failure_category']==expected


class InstrumentedColorBackend(ColorBackend):
    def process(self,frame,timestamp_s,oracle_boxes=None):
        predictions=super().process(frame,timestamp_s,oracle_boxes)
        self.accepted=[{'bbox':list(p.bbox),'confidence':.9} for p in predictions]
        self.described=[item for item in self.accepted if item['bbox'][0]<frame.shape[1]/2]
        result=[Observation(p.track_id,p.bbox,True,timestamp_s>=.2) for p in predictions if p.track_id=='red']
        result.append(Observation('ghost',(70,20,90,40),False,True))
        return result

    def stage_snapshot(self):
        return {'raw_detector_probe':self.accepted+[{'bbox':(1,1,5,5),'confidence':.1}],
                'baseline_detector_output':self.accepted,'confidence_accepted':self.accepted,'descriptor_success':self.described,
                'baseline_detector_tracking_ms':1.0,'diagnostic_probe_ms':1.0}


def prepared(tmp_path):
    raw=synthetic_ava(tmp_path);folder=tmp_path/'prepared';payload=prepare_ava(raw,folder)
    return folder,load_sequence(payload['videos'][0],folder)


def test_actual_stage_counts_use_separate_denominators_and_never_score_predictions_as_detections(tmp_path):
    folder,_=prepared(tmp_path)
    run_benchmark(folder/'manifest.json','ava',tmp_path/'out',BenchmarkConfig(stage_diagnostics=True),
                  backend_factory=InstrumentedColorBackend)
    report=json.loads((tmp_path/'out/stage_metrics.json').read_text())
    stages=report['stages']
    assert stages['confidence_accepted']['output_boxes_all_frames']==16
    assert stages['confidence_accepted']['recall']==1
    assert stages['descriptor_success']['matched_faces']==8 and stages['descriptor_success']['recall']==.5
    assert stages['tentative_tracks']['output_boxes_all_frames']==2
    assert stages['confirmed_observed']['matched_faces']==6 and stages['confirmed_observed']['recall']==6/16
    assert stages['confirmed_tracks']['recall'] is None
    assert stages['confirmed_tracks']['output_boxes_all_frames']==14
    with (tmp_path/'out/diagnostic_stages.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert sum(int(row['spatially_matched_observations']) for row in rows)==6
    assert report['initialization']['status']=='initialized'
    assert report['examples'] and all((tmp_path/'out/per_video/toy'/item['image']).is_file() for item in report['examples'])


def test_diagnostics_cannot_change_predictions_when_future_ground_truth_is_changed(tmp_path):
    _,sequence=prepared(tmp_path)
    config=BenchmarkConfig(stage_diagnostics=True)
    indices=list(range(8))
    first_backend=InstrumentedColorBackend(config)
    first,first_audit,_=infer_video(sequence.video,indices,config,first_backend,plan_initialization(sequence,indices,config))
    for i in range(4,8):sequence.ground_truth[i]=[GroundTruth('toy',i,i/10,'future',(0,0,5,5))]
    second_backend=InstrumentedColorBackend(config)
    second,second_audit,_=infer_video(sequence.video,indices,config,second_backend,plan_initialization(sequence,indices,config))
    assert [(f.observations,f.stages,f.selected_track_id) for f in first]==[(f.observations,f.stages,f.selected_track_id) for f in second]
    assert first_audit==second_audit
    assert all(value is None for value in first_backend.oracle_calls+second_backend.oracle_calls)


def test_alignment_failure_is_explicit_before_inference(tmp_path,monkeypatch):
    import cv2,numpy as np
    _,sequence=prepared(tmp_path)
    class BadCapture:
        def isOpened(self):return True
        def set(self,*args):return True
        def read(self):return True,np.zeros((64,100,3),np.uint8)
        def get(self,prop):return 1000 if prop==cv2.CAP_PROP_POS_MSEC else 1
        def release(self):pass
    monkeypatch.setattr(cv2,'VideoCapture',lambda path:BadCapture())
    backend=InstrumentedColorBackend(BenchmarkConfig())
    with pytest.raises(FrameAlignmentError) as error:
        infer_video(sequence.video,list(range(8)),BenchmarkConfig(),backend)
    assert error.value.frame_index==0 and error.value.failure_category=='frame_timestamp_alignment_failure'
    assert backend.oracle_calls==[]
