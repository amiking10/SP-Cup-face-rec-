"""Native eye-point diagnostics use generated pixels/XML, never downloaded faces."""
from dataclasses import replace
import csv
import json

import cv2
import numpy as np
import pytest

from c3_tracker.benchmark import chokepoint
from c3_tracker.benchmark.chokepoint import EyeAnnotation, load_eyes, match_eyes, score_eyes, run_sequence
from c3_tracker.benchmark.inference import BenchmarkConfig, PredictedFrame
from c3_tracker.tracking import Observation
from benchmark_helpers import ColorBackend


def xml_frame(index, person='a', left=(12,25), right=(22,25)):
    return (f'<frame number="{index:08d}"><person id="{person}">'
            f'<leftEye x="{left[0]}" y="{left[1]}"/>'
            f'<rightEye x="{right[0]}" y="{right[1]}"/></person></frame>')


def write_xml(path, frames, name='camera'):
    path.write_text(f'<!DOCTYPE GroundTruth><dataset name="{name}">{frames}</dataset>')


def prediction(index, track='1', box=(10,20,30,40)):
    return PredictedFrame(index,index,index/30,(Observation(track,box,True,True),),1,None,'NOT SELECTED',100,64)


def test_verified_native_eye_schema_preserves_gaps_without_inventing_boxes(tmp_path):
    path=tmp_path/'camera.xml'
    write_xml(path,xml_frame(0)+'<frame number="00000004"/>')
    frames=load_eyes(path,'camera',100,64)
    assert sorted(frames)==[0,4] and frames[4]==()
    assert frames[0][0]==EyeAnnotation('a',(12,25),(22,25))
    assert not hasattr(frames[0][0],'box')


@pytest.mark.parametrize('content',[
    xml_frame(0)+xml_frame(0),
    xml_frame(0,left=(-1,25)),
    xml_frame(0,right=(12,25)),
    '<frame number="1"/>',
    '<frame number="00000000"><person id="a"><leftEye x="1" y="2"/></person></frame>',
    '<frame number="00000000"><person id="a"><bbox x="1" y="2"/></person></frame>',
    '<frame number="00000000"><person id="a"><leftEye x="1"/><rightEye x="2" y="3"/></person></frame>',
    xml_frame(0,left=('nan',25)),
])
def test_bad_or_incomplete_point_annotations_are_rejected(tmp_path,content):
    path=tmp_path/'camera.xml';write_xml(path,content)
    with pytest.raises(ValueError):load_eyes(path,'camera',100,64)


def test_wrong_camera_xml_cannot_be_silently_joined(tmp_path):
    path=tmp_path/'camera.xml';write_xml(path,xml_frame(0),name='other_camera')
    with pytest.raises(ValueError,match='match'):load_eyes(path,'camera',100,64)


def test_both_eyes_one_to_one_and_scaled_coordinates_are_required():
    people=[EyeAnnotation('a',(12,25),(22,25)),EyeAnnotation('b',(72,25),(82,25))]
    boxes=[Observation('1',(20,40,60,80),True,True),Observation('2',(140,40,180,80),True,True)]
    assert match_eyes(people,boxes,scale=(2,2))==({0:0,1:1},[])
    assert match_eyes(people,[Observation('big',(0,0,100,64),True,True)])==({},[0,1])
    assert match_eyes(people[:1],[Observation('left_eye_only',(10,20,20,40),True,True)])==({},[])
    assert match_eyes(people[:1],[Observation('1',(10,20,30,40),True,True)]*2)==({},[0])


def test_point_metrics_distinguish_id_changes_and_cross_person_reuse():
    a=EyeAnnotation('a',(12,25),(22,25));b=replace(a,person_id='b')
    frames=[prediction(0,'1'),prediction(3,'2'),prediction(6,'2')]
    result,_,_=score_eyes(frames,{0:(a,),3:(a,),6:(b,)},(100,64),30,10)
    assert result['eye_containment_recall']==1
    assert result['observed_id_changes_within_annotated_runs']==1
    assert result['cross_person_track_reassignment_events']==1
    assert result['track_ids_matched_to_multiple_people']=={'2':['a','b']}
    assert 'iou' in result['unavailable'] and 'idf1' in result['unavailable']
    frames[1]=prediction(4,'2')
    result,_,_=score_eyes(frames[:2],{0:(a,),4:(a,)},(100,64),30,10)
    assert result['observed_id_changes_within_annotated_runs']==0  # Unknown source gap resets continuity.


def test_native_raw_image_pipeline_keeps_source_time_and_parses_gt_only_after_inference(tmp_path,monkeypatch):
    images=tmp_path/'camera';images.mkdir()
    path=tmp_path/'camera.xml'
    indices=[0,1,4]
    write_xml(path,''.join(xml_frame(index) for index in indices))
    image=np.zeros((64,100,3),np.uint8);image[20:40,10:30]=(0,0,255)
    for index in indices:assert cv2.imwrite(str(images/f'{index:08d}.jpg'),image)
    backends=[];events=[]
    class PixelBackend(ColorBackend):
        def process(self,frame,timestamp_s,oracle_boxes=None):
            events.append(('raw',timestamp_s))
            assert oracle_boxes is None
            return super().process(frame,timestamp_s,oracle_boxes)
    def factory(config):
        backend=PixelBackend(config);backends.append(backend);return backend
    original=chokepoint.load_eyes
    def scoring_only(*args):
        assert events[-3:]==[('raw',0),('raw',1/30),('raw',4/30)]
        events.append(('ground_truth',None))
        return original(*args)
    monkeypatch.setattr(chokepoint,'load_eyes',scoring_only)
    config=BenchmarkConfig(target_enabled=False)
    result=run_sequence(images,path,tmp_path/'first',config,end=6,overlay=True,backend_factory=factory)
    assert result['eye_containment_recall']==1
    saved=json.loads((tmp_path/'first'/'experiment_config.json').read_text())
    assert saved['processed_source_indices']==indices and saved['missing_requested_source_frames']==3
    assert (tmp_path/'first'/'annotated_video.mp4').is_file()
    write_xml(path,''.join(xml_frame(index,person='renamed') for index in indices))
    run_sequence(images,path,tmp_path/'second',config,end=6,backend_factory=factory)
    with (tmp_path/'first'/'predictions.csv').open() as a,(tmp_path/'second'/'predictions.csv').open() as b:
        first,second=list(csv.DictReader(a)),list(csv.DictReader(b))
    for a,b in zip(first,second):
        a.pop('processing_ms');b.pop('processing_ms')
        assert a==b
    assert all(backend.designations==[] and backend.closed for backend in backends)


def test_oracle_or_target_mode_and_existing_output_are_rejected(tmp_path):
    with pytest.raises(ValueError,match='target designation disabled'):
        run_sequence(tmp_path,tmp_path/'missing.xml',tmp_path/'out',BenchmarkConfig())
    output=tmp_path/'out';output.mkdir();(output/'old.txt').write_text('keep')
    with pytest.raises(ValueError,match='fresh'):
        run_sequence(tmp_path,tmp_path/'missing.xml',output,BenchmarkConfig(target_enabled=False))


def test_box_benchmark_points_to_the_verified_eye_diagnostic_entry_point(tmp_path,capsys):
    from c3_tracker.evaluate import main
    assert main(['--dataset','chokepoint','--output',str(tmp_path/'out')])==1
    assert 'c3_tracker.benchmark.chokepoint' in capsys.readouterr().err
