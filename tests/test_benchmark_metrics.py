import json
import os
from pathlib import Path
import subprocess
import math
import pytest
from c3_tracker.benchmark.inference import BenchmarkConfig
from c3_tracker.benchmark.matching import match_boxes,iou_matrix
from c3_tracker.benchmark.metrics import score_sequence
from benchmark_helpers import metric_sequence,gt,obs,prediction,audit


def test_hungarian_iou_is_one_to_one():
    matches,misses,extras,ious=match_boxes([(0,0,10,10)],[(0,0,10,10),(0,0,10,10)])
    assert len(matches)==1 and len(extras)==1 and misses==[]
    assert ious[0,0]==1
    assert iou_matrix([(0,0,10,10)],[(5,0,15,10)])[0,0]==pytest.approx(1/3)


def test_identity_swap_metrics_and_wrong_person_denominators(tmp_path):
    ground_truth={i:[gt(i),gt(i,'b',(60,20,80,40))] for i in range(2)}
    frames=[prediction(0,[obs(),obs('2',(60,20,80,40))]),
            prediction(1,[obs('2'),obs('1',(60,20,80,40))])]
    result,rows,_,_=score_sequence(metric_sequence(tmp_path,ground_truth),frames,audit(),BenchmarkConfig())
    assert result['detection']['precision']==result['detection']['recall']==1
    assert result['detection']['mean_matched_iou']==1
    assert result['tracking']['idsw']==2
    assert result['tracking']['idf1']==pytest.approx(.5)
    assert result['target']['target_recall']==0
    assert result['target']['wrong_person_assignment_rate']==1
    assert result['target']['bystander_replacement_frames']==1
    assert result['target']['target_loss_duration_s']==pytest.approx(.1)
    assert rows[1]['matched_person_id']=='b' and rows[1]['target_outcome']=='wrong_person'


def test_missing_target_counts_loss_and_fragmentation(tmp_path):
    truth={i:[gt(i)] for i in range(3)}
    frames=[prediction(0,[obs()]),prediction(1,[]),prediction(2,[obs()],status='REACQUIRED')]
    result,_,_,_=score_sequence(metric_sequence(tmp_path,truth),frames,audit(),BenchmarkConfig())
    assert result['detection']['missed_faces']==1
    assert result['tracking']['fragmentations']==1
    assert result['target']['target_recall']==.5
    assert result['target']['target_loss_duration_s']==pytest.approx(.1)
    assert result['target']['target_loss_episodes'][0]['censored'] is False
    assert result['target']['correct_target_reacquisition_rate'] is None  # no real absence
    assert result['target']['false_reacquisition_rate']==0


def test_physical_return_is_only_scored_with_known_identity_and_absence(tmp_path):
    truth={0:[gt(0)],1:[],2:[gt(2)],3:[gt(3)]}
    frames=[prediction(0,[obs()]),prediction(1,[]),prediction(2,[]),prediction(3,[obs()],status='REACQUIRED')]
    result,_,_,_=score_sequence(metric_sequence(tmp_path,truth),frames,audit(),BenchmarkConfig())
    assert result['target']['correct_target_reacquisition_rate']==1
    assert result['target']['successful_return_opportunities']==1
    local,_,_,_=score_sequence(metric_sequence(tmp_path,truth,scope='track_local'),frames,audit(),BenchmarkConfig())
    assert local['target']['correct_target_reacquisition_rate'] is None
    assert local['target']['false_reacquisition_rate'] is None
    assert 'Track-local' in local['target']['reacquisition_unavailable_reason']


def test_false_reacquisition_and_bystander_replacement(tmp_path):
    truth={0:[gt(0),gt(0,'b',(60,20,80,40))],1:[gt(1,'b',(60,20,80,40))],2:[gt(2,'b',(60,20,80,40))]}
    frames=[prediction(0,[obs(),obs('2',(60,20,80,40))]),prediction(1,[obs('2',(60,20,80,40))]),
            prediction(2,[obs('1',(60,20,80,40))],status='REACQUIRED')]
    result,_,_,_=score_sequence(metric_sequence(tmp_path,truth),frames,audit(),BenchmarkConfig())
    assert result['target']['false_reacquisition_rate']==1
    assert result['target']['bystander_replacement_frames']==1


def test_partial_annotations_do_not_invent_false_positives_or_identity_scores(tmp_path):
    truth={0:[gt(0)],1:[gt(1)]}
    frames=[prediction(i,[obs(),obs('2',(60,20,80,40))]) for i in range(2)]
    result,_,_,_=score_sequence(metric_sequence(tmp_path,truth,coverage='partial'),frames,audit(),BenchmarkConfig())
    assert result['detection']['precision'] is None
    assert result['detection']['false_positives'] is None
    assert result['detection']['unmatched_predictions']==2
    assert result['tracking']['idsw'] is None and result['tracking']['idf1'] is None


def test_unknown_frames_break_identity_continuity_and_are_not_empty_frames(tmp_path):
    truth={0:[gt(0)],2:[gt(2)]}
    frames=[prediction(0,[obs()]),prediction(1,[obs('2')]),prediction(2,[obs('3')])]
    result,rows,_,_=score_sequence(metric_sequence(tmp_path,truth),frames,audit(),BenchmarkConfig())
    assert rows[1]['evaluated']==0 and rows[1]['false_positives']==''
    assert result['tracking']['idsw'] is None
    assert result['detection']['unknown_frames']==1


def test_predicted_only_boxes_are_not_observed_detections(tmp_path):
    truth={0:[gt(0)],1:[gt(1)]}
    result,_,_,_=score_sequence(metric_sequence(tmp_path,truth),
        [prediction(0,[obs()]),prediction(1,[obs(observed=False)],status='LOST')],audit(),BenchmarkConfig())
    assert result['detection']['missed_faces']==1
    assert result['target']['target_recall']==0


@pytest.mark.skipif(not os.environ.get('C3_HOTA_PYTHON'),reason='optional isolated TrackEval interpreter')
def test_reference_hota_perfect_and_swap(tmp_path):
    from c3_tracker.benchmark import schema
    truth={i:[gt(i),gt(i,'b',(60,20,80,40))] for i in range(2)}
    perfect=[prediction(i,[obs(),obs('2',(60,20,80,40))]) for i in range(2)]
    swapped=[perfect[0],prediction(1,[obs('2'),obs('1',(60,20,80,40))])]
    sequence=metric_sequence(tmp_path,truth)
    _,_,_,first=score_sequence(sequence,perfect,audit(),BenchmarkConfig())
    _,_,_,second=score_sequence(sequence,swapped,audit(),BenchmarkConfig())
    input_path,output_path=tmp_path/'input.json',tmp_path/'hota.json'
    input_path.write_text(json.dumps({'groups':{'perfect':first,'swap':second}}))
    worker=Path(schema.__file__).with_name('hota_worker.py')
    subprocess.run([os.environ['C3_HOTA_PYTHON'],str(worker),str(input_path),str(output_path)],check=True)
    result=json.loads(output_path.read_text())
    assert result['per_video']['perfect']['hota']==pytest.approx(1)
    assert result['per_video']['swap']['hota']==pytest.approx(math.sqrt(1/3))
