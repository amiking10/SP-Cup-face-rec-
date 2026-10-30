"""Generated fixtures test software; opt-in native-file audit makes no accuracy claim."""
import csv
from dataclasses import replace
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from c3_tracker.benchmark.mobiface import (META_FIELDS, ANNOT_FIELDS, MobiFaceMetadata, read_metadata,
                                          read_annotations, audit_annotations, prepare_mobiface)
from c3_tracker.benchmark.inference import BenchmarkConfig
from c3_tracker.benchmark.runner import run_benchmark
from c3_tracker.benchmark.schema import load_sequence
from c3_tracker.evaluate import main
from benchmark_helpers import ColorBackend, metric_sequence, gt, prediction, obs, audit
from c3_tracker.benchmark.target_localization import score_target_localization


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def native_fixture(tmp_path, layout='clip'):
    root=tmp_path/'native';root.mkdir()
    name='abcdefghijk_0'
    row=dict(zip(META_FIELDS,(name,'abcdefghijk',100,64,10,2,9)))
    other={**row,'name':'lmnopqrstuv_0','youtube_video_id':'lmnopqrstuv'}
    write_csv(root/'mobiface80.meta.csv',META_FIELDS,[row,other])
    write_csv(root/'mobiface80/test.meta.csv',META_FIELDS,[row])
    write_csv(root/'mobiface80/train.meta.csv',META_FIELDS,[other])
    annotations=[]
    for index in range(8):
        if index==6:continue  # Unknown, not absent.
        x,y,w,h=(0,0,0,0) if index==3 else (12+index,20,20,20)
        annotations.append(dict(zip(ANNOT_FIELDS,(index,x,y,w,h))))
    write_csv(root/f'mobiface80/test/{name}.annot.csv',ANNOT_FIELDS,annotations)
    write_csv(root/f'mobiface80/train/{other["name"]}.annot.csv',ANNOT_FIELDS,annotations)
    video=root/'video.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'FFV1'),10,(100,64))
    assert writer.isOpened()
    for original in (range(2,10) if layout=='clip' else range(12)):
        image=np.zeros((64,100,3),np.uint8)
        if original!=5:image[20:40,10+original:30+original]=(0,0,255)
        image[20:40,70:90]=(0,255,0)
        writer.write(image)
    writer.release()
    source={'dataset':'mobiface','split':'test','videos':[{'video_id':name,'video_path':'video.avi',
            'video_layout':layout,'cfr':True,'alignment_provenance':'Generated fixture: source indices 2..9 inclusive'}]}
    path=root/'manifest.json';path.write_text(json.dumps(source))
    return root,path,name


def test_metadata_split_native_xywh_absence_and_missing_rows(tmp_path):
    root,_,name=native_fixture(tmp_path)
    meta=read_metadata(root)
    rows=read_annotations(root/f'mobiface80/test/{name}.annot.csv',meta[name])
    assert meta[name].split=='test' and meta[name].frame_count==8
    assert rows[0]==(12,20,32,40) and rows[3] is None and 6 not in rows
    report=audit_annotations(root)
    assert report['split_counts']=={'train':1,'test':1}
    assert report['missing_annotation_frames']==2 and report['absent_rows']==2


@pytest.mark.parametrize('values',[
    (0,10,10,-1,20),(0,-1,10,20,20),(0,90,10,20,20),(0,10,10,0,0),
    (0,'nan',10,20,20),(8,10,10,20,20),(0,'',10,20,20),
])
def test_invalid_annotations_rejected_with_line_context(tmp_path,values):
    path=tmp_path/'bad.csv';write_csv(path,ANNOT_FIELDS,[dict(zip(ANNOT_FIELDS,values))])
    meta=MobiFaceMetadata('abcdefghijk_0','abcdefghijk',100,64,10,2,9,'test')
    with pytest.raises(ValueError,match='bad.csv:2'):
        read_annotations(path,meta)


def test_duplicate_and_empty_annotations_are_errors(tmp_path):
    path=tmp_path/'bad.csv';row=dict(zip(ANNOT_FIELDS,(0,1,1,10,10)))
    meta=MobiFaceMetadata('abcdefghijk_0','abcdefghijk',100,64,10,2,9,'test')
    write_csv(path,ANNOT_FIELDS,[row,row])
    with pytest.raises(ValueError,match='duplicate'):read_annotations(path,meta)
    write_csv(path,ANNOT_FIELDS,[])
    with pytest.raises(ValueError,match='empty'):read_annotations(path,meta)


@pytest.mark.parametrize('layout,offset',[('clip',0),('full_source',2)])
def test_prepare_reuses_common_schema_with_explicit_video_offset(tmp_path,layout,offset):
    root,path,name=native_fixture(tmp_path,layout)
    output=tmp_path/'prepared';payload=prepare_mobiface(path,output)
    entry=payload['videos'][0];sequence=load_sequence(entry,output)
    assert payload['dataset']=='mobiface' and payload['split']=='test'
    assert entry['coverage']=='partial' and entry['identity_annotations_complete'] is False
    assert entry['start_frame']==offset and entry['end_frame_exclusive']==offset+8
    assert sequence.ground_truth[offset][0].box==(12,20,32,40)
    assert sequence.ground_truth[offset][0].timestamp_s==offset/10
    assert not sequence.ground_truth[offset+3][0].visible
    assert offset+6 not in sequence.ground_truth


def test_alignment_split_and_media_version_are_never_guessed(tmp_path):
    _,path,_=native_fixture(tmp_path)
    source=json.loads(path.read_text());source['videos'][0].pop('video_layout');path.write_text(json.dumps(source))
    with pytest.raises(ValueError,match='explicitly'):prepare_mobiface(path,tmp_path/'a')
    source['videos'][0]['video_layout']='full_source';path.write_text(json.dumps(source))
    with pytest.raises(ValueError,match='Full source'):prepare_mobiface(path,tmp_path/'b')
    source['videos'][0]['video_layout']='clip';source['split']='train';path.write_text(json.dumps(source))
    with pytest.raises(ValueError,match='official split'):prepare_mobiface(path,tmp_path/'c')


def test_inconsistent_split_metadata_rejected(tmp_path):
    root,_,_=native_fixture(tmp_path)
    path=root/'mobiface80.meta.csv';text=path.read_text();path.write_text(text.replace('100,64','101,64',1))
    with pytest.raises(ValueError,match='mismatch'):read_metadata(root)


def test_single_target_integration_and_future_gt_isolation(tmp_path):
    root,path,name=native_fixture(tmp_path)
    folder=tmp_path/'prepared';prepare_mobiface(path,folder)
    backends=[]
    def factory(config):
        value=ColorBackend(config);backends.append(value);return value
    result=run_benchmark(folder/'manifest.json','mobiface',tmp_path/'first',BenchmarkConfig(),backend_factory=factory)
    metrics=json.loads((tmp_path/'first/tracking_metrics.json').read_text())['per_video'][name]
    assert metrics['initialization']['status']=='initialized' and metrics['initialization']['frame_index']==2
    assert metrics['target_localization']['visible_annotated_samples']==6
    assert metrics['target_localization']['target_box_recall']['0.5']==1
    assert metrics['target_localization']['visible_target_retention']==1
    assert metrics['target_localization']['unknown_samples']==1 and metrics['target_localization']['annotated_absent_samples']==1
    assert result['metrics']['detection']['precision'] is None and metrics['tracking']['idf1'] is None
    assert metrics['target']['correct_target_reacquisition_rate'] is None
    assert metrics['target']['bystander_metrics_unavailable_reason']
    for key in ('wrong_person_assignment_rate','false_target_assignment_rate',
                'bystander_replacement_frames','false_positive_target_frames'):
        assert metrics['target'][key] is None
        if key in result['metrics']['target']:
            assert result['metrics']['target'][key] is None
    from c3_tracker.benchmark.schema import fingerprint
    annotation=folder/name/'annotations.csv'
    with annotation.open() as stream:rows=list(csv.DictReader(stream))
    for row in rows:
        if int(row['frame_index'])>2 and row['visible']=='1':
            row.update(x1='0',y1='0',x2='5',y2='5')
    write_csv(annotation,rows[0].keys(),rows)
    manifest=json.loads((folder/'manifest.json').read_text());manifest['videos'][0]['annotations_sha256']=fingerprint(annotation)
    (folder/'manifest.json').write_text(json.dumps(manifest))
    run_benchmark(folder/'manifest.json','mobiface',tmp_path/'second',BenchmarkConfig(),backend_factory=factory)
    def predictions(output):
        with output.open() as stream:rows=list(csv.DictReader(stream))
        for row in rows:row.pop('processing_ms')
        return rows
    assert predictions(tmp_path/'first/predictions.csv')==predictions(tmp_path/'second/predictions.csv')
    assert backends[0].designations==backends[1].designations==['red']
    assert all(value is None for backend in backends for value in backend.oracle_calls)


def test_target_coverage_interruptions_and_unknown_gaps_have_explicit_denominators(tmp_path):
    ground_truth={i:[gt(i,'target')] for i in (0,1,2,3,5)}
    sequence=metric_sequence(tmp_path,ground_truth,scope='track_local',coverage='partial')
    sequence.metadata.update(annotation_type='single_target',designated_person_id='target')
    frames=[prediction(i,[obs()] if i in (0,1,3,5) else [],selected='1') for i in range(6)]
    report=score_target_localization(sequence,frames,audit(frame=0,person='target'),BenchmarkConfig())
    assert report['target_box_recall']['0.5']==4/5
    assert report['target_detection_interruptions']==1 and report['unknown_samples']==1
    assert report['selected_target_tracking_interruptions']==1
    assert report['visible_target_retention']==3/4
    assert report['physical_reacquisition_evaluable'] is False


@pytest.mark.skipif(os.environ.get('C3_RUN_DATASET_TESTS')!='1',reason='opt-in actual provided native MobiFace files')
def test_actual_mobiface_source_annotation_audit():
    report=audit_annotations(Path('datasets/data/raw/mobiface'))
    assert len(report['sequences'])==80 and report['split_counts']=={'train':64,'test':16}
    assert report['annotation_rows']==94910 and report['absent_rows']==19036
    assert report['missing_annotation_frames']==731
