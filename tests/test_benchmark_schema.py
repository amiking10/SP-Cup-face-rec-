import csv
import json
import pytest
from c3_tracker.benchmark.ava import parse_ava_row,prepare_ava
from c3_tracker.benchmark.schema import aligned_frame,normalized_box,scale_box,load_sequence,read_manifest
from benchmark_helpers import synthetic_ava


@pytest.mark.parametrize('fields',[5,8])
def test_verified_ava_formats_and_pixel_coordinates(tmp_path,fields):
    raw=synthetic_ava(tmp_path,fields)
    output=tmp_path/'processed'
    result=prepare_ava(raw,output)
    assert result['videos'][0]['identity_scope']=='track_local'
    sequence=load_sequence(result['videos'][0],output)
    assert len(sequence.ground_truth)==8
    assert sequence.ground_truth[0][0].box==pytest.approx((10,20,30,40))
    assert sequence.ground_truth[7][0].frame_index==7
    assert sequence.ground_truth[7][0].timestamp_s==.7
    assert result['videos'][0]['maximum_alignment_error_s']<1e-6


def test_pixel_scaling_and_half_open_coordinates():
    box=normalized_box((.1,.2,.3,.4),640,480)
    assert box==pytest.approx((64,96,192,192))
    assert scale_box(box,(640,480),(320,240))==pytest.approx((32,48,96,96))


@pytest.mark.parametrize('row',[
    ['id','nan','.1','.2','.4','.5','NOT_SPEAKING','a'],
    ['id','0','.6','.2','.4','.5','NOT_SPEAKING','a'],
    ['id','0','.1','.2','1.1','.5','NOT_SPEAKING','a'],
    ['id','0','.1','.2','.4','.5','NOT_SPEAKING',''],
    ['id','0','(0.1,0.2,0.3)','NOT_SPEAKING','a'],
    ['id','0','.1','.2','.4','.5','12','a'],
    ['id','0','(nan,0.2,0.3,0.4)','NOT_SPEAKING','a'],
])
def test_incorrect_annotations_rejected(row):
    with pytest.raises(ValueError): parse_ava_row(row)


def test_alignment_rejects_wrong_offset_fps_and_out_of_range():
    assert aligned_frame(900.1,900,10,10,.0051)[:2]==(1,.1)
    with pytest.raises(ValueError,match='outside'): aligned_frame(900.1,0,10,10,.0051)
    with pytest.raises(ValueError,match='FPS/offset'): aligned_frame(900.03,900,25,100,.0051)
    with pytest.raises(ValueError,match='half'): aligned_frame(0,0,30,100,1)


def test_duplicate_rows_are_not_silently_dropped(tmp_path):
    raw=synthetic_ava(tmp_path)
    annotation=raw.parent/'toy.csv'
    lines=annotation.read_text().splitlines()
    annotation.write_text('\n'.join(lines+[lines[0]])+'\n')
    with pytest.raises(ValueError,match='duplicate'): prepare_ava(raw,tmp_path/'processed')


def test_offset_and_cfr_must_be_declared(tmp_path):
    raw=synthetic_ava(tmp_path)
    payload=json.loads(raw.read_text());del payload['videos'][0]['video_start_timestamp_s']
    raw.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='explicit'): prepare_ava(raw,tmp_path/'processed')


def test_ava_cannot_claim_persistent_physical_identity(tmp_path):
    raw=synthetic_ava(tmp_path)
    payload=json.loads(raw.read_text());payload['videos'][0]['identity_scope']='persistent'
    raw.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='cannot be declared persistent'): prepare_ava(raw,tmp_path/'processed')


def test_missing_video_and_manifest_are_useful_errors(tmp_path):
    with pytest.raises(FileNotFoundError,match='manifest'): read_manifest(tmp_path/'missing.json','ava')
    raw=synthetic_ava(tmp_path)
    payload=json.loads(raw.read_text());payload['videos'][0]['video_path']='missing.mp4';raw.write_text(json.dumps(payload))
    with pytest.raises(FileNotFoundError,match='Video not found'): prepare_ava(raw,tmp_path/'processed')


def test_normalized_timestamp_and_fingerprints_are_checked(tmp_path):
    raw=synthetic_ava(tmp_path);out=tmp_path/'processed';payload=prepare_ava(raw,out)
    entry=payload['videos'][0];csv_path=out/entry['annotations_path']
    rows=list(csv.DictReader(csv_path.open()));rows[0]['timestamp_s']='.2'
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError,match='timestamp does not match'): load_sequence(entry,out)


def test_prediction_file_frame_mismatch_is_rejected(tmp_path):
    from c3_tracker.benchmark.schema import PRED_FIELDS,read_predictions,write_csv,VideoSpec
    from benchmark_helpers import prediction,obs
    video=VideoSpec('toy',tmp_path/'unused.avi',100,64,10,8,0)
    rows=list(prediction(0,[obs()]).rows('toy'))
    rows[0]['timestamp_s']=.1
    path=tmp_path/'predictions.csv';write_csv(path,PRED_FIELDS,rows)
    with pytest.raises(ValueError,match='alignment mismatch'):read_predictions(path,video)


def test_invisible_and_incomplete_annotations_are_validated(tmp_path):
    raw=synthetic_ava(tmp_path);out=tmp_path/'processed';payload=prepare_ava(raw,out)
    entry=payload['videos'][0]
    del entry['coverage_provenance']
    with pytest.raises(ValueError,match='coverage_provenance'):load_sequence(entry,out)
