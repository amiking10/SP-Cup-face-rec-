from dataclasses import replace
import csv
import json
from pathlib import Path
import pytest
import cv2
from c3_tracker.benchmark.ava import prepare_ava
from c3_tracker.benchmark.schema import load_sequence,GroundTruth,fingerprint
from c3_tracker.benchmark.inference import BenchmarkConfig,sampled_indices,plan_initialization,infer_video
from c3_tracker.benchmark.runner import run_benchmark
from c3_tracker.benchmark.backend import ExistingTrackerBackend
from c3_tracker.tracking import Observation
from c3_tracker.evaluate import main
from benchmark_helpers import synthetic_ava,ColorBackend


def prepared(tmp_path,coverage='complete'):
    raw=synthetic_ava(tmp_path,coverage=coverage)
    folder=tmp_path/'processed';payload=prepare_ava(raw,folder)
    return folder,load_sequence(payload['videos'][0],folder)


def test_synthetic_integration_runs_full_schema_metrics_and_overlay_without_dataset_download(tmp_path):
    folder,sequence=prepared(tmp_path)
    backends=[]
    def factory(config):
        backend=ColorBackend(config);backends.append(backend);return backend
    output=tmp_path/'results'
    result=run_benchmark(folder/'manifest.json','ava',output,BenchmarkConfig(resolution=(320,240)),
                         overlay=True,backend_factory=factory)
    expected=['predictions.csv','frame_metrics.csv','tracking_metrics.json','performance.json','experiment_config.json','annotated_video.mp4']
    assert all((output/name).is_file() for name in expected)
    assert result['metrics']['detection']['precision']==1
    assert result['metrics']['tracking']['idf1']==1
    per_video=json.loads((output/'tracking_metrics.json').read_text())['per_video']['toy']
    assert per_video['target']['target_recall']==1
    assert per_video['target']['correct_target_reacquisition_rate'] is None
    configuration=json.loads((output/'experiment_config.json').read_text())
    assert configuration['status']=='complete'
    initialization=configuration['videos']['toy']['initialization']
    assert initialization['frame_index']==2 and initialization['person_id']=='a'
    assert initialization['selected_track_id']=='red'
    assert backends[0].designations==['red'] and backends[0].closed
    assert all(value is None for value in backends[0].oracle_calls)
    predictions=list(csv.DictReader((output/'predictions.csv').open()))
    assert predictions[0]['width']=='320' and predictions[0]['height']=='240'
    assert float(predictions[0]['x1'])==pytest.approx(32,abs=1)
    cap=cv2.VideoCapture(str(output/'annotated_video.mp4'))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT)==8
    cap.release()


def test_future_ground_truth_cannot_change_normal_predictions_or_trigger_reselection(tmp_path):
    folder,sequence=prepared(tmp_path)
    config=BenchmarkConfig()
    indices=sampled_indices(sequence.video,0,8)
    plan=plan_initialization(sequence,indices,config)
    first_backend=ColorBackend(config)
    first,first_audit,_=infer_video(sequence.video,indices,config,first_backend,plan)
    for index in range(4,8):
        sequence.ground_truth[index]=[GroundTruth('toy',index,index/10,'changed_future_identity',(0,0,5,5),True)]
    second_backend=ColorBackend(config)
    second,second_audit,_=infer_video(sequence.video,indices,config,second_backend,
                                    plan_initialization(sequence,indices,config))
    assert [(f.observations,f.selected_track_id,f.target_status) for f in first]==[
        (f.observations,f.selected_track_id,f.target_status) for f in second]
    assert first_backend.designations==second_backend.designations==['red']
    assert first_audit==second_audit


def test_initialization_failure_is_recorded_once_without_future_retry(tmp_path):
    _,sequence=prepared(tmp_path)
    config=BenchmarkConfig()
    class LateBackend(ColorBackend):
        def process(self,frame,timestamp_s,oracle_boxes=None):
            found=super().process(frame,timestamp_s,oracle_boxes)
            return [Observation(o.track_id,o.bbox,True,timestamp_s!=.2) for o in found]
    backend=LateBackend(config)
    indices=list(range(8))
    frames,audit,_=infer_video(sequence.video,indices,config,backend,plan_initialization(sequence,indices,config))
    assert audit['status']=='no_confirmed_prediction_for_designated_face'
    assert backend.designations==[]
    assert all(frame.selected_track_id is None for frame in frames)


def test_oracle_is_explicit_and_passes_only_boxes_not_identities(tmp_path):
    folder,_=prepared(tmp_path)
    backends=[]
    def factory(config):
        value=ColorBackend(config);backends.append(value);return value
    output=tmp_path/'oracle'
    run_benchmark(folder/'manifest.json','ava',output,
                  BenchmarkConfig(experiment='oracle',appearance_model='mobilenet'),backend_factory=factory)
    assert all(len(value)==2 for value in backends[0].oracle_calls)
    assert all(len(box)==4 and all(isinstance(v,float) for v in box) for values in backends[0].oracle_calls for box in values)
    config=json.loads((output/'experiment_config.json').read_text())
    assert config['mode']=='oracle' and 'not end-to-end' in config['oracle_warning']


def test_oracle_rejects_partial_annotations_and_sface_detector_dependency(tmp_path):
    folder,_=prepared(tmp_path,coverage='partial')
    with pytest.raises(ValueError,match='exhaustive boxes'):
        run_benchmark(folder/'manifest.json','ava',tmp_path/'oracle',BenchmarkConfig(experiment='oracle'),backend_factory=ColorBackend)
    with pytest.raises(ValueError,match='mobilenet'):
        ExistingTrackerBackend(BenchmarkConfig(experiment='oracle'))


def test_oracle_box_order_cannot_encode_ground_truth_identity(tmp_path):
    folder,_=prepared(tmp_path)
    backends=[]
    def factory(config):
        backend=ColorBackend(config);backends.append(backend);return backend
    config=BenchmarkConfig(experiment='oracle',appearance_model='mobilenet',target_enabled=False)
    run_benchmark(folder/'manifest.json','ava',tmp_path/'first',config,backend_factory=factory)
    annotations=folder/'toy'/'annotations.csv'
    with annotations.open(newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    for row in rows:
        row['person_id']={'a':'z','b':'a'}[row['person_id']]
    with annotations.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    manifest=json.loads((folder/'manifest.json').read_text())
    manifest['videos'][0]['annotations_sha256']=fingerprint(annotations)
    (folder/'manifest.json').write_text(json.dumps(manifest))
    run_benchmark(folder/'manifest.json','ava',tmp_path/'renamed',config,backend_factory=factory)
    assert backends[0].oracle_calls==backends[1].oracle_calls
    assert backends[0].designations==backends[1].designations==[]


def test_subsampling_keeps_source_indices_and_timestamps(tmp_path):
    _,sequence=prepared(tmp_path)
    config=BenchmarkConfig(fps=5)
    indices=sampled_indices(sequence.video,0,8,5)
    assert indices==[0,2,4,6]
    frames,_,_=infer_video(sequence.video,indices,config,ColorBackend(config))
    assert [f.timestamp_s for f in frames]==[0,.2,.4,.6]
    assert [f.processed_frame_index for f in frames]==[0,1,2,3]
    with pytest.raises(ValueError,match='no greater'): sampled_indices(sequence.video,0,8,20)


def test_normal_mode_rejects_oracle_input(tmp_path):
    _,sequence=prepared(tmp_path)
    config=BenchmarkConfig()
    with pytest.raises(ValueError,match='must not receive oracle'):
        infer_video(sequence.video,list(range(8)),config,ColorBackend(config),oracle_boxes={})


def test_unverified_dataset_loader_and_missing_annotations_are_clear(tmp_path,capsys):
    assert main(['--dataset','vpcd','--output',str(tmp_path/'out')])==1
    assert 'No loader is claimed' in capsys.readouterr().err
    assert main(['--dataset','ava','--data-root',str(tmp_path),'--output',str(tmp_path/'out')])==1
    assert 'manifest not found' in capsys.readouterr().err


def test_output_is_never_silently_overwritten(tmp_path):
    folder,_=prepared(tmp_path)
    output=tmp_path/'out';output.mkdir();(output/'previous.txt').write_text('keep')
    with pytest.raises(ValueError,match='not empty'):
        run_benchmark(folder/'manifest.json','ava',output,BenchmarkConfig(),backend_factory=ColorBackend)
    assert (output/'previous.txt').read_text()=='keep'


def test_video_resampling_is_explicit_and_records_source_mapping(tmp_path):
    from c3_tracker.benchmark.video import prepare_video
    raw=synthetic_ava(tmp_path)
    output=tmp_path/'clip.mp4'
    result=prepare_video(raw.parent/'toy.avi',output,.1,.4,20)
    assert len(result['frame_mapping'])==8
    assert result['frame_mapping'][0]['source_frame_index']==1
    assert result['maximum_source_time_error_s']<=.05+1e-9
    assert output.with_suffix('.provenance.json').is_file()
    with pytest.raises(ValueError,match='overwrite'):
        prepare_video(raw.parent/'toy.avi',output,.1,.4,20)


def test_common_format_persistent_identity_requires_provenance(tmp_path):
    folder,sequence=prepared(tmp_path)
    metadata=dict(sequence.metadata)
    metadata.update(identity_scope='persistent',absence_annotations_complete=True)
    with pytest.raises(ValueError,match='identity_provenance'):
        load_sequence(metadata,folder)
    metadata['identity_provenance']='Synthetic generator person labels verified at every frame'
    assert load_sequence(metadata,folder).metadata['absence_annotations_complete'] is True


def test_initialization_never_treats_single_face_dataset_as_bystander_test(tmp_path):
    _,sequence=prepared(tmp_path)
    sequence.ground_truth={index:records[:1] for index,records in sequence.ground_truth.items()}
    assert plan_initialization(sequence,list(range(8)),BenchmarkConfig()) is None
