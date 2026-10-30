"""Reproducible orchestration: normalize, infer without GT, then score/render."""
from __future__ import annotations
from dataclasses import asdict
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import cv2
import numpy as np
from .backend import make_backend
from .inference import infer_video, sampled_indices, plan_initialization
from .metrics import score_sequence, FRAME_FIELDS, ratio
from .schema import PRED_FIELDS, read_manifest, load_sequence, read_predictions, fingerprint, scale_box, write_json, write_csv
from .visualization import render_overlay


def environment():
    packages = {}
    for name in ('c3-tracker','mediapipe','deep-sort-realtime','numpy','scipy','opencv-contrib-python','torch','torchvision','motmetrics','pandas'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    source = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(source.rglob('*.py')):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    git = shutil.which('git')
    if git is None and os.name == 'nt':
        candidate = Path(os.environ.get('ProgramFiles','C:/Program Files')) / 'Git/cmd/git.exe'
        git = str(candidate) if candidate.is_file() else None
    revision, dirty = None, None
    if git:
        try:
            revision = subprocess.check_output([git,'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
            dirty = bool(subprocess.check_output([git,'status','--porcelain'],text=True,stderr=subprocess.DEVNULL))
        except subprocess.CalledProcessError:
            pass
    return {'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),
            'logical_cpu_count':os.cpu_count(),'packages':packages,'git_commit':revision,'git_dirty':dirty,
            'python_source_sha256':digest.hexdigest(),'opencv_threads':cv2.getNumThreads(),
            'frame_color_convention':'decoded OpenCV BGR; detector adapters perform their own RGB conversion'}


def combine_metrics(per_video):
    results = list(per_video.values())
    detection_keys = ('evaluated_frames','unknown_frames','gt_faces','predicted_faces','matches','unmatched_predictions','missed_faces','iou_sum')
    detection = {key:sum(r['detection'][key] for r in results) for key in detection_keys}
    complete = all(r['detection']['false_positives'] is not None for r in results)
    detection.update(precision=ratio(detection['matches'],detection['predicted_faces']) if complete else None,
                     recall=ratio(detection['matches'],detection['gt_faces']),
                     mean_matched_iou=ratio(detection['iou_sum'],detection['matches']),
                     false_positives=detection['unmatched_predictions'] if complete else None,
                     precision_unavailable_reason=None if complete else 'At least one sequence has partial annotation coverage')
    identity = all(r['tracking']['supported'] for r in results)
    tracking = {'supported':identity,'idsw':None,'idf1':None,'fragmentations':None,'hota':None,
                'unavailable_reason':None if identity else 'At least one sequence lacks exhaustive/appropriate identity annotation',
                'hota_unavailable_reason':'Supply isolated TrackEval interpreter; each sequence also needs complete identity annotations'}
    if identity:
        for key in ('idsw','fragmentations','idtp','idfp','idfn'):
            tracking[key] = sum(r['tracking'][key] for r in results)
        tracking['idf1'] = ratio(2*tracking['idtp'],2*tracking['idtp']+tracking['idfp']+tracking['idfn'])
    target = {'aggregation':'Per-video C3 results and denominators are authoritative; initialization/coverage may differ.'}
    planned=sum(r['initialization']['frame_index'] is not None for r in results)
    initialized=sum(r['initialization']['status']=='initialized' for r in results)
    target.update(initialization_candidates=planned,successful_initializations=initialized,
                  initialization_success_rate=ratio(initialized,planned),
                  sequences_without_eligible_multiface_frame=sum(r['initialization']['frame_index'] is None for r in results))
    for key in ('target_visible_annotated_frames','correct_target_frames','evaluable_target_assignments'):
        target[key] = sum(r['target'][key] for r in results)
    for key in ('bystander_replacement_frames','false_positive_target_frames'):
        values = [r['target'][key] for r in results]
        target[key] = sum(values) if all(value is not None for value in values) else None
    target['target_recall'] = ratio(target['correct_target_frames'],target['target_visible_annotated_frames'])
    target['wrong_person_assignment_rate'] = (ratio(target['bystander_replacement_frames'],target['evaluable_target_assignments'])
                                             if target['bystander_replacement_frames'] is not None else None)
    if target['bystander_replacement_frames'] is None:
        target['bystander_metrics_unavailable_reason'] = 'At least one sequence has single-target annotations without exhaustive bystander identities'
    return {'detection':detection,'tracking':tracking,'target':target}


def run_benchmark(manifest_path, dataset, output, config, video_ids=None, overlay=False,
                  hota_python=None, backend_factory=make_backend):
    try:
        import motmetrics
    except ImportError as exc:
        raise RuntimeError('Benchmark dependency missing. Install: python -m pip install -e ".[benchmark]"') from exc
    payload, entries, root = read_manifest(manifest_path,dataset,video_ids)
    if hota_python is not None and not Path(hota_python).is_file():
        raise FileNotFoundError(f'TrackEval interpreter not found: {hota_python}; create the isolated metrics environment')
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output is not empty: {output}; use a fresh directory')
    if config.experiment not in ('end_to_end','oracle'):
        raise ValueError('Unknown experiment')
    random.seed(config.seed)
    np.random.seed(config.seed)
    cv2.setRNGSeed(config.seed)
    cv2.setNumThreads(config.threads)
    metadata = environment()
    output.mkdir(parents=True, exist_ok=True)
    experiment = {'schema_version':1,'status':'incomplete','dataset':dataset,'release':payload.get('release'),
                  'split':payload.get('split'),'mode':config.experiment,
                  'oracle_warning':'Ground-truth BOXES supplied; not end-to-end system performance' if config.experiment=='oracle' else None,
                  'config':asdict(config),'manifest_path':str(Path(manifest_path).resolve()),
                  'manifest_sha256':fingerprint(manifest_path),'environment':metadata,'videos':{},
                  'ground_truth_boundary':'Normal inference receives raw frames plus a single recorded initialization; scoring/rendering run afterward.'}
    write_json(output/'experiment_config.json',experiment)
    performance, results, hota_groups = {}, {}, {}
    prediction_file = (output/'predictions.csv').open('w',newline='',encoding='utf-8')
    frame_file = (output/'frame_metrics.csv').open('w',newline='',encoding='utf-8')
    import csv
    prediction_writer = csv.DictWriter(prediction_file,fieldnames=PRED_FIELDS);prediction_writer.writeheader()
    frame_writer = csv.DictWriter(frame_file,fieldnames=FRAME_FIELDS);frame_writer.writeheader()
    try:
        for entry in entries:
            sequence = load_sequence(entry,root)
            video = sequence.video
            start = int(entry.get('start_frame',max(0,min(sequence.ground_truth)-math.ceil(2*video.fps))))
            end = int(entry.get('end_frame_exclusive',min(video.frame_count,max(sequence.ground_truth)+1)))
            indices = sampled_indices(video,start,end,config.fps,config.max_frames)
            initialization = plan_initialization(sequence,indices,config)
            oracle = None
            if config.experiment == 'oracle':
                if sequence.metadata['coverage'] != 'complete' or any(i not in sequence.ground_truth for i in indices):
                    raise ValueError('Oracle needs exhaustive boxes on every processed frame; choose a dense annotated interval. Unknown frames must not become empty detections.')
                size = config.resolution or (video.width,video.height)
                # Strip annotation identity even from ordering: a label rename must
                # not change the detections delivered to an oracle tracker.
                oracle = {i:tuple(sorted(scale_box(gt.box,(video.width,video.height),size)
                                        for gt in sequence.ground_truth[i] if gt.visible)) for i in indices}
            print(f'{video.video_id}: {config.experiment}, {len(indices)} source frames',flush=True)
            backend = backend_factory(config)
            try:
                frames,audit,speed = infer_video(video,indices,config,backend,initialization,oracle)
                speed['backend'] = backend.runtime_info() if hasattr(backend,'runtime_info') else {'name':config.backend_factory,'actual_device':'not reported by custom backend'}
            finally:
                backend.close()
            # GT does not enter normal inference. Evaluation starts only here.
            for frame in frames:
                prediction_writer.writerows(frame.rows(video.video_id))
            prediction_file.flush()
            per_video_folder=output/'per_video'/video.video_id
            per_video_folder.mkdir(parents=True,exist_ok=True)
            per_video_predictions=per_video_folder/'predictions.csv'
            write_csv(per_video_predictions,PRED_FIELDS,(row for frame in frames for row in frame.rows(video.video_id)))
            stage_snapshots = {frame.frame_index:frame.stages for frame in frames}
            frames=read_predictions(per_video_predictions,video,config.resolution or (video.width,video.height))
            for frame in frames:
                frame.stages = stage_snapshots[frame.frame_index]
            result,frame_rows,visual,hota_data = score_sequence(sequence,frames,audit,config)
            write_json(per_video_folder/'initialization_diagnostics.json',audit)
            if audit['status']!='initialized':
                write_json(per_video_folder/'initialization_failure.json',audit)
            if config.stage_diagnostics:
                from .diagnostics import write_stage_diagnostics
                stage_result = write_stage_diagnostics(sequence,frames,audit,config,per_video_folder)
                result['stage_diagnostics'] = stage_result['stages']
                if len(entries)==1:
                    for filename in ('stage_metrics.json','diagnostic_stages.csv','stage_detections.csv',
                                     'initialization_diagnostics.json','initialization_failure.json','diagnostic_report.md',
                                     'stage_snapshots.jsonl','initialization_frame.png'):
                        if (per_video_folder/filename).is_file():
                            shutil.copy2(per_video_folder/filename,output/filename)
                    for image in per_video_folder.glob('example_*.png'):
                        shutil.copy2(image,output/image.name)
            frame_writer.writerows(frame_rows);frame_file.flush()
            results[video.video_id],performance[video.video_id] = result,speed
            hota_groups[video.video_id] = hota_data
            experiment['videos'][video.video_id] = {
                'video_path':str(video.path),'video_sha256':fingerprint(video.path),
                'annotations_path':str(sequence.annotation_path),'annotations_sha256':fingerprint(sequence.annotation_path),
                'annotation_policy':sequence.metadata,'initialization':audit,
                'processed_start_frame':indices[0],'processed_end_frame':indices[-1]}
            if overlay:
                destination = output/'annotated_video.mp4' if len(entries)==1 else output/'per_video'/video.video_id/'annotated_video.mp4'
                destination.parent.mkdir(parents=True,exist_ok=True)
                render_overlay(video,frames,visual,destination,config.fps or video.fps)
        combined = combine_metrics(results)
        if hota_python:
            eligible = {key:value for key,value in hota_groups.items() if value}
            if eligible:
                input_path,output_path = output/'hota_input.json',output/'hota_metrics.json'
                write_json(input_path,{'groups':eligible})
                command = [str(Path(hota_python).resolve()),str(Path(__file__).with_name('hota_worker.py')),str(input_path),str(output_path)]
                completed = subprocess.run(command,text=True,capture_output=True)
                if completed.returncode:
                    raise RuntimeError(f'TrackEval failed: {completed.stderr[-2000:]}; install datasets/requirements-hota.txt in a separate environment')
                hota = json.loads(output_path.read_text(encoding='utf-8'))
                for key,value in hota['per_video'].items():
                    results[key]['tracking']['hota'] = value['hota']
                    results[key]['tracking']['hota_details'] = value
                    results[key]['tracking']['hota_unavailable_reason'] = None
                if combined['tracking']['supported']:
                    combined['tracking']['hota'] = hota['overall']['hota']
                    combined['tracking']['hota_unavailable_reason'] = None
            else:
                experiment['hota_note'] = 'No identity-complete, exhaustive sequences eligible for HOTA'
        total_frames = sum(value['frames_processed'] for value in performance.values())
        elapsed = sum(value['inference_elapsed_s'] for value in performance.values())
        if environment()['python_source_sha256'] != metadata['python_source_sha256']:
            raise RuntimeError('Python source changed during the benchmark; this run is not reproducible. Rerun from a fixed revision.')
        write_json(output/'tracking_metrics.json',{'overall':combined,'per_video':results})
        write_json(output/'performance.json',{'overall':{'frames_processed':total_frames,'inference_elapsed_s':elapsed,
                                                        'processing_fps':ratio(total_frames,elapsed)},
                                               'per_video':performance,'environment':metadata,
                                               'execution':{'gpu_requested':config.gpu,'appearance_model':config.appearance_model,
                                                            'sface_device':'CPU' if config.appearance_model=='sface' else None}})
        experiment['status'] = 'complete'
        write_json(output/'experiment_config.json',experiment)
        report = ['# Benchmark report', '', f'Dataset: {dataset}. Experiment: {config.experiment}.',
                  '', 'These are annotation-relative results for the recorded sequences and configuration.',
                  'Oracle results bypass the face detector and are not end-to-end system performance.' if config.experiment=='oracle' else '',
                  '', '| Video | Scored frames | Recall | Precision | Matched IoU | IDSW | IDF1 | HOTA | Target init |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
        def display(value):
            return 'not evaluable' if value is None else (f'{value:.4f}' if isinstance(value,float) else str(value))
        for key,value in results.items():
            d,t=value['detection'],value['tracking']
            report.append(f'| {key} | {d["evaluated_frames"]} | {display(d["recall"])} | {display(d["precision"])} | {display(d["mean_matched_iou"])} | {display(t["idsw"])} | {display(t["idf1"])} | {display(t["hota"])} | {value["initialization"]["status"]} |')
        report.extend(['','Null metrics are unavailable, not zero. See per-video reasons and target-metric denominators in tracking_metrics.json.',
                       'AVA track-local entity IDs do not establish persistent physical identity after disappearance.',
                       'Models were not trained or tuned by this runner. Mocked/synthetic tests are not real-world performance evidence.',
                       '', 'Reproducibility and initialization audit: experiment_config.json. Timing definitions and hardware: performance.json.'])
        if any('target_localization' in value for value in results.values()):
            report.extend(['','Single-target localization/coverage and selected-ID retention are in each video\'s target_localization object.',
                           'Best-box coverage is a post-inference diagnostic, not a GT-assisted tracking prediction. This warmup/confirmed-track initialization differs from official first-frame-box SOT protocols.'])
        (output/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
        return {'output':str(output),'metrics':combined,'videos':len(results)}
    except BaseException as exc:
        experiment.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        if hasattr(exc,'failure_category'):
            detail = {'failure_category':exc.failure_category,'frame_index':exc.frame_index,
                      'initialization':exc.initialization_audit,
                      'scope':'Input alignment failure; initialization may not yet have been attempted'}
            experiment['failure_details']=detail
            write_json(output/'alignment_failure.json',detail)
        write_json(output/'experiment_config.json',experiment)
        raise
    finally:
        prediction_file.close();frame_file.close()
