"""ChokePoint eye-point diagnostics, deliberately separate from box/MOT scoring.

Verified against the official Zenodo 815657 groundtruth and P1E_S1 archives.
The tracker sees decoded source JPEGs only. XML is parsed after inference.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .backend import make_backend
from .inference import BenchmarkConfig, PredictedFrame, sampled_indices
from .runner import environment
from .schema import PRED_FIELDS, VideoSpec, finite, fingerprint, read_predictions, validate_box, write_csv, write_json


@dataclass(frozen=True)
class EyeAnnotation:
    person_id: str
    left: tuple[float, float]
    right: tuple[float, float]


def load_eyes(path, sequence, width, height):
    root = ET.parse(path).getroot()
    if root.tag != 'dataset' or root.get('name') != sequence:
        raise ValueError('ChokePoint XML dataset name must match the image sequence folder')
    frames = {}
    for frame in root:
        if frame.tag != 'frame' or not re.fullmatch(r'\d{8}', frame.get('number', '')):
            raise ValueError('Expected zero-based eight-digit ChokePoint frame numbers')
        index = int(frame.get('number'))
        if index in frames:
            raise ValueError('Duplicate XML frame')
        people, seen = [], set()
        for person in frame:
            name = person.get('id', '').strip()
            if person.tag != 'person' or not name or name in seen:
                raise ValueError('Expected unique nonempty person IDs per frame')
            seen.add(name)
            if sorted(child.tag for child in person) != ['leftEye', 'rightEye']:
                raise ValueError('Each ChokePoint person must contain exactly two eye points')
            eyes = []
            for tag in ('leftEye', 'rightEye'):
                node = person.find(tag)
                try:
                    point = (finite(node.get('x'), tag), finite(node.get('y'), tag))
                except (ValueError,TypeError) as exc:
                    raise ValueError(f'{tag} needs finite x and y coordinates') from exc
                if not (0 <= point[0] < width and 0 <= point[1] < height):
                    raise ValueError('Eye point outside source image')
                eyes.append(point)
            if eyes[0] == eyes[1]:
                raise ValueError('Eye points must be distinct')
            people.append(EyeAnnotation(name, *eyes))
        frames[index] = tuple(people)
    if not frames:
        raise ValueError('No ChokePoint annotation frames')
    return frames


def match_eyes(people, predictions, scale=(1, 1)):
    """Accept only unambiguous one-person/one-box containment of BOTH eyes.

    Large boxes can contain eyes while localizing a face badly. This is not IoU.
    Overlapping predictions or boxes containing multiple people remain ambiguous.
    """
    candidates = []
    for person in people:
        candidates.append([j for j, prediction in enumerate(predictions)
                           if all(prediction.bbox[0] <= point[0]*scale[0] < prediction.bbox[2]
                                  and prediction.bbox[1] <= point[1]*scale[1] < prediction.bbox[3]
                                  for point in (person.left, person.right))])
    owners = {j: sum(j in values for values in candidates) for j in range(len(predictions))}
    matches = {i: values[0] for i, values in enumerate(candidates)
               if len(values) == 1 and owners[values[0]] == 1}
    ambiguous = [i for i, values in enumerate(candidates)
                 if values and i not in matches]
    return matches, ambiguous


def score_eyes(frames, ground_truth, source_size, source_fps, sample_fps):
    rows, people_summary, track_people, visual = [], {}, {}, {}
    last_person, last_owner, previous_index = {}, {}, None
    matches_total = annotations_total = ambiguous_total = changes = reuse = 0
    nominal_step = math.ceil(source_fps / (sample_fps or source_fps))
    for frame in frames:
        records = ground_truth.get(frame.frame_index, ())
        predictions = [p for p in frame.observations if p.observed and p.confirmed]
        matches, ambiguous = match_eyes(records, predictions, (frame.width/source_size[0], frame.height/source_size[1]))
        annotations_total += len(records)
        matches_total += len(matches)
        ambiguous_total += len(ambiguous)
        if previous_index is not None and frame.frame_index-previous_index > nominal_step:
            last_person.clear()  # Missing source frames cannot establish continuous tracking.
        previous_index = frame.frame_index
        current_people = {person.person_id for person in records}
        last_person = {key:value for key,value in last_person.items() if key in current_people}
        assignments = {}
        for i, person in enumerate(records):
            summary = people_summary.setdefault(person.person_id, {'annotated_samples':0, 'matched_samples':0,
                                                                  'track_ids':set(), 'observed_id_changes':0})
            summary['annotated_samples'] += 1
            if i not in matches:
                continue
            track = predictions[matches[i]].track_id
            assignments[person.person_id] = track
            summary['matched_samples'] += 1
            summary['track_ids'].add(track)
            track_people.setdefault(track, set()).add(person.person_id)
            if person.person_id in last_person and last_person[person.person_id] != track:
                summary['observed_id_changes'] += 1
                changes += 1
            if track in last_owner and last_owner[track] != person.person_id:
                reuse += 1
            last_person[person.person_id], last_owner[track] = track, person.person_id
        rows.append({'video_id':'', 'frame_index':frame.frame_index, 'timestamp_s':frame.timestamp_s,
                     'xml_frame_present':int(frame.frame_index in ground_truth), 'annotated_eye_pairs':len(records),
                     'observed_confirmed_predictions':len(predictions), 'matched_eye_pairs':len(matches),
                     'unmatched_eye_pairs':len(records)-len(matches), 'ambiguous_eye_pairs':len(ambiguous),
                     'eye_assignments':json.dumps(assignments,sort_keys=True)})
        visual[frame.frame_index] = (records, matches)
    for summary in people_summary.values():
        summary['track_ids'] = sorted(summary['track_ids'])
        summary['eye_containment_recall'] = summary['matched_samples']/summary['annotated_samples']
    result = {'metric_family':'ChokePoint eye-point diagnostics, not face-box MOT',
              'annotated_eye_pairs':annotations_total, 'matched_eye_pairs':matches_total,
              'unmatched_eye_pairs':annotations_total-matches_total, 'ambiguous_eye_pairs':ambiguous_total,
              'eye_containment_recall':matches_total/annotations_total if annotations_total else None,
              'observed_id_changes_within_annotated_runs':changes,
              'cross_person_track_reassignment_events':reuse,
              'track_ids_matched_to_multiple_people':{key:sorted(value) for key,value in track_people.items() if len(value)>1},
              'per_person':people_summary,
              'unavailable':dict.fromkeys(('precision','false_positives','iou','idsw','idf1','hota','target_recall','target_reacquisition'),
                                         'Eye points do not supply exhaustive face boxes; no target was designated'),
              'definitions':{
                  'eye_containment_recall':'Unambiguous both-eye containment in one observed confirmed prediction / annotated eye-pair samples',
                  'ambiguity':'An eye pair fits multiple boxes, or one box contains eye pairs from multiple people; counted as unmatched',
                  'observed_id_changes':'Different prediction IDs at successive matched samples of a continuously annotated person; unknown/source gaps reset continuity; not CLEAR IDSW',
                  'cross_person_track_reassignment_events':'A prediction ID matches another annotated person after its last eye-based match; not a physical-reacquisition accuracy score',
                  'coverage':'Only provided eye labels are scored; absent labels are never evidence of physical absence',
                  'limitation':'Eye containment can reward oversized boxes and cannot replace localization or exhaustive multi-person ground truth'}}
    return result, rows, visual


def render_eyes(paths, frames, visual, output, source_size, fps, expected_hashes):
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                             (frames[0].width, frames[0].height+54))
    try:
        if not writer.isOpened():
            raise RuntimeError('Cannot write ChokePoint overlay')
        for frame in frames:
            if fingerprint(paths[frame.frame_index]) != expected_hashes[frame.frame_index]:
                raise ValueError('Source JPEG changed before post-inference visualization')
            image = cv2.imread(str(paths[frame.frame_index]))
            image = cv2.resize(image, (frame.width,frame.height))
            canvas = cv2.copyMakeBorder(image,54,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
            cv2.putText(canvas,f'frame {frame.frame_index} | {frame.timestamp_s:.3f}s',(5,17),0,.4,(255,255,255),1)
            cv2.putText(canvas,'Eye points cyan/red | pred orange | IoU unavailable',(5,37),0,.32,(255,255,255),1)
            records,matches = visual[frame.frame_index]
            for i, person in enumerate(records):
                color = (255,255,0) if i in matches else (0,0,255)
                left = tuple(int(v*s) for v,s in zip(person.left,(frame.width/source_size[0],frame.height/source_size[1])))
                for point in (person.left,person.right):
                    x,y = (int(v*s) for v,s in zip(point,(frame.width/source_size[0],frame.height/source_size[1])))
                    cv2.circle(canvas,(x,y+54),3,color,-1)
                cv2.putText(canvas,'GT '+person.person_id,(left[0],max(65,left[1]+45)),0,.35,color,1)
            for p in frame.observations:
                if p.observed and p.confirmed:
                    x1,y1,x2,y2 = map(int,p.bbox)
                    cv2.rectangle(canvas,(x1,y1+54),(x2,y2+54),(0,150,255),1)
                    cv2.putText(canvas,'ID '+p.track_id,(x1,min(canvas.shape[0]-3,y2+66)),0,.35,(0,150,255),1)
            writer.write(canvas)
    finally:
        writer.release()


def run_sequence(images, annotations, output, config, start=0, end=900, source_fps=30,
                 overlay=False, backend_factory=make_backend):
    images,annotations,output = Path(images).resolve(),Path(annotations).resolve(),Path(output).resolve()
    if config.experiment != 'end_to_end' or config.target_enabled:
        raise ValueError('Eye-point diagnostics require raw inference with target designation disabled')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory')
    if not annotations.is_file() or not images.is_dir():
        raise FileNotFoundError('Need actual ChokePoint source JPEG folder and matching XML')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', images.name):
        raise ValueError('Invalid ChokePoint sequence folder name')
    paths = {int(path.stem):path for path in images.glob('*.jpg') if re.fullmatch(r'\d{8}',path.stem)}
    if not paths or not 0 <= start < end or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError('Need eight-digit source JPEGs and a valid frame interval/FPS')
    requested = sampled_indices(VideoSpec(images.name,images,1,1,source_fps,max(end,max(paths)+1),0),start,end,config.fps)
    indices = [index for index in requested if index in paths]
    if not indices:
        raise ValueError('No available source frames in requested sampling interval')
    random.seed(config.seed);np.random.seed(config.seed);cv2.setRNGSeed(config.seed);cv2.setNumThreads(config.threads)
    metadata = environment()
    annotation_sha = fingerprint(annotations)
    output.mkdir(parents=True,exist_ok=True)
    audit = {'status':'running','dataset':'chokepoint','experiment':'raw-video eye-point diagnostics',
             'config':asdict(config),'environment':metadata,'images':str(images),'annotations':str(annotations),
             'annotations_sha256':annotation_sha,'start_frame':start,'end_frame_exclusive':end,
             'source_fps':source_fps,'processed_source_indices':indices,
             'missing_requested_source_frames':len(requested)-len(indices),
             'ground_truth_boundary':'XML parsed only after inference; no initialization, oracle boxes, IDs or eye points enter the backend',
             'overlay_timing':'Available sampled images only; gaps compress playback but source indices/times remain displayed'}
    write_json(output/'experiment_config.json',audit)
    frames,hashes,digest = [],{},hashlib.sha256()
    try:
        backend = backend_factory(config)
        try:
            started_wall = time.perf_counter()
            source_size = None
            for index in indices:
                started = time.perf_counter()
                data = paths[index].read_bytes()
                hashes[index] = hashlib.sha256(data).hexdigest()
                digest.update(f'{index}:{hashes[index]}\n'.encode())
                image = cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f'Cannot decode source JPEG {paths[index]}')
                native = image.shape[1],image.shape[0]
                if source_size is not None and native != source_size:
                    raise ValueError('Source image resolution changed')
                source_size = native
                size = config.resolution or native
                if size != native:
                    image = cv2.resize(image,size)
                predictions = tuple(backend.process(image,index/source_fps))
                if len({p.track_id for p in predictions}) != len(predictions):
                    raise ValueError('Duplicate predicted ID in frame')
                for prediction in predictions:
                    validate_box(prediction.bbox,*size)
                frames.append(PredictedFrame(index,len(frames),index/source_fps,predictions,
                                             (time.perf_counter()-started)*1000,None,'NOT SELECTED',*size))
            elapsed = time.perf_counter()-started_wall
            runtime = backend.runtime_info() if hasattr(backend,'runtime_info') else {'name':config.backend_factory}
        finally:
            backend.close()
        # No source annotation data has been loaded by the inference loop above.
        gt = load_eyes(annotations,images.name,*source_size)
        if max(gt) < indices[0] or min(gt) > indices[-1]:
            raise ValueError('XML does not overlap the inference interval')
        write_csv(output/'predictions.csv',PRED_FIELDS,(row for f in frames for row in f.rows(images.name)))
        video = VideoSpec(images.name,images,*source_size,source_fps,max(end,max(paths)+1),0)
        frames = read_predictions(output/'predictions.csv',video,config.resolution or source_size)
        metrics,rows,visual = score_eyes(frames,gt,source_size,source_fps,config.fps)
        for row in rows:
            row['video_id'] = images.name
        write_csv(output/'frame_metrics.csv',list(rows[0]),rows)
        if overlay:
            render_eyes(paths,frames,visual,output/'annotated_video.mp4',source_size,config.fps or source_fps,hashes)
        if fingerprint(annotations) != annotation_sha or environment()['python_source_sha256'] != metadata['python_source_sha256']:
            raise RuntimeError('Annotations or benchmark source changed during the experiment')
        latencies = [f.processing_ms for f in frames]
        performance = {'frames_processed':len(frames),'inference_elapsed_s':elapsed,'processing_fps':len(frames)/elapsed,
                       'latency_mean_ms':float(np.mean(latencies)),'latency_p95_ms':float(np.percentile(latencies,95)),
                       'source_resolution':source_size,'source_fps':source_fps,'requested_sampling_fps':config.fps,
                       'output_resolution':[frames[0].width,frames[0].height],'backend':runtime,
                       'definition':'Read/hash/decode/resize source JPEG and infer; excludes model startup, XML parsing, export, scoring and overlays'}
        write_json(output/'tracking_metrics.json',metrics)
        write_json(output/'performance.json',performance)
        audit.update(status='complete',source_images_sha256=digest.hexdigest(),source_image_hashes=hashes)
        write_json(output/'experiment_config.json',audit)
        recall = metrics['eye_containment_recall']
        report = ['# ChokePoint eye-point diagnostic report','',f'Sequence: {images.name}. Raw images, existing tracker.',
                  f'Source interval [{start},{end}) at {source_fps} FPS; {len(frames)} sampled JPEGs.',
                  '',f'Annotated eye pairs: {metrics["annotated_eye_pairs"]}. Unambiguous matches: {metrics["matched_eye_pairs"]}.',
                  f'Eye-containment recall: {recall:.6f}.' if recall is not None else 'Eye-containment recall unavailable: no eye labels in sampled frames.',
                  f'Observed ID changes within annotated runs: {metrics["observed_id_changes_within_annotated_runs"]}.',
                  f'Cross-person prediction-ID reassignment events: {metrics["cross_person_track_reassignment_events"]}.',
                  '', 'These are custom eye-point diagnostics. IoU, standard IDSW/IDF1/HOTA and target reacquisition are unavailable.',
                  'No face boxes were invented from eye points. An oversized box can pass this check; inspect overlays.',
                  'Missing frames preserve their source timeline and break continuity. Unlabeled people/frames are not scored as absent.',
                  'One subject at a time has eye labels; other visible faces may be unlabeled. No target was selected, so C3 retention is not tested.',
                  'All metric definitions and per-person counts are in tracking_metrics.json. Predictions are exported before scoring.']
        (output/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
        return {'output':str(output),'frames':len(frames),'eye_containment_recall':recall,
                'id_changes':metrics['observed_id_changes_within_annotated_runs'],
                'cross_person_reassignments':metrics['cross_person_track_reassignment_events']}
    except BaseException as exc:
        audit.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        write_json(output/'experiment_config.json',audit)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--start-frame',type=int,default=0)
    parser.add_argument('--end-frame',type=int,default=900)
    parser.add_argument('--matrix',action='store_true')
    parser.add_argument('--annotate-video',action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError('Use a fresh output directory')
        config = BenchmarkConfig(target_enabled=False)
        if args.matrix:
            args.output.mkdir(parents=True,exist_ok=True)
            audit = {'status':'running','python_source_sha256':environment()['python_source_sha256'],'variants':{}}
            write_json(args.output/'matrix_manifest.json',audit)
            try:
                for size in (None,(640,480),(320,240)):
                    for fps in (None,10):
                        if environment()['python_source_sha256'] != audit['python_source_sha256']:
                            raise RuntimeError('Source changed between matrix variants')
                        label = ('original' if size is None else f'{size[0]}x{size[1]}')+('_nativefps' if fps is None else '_10fps')
                        result = run_sequence(args.images,args.annotations,args.output/label,replace(config,resolution=size,fps=fps),
                                              args.start_frame,args.end_frame,overlay=args.annotate_video)
                        audit['variants'][label] = result
                        write_json(args.output/'matrix_manifest.json',audit)
                        print(json.dumps(result),flush=True)
                audit['status'] = 'complete'
            except BaseException as exc:
                audit.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                write_json(args.output/'matrix_manifest.json',audit)
        else:
            print(json.dumps(run_sequence(args.images,args.annotations,args.output,config,args.start_frame,args.end_frame,
                                          overlay=args.annotate_video)),flush=True)
        return 0
    except (ValueError,OSError,RuntimeError,ImportError,ET.ParseError) as exc:
        print(f'ChokePoint test error: {exc}',file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
