"""Post-inference stage diagnostics. No GT is supplied to a detector or tracker."""
from pathlib import Path
import json
import cv2
import numpy as np

from .matching import iou_matrix, match_boxes
from .schema import scale_box, write_csv, write_json

STAGES = ('raw_detector_probe','baseline_detector_output','confidence_accepted','descriptor_success','tentative_tracks',
          'confirmed_tracks','confirmed_observed','all_observed')


def observation_dict(observation):
    return {'track_id':str(observation.track_id),'bbox':list(map(float,observation.bbox)),
            'observed':bool(observation.observed),'confirmed':bool(observation.confirmed),
            'detection_confidence':None if observation.detection_confidence is None else float(observation.detection_confidence)}


def best_overlap(box, candidates):
    return float(max(iou_matrix([box],[item['bbox'] for item in candidates])[0],default=0))


def initialization_diagnostic(target_box, observations, snapshot, config):
    predictions = [observation_dict(o) for o in observations]
    eligible = [p for p in predictions if p['observed'] and p['confirmed']]
    tentative = [p for p in predictions if p['observed'] and not p['confirmed']]
    candidates = sum(best_overlap(target_box,[p])>=config.iou_threshold for p in eligible)
    audit = {'available_predictions':predictions,'plausible_confirmed_tracks':candidates,
             'multiple_plausible_tracks':candidates>1,
             'ambiguity_policy':'Existing one-to-one Hungarian assignment; no new ambiguity rejection or future retry',
             'failure_category':'no_matching_predicted_track','stage_max_iou':{}}
    if snapshot is not None:
        audit['available_stage_detections'] = {stage:snapshot.get(stage) for stage in STAGES[:4]}
        for stage in STAGES[:4]:
            values = snapshot.get(stage)
            audit['stage_max_iou'][stage] = None if values is None else best_overlap(target_box,values)
        accepted = audit['stage_max_iou']['baseline_detector_output']
        described = audit['stage_max_iou']['descriptor_success']
        raw = snapshot.get('raw_detector_probe')
        if accepted is not None and accepted<config.iou_threshold:
            overlapping = [item for item in raw or () if best_overlap(target_box,[item])>=config.iou_threshold]
            if overlapping:
                audit['failure_category'] = ('detection_below_confidence_in_probe' if
                    all(item['confidence']<config.confidence for item in overlapping) else 'raw_probe_baseline_disagreement')
            elif raw is not None:
                audit['failure_category'] = ('raw_detection_below_initialization_iou_gate' if
                    best_overlap(target_box,raw)>0 else 'no_raw_detection_overlapping_designated_target')
        elif described is not None and described<config.iou_threshold:
            audit['failure_category'] = 'appearance_extraction_failure'
        elif best_overlap(target_box,tentative)>=config.iou_threshold and not candidates:
            audit['failure_category'] = 'track_still_unconfirmed'
    elif best_overlap(target_box,tentative)>=config.iou_threshold and not candidates:
        audit['failure_category'] = 'track_still_unconfirmed'
    audit['stage_max_iou']['confirmed_observed'] = best_overlap(target_box,eligible)
    return audit


def frame_stages(frame):
    snapshot = frame.stages or {}
    tracks = [observation_dict(o) for o in frame.observations]
    return {**{key:snapshot.get(key) for key in STAGES[:4]},
            'tentative_tracks':[o for o in tracks if not o['confirmed']],
            'confirmed_tracks':[o for o in tracks if o['confirmed']],
            'confirmed_observed':[o for o in tracks if o['confirmed'] and o['observed']],
            'all_observed':[o for o in tracks if o['observed']]}


def _failure_example(video, frame, ground_truth, stages, category, output):
    cap = cv2.VideoCapture(str(video.path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame.frame_index)
        ok,image = cap.read()
        if not ok:
            raise ValueError('Cannot decode post-inference diagnostic illustration')
    finally:
        cap.release()
    image = cv2.resize(image,(frame.width,frame.height))
    image = cv2.copyMakeBorder(image,90,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
    for stage,color in [('baseline_detector_output',(0,150,255)),('descriptor_success',(0,220,0)),('confirmed_observed',(255,0,255))]:
        for item in stages.get(stage) or ():
            x1,y1,x2,y2 = map(int,item['bbox'])
            cv2.rectangle(image,(x1,y1+90),(x2,y2+90),color,1)
    x1,y1,x2,y2 = map(int,ground_truth)
    cv2.rectangle(image,(x1,y1+90),(x2,y2+90),(255,255,0),2)
    lines = [f'frame {frame.frame_index} | {frame.timestamp_s:.3f}s',category,
             'GT cyan | baseline orange | descriptor green', 'confirmed observed magenta | overlays after inference']
    for index,text in enumerate(lines):
        width = cv2.getTextSize(text,0,.38,1)[0][0]
        scale = .38*min(1,(frame.width-10)/max(1,width))
        cv2.putText(image,text,(5,17+index*20),0,scale,(255,255,255),1)
    if not cv2.imwrite(str(output),image):
        raise RuntimeError('Cannot save diagnostic illustration')


def write_stage_diagnostics(sequence, frames, initialization, config, output):
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    gates = (.1,.3,.5,.7)
    totals = {key:{'output_boxes_all_frames':0,'annotated_face_denominator':0,'matched_faces':0,
                   'available_frames':0,'gate_matches':{str(gate):0 for gate in gates},'best_iou_sum':0.0} for key in STAGES}
    confidence_sensitivity = {str(t):{'matched_faces':0,'annotated_face_denominator':0} for t in (.1,.3,.5,.65,.8)}
    rows,detections,examples,categories = [],[],[],{}
    baseline_scores,probe_scores = [],[]
    selected_examples = {}
    snapshots = (output/'stage_snapshots.jsonl').open('w',encoding='utf-8')
    try:
        for frame in frames:
            stages = frame_stages(frame)
            baseline_scores.extend(item['confidence'] for item in stages['baseline_detector_output'] or ())
            probe_scores.extend(item['confidence'] for item in stages['raw_detector_probe'] or ())
            snapshots.write(json.dumps({'frame_index':frame.frame_index,'timestamp_s':frame.timestamp_s,'stages':frame.stages},allow_nan=False)+'\n')
            known = frame.frame_index in sequence.ground_truth
            gt = [g for g in sequence.ground_truth.get(frame.frame_index,()) if g.visible]
            boxes = [scale_box(g.box,(sequence.video.width,sequence.video.height),(frame.width,frame.height)) for g in gt]
            row = {'video_id':sequence.video.video_id,'frame_index':frame.frame_index,'timestamp_s':frame.timestamp_s,
                   'annotation_frame_known':int(known),'annotated_faces':len(gt) if known else None}
            for stage,items in stages.items():
                row[stage+'_count'] = None if items is None else len(items)
                row[stage+'_matched_faces'] = None
                row[stage+'_recall'] = None
                if items is None:
                    continue
                total = totals[stage];total['available_frames']+=1;total['output_boxes_all_frames']+=len(items)
                for ordinal,item in enumerate(items):
                    detections.append({'video_id':sequence.video.video_id,'frame_index':frame.frame_index,
                                       'timestamp_s':frame.timestamp_s,'stage':stage,'detection_index':ordinal,
                                       'track_id':item.get('track_id',''),
                                       **dict(zip(('x1','y1','x2','y2'),item['bbox'])),
                                       'confidence':item.get('confidence',item.get('detection_confidence')),
                                       'observed':item.get('observed',''),'confirmed':item.get('confirmed','')})
                if not known or stage=='confirmed_tracks':
                    continue  # Predicted-only boxes are counted, never given detection recall.
                stage_boxes = [item['bbox'] for item in items]
                matches,_,_,overlaps = match_boxes(boxes,stage_boxes,config.iou_threshold)
                total['annotated_face_denominator']+=len(gt);total['matched_faces']+=len(matches)
                total['best_iou_sum']+=sum(float(max(values,default=0)) for values in overlaps)
                for gate in gates:
                    total['gate_matches'][str(gate)]+=len(match_boxes(boxes,stage_boxes,gate)[0])
                row[stage+'_matched_faces']=len(matches)
                row[stage+'_recall']=len(matches)/len(gt) if gt else None
            row['spatially_matched_observations']=row['confirmed_observed_matched_faces']
            if known:
                probe = stages['raw_detector_probe']
                if probe is not None:
                    for threshold,values in confidence_sensitivity.items():
                        filtered = [p['bbox'] for p in probe if p['confidence']>=float(threshold)]
                        values['matched_faces']+=len(match_boxes(boxes,filtered,config.iou_threshold)[0])
                        values['annotated_face_denominator']+=len(gt)
                final = stages['confirmed_observed']
                misses = match_boxes(boxes,[item['bbox'] for item in final],config.iou_threshold)[1]
                for index in misses:
                    evidence = {stage:None if values is None else best_overlap(boxes[index],values)
                                for stage,values in stages.items() if stage!='confirmed_tracks'}
                    if stages['baseline_detector_output'] is None or stages['descriptor_success'] is None:
                        category = 'stage_telemetry_unavailable'
                    elif evidence['baseline_detector_output']<config.iou_threshold:
                        category = 'no_baseline_box_at_iou_gate'
                    elif evidence['descriptor_success']<config.iou_threshold:
                        category = 'appearance_extraction_loss'
                    elif evidence['all_observed']<config.iou_threshold:
                        category = 'no_current_track_observation'
                    elif evidence['confirmed_observed']<config.iou_threshold:
                        category = 'track_confirmation_loss'
                    else:
                        category = 'one_to_one_assignment_competition'
                    categories[category]=categories.get(category,0)+1
                    if selected_examples.get(category,0)<2:
                        destination=output/f'example_{category}_{frame.frame_index}_{index}.png'
                        _failure_example(sequence.video,frame,boxes[index],stages,category,destination)
                        examples.append({'frame_index':frame.frame_index,'person_id':gt[index].person_id,
                                         'ground_truth_bbox':boxes[index],'category':category,'stage_best_iou':evidence,
                                         'image':destination.name})
                        selected_examples[category]=selected_examples.get(category,0)+1
            rows.append(row)
    finally:
        snapshots.close()
    for stage,total in totals.items():
        denominator = total['annotated_face_denominator']
        total['recall'] = total['matched_faces']/denominator if denominator else None
        total['mean_best_iou_including_misses'] = total['best_iou_sum']/denominator if denominator else None
        total['iou_gate_recall'] = {key:value/denominator if denominator else None for key,value in total['gate_matches'].items()}
        if stage=='confirmed_tracks':
            total['matched_faces']=None;total['unavailable_reason']='Includes predicted-only boxes; detection recall intentionally unavailable'
        elif not total['available_frames']:
            total['matched_faces']=None;total['unavailable_reason']='Backend did not expose this stage'
    for value in confidence_sensitivity.values():
        value['recall']=value['matched_faces']/value['annotated_face_denominator'] if value['annotated_face_denominator'] else None
    result = {'stages':totals,'missed_confirmed_face_diagnostics':categories,'examples':examples,
              'initialization':initialization,'iou_gate':config.iou_threshold,
              'raw_probe_confidence':min(config.diagnostic_probe_confidence,config.confidence),
              'baseline_confidence_configured':config.confidence,
              'baseline_returned_scores_below_configured_confidence':sum(score<config.confidence for score in baseline_scores) if baseline_scores else None,
              'baseline_min_returned_score':min(baseline_scores,default=None),
              'probe_min_returned_score':min(probe_scores,default=None),
              'confidence_filter_scope':'confidence_accepted is an explicit reported-score filter for diagnosis only. The unchanged tracker consumes baseline_detector_output, not this diagnostic subset.',
              'confidence_sensitivity':confidence_sensitivity,
              'confidence_sensitivity_scope':'Post-filtered outputs of one low-floor MediaPipe probe, not separate operational detector runs and not threshold optimization',
              'stage_scope':'Independent per-stage one-to-one IoU matches to provided GT; unknown annotation frames excluded. Counts do not certify exhaustive face coverage.',
              'raw_stage_scope':'Probe is a separate post-NMS MediaPipe pass at a requested score floor. Actual returned scores are measured; no effective floor, pre-NMS or truly unthresholded network output is assumed. Baseline invocation is unchanged.',
              'failure_scope':'Per-final-missed-GT spatial stage evidence, not a guaranteed causal partition; one-to-one assignments can differ between stages.',
              'timing_scope':'Instrumented end-to-end latency includes probe overhead; baseline detector+tracking timing is separately retained in snapshots.'}
    init_frame = next((frame for frame in frames if frame.frame_index==initialization.get('frame_index')),None)
    if init_frame is not None and initialization.get('target_box') is not None:
        destination = output/'initialization_frame.png'
        _failure_example(sequence.video,init_frame,initialization['target_box'],frame_stages(init_frame),
                         initialization.get('failure_category') or initialization['status'],destination)
        result['initialization_example'] = destination.name
    write_json(output/'stage_metrics.json',result)
    write_json(output/'initialization_diagnostics.json',initialization)
    if initialization['status']!='initialized':
        write_json(output/'initialization_failure.json',initialization)
    write_csv(output/'diagnostic_stages.csv',list(rows[0]),rows)
    detection_fields = ('video_id','frame_index','timestamp_s','stage','detection_index','track_id','x1','y1','x2','y2','confidence','observed','confirmed')
    write_csv(output/'stage_detections.csv',detection_fields,detections)
    lines=['# Stage diagnostic report','',f'Video: {sequence.video.video_id}. IoU gate: {config.iou_threshold}.',
           '', '| Stage | Output boxes | GT denominator | Matched GT | Recall | Mean best IoU, misses included |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for stage,total in totals.items():
        display = lambda value:'unavailable' if value is None else (str(value) if isinstance(value,int) else f'{value:.4f}')
        lines.append(f'| {stage} | {total["output_boxes_all_frames"]} | {total["annotated_face_denominator"]} | {display(total["matched_faces"])} | {display(total["recall"])} | {display(total["mean_best_iou_including_misses"])} |')
    lines += ['',f'Initialization: {initialization["status"]}; category: {initialization.get("failure_category")}.',
              'The raw probe is post-NMS at a requested low score floor. Actual baseline detector outputs are reported separately.',
              'confidence_accepted applies an explicit reported-score threshold for diagnosis only; it is not substituted into the baseline pipeline.',
              'Appearance extraction and observation/confirmation stages reuse the actual baseline execution.',
              'Confidence sensitivity and overlap-gate sensitivity are diagnostics; baseline thresholds were not optimized.',
              'Known-frame recall is annotation-relative. Standard identity/precision eligibility remains governed by the main benchmark.',
              'See stage_metrics.json for failure examples, denominators, caveats and complete initialization evidence.']
    (output/'diagnostic_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
