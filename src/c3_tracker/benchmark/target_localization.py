"""Single annotated-target coverage; no exhaustive bystander/physical-ID claims."""
from .matching import iou_matrix
from .schema import scale_box
import math


def score_target_localization(sequence, frames, initialization, config):
    target_id = sequence.metadata['designated_person_id']
    thresholds = sorted(set(config.target_iou_thresholds+(config.iou_threshold,)))
    totals = {threshold:0 for threshold in thresholds}
    selected_totals = {threshold:0 for threshold in thresholds}
    visible = absent = unknown = retained_visible = 0
    best_ious,selected_ious,rows = [],[],[]
    runs,active,previous,interruptions = [],None,None,0
    period = 1/(config.fps or sequence.video.fps)
    initialized = initialization['status']=='initialized'
    init_frame = initialization.get('frame_index')
    max_step = math.ceil(sequence.video.fps/(config.fps or sequence.video.fps))
    selected_previous,selected_index = None,None
    selected_interruptions = selected_run = selected_longest = 0
    def finish(censored):
        nonlocal active
        if active is not None:
            active['censored'] = censored
            active['sampled_duration_s'] = active['samples']*period
            runs.append(active);active=None
    for frame in frames:
        records = sequence.ground_truth.get(frame.frame_index)
        target = next((gt for gt in records or () if gt.person_id==target_id),None)
        if previous is not None and frame.frame_index-previous>max_step:
            finish(True)
        previous = frame.frame_index
        if target is None or not target.visible:
            unknown += int(target is None);absent += int(target is not None)
            finish(True)
            selected_previous,selected_index,selected_run = None,None,0
            continue
        visible += 1
        box = scale_box(target.box,(sequence.video.width,sequence.video.height),(frame.width,frame.height))
        observations = [o for o in frame.observations if o.observed and o.confirmed]
        overlaps = iou_matrix([box],[o.bbox for o in observations])[0]
        best = max(overlaps,default=0.0)
        chosen = next((float(overlaps[i]) for i,o in enumerate(observations) if o.track_id==frame.selected_track_id),0.0)
        best_ious.append(float(best))
        for threshold in thresholds:
            totals[threshold] += int(best>=threshold)
        selected_evaluable = initialized and frame.frame_index>init_frame
        if selected_evaluable:
            retained_visible += 1;selected_ious.append(chosen)
            for threshold in thresholds:
                selected_totals[threshold] += int(chosen>=threshold)
            selected_covered = chosen>=config.iou_threshold
            if selected_index is not None and frame.frame_index-selected_index>max_step:
                selected_previous,selected_run = None,0
            selected_interruptions += int(selected_previous is True and not selected_covered)
            selected_run = selected_run+1 if selected_covered else 0
            selected_longest = max(selected_longest,selected_run)
            selected_previous,selected_index = selected_covered,frame.frame_index
        covered = bool(best>=config.iou_threshold)
        if active is None or active['covered']!=covered:
            if active is not None:
                interruptions += int(active['covered'] and not covered)
                finish(False)
            active = {'covered':covered,'start_frame':frame.frame_index,'end_frame':frame.frame_index,'samples':0}
        active['end_frame']=frame.frame_index;active['samples']+=1
        rows.append({'frame_index':frame.frame_index,'best_observed_iou':float(best),
                     'selected_observed_iou':chosen if selected_evaluable else None})
    finish(True)
    return {'scope':'One native annotated target; other detected faces are not false positives',
            'visible_annotated_samples':visible,'annotated_absent_samples':absent,'unknown_samples':unknown,
            'mean_best_observed_iou':sum(best_ious)/visible if visible else None,
            'target_box_recall':{str(t):totals[t]/visible if visible else None for t in thresholds},
            'matched_target_samples':{str(t):totals[t] for t in thresholds},
            'coverage_gate':config.iou_threshold,'consecutive_coverage_runs':runs,
            'longest_consecutive_covered_samples':max((run['samples'] for run in runs if run['covered']),default=0),
            'target_detection_interruptions':interruptions,
            'selected_target_tracking_interruptions':selected_interruptions if retained_visible else None,
            'longest_consecutive_selected_retention_samples':selected_longest if retained_visible else None,
            'selected_target_visible_samples':retained_visible,
            'mean_selected_observed_iou':sum(selected_ious)/retained_visible if retained_visible else None,
            'visible_target_retention':selected_totals[config.iou_threshold]/retained_visible if retained_visible else None,
            'selected_target_recall':{str(t):selected_totals[t]/retained_visible if retained_visible else None for t in thresholds},
            'selected_target_unavailable_reason':None if retained_visible else 'Initialization failed/unavailable, or no labeled visible samples after initialization',
            'physical_reacquisition_evaluable':False,'per_visible_frame':rows,
            'definitions':{'localization':'Maximum IoU among current observed confirmed boxes; zero when none. Coverage diagnostic, not a GT-assisted tracking output.',
                           'retention':'Selected raw track ID IoU >= gate / labeled target-visible samples after one initialization; never selects a replacement ID',
                           'interruptions':'Covered-to-uncovered transitions within contiguous labeled-visible sampled runs; gaps/absence censor and reset runs',
                           'duration':'Sample count times nominal sampling period, not continuous reconstructed exposure',
                           'unsupported':'No exhaustive bystander precision/IDF1/HOTA or independently verified physical-person reacquisition'}}
