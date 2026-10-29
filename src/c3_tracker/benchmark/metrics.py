"""Annotation-relative detection, established MOT, and explicitly scoped C3 metrics."""
from __future__ import annotations
import math
import numpy as np
from .matching import match_boxes
from .schema import scale_box

FRAME_FIELDS = ['video_id', 'frame_index', 'timestamp_s', 'evaluated', 'skip_reason',
                'gt_faces', 'predicted_faces', 'matches', 'unmatched_predictions', 'false_positives',
                'missed_faces', 'mean_matched_iou', 'id_switches', 'selected_track_id',
                'designated_person_id', 'target_visible_gt', 'target_outcome', 'matched_person_id',
                'target_status', 'processed_frame_index']


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def clean_scalar(value):
    if hasattr(value, 'item'):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _hota_data(block):
    gt_names = sorted({name for gt, _, _ in block for name in gt})
    pred_names = sorted({name for _, pred, _ in block for name in pred})
    gt_map, pred_map = {name:i for i,name in enumerate(gt_names)}, {name:i for i,name in enumerate(pred_names)}
    return {'num_timesteps': len(block), 'num_gt_ids': len(gt_names), 'num_tracker_ids': len(pred_names),
            'num_gt_dets': sum(len(gt) for gt,_,_ in block), 'num_tracker_dets': sum(len(pred) for _,pred,_ in block),
            'gt_ids': [[gt_map[name] for name in gt] for gt,_,_ in block],
            'tracker_ids': [[pred_map[name] for name in pred] for _,pred,_ in block],
            'similarity_scores': [overlaps.tolist() for _,_,overlaps in block]}


class TargetMetrics:
    """Evaluation-only state; never passed back to the tracker."""
    def __init__(self, enabled, persistent, period):
        self.enabled, self.persistent, self.period = enabled, persistent, period
        self.visible = self.correct = self.assigned = self.wrong = self.false_positive = 0
        self.losses, self.active_loss = [], None
        self.runtime_previous = None
        self.presence_previous = True
        self.pending_return = False
        self.in_return = False
        self.return_successes = self.return_failures = self.return_censored = 0
        self.attempts = self.false_attempts = 0

    def close_loss(self, censored):
        if self.active_loss is not None:
            self.active_loss['censored'] = censored
            self.losses.append(self.active_loss)
            self.active_loss = None

    def update(self, frame, outcome, visible):
        if outcome in ('not_evaluable', 'initialization'):
            self.close_loss(True)
            self.runtime_previous = None
            if self.pending_return:
                self.return_censored += 1
            self.pending_return = self.in_return = False
            self.presence_previous = None
            return
        self.visible += int(visible)
        self.correct += int(outcome == 'correct')
        self.assigned += int(outcome in ('correct', 'wrong_person', 'false_positive'))
        self.wrong += int(outcome == 'wrong_person')
        self.false_positive += int(outcome == 'false_positive')
        if visible and outcome != 'correct':
            if self.active_loss is None:
                self.active_loss = {'start_frame': frame.frame_index, 'start_timestamp_s': frame.timestamp_s,
                                    'end_frame': frame.frame_index, 'duration_s': 0.0}
            self.active_loss['end_frame'] = frame.frame_index
            self.active_loss['duration_s'] += self.period
        else:
            self.close_loss(outcome != 'correct')
        if self.persistent:
            if visible and self.presence_previous is False:
                self.pending_return = self.in_return = True
            if self.pending_return and outcome == 'correct':
                self.return_successes += 1
                self.pending_return = False
            if not visible:
                if self.pending_return:
                    self.return_failures += 1
                self.pending_return = self.in_return = False
            if frame.target_visible and self.runtime_previous is False:
                self.attempts += 1
                self.false_attempts += int(outcome in ('wrong_person', 'false_positive'))
            self.runtime_previous = frame.target_visible
            self.presence_previous = visible

    def finish(self):
        self.close_loss(True)
        if self.pending_return:
            self.return_censored += 1
            self.pending_return = False
        available = self.enabled
        return {'target_recall': ratio(self.correct, self.visible) if available else None,
                'supported': available,
                'unavailable_reason': None if available else 'Target initialization failed/was unavailable or identity annotations have no temporal meaning',
                'initialization_frame_excluded': True,
                'target_visible_annotated_frames': self.visible, 'correct_target_frames': self.correct,
                'evaluable_target_assignments': self.assigned,
                'wrong_person_assignment_rate': ratio(self.wrong, self.assigned) if available else None,
                'false_target_assignment_rate': ratio(self.wrong + self.false_positive, self.assigned) if available else None,
                'bystander_replacement_frames': self.wrong, 'false_positive_target_frames': self.false_positive,
                'target_loss_duration_s': sum(loss['duration_s'] for loss in self.losses) if available else None,
                'target_loss_episodes': self.losses,
                'correct_target_reacquisition_rate': ratio(self.return_successes, self.return_successes + self.return_failures)
                if available and self.persistent else None,
                'false_reacquisition_rate': ratio(self.false_attempts, self.attempts) if available and self.persistent else None,
                'successful_return_opportunities': self.return_successes,
                'failed_completed_return_opportunities': self.return_failures,
                'censored_return_opportunities': self.return_censored,
                'evaluable_reacquisition_attempts': self.attempts, 'false_reacquisition_attempts': self.false_attempts,
                'physical_reacquisition_evaluable': available and self.persistent,
                'reacquisition_unavailable_reason': None if available and self.persistent else
                ('target not initialized or identities unavailable' if not available else
                 'Track-local IDs/unknown absence do not establish physical identity after disappearance'),
                'denominators': {
                    'target_recall': 'correct matched target / annotated target-visible frames after initialization',
                    'wrong_person_assignment_rate': 'assignments matched to another annotated face / evaluable observed target assignments',
                    'false_target_assignment_rate': '(wrong-person + unmatched in exhaustive frames) / evaluable observed target assignments',
                    'target_loss_duration_s': 'incorrect/unassigned annotated target-visible samples * nominal sampling period; gaps break episodes',
                    'correct_target_reacquisition_rate': 'successful returns / (successful + completed unsuccessful physical return episodes); unresolved ends censored',
                    'false_reacquisition_rate': 'wrong/false observed returns after runtime loss / evaluable observed return attempts',
                    'bystander_replacement_frames': 'sampled frames in which selected box matches a different annotated face',
                }}


def score_sequence(sequence, predicted_frames, initialization, config):
    import motmetrics as mm
    video = sequence.video
    complete = sequence.metadata['coverage'] == 'complete'
    identity_enabled = (complete and sequence.metadata['identity_annotations_complete']
                        and sequence.metadata['identity_scope'] != 'none')
    target_enabled = initialization['status'] == 'initialized' and sequence.metadata['identity_scope'] != 'none'
    persistent = (sequence.metadata['identity_scope'] == 'persistent'
                  and sequence.metadata['absence_annotations_complete'] and complete
                  and sequence.metadata['identity_annotations_complete'])
    target_id = initialization.get('person_id')
    start = initialization.get('frame_index')
    period = 1 / (config.fps or video.fps)
    target_metrics = TargetMetrics(target_enabled, persistent, period)
    totals = dict(evaluated_frames=0, unknown_frames=0, gt_faces=0, predicted_faces=0,
                  matches=0, unmatched_predictions=0, missed_faces=0, iou_sum=0.0)
    frame_metrics, visual = [], {}
    blocks, active_block = [], []
    accumulators, accumulator = [], None
    gt_names, prediction_names = {}, {}

    def flush_block():
        nonlocal active_block, accumulator
        if active_block:
            blocks.append(active_block)
            accumulators.append(accumulator)
        active_block, accumulator = [], None

    for frame in predicted_frames:
        evaluated = frame.frame_index in sequence.ground_truth
        predictions = [o for o in frame.observations if o.observed and o.confirmed]
        gt = [g for g in sequence.ground_truth.get(frame.frame_index, []) if g.visible]
        boxes = [scale_box(g.box, (video.width, video.height), (frame.width, frame.height)) for g in gt]
        row = dict(video_id=video.video_id, frame_index=frame.frame_index, timestamp_s=frame.timestamp_s,
                   evaluated=int(evaluated), skip_reason='' if evaluated else 'annotation_coverage_unknown',
                   gt_faces=len(gt) if evaluated else '', predicted_faces=len(predictions), matches='',
                   unmatched_predictions='', false_positives='', missed_faces='', mean_matched_iou='',
                   id_switches='', selected_track_id=frame.selected_track_id or '', designated_person_id=target_id or '',
                   target_visible_gt='', target_outcome='not_evaluable', matched_person_id='',
                   target_status=frame.target_status, processed_frame_index=frame.processed_frame_index)
        switches = []
        if not evaluated:
            totals['unknown_frames'] += 1
            flush_block()
            target_metrics.update(frame, 'not_evaluable', False)
            frame_metrics.append(row)
            visual[frame.frame_index] = {'gt': [], 'row': row, 'switches': []}
            continue
        matches, misses, unmatched, overlaps = match_boxes(boxes, [o.bbox for o in predictions], config.iou_threshold)
        iou_sum = sum(float(overlaps[i, j]) for i, j in matches)
        totals['evaluated_frames'] += 1
        totals['gt_faces'] += len(gt)
        totals['predicted_faces'] += len(predictions)
        totals['matches'] += len(matches)
        totals['unmatched_predictions'] += len(unmatched)
        totals['missed_faces'] += len(misses)
        totals['iou_sum'] += iou_sum
        row.update(matches=len(matches), unmatched_predictions=len(unmatched),
                   false_positives=len(unmatched) if complete else '', missed_faces=len(misses),
                   mean_matched_iou=ratio(iou_sum, len(matches)) if matches else '')
        if identity_enabled:
            if accumulator is None:
                accumulator = mm.MOTAccumulator(auto_id=False)
            for g in gt:
                gt_names.setdefault(g.person_id, len(gt_names))
            for p in predictions:
                prediction_names.setdefault(p.track_id, len(prediction_names))
            distances = 1 - overlaps
            distances[overlaps < config.iou_threshold] = np.nan
            accumulator.update([gt_names[g.person_id] for g in gt],
                               [prediction_names[p.track_id] for p in predictions], distances, frameid=frame.frame_index)
            events = accumulator.events.loc[frame.frame_index]
            switched = events[events['Type'] == 'SWITCH']
            reverse = {value:key for key,value in gt_names.items()}
            switches = [reverse[int(value)] for value in switched['OId']]
            row['id_switches'] = len(switches)
            active_block.append(([g.person_id for g in gt], [p.track_id for p in predictions], overlaps))
        outcome, target_is_visible = 'not_evaluable', False
        if target_enabled and start is not None and frame.frame_index == start:
            outcome = 'initialization'
        elif target_enabled and start is not None and frame.frame_index > start:
            target_is_visible = any(g.person_id == target_id for g in gt)
            if target_is_visible or persistent:
                selected = next((j for j, p in enumerate(predictions) if p.track_id == frame.selected_track_id), None)
                match = next((i for i, j in matches if j == selected), None) if selected is not None else None
                if match is not None:
                    row['matched_person_id'] = gt[match].person_id
                    outcome = 'correct' if gt[match].person_id == target_id else 'wrong_person'
                elif selected is not None:
                    outcome = 'false_positive' if complete else 'unmatched_unjudgeable'
                else:
                    outcome = 'unassigned' if target_is_visible else 'correct_absence'
                row['target_visible_gt'] = int(target_is_visible)
        row['target_outcome'] = outcome
        target_metrics.update(frame, outcome, target_is_visible)
        frame_metrics.append(row)
        visual[frame.frame_index] = {'gt': list(zip(gt, boxes)), 'row': row, 'switches': switches,
                                     'missed_ids': [gt[i].person_id for i in misses]}
    flush_block()
    if not totals['evaluated_frames']:
        raise ValueError(f'No evaluated frames for {video.video_id}; check annotation alignment and FPS sampling')
    supported = identity_enabled and any(len(block) >= 2 for block in blocks)
    mot = {'idsw': None, 'idf1': None, 'fragmentations': None, 'mota': None, 'motp_distance': None,
           'idtp': None, 'idfp': None, 'idfn': None, 'hota': None}
    if supported:
        names = ['num_switches', 'num_fragmentations', 'idf1', 'mota', 'motp', 'idtp', 'idfp', 'idfn']
        result = mm.metrics.create().compute_many(accumulators, names=[str(i) for i in range(len(accumulators))],
                                                 metrics=names, generate_overall=True).loc['OVERALL']
        rename = {'num_switches':'idsw', 'num_fragmentations':'fragmentations', 'motp':'motp_distance'}
        mot.update({rename.get(key,key): clean_scalar(value) for key,value in result.items()})
    mot.update(library='motmetrics 1.4.0', identity_scope=sequence.metadata['identity_scope'],
               supported=supported, contiguous_evaluation_blocks=len(blocks),
               unavailable_reason=None if supported else 'Requires exhaustive scored frames, complete appropriate IDs and at least two contiguous sampled frames',
               matching='CLEAR MOT continuity matching with IoU distance; IDF1 uses global bipartite assignment per contiguous evaluation block',
               hota_unavailable_reason='Supply --hota-python for isolated TrackEval; never install its NumPy 2 requirement into the tracker environment')
    detection = {**totals, 'precision': ratio(totals['matches'], totals['predicted_faces']) if complete else None,
                 'recall': ratio(totals['matches'], totals['gt_faces']),
                 'mean_matched_iou': ratio(totals['iou_sum'], totals['matches']),
                 'false_positives': totals['unmatched_predictions'] if complete else None,
                 'precision_unavailable_reason': None if complete else 'Annotation coverage is partial; unmatched predictions may be unlabeled faces',
                 'prediction_filter': 'observed AND confirmed; predictions without a current detection are not scored',
                 'matching': f'frame-local one-to-one Hungarian IoU >= {config.iou_threshold}',
                 'scope': 'Only explicitly annotated/known-empty source frames; unknown frames excluded, never assumed empty'}
    return {'detection': detection, 'tracking': mot, 'target': target_metrics.finish(),
            'initialization': initialization, 'experiment': config.experiment}, frame_metrics, visual, (
                [_hota_data(block) for block in blocks] if supported else [])
