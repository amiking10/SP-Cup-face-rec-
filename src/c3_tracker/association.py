"""DeepSORT with timed track memory and guarded recovery after detection gaps."""
import time
import numpy as np
from deep_sort_realtime.deep_sort.tracker import Tracker
from deep_sort_realtime.deep_sort.track import Track
from deep_sort_realtime.deep_sort import linear_assignment, iou_matching


class TimedTrack(Track):
    """Confirmed-track expiry is managed by the timestamp-aware tracker."""

    def mark_missed(self):
        if not self.is_confirmed():
            super().mark_missed()


class FaceAssociationTracker(Tracker):
    """Keep raw IDs alive through nearby motion without merging or renaming IDs.

    DeepSORT's IoU fallback considers only the immediately preceding frame.
    After a brief miss, a returning face can fail the Kalman gate and create a
    second track. Recover only unmatched, confirmed tracks with both compatible
    appearance and nearby geometry. Ambiguous alternatives stay unmatched.
    """

    def __init__(self, *args, recovery_frames=None, memory_seconds=None, **kwargs):
        if memory_seconds is not None:
            kwargs['override_track_class'] = TimedTrack
        super().__init__(*args, **kwargs)
        self.recovery_frames = recovery_frames
        self.memory_seconds = memory_seconds
        self.timestamp = None
        self._last_boxes = {}
        self._last_seen = {}
        self._now = 0.0
        self.protected_id = None
        self._target_reference = None
        self._competitor_references = {}
        self._return_candidate = None
        self._target_candidate_diagnostics = {}
        self._target_state = 'not_selected'
        self._diagnostic_last_seen = None

    def protect_target(self, track_id):
        """Freeze the selected appearance before any future association updates."""
        self.protected_id = track_id
        self._target_reference = None
        self._competitor_references = {}
        self._return_candidate = None
        if track_id is not None:
            track = next((t for t in self.tracks if t.track_id == track_id
                          and t.is_confirmed() and t.time_since_update == 0), None)
            if track is None:
                raise ValueError('Select a confirmed track with appearance samples')
            # Anchor to the clicked face, then retain a few distinct recent
            # views that agree closely with it. Freeze the entire bank; never
            # import arbitrary old identities or learn from later rejections.
            reference = [track.get_feature().copy()]
            for feature in reversed(self.metric.samples[track_id][-30:]):
                if (self._distance(reference[:1], feature) <= .25
                        and self._distance(reference, feature) >= .02):
                    reference.append(np.asarray(feature).copy())
                if len(reference) == 8:
                    break
            self._target_reference = np.asarray(reference)
            self._remember_competitors(track)

    def _remember_competitors(self, selected):
        # Different raw IDs are not evidence of different people. In particular,
        # stale IDs from this same face must not veto manual reselection.
        box = selected.to_ltwh(orig=True, orig_strict=True)
        for other in self.tracks:
            if (other.track_id == selected.track_id or not other.is_confirmed()
                    or other.time_since_update != 0):
                continue
            other_box = other.to_ltwh(orig=True, orig_strict=True)
            if iou_matching.iou(box, np.asarray([other_box]))[0] < .2:
                self._competitor_references.setdefault(other.track_id, np.asarray([other.get_feature()]).copy())

    @staticmethod
    def _distance(references, feature):
        references = np.asarray(references)
        feature = np.asarray(feature)
        norms = np.linalg.norm(references, axis=1) * np.linalg.norm(feature)
        if not np.isfinite(feature).all() or np.any(norms <= 0):
            return np.inf
        return float(np.min(1 - references @ feature / norms))

    def _appearance_allowed(self, track_index, detection_index, detections):
        track = self.tracks[track_index]
        if not track.is_confirmed():
            return True
        feature = detections[detection_index].feature
        distance = self._distance(self.metric.samples[track.track_id], feature)
        if track.track_id != self.protected_id:
            return np.isfinite(distance) and distance <= self.metric.matching_threshold
        # A frozen reference prevents a wrong association from gradually
        # rewriting the selected person's appearance in the adaptive gallery.
        reference_distance = self._distance(self._target_reference, feature)
        anchor_limit = min(.6, self.metric.matching_threshold + .15)
        anchor_distance = self._distance(self._target_reference[:1], feature)
        limit = min(self.metric.matching_threshold, .35 if track.time_since_update > 1 else .4)
        margin = min((self._distance(samples, feature) - reference_distance
                      for samples in self._competitor_references.values()), default=np.inf)
        reason = 'appearance_passed'
        if not np.isfinite(distance) or distance > self.metric.matching_threshold:
            reason = 'appearance_gallery'
        elif not np.isfinite(anchor_distance) or anchor_distance > anchor_limit:
            reason = 'selection_anchor'
        elif not np.isfinite(reference_distance) or reference_distance > limit:
            reason = 'selection_reference'
        elif margin < .08 or np.isnan(margin):
            reason = 'competing_face'
        self._target_candidate_diagnostics[detection_index] = {
            'detection_index': detection_index,
            'bbox_ltwh': [float(v) for v in detections[detection_index].get_ltwh()],
            'gallery_distance': self._finite(distance), 'gallery_limit': self.metric.matching_threshold,
            'reference_distance': self._finite(reference_distance), 'reference_limit': limit,
            'anchor_distance': self._finite(anchor_distance), 'anchor_limit': anchor_limit,
            'competitor_margin': self._finite(margin), 'competitor_margin_limit': .08,
            'reason': reason,
        }
        return reason == 'appearance_passed'

    @staticmethod
    def _finite(value):
        return float(value) if np.isfinite(value) else None

    def target_diagnostics(self):
        last_seen = self._last_seen.get(self.protected_id, self._diagnostic_last_seen)
        return {
            'selected_id': self.protected_id, 'state': self._target_state,
            'reference_views': 0 if self._target_reference is None else len(self._target_reference),
            'seconds_since_observation': None if last_seen is None else self._now - last_seen,
            'return_confirmation_count': 0 if self._return_candidate is None else self._return_candidate[2],
            'candidates': list(self._target_candidate_diagnostics.values()),
        }

    def _confirm_target_return(self, matches, unmatched_tracks, unmatched_detections, detections):
        candidate = next(((i, j) for i, j in matches
                          if self.tracks[i].track_id == self.protected_id), None)
        if candidate is None or self.tracks[candidate[0]].time_since_update <= 1:
            self._return_candidate = None
            if candidate is not None:
                self._target_state = 'tracked'
            return matches, unmatched_tracks, unmatched_detections
        i, j = candidate
        box = detections[j].get_ltwh()
        feature = detections[j].feature
        previous = self._return_candidate
        count = 1
        if (previous is not None and self._geometry(previous[0], box) <= .6
                and self._distance([previous[1]], feature) <= .15):
            count = previous[2] + 1
        if count >= 3:
            self._return_candidate = None
            self._target_state = 'verified_return'
            return matches, unmatched_tracks, unmatched_detections
        self._return_candidate = (box.copy(), feature.copy(), count)
        self._target_state = 'verifying_return'
        # Reserve this detection while verifying it, preventing a tentative ID
        # from consuming the returning face. It is not logged as an observation.
        return ([pair for pair in matches if pair != candidate],
                sorted(set(unmatched_tracks) | {i}),
                [index for index in unmatched_detections if index != j])

    def update(self, detections, today=None):
        self._now = time.monotonic() if self.timestamp is None else self.timestamp
        self._target_candidate_diagnostics = {}
        self._target_state = 'not_selected' if self.protected_id is None else 'no_compatible_detection'
        self._diagnostic_last_seen = self._last_seen.get(self.protected_id)
        expired = []
        if self.memory_seconds is not None:
            expired = [t.track_id for t in self.tracks
                       if self._now - self._last_seen[t.track_id] > self.memory_seconds]
            self.tracks = [t for t in self.tracks if t.track_id not in expired]
            # The upstream tracker still uses a frame counter internally. Keep
            # surviving tracks alive until the actual time limit, at any FPS,
            # and bound the matching cascade to the ages actually present.
            self.max_age = max((t.time_since_update for t in self.tracks), default=0) + 1
        super().update(detections, today=today)
        self.del_tracks_ids.extend(expired)
        self._last_boxes = {
            t.track_id: (t.to_ltwh(orig=True, orig_strict=True)
                         if t.time_since_update == 0 else self._last_boxes.get(t.track_id))
            for t in self.tracks
        }
        self._last_seen = {
            t.track_id: self._now if t.time_since_update == 0 else self._last_seen[t.track_id]
            for t in self.tracks
        }
        selected = next((t for t in self.tracks if t.track_id == self.protected_id
                         and t.time_since_update == 0), None)
        if selected is not None:
            self._remember_competitors(selected)

    def _motion_cost(self, track, box):
        reference = self._last_boxes.get(track.track_id)
        if reference is None:
            return np.inf
        motion = min(self._geometry(candidate, box) for candidate in (track.to_ltwh(), reference))
        gap = (self._now - self._last_seen[track.track_id] if self.memory_seconds is not None
               else .08 * max(0, track.time_since_update - 1))
        return motion / min(2.5, .9 + gap)

    def _primary_matches(self, detections):
        """Apply gates before assignment, using upstream DeepSORT matchers."""
        matches = []
        remaining = list(range(len(detections)))
        selected = next((i for i, track in enumerate(self.tracks)
                         if track.track_id == self.protected_id and track.is_confirmed()), None)
        if self.protected_id is not None and selected is None:
            self._target_state = 'track_expired'
        if selected is not None:
            # A clean selected-face return gets first refusal, even when a
            # duplicate ID was created during an earlier rejected head pose.
            candidates = []
            for j, detection in enumerate(detections):
                appearance_ok = self._appearance_allowed(selected, j, detections)
                motion = self._motion_cost(self.tracks[selected], detection.get_ltwh())
                detail = self._target_candidate_diagnostics[j]
                detail['normalized_motion_cost'] = self._finite(motion)
                detail['motion_limit'] = 1.0
                if appearance_ok and motion > 1:
                    detail['reason'] = 'geometry'
                if appearance_ok and motion <= 1:
                    candidates.append((self._distance(self._target_reference, detection.feature), j))
            candidates.sort()
            if candidates and (len(candidates) == 1 or candidates[1][0] - candidates[0][0] >= .08):
                j = candidates[0][1]
                matches.append((selected, j))
                remaining.remove(j)
            elif candidates:
                self._target_state = 'ambiguous_detections'

        confirmed = [i for i, track in enumerate(self.tracks)
                     if track.is_confirmed() and i != selected]
        tentative = [i for i, track in enumerate(self.tracks) if not track.is_confirmed()]

        def appearance_cost(tracks, dets, track_indices, detection_indices):
            costs = self.metric.distance(np.asarray([dets[j].feature for j in detection_indices]),
                                         [tracks[i].track_id for i in track_indices])
            costs = linear_assignment.gate_cost_matrix(self.kf, costs, tracks, dets,
                                                       track_indices, detection_indices,
                                                       only_position=self.gating_only_position)
            for row, i in enumerate(track_indices):
                for col, j in enumerate(detection_indices):
                    if not self._appearance_allowed(i, j, dets):
                        costs[row, col] = linear_assignment.INFTY_COST
            return costs

        matched, unmatched, remaining = linear_assignment.matching_cascade(
            appearance_cost, self.metric.matching_threshold, self.max_age,
            self.tracks, detections, confirmed, remaining)
        matches.extend(matched)
        recent = [i for i in unmatched if self.tracks[i].time_since_update == 1]
        older = [i for i in unmatched if i not in recent]

        def overlap_cost(tracks, dets, track_indices, detection_indices):
            costs = iou_matching.iou_cost(tracks, dets, track_indices, detection_indices)
            for row, i in enumerate(track_indices):
                for col, j in enumerate(detection_indices):
                    if not self._appearance_allowed(i, j, dets):
                        costs[row, col] = linear_assignment.INFTY_COST
            return costs

        matched, unmatched, remaining = linear_assignment.min_cost_matching(
            overlap_cost, self.max_iou_distance, self.tracks, detections, tentative + recent, remaining)
        matches.extend(matched)
        unmatched = older + list(unmatched)
        if selected is not None and not any(i == selected for i, _ in matches):
            unmatched.append(selected)
        return matches, unmatched, list(remaining)

    def _match(self, detections):
        matches, unmatched_tracks, unmatched_detections = self._primary_matches(detections)
        candidates = [i for i in unmatched_tracks
                      if self.tracks[i].is_confirmed()
                      and self.tracks[i].track_id != self.protected_id
                      and self.tracks[i].time_since_update <= (
                          self.max_age if self.recovery_frames is None else min(self.recovery_frames, self.max_age))
                      and self._last_boxes.get(self.tracks[i].track_id) is not None]
        if not candidates or not unmatched_detections:
            return self._confirm_target_return(matches, unmatched_tracks, unmatched_detections, detections)

        appearance = self.metric.distance(
            np.asarray([detections[j].feature for j in unmatched_detections]),
            [self.tracks[i].track_id for i in candidates])
        costs = np.full(appearance.shape, np.inf)
        for row, i in enumerate(candidates):
            track = self.tracks[i]
            for col, j in enumerate(unmatched_detections):
                distance = appearance[row, col]
                if not self._appearance_allowed(i, j, detections):
                    continue
                motion = self._motion_cost(track, detections[j].get_ltwh())
                if motion <= 1:
                    costs[row, col] = (0.75 * max(0, distance) / self.metric.matching_threshold
                                       + 0.25 * motion)

        recovered = []
        for row, i in enumerate(candidates):
            col = int(np.argmin(costs[row]))
            cost = costs[row, col]
            if not np.isfinite(cost) or int(np.argmin(costs[:, col])) != row:
                continue
            # Require a margin over every competing detection AND track.
            alternatives = np.concatenate((np.delete(costs[row], col), np.delete(costs[:, col], row)))
            if alternatives.size and np.min(alternatives) - cost < 0.1:
                continue
            recovered.append((i, unmatched_detections[col]))
        matched_tracks = {i for i, _ in recovered}
        matched_detections = {j for _, j in recovered}
        return self._confirm_target_return(matches + recovered,
                [i for i in unmatched_tracks if i not in matched_tracks],
                [j for j in unmatched_detections if j not in matched_detections], detections)

    @staticmethod
    def _geometry(reference, box):
        if min(*reference[2:], *box[2:]) <= 0:
            return np.inf
        ratios = box[2:] / reference[2:]
        if np.any(ratios < 0.5) or np.any(ratios > 2.0):
            return np.inf
        offset = (box[:2] + box[2:] / 2) - (reference[:2] + reference[2:] / 2)
        return float(np.linalg.norm(offset / np.maximum(reference[2:], box[2:])))
