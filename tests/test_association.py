"""Regression sequences for motion after missed face detections."""
import numpy as np
import pytest
from deep_sort_realtime.deep_sort.detection import Detection
from deep_sort_realtime.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort_realtime.deep_sort.tracker import Tracker
from c3_tracker.association import FaceAssociationTracker


def tracker(cls=FaceAssociationTracker, max_age=30):
    return cls(NearestNeighborDistanceMetric('cosine', .2, 100), max_age=max_age, n_init=3,
               gating_only_position=True)


def face(x, feature=(1, 0, 0), width=80):
    return Detection([x, 100, width, 80], .95, np.asarray(feature, dtype=np.float32))


def step(model, detections):
    model.predict()
    model.update(detections)
    return [t.track_id for t in model.tracks if t.time_since_update == 0]


def confirm(model, detections):
    for _ in range(4):
        ids = step(model, detections)
    return ids


def test_moving_face_keeps_id_after_detector_gap_that_fragmented_baseline():
    baseline, fixed = tracker(Tracker), tracker()
    for model in (baseline, fixed):
        original, = confirm(model, [face(100)])
        step(model, [])
        step(model, [])
        returned, = step(model, [face(145)])
        if model is fixed:
            assert returned == original
            assert len(model.tracks) == 1
        else:
            assert returned != original  # reproduces the old fragmentation


def test_two_faces_recover_distinct_ids_even_when_detection_order_reverses():
    model = tracker()
    first, second = confirm(model, [face(100), face(300, (0, 1, 0))])
    step(model, [])
    assert step(model, [face(255, (0, 1, 0)), face(145)]) == [first, second]
    assert len(model.tracks) == 2


def test_nearby_different_appearance_cannot_recover_missing_id():
    model = tracker()
    original, = confirm(model, [face(100)])
    step(model, [])
    returned, = step(model, [face(145, (0, 1, 0))])
    assert returned != original


def test_similar_appearance_far_away_cannot_recover_missing_id():
    model = tracker()
    original, = confirm(model, [face(100)])
    step(model, [])
    returned, = step(model, [face(450)])
    assert returned != original


def test_ambiguous_two_nearby_faces_do_not_force_recovery():
    model = tracker()
    original, = confirm(model, [face(100)])
    step(model, [])
    returned = step(model, [face(145), face(150)])
    assert original not in returned


def test_expired_track_is_not_revived():
    model = tracker(max_age=2)
    original, = confirm(model, [face(100)])
    for _ in range(3):
        step(model, [])
    assert not model.tracks
    returned, = step(model, [face(100)])
    assert returned != original


@pytest.mark.parametrize('fps', [10, 30, 60])
def test_one_second_absence_keeps_id_while_other_face_stays_visible(fps):
    from c3_tracker.target_manager import TargetManager
    from c3_tracker.tracking import observations_from_tracks
    model = FaceAssociationTracker(NearestNeighborDistanceMetric('cosine', .2, 100),
                                   n_init=3, memory_seconds=3, gating_only_position=True)
    for index in range(4):
        model.timestamp = index / fps
        first, second = step(model, [face(100), face(400, (0, 1, 0))])
    selected = TargetManager()
    selected.select(120, 120, observations_from_tracks(model.tracks, 640, 480))
    assert selected.target_id == first
    for index in range(fps):
        model.timestamp = (index + 4) / fps
        assert step(model, [face(400, (0, 1, 0))]) == [second]
        assert selected.update(observations_from_tracks(model.tracks, 640, 480)) == 'LOST'
    model.timestamp = (fps + 4) / fps
    assert step(model, [face(210), face(400, (0, 1, 0))]) == [first, second]
    assert len(model.tracks) == 2
    assert selected.update(observations_from_tracks(model.tracks, 640, 480)) == 'REACQUIRED'
    assert selected.target_id == first


def test_timed_memory_expires_by_seconds_even_with_few_processed_frames():
    model = FaceAssociationTracker(NearestNeighborDistanceMetric('cosine', .2, 100),
                                   n_init=3, memory_seconds=3, gating_only_position=True)
    for timestamp in [0, .1, .2, .3]:
        model.timestamp = timestamp
        original, = step(model, [face(100)])
    model.timestamp = 3.31
    returned, = step(model, [face(100)])
    assert returned != original
    assert original not in model.metric.samples


def test_upstream_does_not_fill_appearance_gallery_with_missed_frames():
    model = tracker()
    original, = confirm(model, [face(100)])
    samples = len(model.metric.samples[original])
    for _ in range(20):
        step(model, [])
    assert len(model.metric.samples[original]) == samples


def test_overlap_alone_cannot_replace_confirmed_face_with_different_person():
    baseline, fixed = tracker(Tracker), tracker()
    for model in (baseline, fixed):
        original, = confirm(model, [face(100)])
        returned, = step(model, [face(100, (0, 1, 0))])
        assert (returned == original) == (model is baseline)
        if model is fixed:
            # Rejected features must not contaminate the original gallery.
            assert all(np.array_equal(feature, [1, 0, 0]) for feature in model.metric.samples[original])


def test_selected_id_stays_lost_when_second_person_takes_its_position():
    from c3_tracker.target_manager import TargetManager
    from c3_tracker.tracking import observations_from_tracks
    model = tracker()
    first, second = confirm(model, [face(100), face(300, (0, 1, 0))])
    model.protect_target(first)
    target = TargetManager()
    assert target.select(120, 120, observations_from_tracks(model.tracks, 640, 480))
    # B moves into A's exact old box in the very first frame A disappears.
    # This exercises the upstream IoU path, before there is an explicit loss.
    for _ in range(12):
        step(model, [face(100, (0, 1, 0))])
        observations = observations_from_tracks(model.tracks, 640, 480)
        assert target.update(observations) == 'LOST'
        assert not any(o.observed and o.track_id == first for o in observations)
        assert target.target_id == first


def test_selected_return_needs_three_compatible_frames_and_keeps_reference():
    model = tracker()
    first, second = confirm(model, [face(100), face(300, (0, 1, 0))])
    model.protect_target(first)
    frozen = model._target_reference.copy()
    step(model, [face(300, (0, 1, 0))])
    for index in range(2):
        assert step(model, [face(110), face(300, (0, 1, 0))]) == [second]
        assert len(model.tracks) == 2  # no temporary ID for the candidate
        diagnostic = model.target_diagnostics()
        assert diagnostic['state'] == 'verifying_return'
        assert diagnostic['return_confirmation_count'] == index + 1
    assert step(model, [face(110), face(300, (0, 1, 0))]) == [first, second]
    assert model.target_diagnostics()['state'] == 'verified_return'
    assert np.array_equal(model._target_reference, frozen)


def test_return_confirmation_resets_after_a_missing_frame():
    model = tracker()
    original, = confirm(model, [face(100)])
    model.protect_target(original)
    step(model, [])
    assert step(model, [face(100)]) == []
    step(model, [])
    assert step(model, [face(100)]) == []
    assert step(model, [face(100)]) == []
    assert step(model, [face(100)]) == [original]


def test_ambiguous_return_cannot_take_selected_id_even_if_only_one_face_remains():
    model = tracker()
    first, second = confirm(model, [face(100, (1, .2, 0)), face(300, (1, -.2, 0))])
    model.protect_target(first)
    for _ in range(2):
        step(model, [])
    for _ in range(5):
        assert first not in step(model, [face(140, (1, 0, 0))])


def test_selection_uses_clicked_face_even_if_old_gallery_contains_another_face():
    model = tracker()
    original, = confirm(model, [face(100)])
    model.metric.partial_fit(np.asarray([[0, 1, 0]]), np.asarray([original]), [original])
    model.protect_target(original)
    for _ in range(4):
        assert original not in step(model, [face(100, (0, 1, 0))])


def test_rejected_selected_match_does_not_starve_a_tentative_track():
    model = FaceAssociationTracker(NearestNeighborDistanceMetric('cosine', .45, 100),
                                   n_init=3, gating_only_position=True)
    original, = confirm(model, [face(100)])
    model.protect_target(original)
    pose = (np.cos(.95), np.sin(.95), 0)
    observed = [step(model, [face(100, pose)])[0] for _ in range(8)]
    assert original not in observed  # appearance still fails the selected reference
    assert len(set(observed)) == 1  # same tentative track gets a chance to confirm
    assert any(t.track_id == observed[0] and t.is_confirmed() for t in model.tracks)


def test_duplicate_track_cannot_block_verified_selected_return():
    model = FaceAssociationTracker(NearestNeighborDistanceMetric('cosine', .45, 100),
                                   n_init=3, gating_only_position=True)
    original, = confirm(model, [face(100)])
    model.protect_target(original)
    pose = (np.cos(.95), np.sin(.95), 0)
    for _ in range(4):
        step(model, [face(100, pose)])
    for _ in range(2):
        assert original not in step(model, [face(100)])
    assert step(model, [face(100)]) == [original]


def test_reselection_does_not_treat_stale_same_face_gallery_as_another_person():
    model = FaceAssociationTracker(NearestNeighborDistanceMetric('cosine', .45, 100),
                                   n_init=3, gating_only_position=True)
    original, = confirm(model, [face(100)])
    model.protect_target(original)
    pose = (np.cos(.95), np.sin(.95), 0)
    for _ in range(4):
        returned, = step(model, [face(100, pose)])
    model.protect_target(returned)
    intermediate_pose = (np.cos(.5), np.sin(.5), 0)
    assert step(model, [face(100, intermediate_pose)]) == [returned]


def test_frozen_consistent_views_keep_a_head_turn_without_learning_a_stranger():
    model = FaceAssociationTracker(NearestNeighborDistanceMetric('cosine', .45, 100),
                                   n_init=3, gating_only_position=True)
    original, = confirm(model, [face(100, (np.cos(.7), np.sin(.7), 0))])
    step(model, [face(100)])
    model.protect_target(original)
    frozen = model._target_reference.copy()
    assert len(frozen) == 2
    assert step(model, [face(100, (np.cos(1), np.sin(1), 0))]) == [original]
    assert np.array_equal(model._target_reference, frozen)
    # A candidate close to one bank member still cannot bypass the clicked
    # anchor's absolute limit, and must not update any frozen reference.
    assert original not in step(model, [face(100, (0, 1, 0))])
    assert np.array_equal(model._target_reference, frozen)


def test_target_diagnostics_reports_live_rejection_without_saving_embeddings():
    import json
    model = tracker()
    first, second = confirm(model, [face(100), face(300, (0, 1, 0))])
    model.protect_target(first)
    step(model, [face(100, (0, 1, 0))])
    diagnostic = model.target_diagnostics()
    candidate, = diagnostic['candidates']
    assert diagnostic['state'] == 'no_compatible_detection'
    assert candidate['reason'] == 'appearance_gallery'
    assert candidate['gallery_distance'] > candidate['gallery_limit']
    assert 'feature' not in candidate and 'embedding' not in candidate
    json.dumps(diagnostic, allow_nan=False)
