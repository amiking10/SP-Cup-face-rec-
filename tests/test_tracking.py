from c3_tracker.tracking import observations_from_tracks


class MockTrack:
    track_id = 17
    time_since_update = 1

    def to_ltrb(self, orig=False, orig_strict=False):
        return None if orig and orig_strict else (-5, 0, 20, 30)

    def is_confirmed(self):
        return True

    def get_det_conf(self):
        raise AssertionError('predicted track has no detection confidence')


def test_prediction_does_not_count_as_observation():
    o, = observations_from_tracks([MockTrack()], 100, 100)
    assert o.bbox == (0, 0, 20, 30)
    assert o.track_id == '17'
    assert not o.observed
    assert o.detection_confidence is None


def test_current_original_box_used():
    track = MockTrack()
    track.time_since_update = 0
    track.to_ltrb = lambda **kwargs: (10, 20, 30, 40) if kwargs.get('orig') else (0, 0, 40, 50)
    track.get_det_conf = lambda: .8
    o, = observations_from_tracks([track], 100, 100)
    assert o.observed and o.confirmed
    assert o.bbox == (10, 20, 30, 40)
    assert o.detection_confidence == .8
