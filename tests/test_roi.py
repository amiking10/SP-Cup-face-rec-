"""Capture regions must be safe NumPy slices and never sample stale target boxes."""
import numpy as np
import pytest

from c3_tracker.roi import selected_capture_roi
from c3_tracker.target_manager import TargetManager
from c3_tracker.tracking import Observation


def test_capture_at_image_edge_clips_without_negative_slice_wraparound():
    image = np.arange(40 * 20 * 3).reshape(40, 20, 3)
    observation = Observation('1', (0, 3, 20, 33), True, True)
    target = TargetManager()
    assert target.select(5, 5, [observation])
    bounds = selected_capture_roi([observation], target, 20, 40)
    left, top, right, bottom = bounds
    crop = image[top:bottom, left:right]
    assert np.array_equal(crop, image[:33])
    assert observation.bbox == (0, 3, 20, 33)


@pytest.mark.parametrize('current', [[], [Observation('1', (5, 10, 25, 30), False, True)],
                                   [Observation('1', (5, 10, 25, 30), True, False)],
                                   [Observation('2', (5, 10, 25, 30), True, True)]])
def test_stale_selection_state_cannot_supply_a_predicted_unconfirmed_or_other_face(current):
    target = TargetManager()
    assert target.select(10, 15, [Observation('1', (5, 10, 25, 30), True, True)])
    assert selected_capture_roi(current, target, 50, 50) is None


def test_lost_target_never_supplies_roi_even_if_observation_list_is_stale():
    observation = Observation('1', (5, 10, 25, 30), True, True)
    target = TargetManager()
    assert target.select(10, 15, [observation])
    target.update([])
    assert selected_capture_roi([observation], target, 50, 50) is None
