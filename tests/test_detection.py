import math
import pytest
from c3_tracker.detection import clip_box, relative_to_pixels


def test_relative_conversion():
    assert relative_to_pixels(.1, .2, .3, .4, 640, 480) == pytest.approx((64, 96, 256, 288))


def test_clipping():
    assert relative_to_pixels(-.1, -.2, .5, 1.5, 100, 100) == (0, 0, 40, 100)
    assert clip_box((20, 30, 150, 140), 100, 80) == (20, 30, 100, 80)


@pytest.mark.parametrize('box', [(1, 1, 1, 4), (4, 4, 2, 5), (-5, 0, -1, 3),
                                 (0, 0, math.nan, 4), (0, 0, math.inf, 4), (0, 0, 1)])
def test_invalid_box(box):
    assert clip_box(box, 100, 100) is None


@pytest.mark.parametrize('w,h', [(0, 1), (-1, 1), (1, -1)])
def test_invalid_relative(w, h):
    assert relative_to_pixels(0, 0, w, h, 100, 100) is None
