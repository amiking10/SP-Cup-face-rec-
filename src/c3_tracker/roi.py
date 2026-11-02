"""Derived capture geometry; detector/association boxes stay in their native form."""
import math

from .detection import clip_box

DEFAULT_TOP_EXTENSION = 0.2


def validate_roi_margins(top=0.0, bottom=0.0, left=0.0, right=0.0):
    margins = (top, bottom, left, right)
    if any(not math.isfinite(value) or not -1 <= value <= 1 for value in margins):
        raise ValueError('ROI margins must be finite fractions in [-1, 1]')
    if top + bottom <= -1 or left + right <= -1:
        raise ValueError('ROI margins must leave a positive width and height')


def selected_capture_roi(observations, target, width, height, *, top=0.0, bottom=0.0, left=0.0, right=0.0):
    """Return integer xyxy bounds for frame[top:bottom, left:right], or None.

    The default box extends the observed face top upward by 20% of face height.
    Margins are adjustments to that new default: positive expands outward, negative
    trims inward. Vertical margins use its height, horizontal margins its width,
    before image clipping. Predicted/lost/unconfirmed selections supply no samples.
    """
    validate_roi_margins(top, bottom, left, right)
    if not target.visible:
        return None
    selected = next((o for o in observations if o.track_id == target.target_id
                     and o.observed and o.confirmed), None)
    if selected is None:
        return None
    x1, y1, x2, y2 = selected.bbox
    if x2 <= x1 or y2 <= y1:
        return None
    y1 -= DEFAULT_TOP_EXTENSION * (y2 - y1)
    base_width, base_height = x2 - x1, y2 - y1
    bounds = clip_box((x1 - left * base_width, y1 - top * base_height,
                       x2 + right * base_width, y2 + bottom * base_height), width, height)
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    return math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)
