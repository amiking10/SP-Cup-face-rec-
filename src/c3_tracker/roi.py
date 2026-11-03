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


def capture_frame(frame,roi,size):
    """Fit current capture pixels to the fixed canvas; missing crops are placeholders."""
    import cv2
    import numpy as np
    out_w,out_h=size
    if min(size)<=0: raise ValueError('Capture canvas dimensions must be positive')
    centered=np.zeros((out_h,out_w,3),dtype=frame.dtype)
    if roi is not None:
        left,top,right,bottom=roi
        if not (0<=left<right<=frame.shape[1] and 0<=top<bottom<=frame.shape[0]):
            raise ValueError('Invalid capture ROI')
        crop=frame[top:bottom,left:right]
        scale=min(out_w/crop.shape[1],out_h/crop.shape[0])
        fit_w=min(out_w,max(1,round(crop.shape[1]*scale)))
        fit_h=min(out_h,max(1,round(crop.shape[0]*scale)))
        fitted=cv2.resize(crop,(fit_w,fit_h),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
        x,y=(out_w-fit_w)//2,(out_h-fit_h)//2
        centered[y:y+fit_h,x:x+fit_w]=fitted
    else:
        text='FACE LOST'
        scale=min(.8,max(.05,(out_w-6)/cv2.getTextSize(text,0,1,1)[0][0]))
        cv2.putText(centered,text,(3,max(10,out_h//2)),0,scale,(255,255,255),1)
    return centered
