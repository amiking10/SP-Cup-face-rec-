"""MediaPipe legacy face detection adapter (pinned to 0.10.21)."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]  # pixel left, top, right, bottom
    confidence: float


def clip_box(box, width, height):
    if width <= 0 or height <= 0 or len(box) != 4:
        return None
    if not all(math.isfinite(v) for v in box):
        return None
    left, top, right, bottom = box
    left, right = max(0, left), min(width, right)
    top, bottom = max(0, top), min(height, bottom)
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def relative_to_pixels(x, y, w, h, width, height):
    if w <= 0 or h <= 0:
        return None
    return clip_box((x * width, y * height, (x + w) * width,
                     (y + h) * height), width, height)


class FaceDetector:
    def __init__(self, confidence=0.5, model_selection=0):
        import mediapipe as mp
        try:
            self.detector = mp.solutions.face_detection.FaceDetection(
                model_selection=model_selection, min_detection_confidence=confidence)
        except Exception as exc:
            raise RuntimeError("MediaPipe model loading failed. Install pinned dependencies; "
                               "this baseline requires mediapipe==0.10.21.") from exc

    def detect(self, frame):
        import cv2
        height, width = frame.shape[:2]
        result = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = []
        for item in result.detections or []:
            b = item.location_data.relative_bounding_box
            box = relative_to_pixels(b.xmin, b.ymin, b.width, b.height, width, height)
            if box is not None:
                detections.append(Detection(box, float(item.score[0])))
        return detections

    def close(self):
        self.detector.close()
