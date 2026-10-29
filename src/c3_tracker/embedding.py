"""Five-landmark alignment and local SFace embeddings for DeepSORT."""
import cv2
import numpy as np
from .models import model_path


class FaceEmbedder:
    def __init__(self, directory=None):
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path('face_detection_yunet_2023mar.onnx', directory)), '', (320, 320), .8, .3)
        self.recognizer = cv2.FaceRecognizerSF.create(
            str(model_path('face_recognition_sface_2021dec.onnx', directory)), '')

    def encode(self, detections, frame):
        if not detections:
            return [], []
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return [], []
        # Match landmark detections one-to-one to the MediaPipe face boxes.
        # An unalignable face is missing, never replaced by a generic embedding.
        pairs = []
        for i, detection in enumerate(detections):
            left, top, right, bottom = detection.bbox
            area = (right - left) * (bottom - top)
            for j, face in enumerate(faces):
                x, y, w, h = face[:4]
                overlap = max(0, min(right, x + w) - max(left, x)) * max(0, min(bottom, y + h) - max(top, y))
                iou = overlap / max(area + w * h - overlap, 1)
                if iou >= .3:
                    pairs.append((float(iou), i, j))
        used_detections, used_faces, encoded = set(), set(), []
        for _, i, j in sorted(pairs, reverse=True):
            if i in used_detections or j in used_faces:
                continue
            aligned = self.recognizer.alignCrop(frame, faces[j])
            feature = self.recognizer.feature(aligned).reshape(-1).copy()
            norm = np.linalg.norm(feature)
            if not np.isfinite(feature).all() or norm <= 0:
                continue
            used_detections.add(i)
            used_faces.add(j)
            encoded.append((i, feature / norm))
        encoded.sort(key=lambda pair: pair[0])
        return [detections[i] for i, _ in encoded], [feature for _, feature in encoded]
