"""Replaceable DeepSORT adapter. Only current detections are observed ROIs."""
from dataclasses import dataclass
from .detection import clip_box


@dataclass(frozen=True)
class Observation:
    track_id: str
    bbox: tuple[float, float, float, float]
    observed: bool
    confirmed: bool
    detection_confidence: float | None = None


def observations_from_tracks(tracks, width, height):
    observations = []
    for track in tracks:
        # orig_strict prevents silently falling back to a Kalman prediction.
        original = track.to_ltrb(orig=True, orig_strict=True)
        observed = track.time_since_update == 0 and original is not None
        box = original if observed else track.to_ltrb()
        box = clip_box(box, width, height)
        if box is not None:
            confidence = track.get_det_conf() if observed else None
            observations.append(Observation(str(track.track_id), box, observed,
                                            track.is_confirmed(), confidence))
    return observations


class FaceTracker:
    def __init__(self, max_age=30, n_init=3, max_cosine_distance=0.2, gpu=False):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        try:
            self.tracker = DeepSort(
                max_age=max_age, n_init=n_init, max_cosine_distance=max_cosine_distance,
                embedder="mobilenet", embedder_gpu=gpu, half=False, bgr=True)
        except Exception as exc:
            raise RuntimeError("DeepSORT/MobileNet initialization failed. Check the pinned "
                               "torch/torchvision pair, setuptools, and bundled pretrained weights. "
                               f"Cause: {exc}") from exc

    def update(self, detections, frame):
        raw = [([d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0],
                 d.bbox[3] - d.bbox[1]], d.confidence, "face") for d in detections]
        tracks = self.tracker.update_tracks(raw, frame=frame)
        return observations_from_tracks(tracks, frame.shape[1], frame.shape[0])
