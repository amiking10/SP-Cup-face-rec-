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
    def __init__(self, max_age=None, n_init=3, max_cosine_distance=None, gpu=False,
                 memory_seconds=3.0, appearance_model='sface', model_dir=None):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        from .association import FaceAssociationTracker
        try:
            self.face_embedder = None
            if appearance_model == 'sface':
                from .embedding import FaceEmbedder
                self.face_embedder = FaceEmbedder(model_dir)
            elif appearance_model != 'mobilenet':
                raise ValueError(f'Unknown appearance model: {appearance_model}')
            if max_cosine_distance is None:
                max_cosine_distance = .45 if self.face_embedder is not None else .2
            self.tracker = DeepSort(
                max_age=max_age if max_age is not None else 30,
                n_init=n_init, max_cosine_distance=max_cosine_distance,
                nn_budget=100, gating_only_position=True,
                embedder=None if self.face_embedder is not None else "mobilenet",
                embedder_gpu=gpu, half=False, bgr=True)
            # Retain the upstream embedding/cropping implementation; customize
            # association in project code rather than editing site-packages.
            self.tracker.tracker = FaceAssociationTracker(
                self.tracker.tracker.metric, max_age=max_age if max_age is not None else 30, n_init=n_init,
                memory_seconds=memory_seconds if max_age is None else None,
                gating_only_position=True)
        except Exception as exc:
            raise RuntimeError(f"DeepSORT/appearance model initialization failed: {exc}") from exc

    def update(self, detections, frame, timestamp=None):
        # Video uses its source timeline; webcam uses monotonic elapsed time.
        self.tracker.tracker.timestamp = timestamp
        detected_faces = len(detections)
        embeds = None
        if self.face_embedder is not None:
            detections, embeds = self.face_embedder.encode(detections, frame)
        self._detection_counts = {'detected_faces': detected_faces, 'alignable_faces': len(detections)}
        raw = [([d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0],
                 d.bbox[3] - d.bbox[1]], d.confidence, "face") for d in detections]
        tracks = self.tracker.update_tracks(raw, embeds=embeds, frame=frame)
        return observations_from_tracks(tracks, frame.shape[1], frame.shape[0])

    def protect_target(self, track_id):
        self.tracker.tracker.protect_target(track_id)

    def diagnostics(self):
        return {**self._detection_counts, **self.tracker.tracker.target_diagnostics()}
