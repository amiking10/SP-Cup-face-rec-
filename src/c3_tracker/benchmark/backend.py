"""Tracker plug-in contract: RGB-derived frames, times, optional oracle BOXES only."""
from __future__ import annotations
from typing import Protocol
import importlib
import time
from ..detection import Detection


class TrackerBackend(Protocol):
    def process(self, frame, timestamp_s: float, oracle_boxes=None): ...
    def designate(self, track_id: str): ...
    def close(self): ...


class ExistingTrackerBackend:
    def __init__(self, config):
        from ..detection import FaceDetector
        from ..tracking import FaceTracker
        self.config = config
        if config.experiment == 'oracle' and config.appearance_model != 'mobilenet':
            raise ValueError('Oracle boxes require --appearance-model mobilenet: aligned SFace uses YuNet detection '
                             'and can drop unalignable boxes. Compare both experiments with mobilenet for detector isolation.')
        self.detector = None
        self.detector_probe = None
        self.tracker = None
        try:
            if config.experiment == 'end_to_end':
                self.detector = FaceDetector(config.confidence, config.model_selection)
                if config.stage_diagnostics:
                    self.detector_probe = FaceDetector(min(config.diagnostic_probe_confidence,config.confidence),config.model_selection)
            self.tracker = FaceTracker(n_init=config.n_init, memory_seconds=config.memory_seconds,
                                       max_cosine_distance=config.max_cosine_distance, gpu=config.gpu,
                                       appearance_model=config.appearance_model, model_dir=config.model_dir,
                                       collect_stages=config.stage_diagnostics)
        except BaseException:
            self.close()
            raise

    def process(self, frame, timestamp_s, oracle_boxes=None):
        started = time.perf_counter()
        if self.detector is not None:
            if oracle_boxes is not None:
                raise ValueError('Ground-truth boxes cannot enter end-to-end inference')
            detections = self.detector.detect(frame)
        else:
            if oracle_boxes is None:
                raise ValueError('Oracle experiment needs boxes, including an empty list for no faces')
            detections = [Detection(tuple(box), 1.0) for box in oracle_boxes]
        observations = self.tracker.update(detections, frame, timestamp=timestamp_s)
        self._snapshot = None
        if self.config.stage_diagnostics:
            baseline_ms = (time.perf_counter()-started)*1000
            probe_started = time.perf_counter()
            raw = self.detector_probe.detect(frame) if self.detector_probe is not None else None
            self._snapshot = {**self.tracker.last_stage_snapshot,
                              'raw_detector_probe':None if raw is None else [
                                  {'bbox':list(map(float,d.bbox)),'confidence':float(d.confidence)} for d in raw],
                              'probe_confidence':min(self.config.diagnostic_probe_confidence,self.config.confidence),
                              'baseline_detector_tracking_ms':baseline_ms,
                              'diagnostic_probe_ms':(time.perf_counter()-probe_started)*1000,
                              'raw_probe_semantics':'Separate MediaPipe pass at stated score floor, after MediaPipe NMS; not unfiltered network logits. Baseline detector call is unchanged.'}
        return observations

    def stage_snapshot(self):
        return self._snapshot

    def designate(self, track_id):
        self.tracker.protect_target(track_id)

    def close(self):
        if self.detector is not None:
            self.detector.close()
        if self.detector_probe is not None:
            self.detector_probe.close()

    def runtime_info(self):
        from ..models import MODELS
        if self.config.appearance_model == 'sface':
            return {'backend':'existing MediaPipe + SFace + DeepSORT','actual_device':'cpu',
                    'appearance_model':'sface','model_sha256':{name:digest for name,(_,digest) in MODELS.items()},
                    'oracle':False}
        embedder = self.tracker.tracker.embedder
        return {'backend':'existing MediaPipe + MobileNet + DeepSORT','actual_device':str(embedder.device),
                'appearance_model':'mobilenet','gpu_requested':self.config.gpu,
                'oracle':self.config.experiment=='oracle'}


def make_backend(config):
    if config.backend_factory == 'existing':
        return ExistingTrackerBackend(config)
    try:
        module, function = config.backend_factory.split(':', 1)
        return getattr(importlib.import_module(module), function)(config)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ValueError(f'Cannot load backend factory {config.backend_factory}: {exc}') from exc
