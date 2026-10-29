"""Video-only inference. Future annotations are not available in this module's loop."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from pathlib import Path
import time
import cv2
import numpy as np
from .matching import match_boxes
from .schema import scale_box, validate_box


@dataclass(frozen=True)
class BenchmarkConfig:
    experiment: str = 'end_to_end'
    resolution: tuple[int, int] | None = None
    fps: float | None = None
    confidence: float = .65
    model_selection: int = 0
    n_init: int = 3
    memory_seconds: float = 3.0
    max_cosine_distance: float | None = None
    appearance_model: str = 'sface'
    model_dir: str = '.models'
    gpu: bool = False
    iou_threshold: float = .5
    target_min_processed_frame: int = 2
    target_enabled: bool = True
    max_frames: int | None = None
    backend_factory: str = 'existing'
    seed: int = 0
    threads: int = 1


@dataclass(frozen=True)
class InitializationPlan:
    frame_index: int
    timestamp_s: float
    person_id: str
    target_box: tuple
    visible_boxes: tuple
    visible_ids: tuple


@dataclass
class PredictedFrame:
    frame_index: int
    processed_frame_index: int
    timestamp_s: float
    observations: tuple
    processing_ms: float
    selected_track_id: str | None
    target_status: str
    width: int
    height: int

    @property
    def target_visible(self):
        return self.target_status in ('VISIBLE', 'REACQUIRED')

    def rows(self, video_id):
        for observation in self.observations or (None,):
            yield {'video_id': video_id, 'frame_index': self.frame_index, 'timestamp_s': self.timestamp_s,
                   'track_id': observation.track_id if observation else '',
                   **dict(zip(('x1','y1','x2','y2'), observation.bbox if observation else ('',)*4)),
                   'observed': int(bool(observation and observation.observed)),
                   'confirmed': int(bool(observation and observation.confirmed)),
                   'processing_ms': self.processing_ms, 'selected_track_id': self.selected_track_id or '',
                   'target_visible': int(self.target_visible), 'target_status': self.target_status,
                   'processed_frame_index': self.processed_frame_index, 'width': self.width, 'height': self.height}


def sampled_indices(video, start, end, target_fps=None, max_frames=None):
    if not 0 <= start < end <= video.frame_count:
        raise ValueError('Invalid inference frame interval')
    if target_fps is None:
        indices = list(range(start, end))
    else:
        if not math.isfinite(target_fps) or not 0 < target_fps <= video.fps + 1e-6:
            raise ValueError('Requested FPS must be positive and no greater than source FPS')
        indices = []
        ordinal = max(0, math.floor(start * target_fps / video.fps))
        while True:
            index = math.floor(ordinal * video.fps / target_fps + .5)
            if index >= end:
                break
            if index >= start and (not indices or indices[-1] != index):
                indices.append(index)
            ordinal += 1
    return indices[:max_frames] if max_frames is not None else indices


def plan_initialization(sequence, indices, config):
    if not config.target_enabled:
        return None
    size = config.resolution or (sequence.video.width, sequence.video.height)
    source_size = sequence.video.width, sequence.video.height
    for ordinal, index in enumerate(indices):
        records = [gt for gt in sequence.ground_truth.get(index, []) if gt.visible]
        if ordinal >= config.target_min_processed_frame and len(records) >= 2:
            records.sort(key=lambda gt: gt.person_id)
            chosen = records[0]
            boxes = tuple(scale_box(gt.box, source_size, size) for gt in records)
            return InitializationPlan(index, index / sequence.video.fps, chosen.person_id,
                                      boxes[0], boxes, tuple(gt.person_id for gt in records))
    return None


def infer_video(video, indices, config, backend, initialization=None, oracle_boxes=None):
    """Only initialization may contain one annotated frame. Oracle data has no IDs.

    The complete sequence/ground-truth object is deliberately not an argument.
    Normal backends receive only OpenCV BGR frames and source-local timestamps.
    """
    if config.experiment == 'end_to_end' and oracle_boxes is not None:
        raise ValueError('End-to-end inference must not receive oracle boxes')
    if config.experiment == 'oracle' and oracle_boxes is None:
        raise ValueError('Oracle inference requires a boxes-only stream')
    if not indices:
        raise ValueError('No source frames selected')
    cap = cv2.VideoCapture(str(video.path))
    frames = []
    selected_id = None
    was_lost = False
    size = config.resolution or (video.width, video.height)
    audit = {'policy': 'first sampled multiface frame after warmup; lexicographic annotation ID; one-to-one IoU',
             'ground_truth_used': 'one initialization frame only; labels never sent to backend',
             'status': 'no_eligible_multiface_frame' if initialization is None else 'pending',
             'frame_index': initialization.frame_index if initialization else None,
             'person_id': initialization.person_id if initialization else None,
             'information': asdict(initialization) if initialization else None,
             'selected_track_id': None, 'iou_threshold': config.iou_threshold,
             'replacement_selection': 'never'}
    started_wall = time.perf_counter()
    timestamp_checks = 0
    try:
        if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0]):
            raise ValueError('Cannot open/seek source video')
        wanted = set(indices)
        for index in range(indices[0], indices[-1] + 1):
            if index not in wanted:
                if not cap.grab():
                    raise RuntimeError(f'Unexpected video EOF at source frame {index}')
                continue
            started = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot decode expected source frame {index}')
            if frame.shape[:2] != (video.height, video.width):
                raise ValueError('Video resolution changed; prepared annotations are no longer aligned')
            decoded_position = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if decoded_position and abs(decoded_position - index - 1) > .1:
                raise ValueError(f'Decoder frame position mismatch at {index}')
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if pts > 0:
                if abs(pts - index / video.fps) > .51 / video.fps + .002:
                    raise ValueError(f'Non-CFR or mismatched decoder timestamp at frame {index}: {pts}')
                timestamp_checks += 1
            if size != (video.width, video.height):
                frame = cv2.resize(frame, size)
            supplied = oracle_boxes.get(index, ()) if oracle_boxes is not None else None
            observations = tuple(backend.process(frame, index / video.fps, oracle_boxes=supplied))
            ids = [o.track_id for o in observations]
            if len(set(ids)) != len(ids):
                raise ValueError('Backend returned duplicate track IDs in one frame')
            for observation in observations:
                validate_box(observation.bbox, *size)
            if initialization is not None and index == initialization.frame_index:
                eligible = [o for o in observations if o.observed and o.confirmed]
                matches, _, _, overlaps = match_boxes(list(initialization.visible_boxes),
                                                       [o.bbox for o in eligible], config.iou_threshold)
                selected = next((j for i, j in matches if i == 0), None)
                audit['status'] = 'no_confirmed_prediction_for_designated_face'
                if selected is not None:
                    selected_id = eligible[selected].track_id
                    backend.designate(selected_id)
                    audit.update(status='initialized', selected_track_id=selected_id,
                                 initialization_iou=float(overlaps[0, selected]))
            visible = bool(selected_id and any(o.track_id == selected_id and o.observed and o.confirmed for o in observations))
            status = 'NOT SELECTED' if selected_id is None else (
                ('REACQUIRED' if was_lost else 'VISIBLE') if visible else 'LOST')
            was_lost = selected_id is not None and not visible
            frames.append(PredictedFrame(index, len(frames), index / video.fps, observations,
                                         (time.perf_counter() - started) * 1000, selected_id, status, *size))
    finally:
        cap.release()
    elapsed = time.perf_counter() - started_wall
    latencies = np.asarray([frame.processing_ms for frame in frames])
    performance = {'frames_processed': len(frames), 'inference_elapsed_s': elapsed,
                   'processing_fps': len(frames) / elapsed if elapsed else None,
                   'latency_mean_ms': float(latencies.mean()) if len(latencies) else None,
                   'latency_p95_ms': float(np.percentile(latencies, 95)) if len(latencies) else None,
                   'source_fps': video.fps, 'requested_sampling_fps': config.fps,
                   'source_resolution': [video.width, video.height], 'output_resolution': list(size),
                   'source_frame_indices': [frame.frame_index for frame in frames],
                   'decoder_timestamp_checks': timestamp_checks,
                   'timestamp_policy': 'source frame index / declared CFR FPS; positive decoder PTS checked',
                   'latency_definition': 'selected-frame read + resize + detection/embedding/tracking + target state; excludes exports/scoring/model startup',
                   'fps_definition': 'all inference frames / inference wall time, including skipped-frame decoding; excludes model startup/exports/evaluation'}
    return frames, audit, performance
