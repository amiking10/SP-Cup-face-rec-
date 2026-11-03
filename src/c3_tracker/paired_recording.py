"""Synchronized clean/full-frame and selected-face exports from a CFR input video."""
import math
import csv
import hashlib
from pathlib import Path

import cv2
from .logging_utils import write_json
from .roi import DEFAULT_TOP_EXTENSION, capture_frame


class PairedRecorder:
    def __init__(self, directory, fps, source, source_size, margins, loss_policy='stop',video_format='mp4'):
        if loss_policy not in ('stop', 'mark'):
            raise ValueError('Pair loss policy must be stop or mark')
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Paired recording needs a positive source FPS')
        self.directory = Path(directory)
        if video_format not in ('mp4','lossless'): raise ValueError('Unknown pair video format')
        extension, self.codec=('mp4','mp4v') if video_format=='mp4' else ('mkv','FFV1')
        self.video_files={'original':f'original.{extension}','cropped':f'face_centered.{extension}'}
        self.fps = fps
        self.status = 'waiting'
        self.writers = []
        self.count = 0
        self.valid_count = 0
        self.loss_policy = loss_policy
        self.validity_file = None
        self.validity_writer = None
        width, height = source_size
        self.source_size = width, height
        self.original_size = width + width % 2, height + height % 2
        self.crop_size = None
        self.target_id = None
        self.first = self.last = None
        self.metadata = {
            'status': self.status, 'source_video': str(Path(source).resolve()),
            'fps': fps, 'source_resolution': list(source_size),
            'original_resolution': list(self.original_size), 'roi_margins': margins,
            'default_top_extension': DEFAULT_TOP_EXTENSION,
            'tracking_log': 'tracking_log.csv',
            'codec': self.codec, 'video_files':self.video_files,'audio': False, 'overlays': False,
            'crop_policy': 'First valid ROI dimensions rounded up to even pixels; later ROIs fitted with aspect ratio preserved and centered black padding',
            'original_padding_right_bottom': [width % 2, height % 2],
            'clock_policy': 'CFR input required; source index / FPS, checked against decoder timestamps',
            'frame_mapping': 'output index = source index - first_source_frame',
            'restart_policy': ('One selected-ID timeline; continue through loss and resume valid crops' if loss_policy=='mark'
                               else 'One contiguous selected-ID segment; never restart after stopping'),
            'schema_version': 2, 'loss_policy': loss_policy,
            'validity_log': 'frame_validity.csv',
            'missing_frame_policy': 'FACE LOST placeholders are invalid data, never pulse samples',
            'invalid_crop_placeholder': 'FACE LOST' if loss_policy=='mark' else None,
        }
        for name in (*self.video_files.values(), 'paired_recording.json', 'frame_validity.csv'):
            if (self.directory / name).exists():
                raise ValueError(f'Paired output already exists: {name}')

    def _save_metadata(self, reason=None):
        self.metadata.update(status=self.status, stop_reason=reason,
                             selected_track_id=self.target_id, frames_written=self.count,
                             first_source_frame=self.first, last_source_frame=self.last,
                             source_start_s=self.first / self.fps if self.first is not None else None,
                             source_end_exclusive_s=(self.last + 1) / self.fps if self.last is not None else None,
                             duration_s=self.count / self.fps,
                             valid_face_frames=self.valid_count, lost_face_frames=self.count - self.valid_count,
                             face_centered_resolution=list(self.crop_size) if self.crop_size else None)
        write_json(self.directory / 'paired_recording.json', self.metadata)

    def _start(self, target_id, roi):
        left, top, right, bottom = roi
        width, height = right - left, bottom - top
        self.crop_size = width + width % 2, height + height % 2
        self.target_id = str(target_id)
        try:
            for name, size in ((self.video_files['original'], self.original_size), (self.video_files['cropped'], self.crop_size)):
                writer = cv2.VideoWriter(str(self.directory / name), cv2.VideoWriter_fourcc(*self.codec), self.fps, size)
                self.writers.append(writer)
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot initialize paired video writer: {name}')
        except BaseException as exc:
            for writer in self.writers:
                writer.release()
            self.writers.clear()
            self.status = 'failed'
            self.metadata['error'] = f'{type(exc).__name__}: {exc}'
            self._save_metadata('writer_initialization_failed')
            raise
        self.status = 'recording'
        self.validity_file = (self.directory / 'frame_validity.csv').open('w', newline='', encoding='utf-8')
        self.validity_writer = csv.writer(self.validity_file)
        self.validity_writer.writerow(['frame_index', 'source_frame_index', 'timestamp_s', 'source_timestamp_s',
                                      'target_id', 'face_valid', 'reason'])
        print(f'Paired recording started: target {target_id} -> {self.directory.resolve()}', flush=True)

    def begin(self,target_id,roi):
        """Start a fixed benchmark timeline, including an initially unavailable target."""
        if self.status!='waiting' or target_id is None: raise ValueError('Pair cannot start in this state')
        left,top,right,bottom=roi
        if not (0<=left<right<=self.source_size[0] and 0<=top<bottom<=self.source_size[1]):
            raise ValueError('Invalid initial crop canvas bounds')
        self._start(target_id,roi)

    def update(self, frame, source_index, target_id, visible, roi, source_timestamp_s):
        """Write the same valid source pixels to both streams without overlays."""
        if self.status not in ('waiting', 'recording'):
            return
        if self.status == 'recording':
            if str(target_id) != self.target_id:
                self.finish('target_changed')
                return
            if (not visible or roi is None) and self.loss_policy == 'stop':
                self.finish('target_lost' if not visible else 'capture_roi_unavailable')
                return
        elif not visible or target_id is None or roi is None:
            return
        if (frame.shape[1], frame.shape[0]) != self.source_size:
            raise RuntimeError('Source resolution changed during paired recording')
        if (not math.isfinite(source_timestamp_s) or
                abs(source_timestamp_s - source_index / self.fps) > .51 / self.fps + .002):
            raise RuntimeError('Paired recording requires CFR input with consistent decoder timestamps')
        if self.last is not None and source_index != self.last + 1:
            raise RuntimeError('Paired recording cannot skip source frames')
        valid = visible and roi is not None
        if valid:
            left, top, right, bottom = roi
            if not (0 <= left < right <= frame.shape[1] and 0 <= top < bottom <= frame.shape[0]):
                raise ValueError('Invalid selected capture bounds for paired recording')
        if self.status == 'waiting':
            self._start(target_id, roi)
        centered=capture_frame(frame,roi if valid else None,self.crop_size)
        original = cv2.copyMakeBorder(frame, 0, frame.shape[0] % 2, 0, frame.shape[1] % 2,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
        self.writers[0].write(original)
        self.writers[1].write(centered)
        self.validity_writer.writerow([self.count, source_index, self.count / self.fps, source_index / self.fps,
                                       self.target_id, int(valid), 'valid' if valid else
                                       ('target_lost' if not visible else 'capture_roi_unavailable')])
        self.validity_file.flush()
        self.valid_count += int(valid)
        self.count += 1
        self.first = source_index if self.first is None else self.first
        self.last = source_index
        if self.count == 1:
            self._save_metadata()

    def _verify(self):
        checks = {}
        for name, size in ((self.video_files['original'], self.original_size), (self.video_files['cropped'], self.crop_size)):
            cap = cv2.VideoCapture(str(self.directory / name))
            try:
                if not cap.isOpened():
                    raise RuntimeError(f'Cannot reopen saved pair video: {name}')
                fps = cap.get(cv2.CAP_PROP_FPS)
                count = 0
                expected_hash=self.metadata.get('expected_pixel_sha256',{}).get(name)
                pixel_hash=hashlib.sha256() if expected_hash else None
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if (frame.shape[1], frame.shape[0]) != size:
                        raise RuntimeError(f'Saved pair resolution mismatch: {name}')
                    if pixel_hash is not None: pixel_hash.update(frame.tobytes())
                    count += 1
                checks[name] = {'decoded_frames': count, 'fps': fps,
                                'duration_s': count / fps if fps > 0 else None,
                                'resolution': list(size)}
                if pixel_hash is not None:
                    checks[name]['pixel_sha256']=pixel_hash.hexdigest()
                    if pixel_hash.hexdigest()!=expected_hash: raise RuntimeError(f'Lossless pixel mismatch: {name}')
                if count != self.count or not math.isfinite(fps) or abs(fps - self.fps) > .001:
                    raise RuntimeError(f'Saved pair frame count/FPS mismatch: {name}')
            finally:
                cap.release()
        return checks

    def finish(self, reason):
        if self.status not in ('waiting', 'recording'):
            return
        for writer in self.writers:
            writer.release()
        self.writers.clear()
        if self.validity_file is not None:
            self.validity_file.close()
        if not self.count:
            self.status = 'not_recorded'
            self._save_metadata(reason)
            return
        try:
            self.metadata['verification'] = self._verify()
            self.status = 'saved'
        except BaseException as exc:
            self.status = 'failed'
            self.metadata['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            self._save_metadata(reason)
        print(f'Paired videos saved: {self.count} frames each; {reason}. Folder: {self.directory.resolve()}', flush=True)
