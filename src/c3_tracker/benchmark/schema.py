"""Common, zero-based, source-frame schemas with continuous pixel xyxy boxes."""
from __future__ import annotations
import csv
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import re

GT_FIELDS = ['video_id', 'frame_index', 'timestamp_s', 'person_id', 'x1', 'y1', 'x2', 'y2', 'visible']
PRED_FIELDS = ['video_id', 'frame_index', 'timestamp_s', 'track_id', 'x1', 'y1', 'x2', 'y2',
               'observed', 'confirmed', 'processing_ms', 'selected_track_id', 'target_visible',
               'target_status', 'processed_frame_index', 'width', 'height']


def finite(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'{name} must be finite')
    return number


def validate_box(box, width, height):
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError('box must contain four finite coordinates')
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f'invalid pixel xyxy box {box} for {width}x{height}')
    return tuple(float(v) for v in box)


def normalized_box(box, width, height):
    validate_box(box, 1, 1)
    return (box[0] * width, box[1] * height, box[2] * width, box[3] * height)


def scale_box(box, source_size, output_size):
    sx, sy = output_size[0] / source_size[0], output_size[1] / source_size[1]
    return (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)


def aligned_frame(timestamp, offset, fps, frame_count, tolerance):
    if not 0 < tolerance <= .5 / fps + 1e-9:
        raise ValueError('alignment tolerance must be positive and at most half a source frame')
    local = finite(timestamp, 'timestamp') - finite(offset, 'video start timestamp')
    index = math.floor(local * fps + .5)
    if not 0 <= index < frame_count:
        raise ValueError(f'annotation timestamp {timestamp} maps outside video; check video_start_timestamp_s')
    error = abs(local - index / fps)
    if error > tolerance + 1e-9:
        raise ValueError(f'timestamp {timestamp} is {error:.6f}s from frame {index}; check FPS/offset/CFR')
    return index, index / fps, error


def fingerprint(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, fields, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class GroundTruth:
    video_id: str
    frame_index: int
    timestamp_s: float
    person_id: str
    box: tuple[float, float, float, float] | None
    visible: bool = True

    def row(self):
        return dict(video_id=self.video_id, frame_index=self.frame_index, timestamp_s=self.timestamp_s,
                    person_id=self.person_id, **dict(zip(('x1','y1','x2','y2'), self.box or ('',)*4)), visible=int(self.visible))


@dataclass(frozen=True)
class VideoSpec:
    video_id: str
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    video_start_timestamp_s: float


@dataclass
class Sequence:
    video: VideoSpec
    ground_truth: dict[int, list[GroundTruth]]
    metadata: dict
    annotation_path: Path


def probe_video(path, video_id, offset):
    import cv2
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', video_id):
        raise ValueError('video_id must be a safe filename stem (letters, digits, underscore, hyphen)')
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Video not found: {path}')
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot decode video: {path}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or min(count, width, height) <= 0:
            raise ValueError('Video needs a valid CFR FPS, resolution and frame count')
        return VideoSpec(video_id, path, width, height, fps, count, finite(offset, 'video offset'))
    finally:
        cap.release()


def load_sequence(entry, root):
    required = ['video_id', 'video_path', 'annotations_path', 'video_start_timestamp_s', 'cfr',
                'coverage', 'identity_scope', 'identity_annotations_complete', 'absence_annotations_complete']
    missing = [key for key in required if key not in entry]
    if missing:
        raise ValueError(f'Manifest sequence is missing {missing}')
    if entry['cfr'] is not True:
        raise ValueError('Only verified CFR videos are supported; transcode VFR and remap annotations first')
    if entry['coverage'] not in ('partial', 'complete'):
        raise ValueError('coverage must be partial or complete; do not infer exhaustive annotation from missing rows')
    if entry['coverage'] == 'complete' and not entry.get('coverage_provenance'):
        raise ValueError('complete coverage requires coverage_provenance describing how exhaustive annotation was verified')
    if entry['identity_scope'] not in ('none', 'track_local', 'persistent'):
        raise ValueError('identity_scope must be none, track_local or persistent')
    for flag in ('identity_annotations_complete', 'absence_annotations_complete'):
        if type(entry[flag]) is not bool:
            raise ValueError(f'{flag} must be a JSON boolean')
    if entry['absence_annotations_complete'] and (entry['coverage'] != 'complete' or entry['identity_scope'] != 'persistent'):
        raise ValueError('known physical absence needs complete coverage and persistent identities')
    video = probe_video(root / entry['video_path'], entry['video_id'], entry['video_start_timestamp_s'])
    annotation_path = (root / entry['annotations_path']).resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f'Normalized annotations missing: {annotation_path}; run the AVA prepare command')
    by_frame = {}
    seen = set()
    with annotation_path.open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        if not set(GT_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f'{annotation_path}: expected common ground-truth schema {GT_FIELDS}')
        for line, row in enumerate(reader, 2):
            try:
                index = int(row['frame_index'])
                if str(index) != row['frame_index'].strip() or not 0 <= index < video.frame_count:
                    raise ValueError('frame_index must be a zero-based source integer within the video')
                timestamp = finite(row['timestamp_s'], 'timestamp_s')
                if abs(timestamp - index / video.fps) > 1e-6:
                    raise ValueError('normalized timestamp does not match frame_index/source FPS')
                if row['video_id'] != video.video_id or not row['person_id'].strip():
                    raise ValueError('wrong video_id or empty person_id')
                if row['visible'] not in ('0', '1'):
                    raise ValueError('visible must be 0 or 1')
                visible = row['visible'] == '1'
                if not visible and any(row[k].strip() for k in ('x1','y1','x2','y2')):
                    raise ValueError('invisible rows must have empty box coordinates')
                box = validate_box(tuple(finite(row[k], k) for k in ('x1','y1','x2','y2')), video.width, video.height) if visible else None
                key = (index, row['person_id'])
                if key in seen:
                    raise ValueError('duplicate person/frame annotation')
                seen.add(key)
                gt = GroundTruth(video.video_id, index, timestamp, row['person_id'], box, visible)
                by_frame.setdefault(index, []).append(gt)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'{annotation_path}:{line}: {exc}') from exc
    for index in entry.get('known_empty_frames', []):
        if type(index) is not int or not 0 <= index < video.frame_count or index in by_frame:
            raise ValueError('known_empty_frames must be unique, in-range, and have no annotation rows')
        by_frame[index] = []
    if not by_frame:
        raise ValueError(f'No annotated frames for {video.video_id}')
    for records in by_frame.values():
        records.sort(key=lambda gt: gt.person_id)
    if entry['identity_scope'] == 'persistent' and not entry.get('identity_provenance'):
        raise ValueError('persistent identities require identity_provenance, not inferred track IDs')
    expected_video = entry.get('video_sha256')
    if expected_video and fingerprint(video.path) != expected_video:
        raise ValueError('video fingerprint differs from prepared annotations; prepare again')
    expected_annotations = entry.get('annotations_sha256')
    if expected_annotations and fingerprint(annotation_path) != expected_annotations:
        raise ValueError('normalized annotations fingerprint mismatch')
    return Sequence(video, by_frame, dict(entry), annotation_path)


def read_manifest(path, dataset, video_ids=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Benchmark manifest not found: {path}; see datasets/BENCHMARK_GUIDE.md')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema_version') != 1 or payload.get('dataset') != dataset:
        raise ValueError('Manifest schema_version/dataset mismatch')
    entries = payload.get('videos', [])
    ids = [entry.get('video_id') for entry in entries]
    if not entries or len(set(ids)) != len(ids):
        raise ValueError('Manifest needs unique nonempty video entries')
    if video_ids:
        missing = set(video_ids) - set(ids)
        if missing:
            raise ValueError(f'Unknown video IDs: {sorted(missing)}')
        entries = [entry for entry in entries if entry['video_id'] in video_ids]
    return payload, entries, path.parent


def read_predictions(path, video, expected_size=None):
    """Validate exported/imported predictions before joining them with annotations."""
    from ..tracking import Observation
    from .inference import PredictedFrame
    groups, properties = {}, {}
    path = Path(path)
    with path.open(newline='',encoding='utf-8') as source:
        reader=csv.DictReader(source)
        if not set(PRED_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: prediction schema is missing fields')
        for line,row in enumerate(reader,2):
            if row['video_id']!=video.video_id:
                continue
            try:
                index=int(row['frame_index']);ordinal=int(row['processed_frame_index'])
                if str(index)!=row['frame_index'].strip() or not 0<=index<video.frame_count or ordinal<0:
                    raise ValueError('invalid prediction source/processed frame index')
                timestamp=finite(row['timestamp_s'],'prediction timestamp')
                if abs(timestamp-index/video.fps)>1e-6:
                    raise ValueError('prediction timestamp/frame alignment mismatch')
                size=(int(row['width']),int(row['height']))
                if min(size)<=0 or expected_size is not None and tuple(expected_size)!=size:
                    raise ValueError('prediction output resolution mismatch')
                latency=finite(row['processing_ms'],'processing_ms')
                if latency<0:
                    raise ValueError('negative processing latency')
                if any(row[k] not in ('0','1') for k in ('observed','confirmed','target_visible')):
                    raise ValueError('prediction flags must be 0 or 1')
                selected=row['selected_track_id'] or None
                status=row['target_status']
                if status not in ('NOT SELECTED','VISIBLE','REACQUIRED','LOST'):
                    raise ValueError('invalid target status')
                prop=(timestamp,ordinal,size,latency,selected,status,row['target_visible'])
                if index in properties and properties[index]!=prop:
                    raise ValueError('inconsistent per-frame properties across prediction rows')
                properties[index]=prop
                track=row['track_id']
                observation=None
                if track:
                    box=validate_box(tuple(finite(row[k],k) for k in ('x1','y1','x2','y2')),*size)
                    observation=Observation(track,box,row['observed']=='1',row['confirmed']=='1')
                elif row['observed']!='0' or row['confirmed']!='0' or any(row[k] for k in ('x1','y1','x2','y2')):
                    raise ValueError('empty-frame sentinel must not contain a box or observation')
                groups.setdefault(index,[]).append(observation)
            except (ValueError,TypeError) as exc:
                raise ValueError(f'{path}:{line}: {exc}') from exc
    if not groups:
        raise ValueError(f'No predictions (including empty-frame sentinels) for {video.video_id}')
    result=[]
    for index,observations in sorted(groups.items()):
        timestamp,ordinal,size,latency,selected,status,visible=properties[index]
        if ordinal!=len(result):
            raise ValueError('processed_frame_index must be contiguous and zero-based')
        ids=[o.track_id for o in observations if o is not None]
        if len(set(ids))!=len(ids) or None in observations and len(observations)!=1:
            raise ValueError('duplicate track ID or mixed empty-frame sentinel')
        observations=tuple(o for o in observations if o is not None)
        actual=bool(selected and any(o.track_id==selected and o.observed and o.confirmed for o in observations))
        if actual!=(visible=='1') or actual!=(status in ('VISIBLE','REACQUIRED')) or (selected is None)!=(status=='NOT SELECTED'):
            raise ValueError('target visibility/status disagrees with observed confirmed selected track')
        result.append(PredictedFrame(index,ordinal,timestamp,observations,latency,selected,status,*size))
    return result
