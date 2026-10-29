"""Strict adapter for verified AVA ActiveSpeaker v1.0 five/eight-field CSV rows."""
from __future__ import annotations
import csv
import ast
import json
from pathlib import Path
import re
from .schema import GT_FIELDS, GroundTruth, aligned_frame, finite, normalized_box, probe_video, fingerprint, write_csv, write_json


def parse_ava_row(row):
    if len(row) == 8:
        video_id, timestamp, *rest = row
        coordinates = tuple(finite(v, 'normalized coordinate') for v in rest[:4])
        label, entity_id = rest[4:]
    elif len(row) == 5:
        video_id, timestamp, box, label, entity_id = row
        try:
            tokens = ast.literal_eval(box)
        except (ValueError,SyntaxError) as exc:
            raise ValueError('encoded entity_box must be a numeric tuple/list') from exc
        if not isinstance(tokens,(list,tuple)) or len(tokens) != 4:
            raise ValueError('encoded entity_box must contain exactly four coordinates')
        coordinates = tuple(finite(v, 'normalized coordinate') for v in tokens)
    else:
        raise ValueError(f'AVA ActiveSpeaker expects 5 or 8 fields, received {len(row)}')
    if not video_id.strip() or not entity_id.strip():
        raise ValueError('video/entity IDs must be nonempty')
    if label.strip() not in ('NOT_SPEAKING','SPEAKING_AUDIBLE','SPEAKING_NOT_AUDIBLE',
                             'SPEAKING_AND_AUDIBLE','SPEAKING_BUT_NOT_AUDIBLE'):
        raise ValueError('Not an AVA ActiveSpeaker label; AVA Actions/person-body CSVs are not face annotations')
    normalized_box(coordinates, 1, 1)
    timestamp = finite(timestamp, 'frame_timestamp')
    if timestamp < 0:
        raise ValueError('negative annotation timestamp')
    return video_id.strip(), timestamp, coordinates, label.strip(), entity_id.strip()


def prepare_ava(manifest_path, output, tolerance=.0051):
    """Explicit source offsets and CFR declarations are mandatory, never guessed."""
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    source = json.loads(manifest_path.read_text(encoding='utf-8'))
    if source.get('dataset') != 'ava' or not source.get('videos'):
        raise ValueError('Raw manifest must declare dataset=ava and a nonempty videos list')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Preparation output is not empty: {output}; use a new directory')
    normalized_entries = []
    pending = []
    ids = set()
    for entry in source['videos']:
        video_id = entry['video_id']
        if video_id in ids:
            raise ValueError('duplicate video_id in raw manifest')
        ids.add(video_id)
        if entry.get('cfr') is not True or 'video_start_timestamp_s' not in entry:
            raise ValueError('AVA requires explicit cfr=true and video_start_timestamp_s; no offset inference')
        video = probe_video(manifest_path.parent / entry['video_path'], video_id, entry['video_start_timestamp_s'])
        csv_path = (manifest_path.parent / entry['annotation_path']).resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f'AVA CSV missing: {csv_path}')
        window = entry.get('annotation_window_s')
        if window and (len(window) != 2 or window[0] >= window[1]):
            raise ValueError('annotation_window_s must be explicit [start,end) source seconds')
        records, seen = [], set()
        excluded = 0
        max_error = 0.0
        with csv_path.open(newline='', encoding='utf-8-sig') as stream:
            for line, row in enumerate(csv.reader(stream), 1):
                if not row:
                    continue
                try:
                    row_video, timestamp, box, label, entity = parse_ava_row(row)
                    if row_video != video_id:
                        raise ValueError('CSV video_id does not match manifest video')
                    if window and not window[0] <= timestamp < window[1]:
                        excluded += 1
                        continue
                    index, local_time, error = aligned_frame(timestamp, video.video_start_timestamp_s,
                                                             video.fps, video.frame_count, tolerance)
                    if (index, entity) in seen:
                        raise ValueError('duplicate entity/frame (or timestamps collapsed onto the same frame)')
                    seen.add((index, entity))
                    max_error = max(max_error, error)
                    gt = GroundTruth(video_id, index, local_time, entity,
                                     normalized_box(box, video.width, video.height))
                    records.append(gt)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'{csv_path}:{line}: {exc}') from exc
        if not records:
            raise ValueError(f'No AVA annotations remain for {video_id}')
        records.sort(key=lambda gt: (gt.frame_index, gt.person_id))
        destination = output / video_id / 'annotations.csv'
        normalized = dict(entry)
        normalized.pop('annotation_path', None)
        normalized.update({
            'video_path': str(video.path), 'annotations_path': f'{video_id}/annotations.csv',
            'identity_scope': 'track_local', 'coverage': entry.get('coverage', 'partial'),
            'identity_annotations_complete': entry.get('identity_annotations_complete', False),
            'absence_annotations_complete': False,
            'source_annotation_path': str(csv_path), 'source_annotations_sha256': fingerprint(csv_path),
            'video_sha256': fingerprint(video.path),
            'source_width': video.width, 'source_height': video.height, 'source_fps': video.fps,
            'source_frame_count': video.frame_count, 'alignment_tolerance_s': tolerance,
            'maximum_alignment_error_s': max_error, 'explicitly_excluded_annotations': excluded,
            'identity_note': 'AVA entity_id is a local face track, not a persistent physical-person identity.',
        })
        if entry.get('identity_scope', 'track_local') != 'track_local':
            raise ValueError('AVA entity IDs cannot be declared persistent; use independently verified common-format identities')
        pending.append((destination, records, normalized))
    output.mkdir(parents=True, exist_ok=True)
    for destination, records, entry in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_csv(destination, GT_FIELDS, (gt.row() for gt in records))
        entry['annotations_sha256'] = fingerprint(destination)
        normalized_entries.append(entry)
    payload = {'schema_version': 1, 'dataset': 'ava', 'release': source.get('release', 'v1.0'),
               'split': source.get('split', 'unspecified'), 'source_manifest': str(manifest_path),
               'source_manifest_sha256': fingerprint(manifest_path), 'videos': normalized_entries}
    write_json(output / 'manifest.json', payload)
    return payload
