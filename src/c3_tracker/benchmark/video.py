"""Explicit CFR clip preparation. No annotations are used to produce video frames."""
from pathlib import Path
import math
import cv2
from .schema import probe_video, fingerprint, write_json


def prepare_video(source, output, start_s, duration_s, fps):
    if not all(math.isfinite(v) for v in (start_s, duration_s, fps)) or start_s < 0 or duration_s <= 0 or fps <= 0:
        raise ValueError('Need finite start >= 0, duration > 0 and output FPS > 0')
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f'Will not overwrite video: {output}')
    video = probe_video(source, source.stem, 0)
    count = math.floor(duration_s * fps + 1e-9)
    if count < 1 or start_s + duration_s > video.frame_count / video.fps + 1e-6:
        raise ValueError('Requested clip is empty or extends outside source video')
    output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (video.width, video.height))
    mapping = []
    try:
        if not writer.isOpened():
            raise RuntimeError('Cannot initialize output MP4 writer')
        source_index, frame = -1, None
        first = math.floor(start_s * video.fps + .5)
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, first):
            raise ValueError('Cannot seek source clip start')
        source_index = first - 1
        for ordinal in range(count):
            timestamp = start_s + ordinal / fps
            wanted = math.floor(timestamp * video.fps + .5)
            while source_index < wanted:
                ok, frame = cap.read()
                source_index += 1
                if not ok:
                    raise ValueError(f'Unexpected source EOF at frame {source_index}')
                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if pts > 0 and abs(pts - source_index / video.fps) > .51 / video.fps + .002:
                    raise ValueError('Source PTS is not consistent with CFR; use a PTS-aware transcoder')
            writer.write(frame)
            mapping.append({'output_frame_index': ordinal, 'source_frame_index': source_index,
                            'requested_source_timestamp_s': timestamp, 'actual_source_timestamp_s': source_index / video.fps})
    finally:
        writer.release()
        cap.release()
    provenance = {'source_video': str(source), 'source_sha256': fingerprint(source),
                  'source_fps': video.fps, 'output_fps': fps, 'output_sha256': fingerprint(output),
                  'video_start_timestamp_s': start_s, 'duration_s': count / fps,
                  'method': 'nearest source CFR frame in time; duplicates allowed; original resolution; MP4 lossy encoding',
                  'maximum_source_time_error_s': max(abs(r['requested_source_timestamp_s']-r['actual_source_timestamp_s']) for r in mapping),
                  'frame_mapping': mapping}
    write_json(output.with_suffix('.provenance.json'), provenance)
    return provenance
