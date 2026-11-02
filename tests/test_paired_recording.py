"""Decode real MP4 outputs to verify selection boundaries, timing and clean pixels."""
import json

import cv2
import numpy as np
import pytest

from c3_tracker.paired_recording import PairedRecorder


def decode(path):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    frames = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames, fps


def recorder(tmp_path, size=(63, 47)):
    return PairedRecorder(tmp_path, 25.0, tmp_path / 'base.mp4', size,
                          {'top': 0, 'bottom': 0, 'left': 0, 'right': 0})


def test_pair_stops_before_first_lost_frame_and_never_restarts(tmp_path):
    pair = recorder(tmp_path)
    for index in range(8):
        frame = np.full((47, 63, 3), index * 25, np.uint8)
        unchanged = frame.copy()
        visible = index >= 2 and index != 5
        roi = (8, 10, 40, 26) if index == 3 else (8, 10, 29, 37)
        pair.update(frame, index, '7' if index >= 2 else None, visible,
                    roi if visible else None, index / 25)
        assert np.array_equal(frame, unchanged)
    pair.finish('end_of_input')
    metadata = json.loads((tmp_path / 'paired_recording.json').read_text())
    assert metadata['status'] == 'saved' and metadata['stop_reason'] == 'target_lost'
    assert metadata['selected_track_id'] == '7'
    assert (metadata['first_source_frame'], metadata['last_source_frame']) == (2, 4)
    assert metadata['source_start_s'] == 2 / 25
    assert metadata['source_end_exclusive_s'] == 5 / 25
    originals, fps = decode(tmp_path / 'original.mp4')
    crops, crop_fps = decode(tmp_path / 'face_centered.mp4')
    assert len(originals) == len(crops) == metadata['frames_written'] == 3
    assert fps == crop_fps == 25
    assert metadata['duration_s'] == pytest.approx(3 / 25)
    assert all(image.shape == (48, 64, 3) for image in originals)
    assert all(image.shape == (28, 22, 3) for image in crops)
    for index, (original, crop) in enumerate(zip(originals, crops), 2):
        assert abs(float(original[20, 30].mean()) - index * 25) < 8
        assert abs(float(crop[14, 11].mean()) - index * 25) < 8
    assert crops[1][0].mean() < 8  # Wider ROI is letterboxed, rather than stretched.
    assert set(metadata['verification']) == {'original.mp4', 'face_centered.mp4'}


def test_no_video_created_without_selection(tmp_path):
    pair = recorder(tmp_path)
    pair.update(np.zeros((47, 63, 3), np.uint8), 0, None, False, None, 0)
    pair.finish('user_quit')
    assert not list(tmp_path.glob('*.mp4'))
    metadata = json.loads((tmp_path / 'paired_recording.json').read_text())
    assert metadata['status'] == 'not_recorded' and metadata['frames_written'] == 0


def test_new_selected_id_is_excluded_from_existing_pair(tmp_path):
    pair = recorder(tmp_path)
    frame = np.full((47, 63, 3), 90, np.uint8)
    roi = (8, 10, 29, 37)
    pair.update(frame, 0, '7', True, roi, 0)
    pair.update(frame, 1, '8', True, roi, 1 / 25)
    pair.finish('end_of_input')
    metadata = json.loads((tmp_path / 'paired_recording.json').read_text())
    assert metadata['stop_reason'] == 'target_changed' and metadata['frames_written'] == 1
    assert len(decode(tmp_path / 'original.mp4')[0]) == 1
    assert len(decode(tmp_path / 'face_centered.mp4')[0]) == 1


@pytest.mark.parametrize('index,timestamp', [(1, .5), (2, 2 / 25)])
def test_bad_timing_or_skipped_source_frame_is_rejected_without_discarding_valid_segment(tmp_path, index, timestamp):
    pair = recorder(tmp_path)
    frame = np.full((47, 63, 3), 90, np.uint8)
    roi = (8, 10, 29, 37)
    pair.update(frame, 0, '7', True, roi, 0)
    with pytest.raises(RuntimeError, match='CFR|skip source'):
        pair.update(frame, index, '7', True, roi, timestamp)
    pair.finish('run_error')
    assert len(decode(tmp_path / 'original.mp4')[0]) == 1
    assert len(decode(tmp_path / 'face_centered.mp4')[0]) == 1


def test_second_writer_failure_releases_both_and_reports_failure(tmp_path, monkeypatch):
    writers = []
    class Writer:
        def __init__(self, *args):
            self.good = not writers
            self.released = False
            writers.append(self)
        def isOpened(self):
            return self.good
        def release(self):
            self.released = True
    monkeypatch.setattr(cv2, 'VideoWriter', Writer)
    pair = recorder(tmp_path)
    with pytest.raises(RuntimeError, match='face_centered'):
        pair.update(np.zeros((47, 63, 3), np.uint8), 0, '7', True, (8, 10, 29, 37), 0)
    pair.finish('user_quit')
    assert len(writers) == 2 and all(writer.released for writer in writers)
    assert json.loads((tmp_path / 'paired_recording.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('stop,expected_count,reason', [
    ('loss', 2, 'target_lost'), ('quit', 2, 'user_quit'),
    ('reset', 2, 'selection_reset'), ('end', 4, 'end_of_input'), ('mark', 4, 'end_of_input'),
])
def test_desktop_pair_uses_real_video_clock_and_finalizes_on_each_stop(tmp_path, monkeypatch, stop, expected_count, reason):
    from c3_tracker import app
    from c3_tracker.tracking import Observation
    video = tmp_path / 'base.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'FFV1'), 10, (640, 300))
    assert writer.isOpened()
    for index in range(6):
        writer.write(np.full((300, 640, 3), 45 + 5 * index, np.uint8))
    writer.release()

    class Detector:
        def __init__(self, *args): pass
        def detect(self, frame): return []
        def close(self): pass
    class Tracker:
        def __init__(self, *args, **kwargs): pass
        def update(self, detections, frame, timestamp=None):
            observed = not (stop in ('loss','mark') and round(timestamp * 10) == 4)
            return [Observation('7', (220, 120, 280, 200), observed, True)]
        def protect_target(self, track_id): pass
    mouse = []
    events = iter(range(6))
    def wait_key(_):
        index = next(events)
        if index == 1:
            mouse[0](cv2.EVENT_LBUTTONDOWN, 240, 140, 0, None)
        if index == 3 and stop in ('quit', 'reset'):
            return ord('q' if stop == 'quit' else 'r')
        return -1
    monkeypatch.setattr(app, 'FaceDetector', Detector)
    monkeypatch.setattr(app, 'FaceTracker', Tracker)
    monkeypatch.setattr(cv2, 'namedWindow', lambda *args: None)
    monkeypatch.setattr(cv2, 'setMouseCallback', lambda window, callback: mouse.append(callback))
    monkeypatch.setattr(cv2, 'imshow', lambda *args: None)
    monkeypatch.setattr(cv2, 'waitKey', wait_key)
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *args: 1)
    monkeypatch.setattr(cv2, 'destroyAllWindows', lambda: None)
    options = ['--pair-loss-policy','mark'] if stop=='mark' else []
    assert app.main(['--source', 'video', '--input', str(video), '--record-pair',
                     '--output-dir', str(tmp_path / 'runs'), *options]) == 0
    output, = (tmp_path / 'runs').iterdir()
    metadata = json.loads((output / 'paired_recording.json').read_text())
    assert metadata['stop_reason'] == reason and metadata['status'] == 'saved'
    assert metadata['first_source_frame'] == 2 and metadata['last_source_frame'] == expected_count + 1
    originals, fps = decode(output / 'original.mp4')
    crops, crop_fps = decode(output / 'face_centered.mp4')
    assert len(originals) == len(crops) == expected_count
    assert fps == crop_fps == 10
    assert crops[0].shape == (96, 60, 3)
    if stop=='mark':
        assert metadata['lost_face_frames']==1 and crops[2].mean()<40
    assert originals[0][10, 10].mean() > 40  # No UI header drawn into either output.
    assert json.loads((output / 'summary.json').read_text())['paired_recording_status'] == 'saved'
