"""Desktop actions without relying on manual keyboard timing or a webcam."""
import csv
import json
import cv2
import numpy as np
from c3_tracker.app import main
from c3_tracker.controls import Controls, SAVE_BUTTON
from c3_tracker.target_manager import TargetManager
from c3_tracker.tracking import Observation
from c3_tracker.visualization import annotate


def test_screenshot_png_and_feedback_with_unicode_directory(tmp_path, capsys):
    directory = tmp_path / '\u05ea\u05de\u05d5\u05e0\u05d5\u05ea'
    directory.mkdir()
    controls = Controls(directory)
    image = np.full((120, 200, 3), 90, dtype=np.uint8)
    path = controls.save(image, 17)
    assert path.is_file()
    decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(decoded, image)
    assert 'Saved:' in controls.message
    assert str(path) in capsys.readouterr().out
    assert controls.save(image, 17) != path  # no overwrite on repeated saves


def test_screenshot_write_error_is_visible_and_does_not_crash(tmp_path):
    controls = Controls(tmp_path / 'missing-folder')
    controls.request_save()
    assert controls.save(np.zeros((20, 20, 3), dtype=np.uint8), 0) is None
    assert 'failed' in controls.message
    assert not controls.save_requested


def test_open_folder_button_and_face_click_are_separate(tmp_path, monkeypatch):
    from c3_tracker.controls import FOLDER_BUTTON
    controls = Controls(tmp_path)
    opened = []
    monkeypatch.setattr(controls, 'open_folder', lambda: opened.append(controls.directory))
    assert controls.click(FOLDER_BUTTON[0] + 3, FOLDER_BUTTON[1] + 3)
    assert opened == [tmp_path.resolve()]
    assert not controls.click(200, 200)


def test_clean_overlay_hides_unobserved_and_unconfirmed_boxes():
    image = np.zeros((300, 640, 3), dtype=np.uint8)
    tracks = [Observation('1', (20, 120, 80, 200), False, True),
              Observation('2', (120, 120, 180, 200), True, False),
              Observation('3', (220, 120, 280, 200), True, True)]
    clean = annotate(image, [], tracks, TargetManager(), 20)
    debug = annotate(image, [], tracks, TargetManager(), 20, debug=True)
    assert not clean[120:201, 20:81].any()
    assert not clean[120:201, 120:181].any()
    assert clean[120:201, 220:281].any()
    assert debug[120:201, 20:81].any()
    assert debug[120:201, 120:181].any()


def test_desktop_loop_saves_and_protects_target_selection(tmp_path, monkeypatch):
    from c3_tracker import app
    image = np.full((300, 640, 3), 100, dtype=np.uint8)

    class Capture:
        released = False

        def read(self):
            return True, image.copy()

        def get(self, prop):
            return 30

        def release(self):
            self.released = True

    class Detector:
        def __init__(self, *args):
            pass

        def detect(self, frame):
            return []

        def close(self):
            pass

    class Tracker:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, detections, frame, timestamp=None):
            # Stale tracks must remain in the log even though the UI hides them.
            return [Observation('7', (20, 120, 80, 200), False, True),
                    Observation('8', (220, 120, 280, 200), True, True)]

        def protect_target(self, track_id):
            protected.append(track_id)

    capture = Capture()
    mouse = []
    events = iter(['s', ' ', 'select', 'r', 'click', 'q'])
    displayed = []
    protected = []

    def wait_key(_):
        event = next(events)
        if event == 'select':
            mouse[0](cv2.EVENT_LBUTTONDOWN, 240, 140, 0, None)
            return -1
        if event == 'click':
            mouse[0](cv2.EVENT_LBUTTONDOWN, SAVE_BUTTON[0] + 5, SAVE_BUTTON[1] + 5, 0, None)
            return -1
        return ord(event)

    monkeypatch.setattr(app, 'open_capture', lambda args: capture)
    monkeypatch.setattr(app, 'FaceDetector', Detector)
    monkeypatch.setattr(app, 'FaceTracker', Tracker)
    monkeypatch.setattr(cv2, 'namedWindow', lambda *args: None)
    monkeypatch.setattr(cv2, 'setMouseCallback', lambda window, callback: mouse.append(callback))
    monkeypatch.setattr(cv2, 'imshow', lambda window, frame: displayed.append(frame.copy()))
    monkeypatch.setattr(cv2, 'waitKey', wait_key)
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *args: 1)
    monkeypatch.setattr(cv2, 'destroyAllWindows', lambda: None)
    assert main(['--source', 'webcam', '--output-dir', str(tmp_path)]) == 0
    run, = tmp_path.iterdir()
    shots = sorted(run.glob('screenshot_*.png'))
    assert len(shots) == 3
    for i, shot in zip((0, 1, 4), shots):
        decoded = cv2.imdecode(np.frombuffer(shot.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(decoded, displayed[i])
    with (run / 'tracking_log.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    assert all(row['has_detection'] == '0' for row in rows if row['track_id'] == '7')
    assert protected == ['8', None]
    assert json.loads((run / 'summary.json').read_text())['error'] is None
    assert capture.released
