"""Opt-in actual model tests; sample-photo fixtures do not validate person identity."""
import csv
import json
import os
from pathlib import Path
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('C3_RUN_MODEL_TESTS') != '1',
                                reason='opt-in pretrained model smoke checks')


def sample_frame():
    import cv2
    import matplotlib
    image = cv2.imread(str(Path(matplotlib.get_data_path()) / 'sample_data' / 'grace_hopper.jpg'))
    assert image is not None
    face = cv2.resize(image, (320, 480))
    return cv2.hconcat([face, face])


def test_actual_models_observation_loss_and_expiry():
    import numpy as np
    from c3_tracker.detection import FaceDetector
    from c3_tracker.tracking import FaceTracker
    from c3_tracker.target_manager import TargetManager
    frame = sample_frame()
    detector = FaceDetector()
    try:
        detections = detector.detect(frame)
        assert len(detections) == 2
        tracker = FaceTracker(max_age=2)
        for _ in range(3):
            observations = tracker.update(detections, frame)
        assert len(observations) == 2
        assert all(o.observed and o.confirmed for o in observations)
        target = TargetManager()
        l, t, r, b = observations[0].bbox
        assert target.select((l + r) / 2, (t + b) / 2, observations)
        raw_id = target.target_id
        blank = np.zeros_like(frame)
        assert detector.detect(blank) == []
        predictions = tracker.update([], blank)
        assert predictions and all(not o.observed for o in predictions)
        assert target.update(predictions) == 'LOST'
        observations = tracker.update(detections, frame)
        assert target.update(observations) == 'REACQUIRED'
        assert target.target_id == raw_id
        for _ in range(3):
            observations = tracker.update([], blank)
            target.update(observations)
        assert observations == []
        for _ in range(3):
            observations = tracker.update(detections, frame)
        assert raw_id not in {o.track_id for o in observations}
        assert target.update(observations) == 'LOST'
    finally:
        detector.close()


def test_headless_video_artifacts(tmp_path):
    import cv2
    from c3_tracker.app import main
    frame = sample_frame()
    video = tmp_path / 'input.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))
    assert writer.isOpened()
    for _ in range(8):
        writer.write(frame)
    writer.release()
    output = tmp_path / 'results'
    assert main(['--source', 'video', '--input', str(video), '--headless',
                 '--record', '--record-raw', '--diagnostics', '--output-dir', str(output)]) == 0
    run, = output.iterdir()
    summary = json.loads((run / 'summary.json').read_text())
    assert summary['frames_processed'] == 8 and summary['error'] is None
    assert summary['end_to_end_processing_fps'] > 0
    assert (run / 'run_metadata.json').is_file()
    with (run / 'tracking_log.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len({row['frame_index'] for row in rows}) == 8
    assert len({row['track_id'] for row in rows}) == 2
    assert any(row['confirmed'] == '1' for row in rows)
    diagnostic_rows = [json.loads(line) for line in (run / 'association_diagnostics.jsonl').read_text().splitlines()]
    assert [row['frame_index'] for row in diagnostic_rows] == list(range(8))
    assert all(row['detected_faces'] == 2 and row['alignable_faces'] == 2 for row in diagnostic_rows)
    assert all(row['state'] == 'not_selected' for row in diagnostic_rows)
    cap = cv2.VideoCapture(str(run / 'annotated_video.mp4'))
    count = 0
    annotated_first = None
    while True:
        ok, annotated = cap.read()
        if not ok:
            break
        if annotated_first is None:
            annotated_first = annotated
        count += 1
    cap.release()
    assert count == 8
    source_cap = cv2.VideoCapture(str(video))
    ok, original_first = source_cap.read()
    source_cap.release()
    assert ok
    raw_cap = cv2.VideoCapture(str(run / 'source_video.mp4'))
    count = 0
    raw_first = None
    while True:
        ok, raw = raw_cap.read()
        if not ok:
            break
        if raw_first is None:
            raw_first = raw
        count += 1
    raw_cap.release()
    assert count == 8
    import numpy as np
    source_error = np.abs(raw_first.astype(float) - original_first.astype(float)).mean()
    annotated_error = np.abs(annotated_first.astype(float) - original_first.astype(float)).mean()
    assert source_error < annotated_error  # source recording preserves the unannotated input


def test_actual_models_keep_ids_after_gap_and_translation():
    import cv2
    import numpy as np
    from c3_tracker.detection import FaceDetector
    from c3_tracker.tracking import FaceTracker
    frame = sample_frame()
    detector = FaceDetector(confidence=.65)
    try:
        tracker = FaceTracker()
        for _ in range(4):
            original = tracker.update(detector.detect(frame), frame)
        original_ids = {o.track_id for o in original if o.confirmed and o.observed}
        assert len(original_ids) == 2
        blank = np.zeros_like(frame)
        for _ in range(2):
            tracker.update([], blank)
        moved = cv2.warpAffine(frame, np.float32([[1, 0, 45], [0, 1, 0]]), (640, 480))
        returned = tracker.update(detector.detect(moved), moved)
        assert {o.track_id for o in returned if o.observed} == original_ids
        assert len(returned) == 2
    finally:
        detector.close()


def test_actual_models_one_face_returns_after_one_second_while_other_stays():
    import cv2
    import numpy as np
    from c3_tracker.detection import FaceDetector
    from c3_tracker.tracking import FaceTracker
    frame = sample_frame()
    detector = FaceDetector(confidence=.65)
    try:
        tracker = FaceTracker()
        for index in range(4):
            original = tracker.update(detector.detect(frame), frame, timestamp=index / 60)
        original = sorted((o for o in original if o.confirmed and o.observed), key=lambda o: o.bbox[0])
        assert len(original) == 2
        first, second = [o.track_id for o in original]
        hidden = frame.copy()
        hidden[:, :320] = 0
        remaining = detector.detect(hidden)
        assert len(remaining) == 1
        for index in range(60):
            observed = tracker.update(remaining, hidden, timestamp=(index + 4) / 60)
            assert [o.track_id for o in observed if o.observed] == [second]
        moved = frame.copy()
        moved[:, :320] = cv2.warpAffine(frame[:, :320], np.float32([[1, 0, 45], [0, 1, 0]]), (320, 480))
        returned = tracker.update(detector.detect(moved), moved, timestamp=64 / 60)
        assert len(returned) == 2
        assert {o.track_id for o in returned if o.observed and o.confirmed} == {first, second}
    finally:
        detector.close()


def test_different_real_face_cannot_take_selected_id_and_original_can_return():
    import cv2
    import numpy as np
    from c3_tracker.detection import FaceDetector
    from c3_tracker.tracking import FaceTracker
    from c3_tracker.target_manager import TargetManager
    other_path = Path('.models/testdata/other_face.jpg')
    assert other_path.is_file(), 'Run python -m c3_tracker.models --download --test-fixtures'
    other = cv2.imdecode(np.frombuffer(other_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    first_image = sample_frame()[:, :320]
    other = cv2.copyMakeBorder(cv2.resize(other[130:450, 140:450], (320, 320)),
                              80, 80, 0, 0, cv2.BORDER_CONSTANT)
    frame = cv2.hconcat([first_image, other])
    detector = FaceDetector(.65)
    try:
        tracker = FaceTracker()
        for index in range(4):
            observations = tracker.update(detector.detect(frame), frame, timestamp=index / 10)
        left, right = sorted((o for o in observations if o.confirmed and o.observed), key=lambda o: o.bbox[0])
        target = TargetManager()
        l, t, r, b = left.bbox
        assert target.select((l + r) / 2, (t + b) / 2, observations)
        tracker.protect_target(target.target_id)
        # The other face occupies the selected face's previous area, rather
        # than simply staying far away while the selected person disappears.
        replacement = cv2.hconcat([other, np.zeros_like(other)])
        for index in range(10):
            observations = tracker.update(detector.detect(replacement), replacement, timestamp=(index + 4) / 10)
            assert target.update(observations) == 'LOST'
            assert not any(o.observed and o.track_id == left.track_id for o in observations)
        for index in range(3):
            observations = tracker.update(detector.detect(frame), frame, timestamp=(index + 14) / 10)
            status = target.update(observations)
            assert status == ('REACQUIRED' if index == 2 else 'LOST')
        assert target.target_id == left.track_id
        visible_target, = [o for o in observations if o.observed and o.track_id == target.target_id]
        assert visible_target.bbox[0] < 320
    finally:
        detector.close()
