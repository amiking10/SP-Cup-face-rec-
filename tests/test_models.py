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
                 '--record', '--output-dir', str(output)]) == 0
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
    cap = cv2.VideoCapture(str(run / 'annotated_video.mp4'))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    assert count == 8
