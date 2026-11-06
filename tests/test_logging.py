import csv
import json
from c3_tracker.logging_utils import RunLogger, FIELDS
from c3_tracker.target_manager import TargetManager
from c3_tracker.tracking import Observation


def test_schema_and_empty_frame(tmp_path):
    logger = RunLogger(tmp_path, {'resolution': [640, 480]})
    t = TargetManager()
    o = Observation('7', (1, 2, 30, 40), True, True, .9)
    t.select(5, 5, [o])
    logger.log_frame(0, 0, [o], t, 10)
    t.update([])
    logger.log_frame(1, .04, [], t, 20)
    logger.close({'frames_processed': 2})
    with (logger.directory / 'tracking_log.csv').open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == FIELDS
        rows = list(reader)
    assert rows[0]['is_target'] == '1'
    assert rows[1]['track_id'] == ''
    assert rows[1]['selected_target_id'] == '7'
    assert rows[1]['target_lost'] == '1'
    assert json.loads((logger.directory / 'summary.json').read_text())['frames_processed'] == 2
