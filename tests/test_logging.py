import csv
import json
from c3_tracker.logging_utils import RunLogger, FIELDS
from c3_tracker.target_manager import TargetManager
from c3_tracker.tracking import Observation
from c3_tracker.evaluation import evaluate


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
    result = evaluate(logger.directory / 'tracking_log.csv')
    assert result['frames'] == 2 and result['latency_mean_ms'] == 15
    assert isinstance(result['identity_metrics'], str)


def test_independent_annotations(tmp_path):
    logger = RunLogger(tmp_path, {})
    t = TargetManager()
    o = Observation('1', (0, 0, 50, 50), True, True)
    t.select(10, 10, [o])
    for i in range(3):
        logger.log_frame(i, i / 25, [o], t, 1)
    logger.close({})
    annotations = tmp_path / 'annotations.csv'
    annotations.write_text('frame_index,target_visible,target_person_id,selected_person_id\n'
                           '0,1,A,A\n1,1,A,B\n2,1,A,A\n')
    result = evaluate(logger.directory / 'tracking_log.csv', annotations)
    assert result['target_recall'] == 2 / 3
    assert result['false_target_assignment_rate'] == 1 / 3
    assert result['selected_person_switches'] == 2
