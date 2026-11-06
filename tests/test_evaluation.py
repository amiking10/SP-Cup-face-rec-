"""Research evaluation stays outside the runtime integration tests."""
from c3_tracker.evaluation import evaluate
from c3_tracker.logging_utils import RunLogger
from c3_tracker.target_manager import TargetManager
from c3_tracker.tracking import Observation


def test_summary_of_a_log_with_an_empty_frame(tmp_path):
    logger = RunLogger(tmp_path, {})
    target = TargetManager()
    observation = Observation('7', (1, 2, 30, 40), True, True, .9)
    target.select(5, 5, [observation])
    logger.log_frame(0, 0, [observation], target, 10)
    target.update([])
    logger.log_frame(1, .04, [], target, 20)
    logger.close({'frames_processed': 2})
    result = evaluate(logger.directory / 'tracking_log.csv')
    assert result['frames'] == 2 and result['latency_mean_ms'] == 15
    assert isinstance(result['identity_metrics'], str)


def test_independent_annotations(tmp_path):
    logger = RunLogger(tmp_path, {})
    target = TargetManager()
    observation = Observation('1', (0, 0, 50, 50), True, True)
    target.select(10, 10, [observation])
    for i in range(3):
        logger.log_frame(i, i / 25, [observation], target, 1)
    logger.close({})
    annotations = tmp_path / 'annotations.csv'
    annotations.write_text('frame_index,target_visible,target_person_id,selected_person_id\n'
                           '0,1,A,A\n1,1,A,B\n2,1,A,A\n')
    result = evaluate(logger.directory / 'tracking_log.csv', annotations)
    assert result['target_recall'] == 2 / 3
    assert result['false_target_assignment_rate'] == 1 / 3
    assert result['selected_person_switches'] == 2
