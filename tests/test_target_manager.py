from c3_tracker.target_manager import TargetManager
from c3_tracker.tracking import Observation


def obs(id='1', box=(0, 0, 100, 100), observed=True, confirmed=True):
    return Observation(id, box, observed, confirmed, .9 if observed else None)


def test_selection_and_overlap():
    t = TargetManager()
    assert t.status == 'NOT SELECTED'
    assert not t.select(200, 200, [obs()])
    assert t.select(25, 25, [obs(), obs('2', (10, 10, 40, 40))])
    assert t.target_id == '2'
    assert not t.select(100, 100, [obs()])
    assert t.target_id == '2'


def test_only_select_current_confirmed_tracks():
    t = TargetManager()
    assert not t.select(10, 10, [obs(observed=False), obs('2', confirmed=False)])


def test_loss_and_same_id_reacquisition():
    t = TargetManager()
    t.select(10, 10, [obs()])
    assert t.update([obs()]) == 'VISIBLE'
    assert t.update([obs(observed=False), obs('2')]) == 'LOST'
    assert not t.visible
    assert t.update([]) == 'LOST'
    assert t.update([obs('2')]) == 'LOST'
    assert t.target_id == '1'
    assert t.update([obs()]) == 'REACQUIRED'
    assert t.update([obs()]) == 'VISIBLE'


def test_unconfirmed_target_is_not_visible_and_reset():
    t = TargetManager()
    t.select(10, 10, [obs()])
    assert t.update([obs(confirmed=False)]) == 'LOST'
    t.reset()
    assert t.target_id is None
    assert t.update([obs()]) == 'NOT SELECTED'
