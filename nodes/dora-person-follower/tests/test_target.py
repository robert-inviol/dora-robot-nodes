from dora_person_follower.people import Person
from dora_person_follower.target import SelectionRule, TargetLock
from dora_rig_messages.status import TrackId

RELEASE_AFTER_S = 1.5


def _person(track_id: int, centre_x: float = 0.5, size: float = 0.3) -> Person:
    return Person(TrackId(track_id), centre_x=centre_x, frame_fill=size, area=size * size)


def _lock(rule: SelectionRule = SelectionRule.LARGEST) -> TargetLock:
    return TargetLock(rule, RELEASE_AFTER_S)


def test_nobody_in_view_means_no_target():
    lock = _lock()

    assert lock.update([], now_s=0.0) is None
    assert lock.locked_id is None


def test_largest_rule_picks_the_biggest_person():
    near, far = _person(1, size=0.8), _person(2, size=0.2)

    assert _lock(SelectionRule.LARGEST).update([far, near], now_s=0.0) == near


def test_most_central_rule_picks_the_person_nearest_the_middle():
    edge, middle = _person(1, centre_x=0.9), _person(2, centre_x=0.45)

    assert _lock(SelectionRule.MOST_CENTRAL).update([edge, middle], now_s=0.0) == middle


def test_first_seen_rule_picks_the_lowest_track_id():
    earlier, later = _person(3), _person(8)

    assert _lock(SelectionRule.FIRST_SEEN).update([later, earlier], now_s=0.0) == earlier


def test_a_bigger_newcomer_does_not_steal_the_lock():
    lock = _lock()
    followed = _person(1, size=0.3)
    lock.update([followed], now_s=0.0)

    assert lock.update([followed, _person(2, size=0.9)], now_s=0.1) == followed


def test_while_the_locked_person_is_briefly_hidden_nobody_is_followed():
    lock = _lock()
    lock.update([_person(1)], now_s=0.0)

    assert lock.update([_person(2)], now_s=RELEASE_AFTER_S - 0.1) is None
    assert lock.locked_id == TrackId(1)


def test_the_locked_person_is_picked_up_again_when_they_reappear():
    lock = _lock()
    followed = _person(1)
    lock.update([followed], now_s=0.0)
    lock.update([], now_s=1.0)

    assert lock.update([followed, _person(2, size=0.9)], now_s=1.2) == followed


def test_after_the_release_time_someone_else_is_chosen():
    lock = _lock()
    lock.update([_person(1)], now_s=0.0)
    newcomer = _person(2)

    assert lock.update([newcomer], now_s=RELEASE_AFTER_S) == newcomer
    assert lock.locked_id == TrackId(2)


def test_after_the_release_time_with_nobody_in_view_the_lock_is_dropped():
    lock = _lock()
    lock.update([_person(1)], now_s=0.0)

    assert lock.update([], now_s=RELEASE_AFTER_S) is None
    assert lock.locked_id is None
