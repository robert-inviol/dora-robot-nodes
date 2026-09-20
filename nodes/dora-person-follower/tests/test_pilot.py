from dora_person_follower.pilot import Pilot, StaleAfter
from dora_person_follower.pursuit import PursuitGains
from dora_person_follower.target import SelectionRule, TargetLock
from dora_rig_messages.drive import STOPPED, DriveDemand, PowerCap
from dora_rig_messages.operator import DriveMode, OperatorCommand
from dora_rig_messages.status import PilotStatus, TrackId

STALE_AFTER = StaleAfter(detections_s=0.5, operator_s=0.5)
RELEASE_AFTER_S = 1.5
MANUAL_MAX_POWER = PowerCap(50)

FOLLOW = OperatorCommand(DriveMode.FOLLOW, throttle=0.0, steer=0.0)


def _pilot() -> Pilot:
    return Pilot(
        lock=TargetLock(SelectionRule.LARGEST, RELEASE_AFTER_S),
        gains=PursuitGains(
            turn=0.8,
            forward=1.2,
            target_frame_fill=0.55,
            frame_fill_deadband=0.08,
            max_power=PowerCap(20),
        ),
        manual_max_power=MANUAL_MAX_POWER,
        min_confidence=0.5,
        stale_after=STALE_AFTER,
    )


def _following_pilot(now_s: float = 0.0) -> Pilot:
    pilot = _pilot()
    pilot.on_operator(FOLLOW, now_s)
    return pilot


def _manual(throttle: float, steer: float) -> OperatorCommand:
    return OperatorCommand(DriveMode.MANUAL, throttle, steer)


def _person_row(track_id: int, x: float = 0.7, h: float = 0.2) -> dict:
    return {
        "track_id": track_id,
        "class_name": "person",
        "confidence": 0.9,
        "x": x,
        "y": 0.1,
        "w": 0.2,
        "h": h,
    }


def test_nothing_is_asked_for_before_the_operator_is_heard_from():
    pilot = _pilot()

    pilot.on_detections([_person_row(1)], now_s=0.0)

    assert pilot.demand == STOPPED


def test_a_person_in_view_is_not_followed_in_manual_mode():
    pilot = _pilot()
    pilot.on_operator(_manual(0.0, 0.0), now_s=0.0)

    pilot.on_detections([_person_row(1)], now_s=0.1)

    assert pilot.demand == STOPPED
    assert pilot.status.locked_id is None


def test_in_follow_mode_a_person_to_the_right_is_turned_towards():
    pilot = _following_pilot()

    pilot.on_detections([_person_row(1)], now_s=0.1)

    assert pilot.demand.turn > 0


def test_in_follow_mode_an_empty_frame_asks_for_a_stop():
    pilot = _following_pilot()
    pilot.on_detections([_person_row(1)], now_s=0.1)

    pilot.on_detections([], now_s=0.2)

    assert pilot.demand == STOPPED


def test_a_stop_is_asked_for_while_the_followed_person_is_hidden():
    pilot = _following_pilot()
    pilot.on_detections([_person_row(1)], now_s=0.1)

    pilot.on_detections([_person_row(2)], now_s=0.2)

    assert pilot.demand == STOPPED


def test_a_full_stick_asks_for_the_manual_power_cap():
    pilot = _pilot()

    pilot.on_operator(_manual(throttle=1.0, steer=0.0), now_s=0.0)

    assert pilot.demand == DriveDemand(forward=0.5, turn=0.0)


def test_full_steer_asks_for_a_turn_on_the_spot():
    pilot = _pilot()

    pilot.on_operator(_manual(throttle=0.0, steer=1.0), now_s=0.0)

    assert pilot.demand == DriveDemand(forward=0.0, turn=0.5)


def test_a_centred_stick_asks_for_a_stop():
    pilot = _pilot()
    pilot.on_operator(_manual(1.0, 0.0), now_s=0.0)

    pilot.on_operator(_manual(0.0, 0.0), now_s=0.1)

    assert pilot.demand == STOPPED


def test_taking_the_stick_drops_the_person_being_followed():
    pilot = _following_pilot()
    pilot.on_detections([_person_row(4)], now_s=0.1)

    pilot.on_operator(_manual(0.5, 0.0), now_s=0.2)

    assert pilot.status.locked_id is None


def test_a_tick_asks_for_a_stop_when_the_operator_goes_quiet_in_manual_mode():
    pilot = _pilot()
    pilot.on_operator(_manual(1.0, 0.0), now_s=0.0)

    pilot.on_tick(now_s=STALE_AFTER.operator_s + 0.01)

    assert pilot.demand == STOPPED


def test_a_tick_asks_for_a_stop_when_the_operator_goes_quiet_in_follow_mode():
    pilot = _following_pilot()
    pilot.on_detections([_person_row(1)], now_s=0.1)

    pilot.on_tick(now_s=STALE_AFTER.operator_s + 0.01)

    assert pilot.demand == STOPPED


def test_detections_are_not_acted_on_once_the_operator_has_gone_quiet():
    pilot = _following_pilot()

    pilot.on_detections([_person_row(1)], now_s=STALE_AFTER.operator_s + 0.01)

    assert pilot.demand == STOPPED


def test_a_tick_leaves_the_demand_alone_while_the_operator_is_still_heard():
    pilot = _pilot()
    pilot.on_operator(_manual(1.0, 0.0), now_s=0.0)

    pilot.on_tick(now_s=STALE_AFTER.operator_s)

    assert pilot.demand != STOPPED


def test_a_tick_asks_for_a_stop_when_detections_go_stale_in_follow_mode():
    pilot = _following_pilot()
    pilot.on_detections([_person_row(1)], now_s=0.0)
    pilot.on_operator(FOLLOW, now_s=0.4)

    pilot.on_tick(now_s=STALE_AFTER.detections_s + 0.01)

    assert pilot.demand == STOPPED


def test_stale_detections_do_not_interrupt_manual_driving():
    pilot = _pilot()
    pilot.on_operator(_manual(1.0, 0.0), now_s=10.0)

    pilot.on_tick(now_s=10.1)

    assert pilot.demand != STOPPED


def test_starting_to_follow_someone_is_announced():
    pilot = _following_pilot()

    assert pilot.on_detections([_person_row(4)], now_s=0.1) == "Following person 4"


def test_continuing_to_follow_the_same_person_is_not_announced_again():
    pilot = _following_pilot()
    pilot.on_detections([_person_row(4)], now_s=0.1)

    assert pilot.on_detections([_person_row(4)], now_s=0.2) is None


def test_losing_the_followed_person_is_announced_once_the_lock_releases():
    pilot = _following_pilot()
    pilot.on_detections([_person_row(4)], now_s=0.0)

    pilot.on_operator(FOLLOW, now_s=0.9)
    assert pilot.on_detections([], now_s=1.0) is None

    pilot.on_operator(FOLLOW, now_s=1.4)
    assert pilot.on_detections([], now_s=RELEASE_AFTER_S) == "Lost person 4"


def test_the_status_reports_the_mode_and_who_is_followed():
    pilot = _following_pilot()
    pilot.on_detections([_person_row(4)], now_s=0.1)

    assert pilot.status == PilotStatus(DriveMode.FOLLOW, locked_id=TrackId(4))
