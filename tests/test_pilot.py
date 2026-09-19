from follower.drive import STOPPED, DriveCommand, TrackSpeed
from follower.operator import DriveMode, OperatorCommand
from follower.people import TrackId
from follower.pilot import Pilot, StaleAfter
from follower.pursuit import PursuitGains
from follower.target import SelectionRule, TargetLock

STALE_AFTER = StaleAfter(detections_s=0.5, operator_s=0.5)
RELEASE_AFTER_S = 1.5
MANUAL_MAX_SPEED = TrackSpeed(50)
STOP = "stop"

FOLLOW = OperatorCommand(DriveMode.FOLLOW, throttle=0.0, steer=0.0)


class RecordingTank:
    def __init__(self):
        self.commands: list[DriveCommand | str] = []

    def drive(self, command: DriveCommand) -> None:
        self.commands.append(command)

    def stop(self) -> None:
        self.commands.append(STOP)


def _pilot(tank: RecordingTank) -> Pilot:
    return Pilot(
        lock=TargetLock(SelectionRule.LARGEST, RELEASE_AFTER_S),
        gains=PursuitGains(
            turn=0.8,
            forward=1.2,
            target_frame_fill=0.55,
            frame_fill_deadband=0.08,
            max_speed=TrackSpeed(20),
        ),
        manual_max_speed=MANUAL_MAX_SPEED,
        tank=tank,
        armed=False,
        min_confidence=0.5,
        stale_after=STALE_AFTER,
    )


def _following_pilot(tank: RecordingTank, now_s: float = 0.0) -> Pilot:
    pilot = _pilot(tank)
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


def test_nothing_moves_before_the_operator_is_heard_from():
    tank = RecordingTank()
    pilot = _pilot(tank)

    pilot.on_detections([_person_row(1)], now_s=0.0)

    assert tank.commands == []


def test_a_person_in_view_is_not_followed_in_manual_mode():
    tank = RecordingTank()
    pilot = _pilot(tank)
    pilot.on_operator(_manual(0.0, 0.0), now_s=0.0)
    tank.commands.clear()

    pilot.on_detections([_person_row(1)], now_s=0.1)

    assert tank.commands == []
    assert pilot.status.locked_id is None


def test_in_follow_mode_a_person_in_view_is_driven_towards():
    tank = RecordingTank()

    _following_pilot(tank).on_detections([_person_row(1)], now_s=0.1)

    (command,) = tank.commands
    assert command.left.percent > command.right.percent


def test_in_follow_mode_an_empty_frame_stops_the_tank():
    tank = RecordingTank()

    _following_pilot(tank).on_detections([], now_s=0.1)

    assert tank.commands == [STOP]


def test_the_tank_stops_while_the_followed_person_is_hidden():
    tank = RecordingTank()
    pilot = _following_pilot(tank)
    pilot.on_detections([_person_row(1)], now_s=0.1)

    pilot.on_detections([_person_row(2)], now_s=0.2)

    assert tank.commands[-1] == STOP


def test_the_stick_drives_the_tracks_up_to_the_manual_speed_cap():
    tank = RecordingTank()

    _pilot(tank).on_operator(_manual(throttle=1.0, steer=0.0), now_s=0.0)

    assert tank.commands == [DriveCommand(MANUAL_MAX_SPEED, MANUAL_MAX_SPEED)]


def test_full_steer_pivots_the_tank_on_the_spot():
    tank = RecordingTank()

    _pilot(tank).on_operator(_manual(throttle=0.0, steer=1.0), now_s=0.0)

    assert tank.commands == [DriveCommand(left=TrackSpeed(50), right=TrackSpeed(-50))]


def test_a_centred_stick_stops_the_tank():
    tank = RecordingTank()

    _pilot(tank).on_operator(_manual(0.0, 0.0), now_s=0.0)

    assert tank.commands == [STOP]


def test_taking_the_stick_drops_the_person_being_followed():
    pilot = _following_pilot(RecordingTank())
    pilot.on_detections([_person_row(4)], now_s=0.1)

    pilot.on_operator(_manual(0.5, 0.0), now_s=0.2)

    assert pilot.status.locked_id is None


def test_a_tick_stops_the_tank_when_the_operator_goes_quiet_in_manual_mode():
    tank = RecordingTank()
    pilot = _pilot(tank)
    pilot.on_operator(_manual(1.0, 0.0), now_s=0.0)

    pilot.on_tick(now_s=STALE_AFTER.operator_s + 0.01)

    assert tank.commands[-1] == STOP
    assert pilot.status.drive == STOPPED


def test_a_tick_stops_the_tank_when_the_operator_goes_quiet_in_follow_mode():
    tank = RecordingTank()
    pilot = _following_pilot(tank)
    pilot.on_detections([_person_row(1)], now_s=0.1)

    pilot.on_tick(now_s=STALE_AFTER.operator_s + 0.01)

    assert tank.commands[-1] == STOP


def test_detections_are_not_acted_on_once_the_operator_has_gone_quiet():
    tank = RecordingTank()
    pilot = _following_pilot(tank)

    pilot.on_detections([_person_row(1)], now_s=STALE_AFTER.operator_s + 0.01)

    assert tank.commands == []


def test_a_tick_leaves_the_tank_driving_while_the_operator_is_still_heard():
    tank = RecordingTank()
    pilot = _pilot(tank)
    pilot.on_operator(_manual(1.0, 0.0), now_s=0.0)

    pilot.on_tick(now_s=STALE_AFTER.operator_s)

    assert tank.commands[-1] != STOP


def test_a_tick_stops_the_tank_when_detections_go_stale_in_follow_mode():
    tank = RecordingTank()
    pilot = _following_pilot(tank)
    pilot.on_detections([_person_row(1)], now_s=0.0)
    pilot.on_operator(FOLLOW, now_s=0.4)

    pilot.on_tick(now_s=STALE_AFTER.detections_s + 0.01)

    assert tank.commands[-1] == STOP


def test_stale_detections_do_not_interrupt_manual_driving():
    tank = RecordingTank()
    pilot = _pilot(tank)
    pilot.on_operator(_manual(1.0, 0.0), now_s=10.0)

    pilot.on_tick(now_s=10.1)

    assert tank.commands[-1] != STOP


def test_starting_to_follow_someone_is_announced():
    pilot = _following_pilot(RecordingTank())

    assert pilot.on_detections([_person_row(4)], now_s=0.1) == "Following person 4"


def test_continuing_to_follow_the_same_person_is_not_announced_again():
    pilot = _following_pilot(RecordingTank())
    pilot.on_detections([_person_row(4)], now_s=0.1)

    assert pilot.on_detections([_person_row(4)], now_s=0.2) is None


def test_losing_the_followed_person_is_announced_once_the_lock_releases():
    pilot = _following_pilot(RecordingTank())
    pilot.on_detections([_person_row(4)], now_s=0.0)

    pilot.on_operator(FOLLOW, now_s=0.9)
    assert pilot.on_detections([], now_s=1.0) is None

    pilot.on_operator(FOLLOW, now_s=1.4)
    assert pilot.on_detections([], now_s=RELEASE_AFTER_S) == "Lost person 4"


def test_the_status_reports_mode_target_and_track_speeds():
    pilot = _following_pilot(RecordingTank())
    pilot.on_detections([_person_row(4)], now_s=0.1)

    status = pilot.status

    assert status.mode is DriveMode.FOLLOW
    assert status.locked_id == TrackId(4)
    assert status.drive.left.percent > status.drive.right.percent
    assert status.armed is False
