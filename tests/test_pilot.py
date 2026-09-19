from follower.pilot import Pilot
from follower.pursuit import DriveCommand, PursuitGains, TrackSpeed
from follower.target import SelectionRule, TargetLock

STALE_AFTER_S = 0.5
RELEASE_AFTER_S = 1.5
STOP = "stop"


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
        tank=tank,
        min_confidence=0.5,
        stale_detections_after_s=STALE_AFTER_S,
    )


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


def test_a_person_in_view_is_driven_towards():
    tank = RecordingTank()

    _pilot(tank).on_detections([_person_row(1)], now_s=0.0)

    (command,) = tank.commands
    assert command.left.percent > command.right.percent


def test_an_empty_frame_stops_the_tank():
    tank = RecordingTank()

    _pilot(tank).on_detections([], now_s=0.0)

    assert tank.commands == [STOP]


def test_the_tank_stops_while_the_followed_person_is_hidden():
    tank = RecordingTank()
    pilot = _pilot(tank)
    pilot.on_detections([_person_row(1)], now_s=0.0)

    pilot.on_detections([_person_row(2)], now_s=0.1)

    assert tank.commands[-1] == STOP


def test_starting_to_follow_someone_is_announced():
    assert _pilot(RecordingTank()).on_detections([_person_row(4)], now_s=0.0) == "Following person 4"


def test_continuing_to_follow_the_same_person_is_not_announced_again():
    pilot = _pilot(RecordingTank())
    pilot.on_detections([_person_row(4)], now_s=0.0)

    assert pilot.on_detections([_person_row(4)], now_s=0.1) is None


def test_losing_the_followed_person_is_announced_once_the_lock_releases():
    pilot = _pilot(RecordingTank())
    pilot.on_detections([_person_row(4)], now_s=0.0)

    assert pilot.on_detections([], now_s=1.0) is None
    assert pilot.on_detections([], now_s=RELEASE_AFTER_S) == "Lost person 4"


def test_switching_to_a_new_person_is_announced():
    pilot = _pilot(RecordingTank())
    pilot.on_detections([_person_row(4)], now_s=0.0)

    assert pilot.on_detections([_person_row(9)], now_s=RELEASE_AFTER_S) == "Following person 9"


def test_a_tick_stops_the_tank_when_detections_have_gone_stale():
    tank = RecordingTank()
    pilot = _pilot(tank)
    pilot.on_detections([_person_row(1)], now_s=0.0)

    pilot.on_tick(now_s=STALE_AFTER_S + 0.01)

    assert tank.commands[-1] == STOP


def test_a_tick_leaves_the_tank_driving_while_detections_are_fresh():
    tank = RecordingTank()
    pilot = _pilot(tank)
    pilot.on_detections([_person_row(1)], now_s=0.0)

    pilot.on_tick(now_s=STALE_AFTER_S)

    assert tank.commands[-1] != STOP


def test_a_tick_before_any_detections_arrive_stops_the_tank():
    tank = RecordingTank()

    _pilot(tank).on_tick(now_s=0.0)

    assert tank.commands == [STOP]
