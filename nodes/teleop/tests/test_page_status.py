from messages.operator import DriveMode
from messages.status import DriverStatus, PilotStatus, TrackId
from teleop.server import page_status


def test_the_page_gets_one_status_merged_from_the_pilot_and_the_driver():
    status = page_status(
        PilotStatus(DriveMode.FOLLOW, locked_id=TrackId(7)),
        DriverStatus(armed=False, left_pct=12, right_pct=-12),
    )

    assert status == {
        "mode": "follow",
        "locked_track_id": 7,
        "left_pct": 12,
        "right_pct": -12,
        "armed": False,
    }


def test_nobody_being_followed_reaches_the_page_as_minus_one():
    status = page_status(
        PilotStatus(DriveMode.MANUAL, locked_id=None),
        DriverStatus(armed=False, left_pct=0, right_pct=0),
    )

    assert status["locked_track_id"] == -1
