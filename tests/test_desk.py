import pytest

from follower.operator import DriveMode, OperatorCommand
from teleop.desk import ControlDesk

STICK_STALE_AFTER_S = 0.25
IDLE = OperatorCommand.idle()


def _desk_with_a_page_open() -> ControlDesk:
    desk = ControlDesk(STICK_STALE_AFTER_S)
    desk.page_connected()
    return desk


def test_a_desk_nobody_has_touched_asks_for_nothing():
    assert _desk_with_a_page_open().command(now_s=0.0) == IDLE


def test_a_held_stick_is_passed_on():
    desk = _desk_with_a_page_open()

    desk.move_stick(throttle=0.6, steer=-0.2, now_s=1.0)

    assert desk.command(now_s=1.1) == OperatorCommand(DriveMode.MANUAL, 0.6, -0.2)


def test_a_stick_position_that_stops_being_repeated_is_dropped():
    desk = _desk_with_a_page_open()
    desk.move_stick(throttle=1.0, steer=0.0, now_s=1.0)

    assert desk.command(now_s=1.0 + STICK_STALE_AFTER_S) != IDLE
    assert desk.command(now_s=1.0 + STICK_STALE_AFTER_S + 0.01) == IDLE


def test_follow_mode_is_passed_on_with_a_centred_stick():
    desk = _desk_with_a_page_open()

    desk.select_mode(DriveMode.FOLLOW)

    assert desk.command(now_s=0.0) == OperatorCommand(DriveMode.FOLLOW, 0.0, 0.0)


def test_follow_mode_does_not_time_out_while_a_page_is_open():
    desk = _desk_with_a_page_open()
    desk.select_mode(DriveMode.FOLLOW)

    assert desk.command(now_s=3600.0).mode is DriveMode.FOLLOW


def test_touching_the_stick_takes_over_from_follow_mode():
    desk = _desk_with_a_page_open()
    desk.select_mode(DriveMode.FOLLOW)

    desk.move_stick(throttle=0.0, steer=0.5, now_s=2.0)

    assert desk.command(now_s=2.0) == OperatorCommand(DriveMode.MANUAL, 0.0, 0.5)


def test_stop_ends_follow_mode():
    desk = _desk_with_a_page_open()
    desk.select_mode(DriveMode.FOLLOW)

    desk.stop()

    assert desk.command(now_s=0.0) == IDLE


def test_stop_centres_a_held_stick():
    desk = _desk_with_a_page_open()
    desk.move_stick(throttle=1.0, steer=0.0, now_s=1.0)

    desk.stop()

    assert desk.command(now_s=1.0) == IDLE


def test_with_no_page_open_a_held_stick_is_ignored():
    desk = ControlDesk(STICK_STALE_AFTER_S)

    desk.move_stick(throttle=1.0, steer=0.0, now_s=1.0)

    assert desk.command(now_s=1.0) == IDLE


def test_follow_mode_ends_when_the_last_page_closes_and_stays_ended():
    desk = _desk_with_a_page_open()
    desk.select_mode(DriveMode.FOLLOW)

    desk.page_disconnected()
    assert desk.command(now_s=0.0) == IDLE

    desk.page_connected()
    assert desk.command(now_s=0.1) == IDLE


def test_follow_mode_ends_even_when_a_page_reconnects_before_the_controls_are_read():
    desk = _desk_with_a_page_open()
    desk.select_mode(DriveMode.FOLLOW)

    desk.page_disconnected()
    desk.page_connected()

    assert desk.command(now_s=0.0) == IDLE


def test_follow_mode_continues_while_another_page_is_still_open():
    desk = _desk_with_a_page_open()
    desk.page_connected()
    desk.select_mode(DriveMode.FOLLOW)

    desk.page_disconnected()

    assert desk.command(now_s=0.0).mode is DriveMode.FOLLOW


def test_a_stick_position_beyond_full_deflection_is_rejected():
    with pytest.raises(ValueError, match="throttle"):
        _desk_with_a_page_open().move_stick(throttle=1.5, steer=0.0, now_s=0.0)
