import pytest

from follower.drive import STOPPED, TrackSpeed
from follower.people import Person, TrackId
from follower.pursuit import PursuitGains, pursue

GAINS = PursuitGains(
    turn=0.8,
    forward=1.2,
    target_frame_fill=0.55,
    frame_fill_deadband=0.08,
    max_speed=TrackSpeed(20),
)


TURN_ONLY = PursuitGains(
    turn=0.8,
    forward=0.0,
    target_frame_fill=0.55,
    frame_fill_deadband=0.08,
    max_speed=TrackSpeed(20),
)


def _target(centre_x: float = 0.5, frame_fill: float = 0.55) -> Person:
    return Person(TrackId(1), centre_x=centre_x, frame_fill=frame_fill, area=0.1)


def test_a_centred_target_at_the_following_distance_needs_no_movement():
    assert pursue(_target(), GAINS) == STOPPED


def test_a_target_to_the_right_turns_the_tank_right():
    command = pursue(_target(centre_x=0.9), GAINS)

    assert command.left.percent > 0 > command.right.percent


def test_a_target_to_the_left_turns_the_tank_left():
    command = pursue(_target(centre_x=0.1), GAINS)

    assert command.right.percent > 0 > command.left.percent


def test_a_distant_target_is_approached():
    command = pursue(_target(frame_fill=0.2), GAINS)

    assert command.left.percent > 0
    assert command.left == command.right


def test_a_target_that_is_too_close_is_backed_away_from():
    command = pursue(_target(frame_fill=0.95), GAINS)

    assert command.left.percent < 0
    assert command.left == command.right


def test_distance_errors_inside_the_deadband_are_ignored():
    just_inside = GAINS.target_frame_fill - GAINS.frame_fill_deadband + 0.01

    assert pursue(_target(frame_fill=just_inside), GAINS) == STOPPED


def test_distance_errors_beyond_the_deadband_are_acted_on():
    just_outside = GAINS.target_frame_fill - GAINS.frame_fill_deadband - 0.01

    assert pursue(_target(frame_fill=just_outside), GAINS).left.percent > 0


@pytest.mark.parametrize("centre_x", [-0.2, 0.0, 0.3, 0.5, 0.8, 1.0, 1.3])
@pytest.mark.parametrize("frame_fill", [0.0, 0.3, 0.55, 1.0, 1.4])
def test_neither_track_ever_exceeds_the_speed_cap(centre_x, frame_fill):
    command = pursue(_target(centre_x, frame_fill), GAINS)

    assert abs(command.left.percent) <= GAINS.max_speed.percent
    assert abs(command.right.percent) <= GAINS.max_speed.percent


def test_turning_while_approaching_keeps_both_tracks_moving_forward_at_different_speeds():
    command = pursue(_target(centre_x=0.6, frame_fill=0.0), GAINS)

    assert command.left.percent > command.right.percent > 0


@pytest.mark.parametrize("frame_fill", [0.05, 0.55, 1.0])
def test_with_no_forward_gain_an_off_centre_target_is_turned_towards_on_the_spot(frame_fill):
    command = pursue(_target(centre_x=0.9, frame_fill=frame_fill), TURN_ONLY)

    assert command.left.percent > 0
    assert command.right.percent == -command.left.percent


@pytest.mark.parametrize("frame_fill", [0.05, 1.0])
def test_with_no_forward_gain_a_centred_target_needs_no_movement_at_any_distance(frame_fill):
    assert pursue(_target(centre_x=0.5, frame_fill=frame_fill), TURN_ONLY) == STOPPED
