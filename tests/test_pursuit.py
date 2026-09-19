import pytest

from follower.people import Person
from follower.pursuit import PursuitGains, pursue
from messages.drive import STOPPED, PowerCap
from messages.status import TrackId

GAINS = PursuitGains(
    turn=0.8,
    forward=1.2,
    target_frame_fill=0.55,
    frame_fill_deadband=0.08,
    max_power=PowerCap(20),
)


TURN_ONLY = PursuitGains(
    turn=0.8,
    forward=0.0,
    target_frame_fill=0.55,
    frame_fill_deadband=0.08,
    max_power=PowerCap(20),
)


def _target(centre_x: float = 0.5, frame_fill: float = 0.55) -> Person:
    return Person(TrackId(1), centre_x=centre_x, frame_fill=frame_fill, area=0.1)


def test_a_centred_target_at_the_following_distance_needs_no_movement():
    assert pursue(_target(), GAINS) == STOPPED


def test_a_target_to_the_right_turns_the_tank_right():
    demand = pursue(_target(centre_x=0.9), GAINS)

    assert demand.turn > 0
    assert demand.forward == 0


def test_a_target_to_the_left_turns_the_tank_left():
    demand = pursue(_target(centre_x=0.1), GAINS)

    assert demand.turn < 0
    assert demand.forward == 0


def test_a_distant_target_is_approached():
    demand = pursue(_target(frame_fill=0.2), GAINS)

    assert demand.forward > 0
    assert demand.turn == 0


def test_a_target_that_is_too_close_is_backed_away_from():
    demand = pursue(_target(frame_fill=0.95), GAINS)

    assert demand.forward < 0
    assert demand.turn == 0


def test_distance_errors_inside_the_deadband_are_ignored():
    just_inside = GAINS.target_frame_fill - GAINS.frame_fill_deadband + 0.01

    assert pursue(_target(frame_fill=just_inside), GAINS) == STOPPED


def test_distance_errors_beyond_the_deadband_are_acted_on():
    just_outside = GAINS.target_frame_fill - GAINS.frame_fill_deadband - 0.01

    assert pursue(_target(frame_fill=just_outside), GAINS).forward > 0


@pytest.mark.parametrize("centre_x", [-0.2, 0.0, 0.3, 0.5, 0.8, 1.0, 1.3])
@pytest.mark.parametrize("frame_fill", [0.0, 0.3, 0.55, 1.0, 1.4])
def test_the_demand_never_exceeds_the_power_cap(centre_x, frame_fill):
    demand = pursue(_target(centre_x, frame_fill), GAINS)

    assert abs(demand.forward) + abs(demand.turn) <= GAINS.max_power.share + 1e-9


def test_an_off_centre_distant_target_is_approached_while_turning_towards_it():
    demand = pursue(_target(centre_x=0.6, frame_fill=0.0), GAINS)

    assert demand.forward > demand.turn > 0


@pytest.mark.parametrize("frame_fill", [0.05, 0.55, 1.0])
def test_with_no_forward_gain_an_off_centre_target_is_turned_towards_on_the_spot(frame_fill):
    demand = pursue(_target(centre_x=0.9, frame_fill=frame_fill), TURN_ONLY)

    assert demand.turn > 0
    assert demand.forward == 0


@pytest.mark.parametrize("frame_fill", [0.05, 1.0])
def test_with_no_forward_gain_a_centred_target_needs_no_movement_at_any_distance(frame_fill):
    assert pursue(_target(centre_x=0.5, frame_fill=frame_fill), TURN_ONLY) == STOPPED
