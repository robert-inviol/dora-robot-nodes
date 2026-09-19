"""Turning where the target is in the frame into track speeds."""

from dataclasses import dataclass

from .people import Person
from .target import FRAME_CENTRE_X

FULL_POWER_PCT = 100


@dataclass(frozen=True)
class TrackSpeed:
    """Signed share of full motor power, in percent. Positive drives the track forward."""

    percent: int

    def __post_init__(self):
        if not -FULL_POWER_PCT <= self.percent <= FULL_POWER_PCT:
            raise ValueError(f"track speed {self.percent} is outside -100..100 percent")


@dataclass(frozen=True)
class DriveCommand:
    left: TrackSpeed
    right: TrackSpeed


@dataclass(frozen=True)
class PursuitGains:
    turn: float
    forward: float
    # Share of the frame height the target's box fills at the distance the tank holds.
    target_frame_fill: float
    frame_fill_deadband: float
    max_speed: TrackSpeed


def _clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))


def pursue(target: Person, gains: PursuitGains) -> DriveCommand:
    """Steer towards the target and close to the distance where it fills the chosen share of the frame."""
    offset_from_centre = (target.centre_x - FRAME_CENTRE_X) / FRAME_CENTRE_X
    turn = _clamp_unit(gains.turn * offset_from_centre)

    fill_error = gains.target_frame_fill - target.frame_fill
    within_deadband = abs(fill_error) < gains.frame_fill_deadband
    forward = 0.0 if within_deadband else _clamp_unit(gains.forward * fill_error)

    left, right = forward + turn, forward - turn
    # Dividing by the larger track keeps the turn ratio when the sum would exceed full power.
    scale = gains.max_speed.percent / max(1.0, abs(left), abs(right))
    return DriveCommand(TrackSpeed(round(left * scale)), TrackSpeed(round(right * scale)))
