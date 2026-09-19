"""Turning where the target is in the frame into track speeds."""

from dataclasses import dataclass

from .drive import DriveCommand, TrackSpeed, mix
from .people import Person
from .target import FRAME_CENTRE_X


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

    return mix(forward, turn, gains.max_speed)
