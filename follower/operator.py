"""The wire contract between the teleop node and the follower node."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .drive import DriveCommand, TrackSpeed
from .people import TrackId

NOBODY_LOCKED = -1


class DriveMode(Enum):
    MANUAL = "manual"
    FOLLOW = "follow"


def _unit(name: str, value: float) -> float:
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"{name} is {value}, expected -1 to 1")
    return float(value)


@dataclass(frozen=True)
class OperatorCommand:
    """What the person at the controls wants. Throttle is positive forward, steer positive right."""

    mode: DriveMode
    throttle: float
    steer: float

    def __post_init__(self):
        _unit("throttle", self.throttle)
        _unit("steer", self.steer)

    @classmethod
    def idle(cls) -> "OperatorCommand":
        return cls(DriveMode.MANUAL, throttle=0.0, steer=0.0)

    def to_row(self) -> dict[str, Any]:
        return {"mode": self.mode.value, "throttle": self.throttle, "steer": self.steer}

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "OperatorCommand":
        return cls(DriveMode(row["mode"]), throttle=row["throttle"], steer=row["steer"])


@dataclass(frozen=True)
class PilotStatus:
    """What the follower is doing, for the control page to show."""

    mode: DriveMode
    locked_id: TrackId | None
    drive: DriveCommand
    armed: bool

    def to_row(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "locked_track_id": NOBODY_LOCKED if self.locked_id is None else self.locked_id,
            "left_pct": self.drive.left.percent,
            "right_pct": self.drive.right.percent,
            "armed": self.armed,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "PilotStatus":
        locked = row["locked_track_id"]
        return cls(
            mode=DriveMode(row["mode"]),
            locked_id=None if locked == NOBODY_LOCKED else TrackId(locked),
            drive=DriveCommand(TrackSpeed(row["left_pct"]), TrackSpeed(row["right_pct"])),
            armed=row["armed"],
        )
