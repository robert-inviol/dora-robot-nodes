"""What the pilot and the robot's driver report, for a control page to show."""

from dataclasses import dataclass
from typing import NewType

import pyarrow as pa

from .operator import DriveMode
from .wire import one_row, only_row

TrackId = NewType("TrackId", int)

PILOT_STATUS_TYPE = pa.struct([("mode", pa.string()), ("locked_track_id", pa.int32())])
DRIVER_STATUS_TYPE = pa.struct(
    [("armed", pa.bool_()), ("left_pct", pa.int8()), ("right_pct", pa.int8())]
)

NOBODY_LOCKED = -1


@dataclass(frozen=True)
class PilotStatus:
    mode: DriveMode
    locked_id: TrackId | None

    def to_arrow(self) -> pa.Array:
        locked = NOBODY_LOCKED if self.locked_id is None else self.locked_id
        return one_row({"mode": self.mode.value, "locked_track_id": locked}, PILOT_STATUS_TYPE)

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "PilotStatus":
        row = only_row(array, PILOT_STATUS_TYPE)
        locked = row["locked_track_id"]
        return cls(DriveMode(row["mode"]), None if locked == NOBODY_LOCKED else TrackId(locked))


@dataclass(frozen=True)
class DriverStatus:
    """Whether the motors are live, and the signed share of full power on each track in percent."""

    armed: bool
    left_pct: int
    right_pct: int

    def to_arrow(self) -> pa.Array:
        return one_row(
            {"armed": self.armed, "left_pct": self.left_pct, "right_pct": self.right_pct},
            DRIVER_STATUS_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "DriverStatus":
        row = only_row(array, DRIVER_STATUS_TYPE)
        return cls(armed=row["armed"], left_pct=row["left_pct"], right_pct=row["right_pct"])
