"""What the person at the controls wants, as published by a control page."""

from dataclasses import dataclass
from enum import Enum

import pyarrow as pa

from .wire import one_row, only_row

OPERATOR_COMMAND_TYPE = pa.struct(
    [("mode", pa.string()), ("throttle", pa.float32()), ("steer", pa.float32())]
)


class DriveMode(Enum):
    MANUAL = "manual"
    FOLLOW = "follow"


def _unit(name: str, value: float) -> float:
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"{name} is {value}, expected -1 to 1")
    return float(value)


@dataclass(frozen=True)
class OperatorCommand:
    """The selected mode and the stick. Throttle is positive forward, steer positive right."""

    mode: DriveMode
    throttle: float
    steer: float

    def __post_init__(self):
        _unit("throttle", self.throttle)
        _unit("steer", self.steer)

    @classmethod
    def idle(cls) -> "OperatorCommand":
        return cls(DriveMode.MANUAL, throttle=0.0, steer=0.0)

    def to_arrow(self) -> pa.Array:
        return one_row(
            {"mode": self.mode.value, "throttle": self.throttle, "steer": self.steer},
            OPERATOR_COMMAND_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "OperatorCommand":
        row = only_row(array, OPERATOR_COMMAND_TYPE)
        return cls(DriveMode(row["mode"]), throttle=row["throttle"], steer=row["steer"])
