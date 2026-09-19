"""The drive demand: what a pilot asks of whatever robot is underneath it."""

from dataclasses import dataclass

import pyarrow as pa

from .wire import one_row, only_row

DRIVE_DEMAND_TYPE = pa.struct([("forward", pa.float32()), ("turn", pa.float32())])

FULL_POWER_PCT = 100
# Room for a demand that summed to exactly 1 before it was narrowed to float32 on the wire.
_FLOAT32_SLACK = 1e-6


@dataclass(frozen=True)
class PowerCap:
    """The most of full power a pilot may ask for, in percent."""

    percent: int

    def __post_init__(self):
        if not 1 <= self.percent <= FULL_POWER_PCT:
            raise ValueError(f"power cap {self.percent} is outside 1..100 percent")

    @property
    def share(self) -> float:
        return self.percent / FULL_POWER_PCT


@dataclass(frozen=True)
class DriveDemand:
    """Shares of full power in -1..1. Positive forward drives ahead, positive turn steers right.

    The two magnitudes sum to at most 1, so however a robot mixes them no wheel is asked for
    more than full power.
    """

    forward: float
    turn: float

    def __post_init__(self):
        if abs(self.forward) + abs(self.turn) > 1 + _FLOAT32_SLACK:
            raise ValueError(
                f"drive demand forward {self.forward} and turn {self.turn} sum to more than full power"
            )

    @classmethod
    def limited(cls, forward: float, turn: float, cap: PowerCap) -> "DriveDemand":
        """Scale a forward and a turn wish to fit under the cap, keeping their ratio."""
        scale = cap.share / max(1.0, abs(forward) + abs(turn))
        return cls(forward * scale, turn * scale)

    def to_arrow(self) -> pa.Array:
        return one_row({"forward": self.forward, "turn": self.turn}, DRIVE_DEMAND_TYPE)

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "DriveDemand":
        row = only_row(array, DRIVE_DEMAND_TYPE)
        return cls(forward=row["forward"], turn=row["turn"])


STOPPED = DriveDemand(forward=0.0, turn=0.0)
