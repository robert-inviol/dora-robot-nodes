"""Range readings parsed from the rows the lidar node publishes."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# The column and parameter names are set by lidar/lidar-node/src/wire.rs.
BEARING_COLUMN = "bearing_deg"
RANGE_COLUMN = "range_m"
SPIN_RATE_PARAMETER = "spin_rev_per_s"


@dataclass(frozen=True)
class RangeReading:
    """One look in one direction. The bearing is clockwise from the lidar's zero mark, seen from above."""

    bearing_deg: float
    # None when nothing came back.
    range_m: float | None


def readings_in(rows: Sequence[Mapping[str, Any]]) -> list[RangeReading]:
    return [RangeReading(row[BEARING_COLUMN], row[RANGE_COLUMN]) for row in rows]
