"""The lidar's scan: every reading of one turn of its head."""

from dataclasses import dataclass

import pyarrow as pa

from .wire import one_row, only_row

# The two lists run in step, one item per reading, in bearing order. Only a range can be missing.
SCAN_TYPE = pa.struct(
    [
        ("spin_rev_per_s", pa.float32()),
        ("bearing_deg", pa.list_(pa.field("item", pa.float32(), nullable=False))),
        ("range_m", pa.list_(pa.field("item", pa.float32()))),
    ]
)


@dataclass(frozen=True)
class RangeReading:
    """One look in one direction. The bearing is clockwise from the lidar's zero mark, seen from above."""

    bearing_deg: float
    # None when nothing came back.
    range_m: float | None


@dataclass(frozen=True)
class Scan:
    spin_rev_per_s: float
    readings: tuple[RangeReading, ...]

    def to_arrow(self) -> pa.Array:
        return one_row(
            {
                "spin_rev_per_s": self.spin_rev_per_s,
                "bearing_deg": [reading.bearing_deg for reading in self.readings],
                "range_m": [reading.range_m for reading in self.readings],
            },
            SCAN_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "Scan":
        row = only_row(array, SCAN_TYPE)
        bearings, ranges = row["bearing_deg"], row["range_m"]
        if len(bearings) != len(ranges):
            raise ValueError(f"scan has {len(bearings)} bearings for {len(ranges)} ranges")
        return cls(
            spin_rev_per_s=row["spin_rev_per_s"],
            readings=tuple(RangeReading(bearing, range_m) for bearing, range_m in zip(bearings, ranges)),
        )
