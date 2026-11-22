"""The lidar's scan: every reading of one turn of its head."""

from dataclasses import dataclass

import pyarrow as pa

from .wire import one_row, only_row

# The lists run in step, one item per reading, in the order measured. Only a range can be missing; the intensity
# list is null as a whole for a lidar that does not report one.
SCAN_TYPE = pa.struct(
    [
        ("spin_rev_per_s", pa.float32()),
        ("first_reading_unix_ns", pa.int64()),
        ("last_reading_unix_ns", pa.int64()),
        ("bearing_deg", pa.list_(pa.field("item", pa.float32(), nullable=False))),
        ("range_m", pa.list_(pa.field("item", pa.float32()))),
        ("intensity", pa.list_(pa.field("item", pa.uint8(), nullable=False))),
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
    # When the first and the last reading were measured, on the host's clock.
    first_reading_unix_ns: int
    last_reading_unix_ns: int
    readings: tuple[RangeReading, ...]
    # The return strength of each reading, 0 to 255 on the lidar's own scale; None when the lidar does not report it.
    intensities: tuple[int, ...] | None

    def __post_init__(self) -> None:
        if self.intensities is not None and len(self.intensities) != len(self.readings):
            raise ValueError(f"scan has {len(self.intensities)} intensities for {len(self.readings)} readings")
        if self.last_reading_unix_ns < self.first_reading_unix_ns:
            raise ValueError(
                f"scan's last reading at {self.last_reading_unix_ns} ns is before its first at "
                f"{self.first_reading_unix_ns} ns"
            )

    def to_arrow(self) -> pa.Array:
        return one_row(
            {
                "spin_rev_per_s": self.spin_rev_per_s,
                "first_reading_unix_ns": self.first_reading_unix_ns,
                "last_reading_unix_ns": self.last_reading_unix_ns,
                "bearing_deg": [reading.bearing_deg for reading in self.readings],
                "range_m": [reading.range_m for reading in self.readings],
                "intensity": None if self.intensities is None else list(self.intensities),
            },
            SCAN_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "Scan":
        row = only_row(array, SCAN_TYPE)
        bearings, ranges, intensities = row["bearing_deg"], row["range_m"], row["intensity"]
        if len(bearings) != len(ranges):
            raise ValueError(f"scan has {len(bearings)} bearings for {len(ranges)} ranges")
        return cls(
            spin_rev_per_s=row["spin_rev_per_s"],
            first_reading_unix_ns=row["first_reading_unix_ns"],
            last_reading_unix_ns=row["last_reading_unix_ns"],
            readings=tuple(RangeReading(bearing, range_m) for bearing, range_m in zip(bearings, ranges)),
            intensities=None if intensities is None else tuple(intensities),
        )
