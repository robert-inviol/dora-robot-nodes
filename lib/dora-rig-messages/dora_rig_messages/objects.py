"""The persistent objects of the map: what the object map reports each time one appears, changes, moves or goes.

Coordinates are in the map frame: z up, gravity aligned, metres.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pyarrow as pa

from .navigation import MapVersion, ObjectId
from .wire import one_row, only_row


class ObjectUpdateKind(Enum):
    # Confirmed after enough sightings; first report of this id.
    APPEARED = "appeared"
    # Seen again; position, size, label or crop may have changed.
    UPDATED = "updated"
    # Gone from where it was and confirmed somewhere else; the id is kept.
    MOVED = "moved"
    # The camera saw clearly past where it should be, often enough to say it has gone.
    DISAPPEARED = "disappeared"
    # Found to be the same object as merged_into, which keeps the identity from now on.
    MERGED = "merged"


MAP_POINT_TYPE = pa.struct([("x_m", pa.float32()), ("y_m", pa.float32())])

FOOTPRINT_CORNERS = 4

OBJECT_UPDATE_TYPE = pa.struct(
    [
        ("kind", pa.string()),
        ("object_id", pa.uint64()),
        # The surviving object; null unless the kind is merged.
        ("merged_into", pa.uint64()),
        # The label of the best-scoring detection.
        ("label", pa.string()),
        # The best segmentation score among the object's detections.
        ("score", pa.float32()),
        # Sightings over sightings plus misses: how sure the map is that the object is there.
        ("confidence", pa.float32()),
        ("seen_count", pa.uint32()),
        # The keyframe whose evidence caused this update.
        ("frame_id", pa.uint64()),
        ("captured_unix_ns", pa.int64()),
        ("map_version", pa.uint64()),
        ("centre_x_m", pa.float32()),
        ("centre_y_m", pa.float32()),
        ("centre_z_m", pa.float32()),
        # Corners of the minimum-area floor-plan rectangle, in order.
        ("footprint", pa.list_(pa.field("item", MAP_POINT_TYPE, nullable=False))),
        ("long_m", pa.float32()),
        ("short_m", pa.float32()),
        ("height_m", pa.float32()),
        # Height of the object's top in the map frame.
        ("top_m", pa.float32()),
        # The best view of the object so far; null unless it is new to the receiver.
        ("crop_jpeg", pa.binary()),
    ]
)


@dataclass(frozen=True)
class MapPoint:
    x_m: float
    y_m: float


@dataclass(frozen=True)
class ObjectUpdate:
    kind: ObjectUpdateKind
    object_id: ObjectId
    merged_into: ObjectId | None
    label: str
    score: float
    confidence: float
    seen_count: int
    frame_id: int
    captured_unix_ns: int
    map_version: MapVersion
    centre_x_m: float
    centre_y_m: float
    centre_z_m: float
    footprint: tuple[MapPoint, ...]
    long_m: float
    short_m: float
    height_m: float
    top_m: float
    crop_jpeg: bytes | None

    def __post_init__(self) -> None:
        if (self.kind is ObjectUpdateKind.MERGED) != (self.merged_into is not None):
            needs = "needs" if self.kind is ObjectUpdateKind.MERGED else "takes no"
            raise ValueError(f"a {self.kind.value} object update {needs} merged_into")
        if len(self.footprint) != FOOTPRINT_CORNERS:
            raise ValueError(f"an object footprint has {len(self.footprint)} corners, not {FOOTPRINT_CORNERS}")
        for name in ("confidence", "score"):
            value = getattr(self, name)
            if not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(f"{name} is {value}, expected a value from 0 to 1")

    def to_arrow(self) -> pa.Array:
        return one_row(
            {
                "kind": self.kind.value,
                "object_id": self.object_id,
                "merged_into": self.merged_into,
                "label": self.label,
                "score": self.score,
                "confidence": self.confidence,
                "seen_count": self.seen_count,
                "frame_id": self.frame_id,
                "captured_unix_ns": self.captured_unix_ns,
                "map_version": self.map_version,
                "centre_x_m": self.centre_x_m,
                "centre_y_m": self.centre_y_m,
                "centre_z_m": self.centre_z_m,
                "footprint": [{"x_m": c.x_m, "y_m": c.y_m} for c in self.footprint],
                "long_m": self.long_m,
                "short_m": self.short_m,
                "height_m": self.height_m,
                "top_m": self.top_m,
                "crop_jpeg": self.crop_jpeg,
            },
            OBJECT_UPDATE_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "ObjectUpdate":
        row = only_row(array, OBJECT_UPDATE_TYPE)
        merged_into = row["merged_into"]
        return cls(
            kind=ObjectUpdateKind(row["kind"]),
            object_id=ObjectId(row["object_id"]),
            merged_into=None if merged_into is None else ObjectId(merged_into),
            label=row["label"],
            score=row["score"],
            confidence=row["confidence"],
            seen_count=row["seen_count"],
            frame_id=row["frame_id"],
            captured_unix_ns=row["captured_unix_ns"],
            map_version=MapVersion(row["map_version"]),
            centre_x_m=row["centre_x_m"],
            centre_y_m=row["centre_y_m"],
            centre_z_m=row["centre_z_m"],
            footprint=tuple(MapPoint(c["x_m"], c["y_m"]) for c in row["footprint"]),
            long_m=row["long_m"],
            short_m=row["short_m"],
            height_m=row["height_m"],
            top_m=row["top_m"],
            crop_jpeg=row["crop_jpeg"],
        )


_SOFA_FOOTPRINT = (MapPoint(1.0, 2.0), MapPoint(2.75, 2.0), MapPoint(2.75, 2.5), MapPoint(1.0, 2.5))

# Golden examples of this module's messages, merged into fixtures.EXAMPLES.
FIXTURE_EXAMPLES: dict[str, Any] = {
    "object_update_appeared": ObjectUpdate(
        kind=ObjectUpdateKind.APPEARED,
        object_id=ObjectId(17),
        merged_into=None,
        label="sofa",
        score=0.875,
        confidence=0.75,
        seen_count=3,
        frame_id=42,
        captured_unix_ns=1_762_560_000_123_456_789,
        map_version=MapVersion(412),
        centre_x_m=1.875,
        centre_y_m=2.25,
        centre_z_m=0.25,
        footprint=_SOFA_FOOTPRINT,
        long_m=1.75,
        short_m=0.5,
        height_m=0.5,
        top_m=0.5,
        crop_jpeg=bytes([0xFF, 0xD8, 0xFF, 0xD9]),
    ),
    "object_update_merged": ObjectUpdate(
        kind=ObjectUpdateKind.MERGED,
        object_id=ObjectId(23),
        merged_into=ObjectId(17),
        label="sofa",
        score=0.625,
        confidence=1.0,
        seen_count=4,
        frame_id=57,
        captured_unix_ns=1_762_560_030_000_000_000,
        map_version=MapVersion(413),
        centre_x_m=1.875,
        centre_y_m=2.25,
        centre_z_m=0.25,
        footprint=_SOFA_FOOTPRINT,
        long_m=1.75,
        short_m=0.5,
        height_m=0.5,
        top_m=0.5,
        crop_jpeg=None,
    ),
}
