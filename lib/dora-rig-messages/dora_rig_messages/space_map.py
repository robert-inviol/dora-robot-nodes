"""The dense map's view of the house: free space on the floor, rooms with their doorways and places, keep-out zones.

Coordinates are the map frame: z up, gravity aligned, metres. Heights are above the floor unless named otherwise.
"""

import math
from dataclasses import dataclass
from typing import Any, NewType

import numpy as np
import pyarrow as pa

from .navigation import MapVersion, ObjectId, PlaceId, RoomId
from .wire import one_row, only_row

DoorwayId = NewType("DoorwayId", int)

# A free_space cell's clearance: 0 is occupied, this is unknown, anything between is millimetres to the nearest
# occupied cell.
OCCUPIED_MM = 0
UNKNOWN_MM = 65535


def _items(item_type: pa.DataType) -> pa.ListType:
    return pa.list_(pa.field("item", item_type, nullable=False))


POINT_TYPE = pa.struct([("x_m", pa.float32()), ("y_m", pa.float32())])

FREE_SPACE_TYPE = pa.struct(
    [
        ("map_version", pa.uint64()),
        ("built_unix_ns", pa.int64()),
        # The outer corner of cell (row 0, column 0); columns run along x, rows along y.
        ("origin_x_m", pa.float32()),
        ("origin_y_m", pa.float32()),
        ("resolution_m", pa.float32()),
        ("width", pa.uint32()),
        ("height", pa.uint32()),
        # The height band above the floor that a cell describes.
        ("band_min_m", pa.float32()),
        ("band_max_m", pa.float32()),
        # Row major from the origin.
        ("clearance_mm", _items(pa.uint16())),
    ]
)

ROOM_TYPE = pa.struct(
    [
        ("room_id", pa.uint64()),
        ("polygon", _items(POINT_TYPE)),
        # Heights in the map frame.
        ("floor_m", pa.float32()),
        # Null where no ceiling was seen over the room.
        ("ceiling_m", pa.float32()),
    ]
)

DOORWAY_TYPE = pa.struct(
    [
        ("doorway_id", pa.uint64()),
        ("room_a", pa.uint64()),
        ("room_b", pa.uint64()),
        ("x_m", pa.float32()),
        ("y_m", pa.float32()),
        ("width_m", pa.float32()),
    ]
)

PLACE_TYPE = pa.struct(
    [
        ("place_id", pa.uint64()),
        ("room_id", pa.uint64()),
        ("x_m", pa.float32()),
        ("y_m", pa.float32()),
        ("clearance_m", pa.float32()),
    ]
)

EDGE_TYPE = pa.struct(
    [
        ("place_a", pa.uint64()),
        ("place_b", pa.uint64()),
        ("length_m", pa.float32()),
        ("min_clearance_m", pa.float32()),
    ]
)

TOPOLOGY_TYPE = pa.struct(
    [
        ("map_version", pa.uint64()),
        ("built_unix_ns", pa.int64()),
        ("rooms", _items(ROOM_TYPE)),
        ("doorways", _items(DOORWAY_TYPE)),
        ("places", _items(PLACE_TYPE)),
        ("edges", _items(EDGE_TYPE)),
    ]
)

ZONE_TYPE = pa.struct(
    [
        ("object_id", pa.uint64()),
        ("label", pa.string()),
        ("polygon", _items(POINT_TYPE)),
        # How far beyond the polygon the robot keeps out.
        ("margin_m", pa.float32()),
    ]
)

KEEP_OUT_TYPE = pa.struct([("map_version", pa.uint64()), ("zones", _items(ZONE_TYPE))])


@dataclass(frozen=True)
class Point:
    x_m: float
    y_m: float

    def to_row(self) -> dict[str, float]:
        return {"x_m": self.x_m, "y_m": self.y_m}

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Point":
        return cls(row["x_m"], row["y_m"])


def _polygon(points: tuple[Point, ...], owner: str) -> None:
    if len(points) < 3:
        raise ValueError(f"{owner} has a polygon of {len(points)} points, expected at least 3")


@dataclass(frozen=True, eq=False)
class FreeSpace:
    """Which floor cells the robot can occupy, sliced through a height band above the floor."""

    map_version: MapVersion
    built_unix_ns: int
    origin_x_m: float
    origin_y_m: float
    resolution_m: float
    band_min_m: float
    band_max_m: float
    # Shape (height, width): rows along y, columns along x.
    clearance_mm: np.ndarray

    def __post_init__(self) -> None:
        if self.clearance_mm.ndim != 2:
            raise ValueError(f"clearance has {self.clearance_mm.ndim} dimensions, expected 2")
        if not (math.isfinite(self.resolution_m) and self.resolution_m > 0.0):
            raise ValueError(f"resolution_m is {self.resolution_m}, expected a finite value above 0")
        if not self.band_min_m < self.band_max_m:
            raise ValueError(f"the band {self.band_min_m}..{self.band_max_m} m is empty")

    @property
    def height(self) -> int:
        return self.clearance_mm.shape[0]

    @property
    def width(self) -> int:
        return self.clearance_mm.shape[1]

    def to_arrow(self) -> pa.Array:
        flat = pa.array(self.clearance_mm.astype(np.uint16, copy=False).ravel(), type=pa.uint16())
        columns = {
            "map_version": pa.array([self.map_version], pa.uint64()),
            "built_unix_ns": pa.array([self.built_unix_ns], pa.int64()),
            "origin_x_m": pa.array([self.origin_x_m], pa.float32()),
            "origin_y_m": pa.array([self.origin_y_m], pa.float32()),
            "resolution_m": pa.array([self.resolution_m], pa.float32()),
            "width": pa.array([self.width], pa.uint32()),
            "height": pa.array([self.height], pa.uint32()),
            "band_min_m": pa.array([self.band_min_m], pa.float32()),
            "band_max_m": pa.array([self.band_max_m], pa.float32()),
            "clearance_mm": pa.ListArray.from_arrays(
                pa.array([0, len(flat)], pa.int32()), flat, type=_items(pa.uint16())
            ),
        }
        return pa.StructArray.from_arrays(list(columns.values()), fields=list(FREE_SPACE_TYPE))

    def __eq__(self, other: object) -> bool:
        scalars = ("map_version", "built_unix_ns", "origin_x_m", "origin_y_m", "resolution_m", "band_min_m", "band_max_m")
        return (
            isinstance(other, FreeSpace)
            and all(getattr(self, name) == getattr(other, name) for name in scalars)
            and np.array_equal(self.clearance_mm, other.clearance_mm)
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "FreeSpace":
        if array.type != FREE_SPACE_TYPE:
            raise ValueError(f"expected wire type {FREE_SPACE_TYPE}, received {array.type}")
        if len(array) != 1:
            raise ValueError(f"expected one row of {FREE_SPACE_TYPE}, received {len(array)}")
        scalar = lambda name: array.field(name)[0].as_py()  # noqa: E731
        width, height = scalar("width"), scalar("height")
        column = array.field("clearance_mm")
        clearance = column.values[column.offsets[0].as_py() : column.offsets[1].as_py()].to_numpy(zero_copy_only=False)
        if clearance.size != width * height:
            raise ValueError(f"free space has {clearance.size} cells for {width}x{height}")
        return cls(
            map_version=MapVersion(scalar("map_version")),
            built_unix_ns=scalar("built_unix_ns"),
            origin_x_m=scalar("origin_x_m"),
            origin_y_m=scalar("origin_y_m"),
            resolution_m=scalar("resolution_m"),
            band_min_m=scalar("band_min_m"),
            band_max_m=scalar("band_max_m"),
            clearance_mm=clearance.reshape(height, width),
        )


@dataclass(frozen=True)
class TopologyRoom:
    room_id: RoomId
    polygon: tuple[Point, ...]
    floor_m: float
    ceiling_m: float | None

    def __post_init__(self) -> None:
        _polygon(self.polygon, f"room {self.room_id}")


@dataclass(frozen=True)
class Doorway:
    doorway_id: DoorwayId
    room_a: RoomId
    room_b: RoomId
    x_m: float
    y_m: float
    width_m: float

    def __post_init__(self) -> None:
        if self.room_a == self.room_b:
            raise ValueError(f"doorway {self.doorway_id} joins room {self.room_a} to itself")


@dataclass(frozen=True)
class Place:
    place_id: PlaceId
    room_id: RoomId
    x_m: float
    y_m: float
    # Distance to the nearest cell that is not known free.
    clearance_m: float


@dataclass(frozen=True)
class PlaceEdge:
    place_a: PlaceId
    place_b: PlaceId
    length_m: float
    min_clearance_m: float


@dataclass(frozen=True)
class Topology:
    """Rooms cut at their doorways, and a graph of places along the middle of the free space."""

    map_version: MapVersion
    built_unix_ns: int
    rooms: tuple[TopologyRoom, ...]
    doorways: tuple[Doorway, ...]
    places: tuple[Place, ...]
    edges: tuple[PlaceEdge, ...]

    def __post_init__(self) -> None:
        rooms = {room.room_id for room in self.rooms}
        places = {place.place_id for place in self.places}
        for doorway in self.doorways:
            if not {doorway.room_a, doorway.room_b} <= rooms:
                raise ValueError(f"doorway {doorway.doorway_id} joins a room that is not in the topology")
        for place in self.places:
            if place.room_id not in rooms:
                raise ValueError(f"place {place.place_id} is in room {place.room_id}, which is not in the topology")
        for edge in self.edges:
            if not {edge.place_a, edge.place_b} <= places:
                raise ValueError(f"edge {edge.place_a}-{edge.place_b} joins a place that is not in the topology")

    def to_arrow(self) -> pa.Array:
        return one_row(
            {
                "map_version": self.map_version,
                "built_unix_ns": self.built_unix_ns,
                "rooms": [
                    {
                        "room_id": room.room_id,
                        "polygon": [point.to_row() for point in room.polygon],
                        "floor_m": room.floor_m,
                        "ceiling_m": room.ceiling_m,
                    }
                    for room in self.rooms
                ],
                "doorways": [
                    {
                        "doorway_id": d.doorway_id,
                        "room_a": d.room_a,
                        "room_b": d.room_b,
                        "x_m": d.x_m,
                        "y_m": d.y_m,
                        "width_m": d.width_m,
                    }
                    for d in self.doorways
                ],
                "places": [
                    {"place_id": p.place_id, "room_id": p.room_id, "x_m": p.x_m, "y_m": p.y_m, "clearance_m": p.clearance_m}
                    for p in self.places
                ],
                "edges": [
                    {"place_a": e.place_a, "place_b": e.place_b, "length_m": e.length_m, "min_clearance_m": e.min_clearance_m}
                    for e in self.edges
                ],
            },
            TOPOLOGY_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "Topology":
        row = only_row(array, TOPOLOGY_TYPE)
        return cls(
            map_version=MapVersion(row["map_version"]),
            built_unix_ns=row["built_unix_ns"],
            rooms=tuple(
                TopologyRoom(
                    RoomId(r["room_id"]), tuple(Point.from_row(p) for p in r["polygon"]), r["floor_m"], r["ceiling_m"]
                )
                for r in row["rooms"]
            ),
            doorways=tuple(
                Doorway(DoorwayId(d["doorway_id"]), RoomId(d["room_a"]), RoomId(d["room_b"]), d["x_m"], d["y_m"], d["width_m"])
                for d in row["doorways"]
            ),
            places=tuple(
                Place(PlaceId(p["place_id"]), RoomId(p["room_id"]), p["x_m"], p["y_m"], p["clearance_m"])
                for p in row["places"]
            ),
            edges=tuple(
                PlaceEdge(PlaceId(e["place_a"]), PlaceId(e["place_b"]), e["length_m"], e["min_clearance_m"])
                for e in row["edges"]
            ),
        )


@dataclass(frozen=True)
class KeepOutZone:
    object_id: ObjectId
    label: str
    polygon: tuple[Point, ...]
    margin_m: float

    def __post_init__(self) -> None:
        _polygon(self.polygon, f"keep-out zone {self.object_id}")
        if not (math.isfinite(self.margin_m) and self.margin_m >= 0.0):
            raise ValueError(f"margin_m is {self.margin_m}, expected a finite value of at least 0")


@dataclass(frozen=True)
class KeepOut:
    """Every zone the robot must not enter, such as glass it cannot see with depth; each message replaces the last."""

    map_version: MapVersion
    zones: tuple[KeepOutZone, ...]

    def to_arrow(self) -> pa.Array:
        return one_row(
            {
                "map_version": self.map_version,
                "zones": [
                    {
                        "object_id": z.object_id,
                        "label": z.label,
                        "polygon": [point.to_row() for point in z.polygon],
                        "margin_m": z.margin_m,
                    }
                    for z in self.zones
                ],
            },
            KEEP_OUT_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "KeepOut":
        row = only_row(array, KEEP_OUT_TYPE)
        return cls(
            map_version=MapVersion(row["map_version"]),
            zones=tuple(
                KeepOutZone(ObjectId(z["object_id"]), z["label"], tuple(Point.from_row(p) for p in z["polygon"]), z["margin_m"])
                for z in row["zones"]
            ),
        )


def _square(x_m: float, y_m: float, side_m: float) -> tuple[Point, ...]:
    return (Point(x_m, y_m), Point(x_m + side_m, y_m), Point(x_m + side_m, y_m + side_m), Point(x_m, y_m + side_m))


# Golden examples of this module's messages, merged into fixtures.EXAMPLES. Every float is exact in float32.
FIXTURE_EXAMPLES: dict[str, Any] = {
    "free_space": FreeSpace(
        map_version=MapVersion(412),
        built_unix_ns=1_762_560_002_000_000_000,
        origin_x_m=-1.25,
        origin_y_m=0.5,
        resolution_m=0.0625,
        band_min_m=0.0625,
        band_max_m=0.625,
        clearance_mm=np.array([[0, 50, 100], [UNKNOWN_MM, 70, 0]], dtype=np.uint16),
    ),
    "topology": Topology(
        map_version=MapVersion(412),
        built_unix_ns=1_762_560_002_500_000_000,
        rooms=(
            TopologyRoom(RoomId(3), _square(0.0, 0.0, 4.0), floor_m=-0.875, ceiling_m=1.5),
            TopologyRoom(RoomId(5), _square(4.0, 0.0, 3.0), floor_m=-0.875, ceiling_m=None),
        ),
        doorways=(Doorway(DoorwayId(1), RoomId(3), RoomId(5), x_m=4.0, y_m=1.5, width_m=0.75),),
        places=(
            Place(PlaceId(10), RoomId(3), x_m=2.0, y_m=2.0, clearance_m=1.5),
            Place(PlaceId(11), RoomId(3), x_m=4.0, y_m=1.5, clearance_m=0.375),
            Place(PlaceId(12), RoomId(5), x_m=5.5, y_m=1.5, clearance_m=1.25),
        ),
        edges=(
            PlaceEdge(PlaceId(10), PlaceId(11), length_m=2.0625, min_clearance_m=0.375),
            PlaceEdge(PlaceId(11), PlaceId(12), length_m=1.5, min_clearance_m=0.375),
        ),
    ),
    "keep_out": KeepOut(
        map_version=MapVersion(412),
        zones=(KeepOutZone(ObjectId(42), "glass door", _square(4.0, 1.0, 1.0), margin_m=0.25),),
    ),
    "keep_out_empty": KeepOut(map_version=MapVersion(413), zones=()),
}
