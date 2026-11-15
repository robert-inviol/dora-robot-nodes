"""The dense map's view of the house: free space on the floor, rooms with their doorways and places, keep-out zones.

Coordinates are the map frame: z up, gravity aligned, metres. Heights are above the floor unless named otherwise.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, NewType

import numpy as np
import pyarrow as pa

from .navigation import MapVersion, ObjectId, PlaceId, RoomId
from .wire import one_row, only_row

DoorwayId = NewType("DoorwayId", int)
# The number of a mesh_blocks message within one run of the dense map.
MeshSequence = NewType("MeshSequence", int)


class MeshBlocksKind(Enum):
    # The blocks whose surface changed since the last message, each with its whole new surface; a mesh block without
    # triangles says its block has no surface now.
    CHANGES = "changes"
    # Every block that has surface, each with its whole surface: it replaces every block the receiver holds.
    SNAPSHOT = "snapshot"

# A occupancy_grid cell's clearance: 0 is occupied, this is unknown, anything between is millimetres to the nearest
# occupied cell.
OCCUPIED_MM = 0
UNKNOWN_MM = 65535


def _items(item_type: pa.DataType) -> pa.ListType:
    return pa.list_(pa.field("item", item_type, nullable=False))


POINT_TYPE = pa.struct([("x_m", pa.float32()), ("y_m", pa.float32())])

OCCUPANCY_GRID_TYPE = pa.struct(
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

MESH_BLOCK_TYPE = pa.struct(
    [
        # The block of the dense map the mesh block covers: block (x, y, z) spans [x, x + 1) * block_voxels * voxel_m
        # along x, and likewise along y and z.
        ("block_x", pa.int32()),
        ("block_y", pa.int32()),
        ("block_z", pa.int32()),
        # x, y and z of every vertex in turn, in the map frame.
        ("positions_m", _items(pa.float32())),
        # Three vertex indices per triangle, anticlockwise seen from the free side of the surface.
        ("triangles", _items(pa.uint32())),
    ]
)

MESH_BLOCKS_TYPE = pa.struct(
    [
        ("map_version", pa.uint64()),
        ("built_unix_ns", pa.int64()),
        # When the dense map that sent the message started: a later value is a new run whose blocks replace every
        # block of the last.
        ("run_started_unix_ns", pa.int64()),
        # Counts the run's messages from 0, one up per message, snapshots included, so a receiver sees when it
        # missed one.
        ("sequence", pa.uint64()),
        # "changes" or "snapshot" (MeshBlocksKind).
        ("kind", pa.string()),
        ("voxel_m", pa.float32()),
        ("block_voxels", pa.uint32()),
        ("blocks", _items(MESH_BLOCK_TYPE)),
    ]
)

# A receiver of mesh_blocks asking the dense map for a snapshot, because it missed a message or joined late.
MESH_SNAPSHOT_REQUEST_TYPE = pa.struct([("requested_unix_ns", pa.int64())])


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
class OccupancyGrid:
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
        return pa.StructArray.from_arrays(list(columns.values()), fields=list(OCCUPANCY_GRID_TYPE))

    def __eq__(self, other: object) -> bool:
        scalars = ("map_version", "built_unix_ns", "origin_x_m", "origin_y_m", "resolution_m", "band_min_m", "band_max_m")
        return (
            isinstance(other, OccupancyGrid)
            and all(getattr(self, name) == getattr(other, name) for name in scalars)
            and np.array_equal(self.clearance_mm, other.clearance_mm)
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "OccupancyGrid":
        if array.type != OCCUPANCY_GRID_TYPE:
            raise ValueError(f"expected wire type {OCCUPANCY_GRID_TYPE}, received {array.type}")
        if len(array) != 1:
            raise ValueError(f"expected one row of {OCCUPANCY_GRID_TYPE}, received {len(array)}")
        scalar = lambda name: array.field(name)[0].as_py()  # noqa: E731
        width, height = scalar("width"), scalar("height")
        column = array.field("clearance_mm")
        clearance = column.values[column.offsets[0].as_py() : column.offsets[1].as_py()].to_numpy(zero_copy_only=False)
        if clearance.size != width * height:
            raise ValueError(f"occupancy grid has {clearance.size} cells for {width}x{height}")
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


@dataclass(frozen=True)
class MeshBlock:
    """The dense map's surface inside one block; a mesh block without triangles says the block has no surface now."""

    block: tuple[int, int, int]
    positions_m: tuple[float, ...]
    triangles: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.positions_m) % 3 or len(self.triangles) % 3:
            raise ValueError(f"block {self.block} has {len(self.positions_m)} coordinates and {len(self.triangles)} "
                             "triangle corners; both must be multiples of 3")
        vertices = len(self.positions_m) // 3
        if any(index >= vertices for index in self.triangles):
            raise ValueError(f"block {self.block} has a triangle corner beyond its {vertices} vertices")


@dataclass(frozen=True)
class MeshBlocks:
    """The dense map's surface, block by block: the blocks whose surface changed since the last message, or, in a
    snapshot, every block that has surface, replacing every block the receiver holds.

    The messages of one run of the dense map share `run_started_unix_ns` and number from `sequence` 0 up by one,
    snapshots included, so a receiver that sees a gap in the sequence knows it missed blocks.
    """

    map_version: MapVersion
    built_unix_ns: int
    run_started_unix_ns: int
    sequence: MeshSequence
    kind: MeshBlocksKind
    voxel_m: float
    block_voxels: int
    blocks: tuple[MeshBlock, ...]

    def __post_init__(self) -> None:
        if not (math.isfinite(self.voxel_m) and self.voxel_m > 0.0):
            raise ValueError(f"voxel_m is {self.voxel_m}, expected a finite value above 0")
        if self.block_voxels < 1:
            raise ValueError(f"block_voxels is {self.block_voxels}, expected at least 1")
        blocks = [mesh_block.block for mesh_block in self.blocks]
        if len(set(blocks)) != len(blocks):
            raise ValueError("a block has more than one mesh block")
        if self.kind is MeshBlocksKind.SNAPSHOT:
            empty = [mesh_block.block for mesh_block in self.blocks if not mesh_block.triangles]
            if empty:
                raise ValueError(f"a snapshot holds only blocks with surface; block {empty[0]} has none")

    def to_arrow(self) -> pa.Array:
        return one_row(
            {
                "map_version": self.map_version,
                "built_unix_ns": self.built_unix_ns,
                "run_started_unix_ns": self.run_started_unix_ns,
                "sequence": self.sequence,
                "kind": self.kind.value,
                "voxel_m": self.voxel_m,
                "block_voxels": self.block_voxels,
                "blocks": [
                    {
                        "block_x": p.block[0],
                        "block_y": p.block[1],
                        "block_z": p.block[2],
                        "positions_m": list(p.positions_m),
                        "triangles": list(p.triangles),
                    }
                    for p in self.blocks
                ],
            },
            MESH_BLOCKS_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "MeshBlocks":
        row = only_row(array, MESH_BLOCKS_TYPE)
        return cls(
            map_version=MapVersion(row["map_version"]),
            built_unix_ns=row["built_unix_ns"],
            run_started_unix_ns=row["run_started_unix_ns"],
            sequence=MeshSequence(row["sequence"]),
            kind=MeshBlocksKind(row["kind"]),
            voxel_m=row["voxel_m"],
            block_voxels=row["block_voxels"],
            blocks=tuple(
                MeshBlock((p["block_x"], p["block_y"], p["block_z"]), tuple(p["positions_m"]), tuple(p["triangles"]))
                for p in row["blocks"]
            ),
        )


@dataclass(frozen=True)
class MeshSnapshotRequest:
    """Asks the dense map to send its whole surface as the next mesh_blocks message, a snapshot."""

    requested_unix_ns: int

    def to_arrow(self) -> pa.Array:
        return one_row({"requested_unix_ns": self.requested_unix_ns}, MESH_SNAPSHOT_REQUEST_TYPE)

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "MeshSnapshotRequest":
        return cls(requested_unix_ns=only_row(array, MESH_SNAPSHOT_REQUEST_TYPE)["requested_unix_ns"])


def _square(x_m: float, y_m: float, side_m: float) -> tuple[Point, ...]:
    return (Point(x_m, y_m), Point(x_m + side_m, y_m), Point(x_m + side_m, y_m + side_m), Point(x_m, y_m + side_m))


# Golden examples of this module's messages, merged into fixtures.EXAMPLES. Every float is exact in float32.
FIXTURE_EXAMPLES: dict[str, Any] = {
    "occupancy_grid": OccupancyGrid(
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
    "mesh_blocks": MeshBlocks(
        map_version=MapVersion(412),
        built_unix_ns=1_762_560_003_000_000_000,
        run_started_unix_ns=1_762_559_990_000_000_000,
        sequence=MeshSequence(17),
        kind=MeshBlocksKind.CHANGES,
        voxel_m=0.0625,
        block_voxels=8,
        blocks=(
            # A square of floor in block (0, 1, -1), facing up.
            MeshBlock(
                (0, 1, -1),
                (0.0, 0.5, -0.125, 0.25, 0.5, -0.125, 0.25, 0.75, -0.125, 0.0, 0.75, -0.125),
                (0, 1, 2, 0, 2, 3),
            ),
            # Block (2, 0, 0) had surface and has none now.
            MeshBlock((2, 0, 0), (), ()),
        ),
    ),
    "mesh_blocks_snapshot": MeshBlocks(
        map_version=MapVersion(413),
        built_unix_ns=1_762_560_004_000_000_000,
        run_started_unix_ns=1_762_559_990_000_000_000,
        sequence=MeshSequence(18),
        kind=MeshBlocksKind.SNAPSHOT,
        voxel_m=0.0625,
        block_voxels=8,
        blocks=(
            MeshBlock(
                (0, 1, -1),
                (0.0, 0.5, -0.125, 0.25, 0.5, -0.125, 0.25, 0.75, -0.125, 0.0, 0.75, -0.125),
                (0, 1, 2, 0, 2, 3),
            ),
        ),
    ),
    "mesh_snapshot_request": MeshSnapshotRequest(requested_unix_ns=1_762_560_003_500_000_000),
}
