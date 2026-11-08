"""What a depth camera sees and what perception makes of it: keyframes, IMU samples, object concepts, segments, scenes.

Images travel as one flat list per row, row major. Coordinates in a scene are the gravity-aligned floor
frame at the camera: x right, y forward, z up, metres.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pyarrow as pa


def _items(item_type: pa.DataType) -> pa.ListType:
    return pa.list_(pa.field("item", item_type, nullable=False))


def _one_list(values: np.ndarray | list, item_type: pa.DataType) -> pa.ListArray:
    flat = pa.array(np.asarray(values).ravel() if not isinstance(values, list) else values, type=item_type)
    return pa.ListArray.from_arrays(pa.array([0, len(flat)], pa.int32()), flat, type=_items(item_type))


def _check(array: pa.Array, wire_type: pa.StructType) -> pa.StructArray:
    if array.type != wire_type:
        raise ValueError(f"expected wire type {wire_type}, received {array.type}")
    if len(array) != 1:
        raise ValueError(f"expected one row of {wire_type}, received {len(array)}")
    return array


def _scalar(array: pa.StructArray, field: str):
    return array.field(field)[0].as_py()


def _list_values(array: pa.StructArray, field: str) -> np.ndarray:
    """The one list in a column, as a numpy view where the type allows."""
    column = array.field(field)
    return column.values[column.offsets[0].as_py() : column.offsets[1].as_py()].to_numpy(zero_copy_only=False)


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    ppx: float
    ppy: float


KEYFRAME_TYPE = pa.struct(
    [
        ("frame_id", pa.uint64()),
        ("captured_unix_ns", pa.int64()),
        ("width", pa.uint16()),
        ("height", pa.uint16()),
        ("fx", pa.float32()),
        ("fy", pa.float32()),
        ("ppx", pa.float32()),
        ("ppy", pa.float32()),
        # The accelerometer at rest, in camera coordinates (x right, y down, z forward): the reaction to gravity.
        ("gravity_reaction", _items(pa.float32())),
        # Aligned to the colour image; 0 where the sensor has no reading.
        ("depth_mm", _items(pa.uint16())),
        ("color_jpeg", pa.binary()),
    ]
)


@dataclass(frozen=True, eq=False)
class Keyframe:
    frame_id: int
    captured_unix_ns: int
    intrinsics: CameraIntrinsics
    gravity_reaction: tuple[float, float, float]
    depth_mm: np.ndarray
    color_jpeg: bytes

    @property
    def height(self) -> int:
        return self.depth_mm.shape[0]

    @property
    def width(self) -> int:
        return self.depth_mm.shape[1]

    def to_arrow(self) -> pa.Array:
        k = self.intrinsics
        columns = {
            "frame_id": pa.array([self.frame_id], pa.uint64()),
            "captured_unix_ns": pa.array([self.captured_unix_ns], pa.int64()),
            "width": pa.array([self.width], pa.uint16()),
            "height": pa.array([self.height], pa.uint16()),
            "fx": pa.array([k.fx], pa.float32()),
            "fy": pa.array([k.fy], pa.float32()),
            "ppx": pa.array([k.ppx], pa.float32()),
            "ppy": pa.array([k.ppy], pa.float32()),
            "gravity_reaction": _one_list(list(self.gravity_reaction), pa.float32()),
            "depth_mm": _one_list(self.depth_mm.astype(np.uint16, copy=False), pa.uint16()),
            "color_jpeg": pa.array([self.color_jpeg], pa.binary()),
        }
        return pa.StructArray.from_arrays(list(columns.values()), fields=list(KEYFRAME_TYPE))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Keyframe)
            and (self.frame_id, self.captured_unix_ns, self.intrinsics, self.gravity_reaction, self.color_jpeg)
            == (other.frame_id, other.captured_unix_ns, other.intrinsics, other.gravity_reaction, other.color_jpeg)
            and np.array_equal(self.depth_mm, other.depth_mm)
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "Keyframe":
        row = _check(array, KEYFRAME_TYPE)
        width, height = _scalar(row, "width"), _scalar(row, "height")
        depth = _list_values(row, "depth_mm")
        if depth.size != width * height:
            raise ValueError(f"keyframe depth has {depth.size} values for {width}x{height}")
        gravity = _list_values(row, "gravity_reaction")
        if gravity.size != 3:
            raise ValueError(f"keyframe gravity reaction has {gravity.size} values, not 3")
        return cls(
            frame_id=_scalar(row, "frame_id"),
            captured_unix_ns=_scalar(row, "captured_unix_ns"),
            intrinsics=CameraIntrinsics(*(_scalar(row, f) for f in ("fx", "fy", "ppx", "ppy"))),
            gravity_reaction=tuple(float(g) for g in gravity),
            depth_mm=depth.reshape(height, width),
            color_jpeg=_scalar(row, "color_jpeg"),
        )


IMU_TYPE = pa.struct(
    [
        # The keyframe these samples lead up to; they cover the time since the previous keyframe.
        ("frame_id", pa.uint64()),
        # Camera timestamps are on the camera's own clock, shared by all its sensors, in nanoseconds, so a keyframe's
        # depth frame and the IMU samples line up. They are not unix times.
        ("keyframe_camera_ns", pa.int64()),
        ("accel_camera_ns", _items(pa.int64())),
        # x y z per sample, in the IMU's axes, which on the D455 are the colour camera's: x right, y down, z forward.
        ("accel_m_s2", _items(pa.float32())),
        ("gyro_camera_ns", _items(pa.int64())),
        ("gyro_rad_s", _items(pa.float32())),
    ]
)


@dataclass(frozen=True, eq=False)
class Imu:
    """Every accelerometer and gyro sample since the previous keyframe, oldest first."""

    frame_id: int
    keyframe_camera_ns: int
    accel_camera_ns: np.ndarray
    # One row of x y z per sample.
    accel_m_s2: np.ndarray
    gyro_camera_ns: np.ndarray
    gyro_rad_s: np.ndarray

    def __post_init__(self) -> None:
        for name, times, samples in (("accel", self.accel_camera_ns, self.accel_m_s2), ("gyro", self.gyro_camera_ns, self.gyro_rad_s)):
            if samples.shape != (len(times), 3):
                raise ValueError(f"imu {name} has {len(times)} timestamps for samples of shape {samples.shape}")

    def to_arrow(self) -> pa.Array:
        columns = [
            pa.array([self.frame_id], pa.uint64()),
            pa.array([self.keyframe_camera_ns], pa.int64()),
            _one_list(self.accel_camera_ns.astype(np.int64, copy=False), pa.int64()),
            _one_list(self.accel_m_s2.astype(np.float32, copy=False), pa.float32()),
            _one_list(self.gyro_camera_ns.astype(np.int64, copy=False), pa.int64()),
            _one_list(self.gyro_rad_s.astype(np.float32, copy=False), pa.float32()),
        ]
        return pa.StructArray.from_arrays(columns, fields=list(IMU_TYPE))

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "Imu":
        row = _check(array, IMU_TYPE)
        triples = lambda field: _list_values(row, field).reshape(-1, 3)
        return cls(
            frame_id=_scalar(row, "frame_id"),
            keyframe_camera_ns=_scalar(row, "keyframe_camera_ns"),
            accel_camera_ns=_list_values(row, "accel_camera_ns"),
            accel_m_s2=triples("accel_m_s2"),
            gyro_camera_ns=_list_values(row, "gyro_camera_ns"),
            gyro_rad_s=triples("gyro_rad_s"),
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Imu)
            and (self.frame_id, self.keyframe_camera_ns) == (other.frame_id, other.keyframe_camera_ns)
            and all(
                np.array_equal(a, b)
                for a, b in (
                    (self.accel_camera_ns, other.accel_camera_ns),
                    (self.accel_m_s2, other.accel_m_s2),
                    (self.gyro_camera_ns, other.gyro_camera_ns),
                    (self.gyro_rad_s, other.gyro_rad_s),
                )
            )
        )


CONCEPTS_TYPE = pa.struct(
    [
        ("model", pa.string()),
        # The keyframe the model looked at.
        ("frame_id", pa.uint64()),
        # Short noun phrases, one per object type, in the order the model named them.
        ("concepts", _items(pa.string())),
        # False while the model is still naming more.
        ("complete", pa.bool_()),
    ]
)


@dataclass(frozen=True)
class Concepts:
    model: str
    frame_id: int
    concepts: tuple[str, ...]
    complete: bool

    def to_arrow(self) -> pa.Array:
        return pa.array(
            [{"model": self.model, "frame_id": self.frame_id, "concepts": list(self.concepts), "complete": self.complete}],
            type=CONCEPTS_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "Concepts":
        row = _check(array, CONCEPTS_TYPE)[0].as_py()
        return cls(row["model"], row["frame_id"], tuple(row["concepts"]), row["complete"])


NO_SEGMENT = -1

SEGMENTS_TYPE = pa.struct(
    [
        ("frame_id", pa.uint64()),
        ("width", pa.uint16()),
        ("height", pa.uint16()),
        ("labels", _items(pa.string())),
        ("scores", _items(pa.float32())),
        # Per pixel, the index into labels, or -1; where masks overlap the smaller one wins.
        ("segment_ids", _items(pa.int16())),
    ]
)


@dataclass(frozen=True, eq=False)
class Segments:
    frame_id: int
    labels: tuple[str, ...]
    scores: tuple[float, ...]
    segment_ids: np.ndarray

    def to_arrow(self) -> pa.Array:
        height, width = self.segment_ids.shape
        columns = [
            pa.array([self.frame_id], pa.uint64()),
            pa.array([width], pa.uint16()),
            pa.array([height], pa.uint16()),
            _one_list(list(self.labels), pa.string()),
            _one_list(list(self.scores), pa.float32()),
            _one_list(self.segment_ids.astype(np.int16, copy=False), pa.int16()),
        ]
        return pa.StructArray.from_arrays(columns, fields=list(SEGMENTS_TYPE))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Segments)
            and (self.frame_id, self.labels, self.scores) == (other.frame_id, other.labels, other.scores)
            and np.array_equal(self.segment_ids, other.segment_ids)
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "Segments":
        row = _check(array, SEGMENTS_TYPE)
        width, height = _scalar(row, "width"), _scalar(row, "height")
        labels = tuple(row.field("labels")[0].as_py())
        scores = tuple(float(s) for s in _list_values(row, "scores"))
        if len(labels) != len(scores):
            raise ValueError(f"segments have {len(labels)} labels for {len(scores)} scores")
        return cls(_scalar(row, "frame_id"), labels, scores, _list_values(row, "segment_ids").reshape(height, width))


class SurfaceKind(str, Enum):
    FLOOR = "floor"
    CEILING = "ceiling"
    WALL = "wall"
    LEVEL_SURFACE = "level surface"


PLANE_TYPE = pa.struct(
    [
        ("kind", pa.string()),
        # normal . p + offset_m = 0; a level plane's normal is exactly up, so its height is -offset_m.
        ("normal", _items(pa.float32())),
        ("offset_m", pa.float32()),
        ("area_m2", pa.float32()),
        # Convex outline on the plane, x y z per corner.
        ("outline_m", _items(pa.float32())),
    ]
)

ROOM_TYPE = pa.struct(
    [
        ("floor_m", pa.float32()),
        ("ceiling_m", pa.float32()),
        # Floor-plan corners, x y per corner, counter-clockwise.
        ("footprint_m", _items(pa.float32())),
        # Per edge ending at the matching corner: whether a wall was seen along it.
        ("edge_seen", _items(pa.bool_())),
    ]
)

SCENE_OBJECT_TYPE = pa.struct(
    [
        ("label", pa.string()),
        # Index into the labels of the segments message this object came from.
        ("segment_index", pa.uint16()),
        ("score", pa.float32()),
        ("point_count", pa.uint32()),
        # Median depth along the optical axis.
        ("distance_m", pa.float32()),
        ("footprint_long_m", pa.float32()),
        ("footprint_short_m", pa.float32()),
        ("height_m", pa.float32()),
        ("top_m", pa.float32()),
        # Footprint rectangle, x y per corner.
        ("corners_m", _items(pa.float32())),
    ]
)

SCENE_TYPE = pa.struct(
    [
        ("frame_id", pa.uint64()),
        ("captured_unix_ns", pa.int64()),
        ("camera_forward", _items(pa.float32())),
        ("camera_up", _items(pa.float32())),
        ("planes", pa.list_(pa.field("item", PLANE_TYPE, nullable=False))),
        # Null when the room cannot be bounded; room_problem then says why.
        ("room", ROOM_TYPE),
        ("room_problem", pa.string()),
        # Only the segments with enough depth for a footprint.
        ("objects", pa.list_(pa.field("item", SCENE_OBJECT_TYPE, nullable=False))),
    ]
)


@dataclass(frozen=True)
class Plane:
    kind: SurfaceKind
    normal: tuple[float, float, float]
    offset_m: float
    area_m2: float
    outline_m: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class Room:
    floor_m: float
    ceiling_m: float
    footprint_m: tuple[tuple[float, float], ...]
    edge_seen: tuple[bool, ...]


@dataclass(frozen=True)
class SceneObject:
    label: str
    segment_index: int
    score: float
    point_count: int
    distance_m: float
    footprint_long_m: float
    footprint_short_m: float
    height_m: float
    top_m: float
    corners_m: tuple[tuple[float, float], ...]


def _pairs(flat: list[float], width: int) -> tuple:
    return tuple(tuple(flat[i : i + width]) for i in range(0, len(flat), width))


@dataclass(frozen=True)
class Scene:
    frame_id: int
    captured_unix_ns: int
    camera_forward: tuple[float, float, float]
    camera_up: tuple[float, float, float]
    planes: tuple[Plane, ...]
    room: Room | None
    room_problem: str | None
    objects: tuple[SceneObject, ...]

    def __post_init__(self) -> None:
        if (self.room is None) == (self.room_problem is None):
            raise ValueError("a scene has either a room or a room problem")

    def to_arrow(self) -> pa.Array:
        row = {
            "frame_id": self.frame_id,
            "captured_unix_ns": self.captured_unix_ns,
            "camera_forward": list(self.camera_forward),
            "camera_up": list(self.camera_up),
            "planes": [
                {"kind": p.kind.value, "normal": list(p.normal), "offset_m": p.offset_m, "area_m2": p.area_m2,
                 "outline_m": [c for corner in p.outline_m for c in corner]}
                for p in self.planes
            ],
            "room": None if self.room is None else {
                "floor_m": self.room.floor_m, "ceiling_m": self.room.ceiling_m,
                "footprint_m": [c for corner in self.room.footprint_m for c in corner], "edge_seen": list(self.room.edge_seen),
            },
            "room_problem": self.room_problem,
            "objects": [
                {"label": o.label, "segment_index": o.segment_index, "score": o.score, "point_count": o.point_count, "distance_m": o.distance_m,
                 "footprint_long_m": o.footprint_long_m, "footprint_short_m": o.footprint_short_m, "height_m": o.height_m,
                 "top_m": o.top_m, "corners_m": [c for corner in o.corners_m for c in corner]}
                for o in self.objects
            ],
        }
        return pa.array([row], type=SCENE_TYPE)

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "Scene":
        row = _check(array, SCENE_TYPE)[0].as_py()
        room = row["room"]
        return cls(
            frame_id=row["frame_id"],
            captured_unix_ns=row["captured_unix_ns"],
            camera_forward=tuple(row["camera_forward"]),
            camera_up=tuple(row["camera_up"]),
            planes=tuple(
                Plane(SurfaceKind(p["kind"]), tuple(p["normal"]), p["offset_m"], p["area_m2"], _pairs(p["outline_m"], 3))
                for p in row["planes"]
            ),
            room=None if room is None else Room(room["floor_m"], room["ceiling_m"], _pairs(room["footprint_m"], 2), tuple(room["edge_seen"])),
            room_problem=row["room_problem"],
            objects=tuple(
                SceneObject(o["label"], o["segment_index"], o["score"], o["point_count"], o["distance_m"], o["footprint_long_m"],
                            o["footprint_short_m"], o["height_m"], o["top_m"], _pairs(o["corners_m"], 2))
                for o in row["objects"]
            ),
        )
