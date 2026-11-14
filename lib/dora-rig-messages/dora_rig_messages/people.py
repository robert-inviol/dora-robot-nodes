"""People in front of the camera: their 2D poses as the pose model sees them, and their tracks placed in 3D.

Both messages carry one row per camera frame and are sent even when nobody is in view, so silence means the
producer has stopped. Pixel coordinates are in the colour image, origin top left. Keypoints are the 17 COCO body
points in COCO order.
"""

from dataclasses import dataclass
from enum import Enum

import pyarrow as pa

from .status import TrackId
from .wire import one_row, only_row

KEYPOINT_COUNT = 17


def _required(name: str, data_type: pa.DataType) -> pa.Field:
    return pa.field(name, data_type, nullable=False)


def _list_of(item_type: pa.DataType, nullable_items: bool = False) -> pa.ListType:
    return pa.list_(pa.field("item", item_type, nullable=nullable_items))


PIXEL_BOX_TYPE = pa.struct([_required(name, pa.float32()) for name in ("x_px", "y_px", "w_px", "h_px")])
KEYPOINT_TYPE = pa.struct([_required(name, pa.float32()) for name in ("x_px", "y_px", "score")])
POINT_TYPE = pa.struct([_required(name, pa.float32()) for name in ("x_m", "y_m", "z_m")])
VELOCITY_TYPE = pa.struct([_required(name, pa.float32()) for name in ("vx_m_s", "vy_m_s", "vz_m_s")])

PERSON_POSE_TYPE = pa.struct(
    [
        # Null until the tracker has numbered the person.
        pa.field("track_id", pa.uint64()),
        _required("score", pa.float32()),
        _required("box", PIXEL_BOX_TYPE),
        _required("keypoints", _list_of(KEYPOINT_TYPE)),
    ]
)

PERSON_POSES_TYPE = pa.struct(
    [
        _required("frame_id", pa.uint64()),
        _required("captured_camera_ns", pa.int64()),
        _required("width", pa.uint16()),
        _required("height", pa.uint16()),
        _required("people", _list_of(PERSON_POSE_TYPE)),
    ]
)

PERSON_TRACK_TYPE = pa.struct(
    [
        _required("track_id", pa.uint64()),
        _required("state", pa.string()),
        _required("score", pa.float32()),
        _required("box", PIXEL_BOX_TYPE),
        _required("keypoints", _list_of(KEYPOINT_TYPE)),
        pa.field("position", POINT_TYPE),
        pa.field("velocity", VELOCITY_TYPE),
        _required("keypoints_3d", _list_of(POINT_TYPE, nullable_items=True)),
    ]
)

PEOPLE_TRACKS_TYPE = pa.struct(
    [
        _required("frame_id", pa.uint64()),
        _required("captured_camera_ns", pa.int64()),
        _required("width", pa.uint16()),
        _required("height", pa.uint16()),
        _required("position_frame", pa.string()),
        # The accelerometer at rest in camera coordinates, as in a keyframe; null once positions are in odom.
        pa.field("gravity_reaction", _list_of(pa.float32())),
        _required("tracks", _list_of(PERSON_TRACK_TYPE)),
    ]
)


@dataclass(frozen=True)
class PixelBox:
    x_px: float
    y_px: float
    w_px: float
    h_px: float


@dataclass(frozen=True)
class Keypoint:
    x_px: float
    y_px: float
    score: float


@dataclass(frozen=True)
class Point:
    x_m: float
    y_m: float
    z_m: float


@dataclass(frozen=True)
class Velocity:
    vx_m_s: float
    vy_m_s: float
    vz_m_s: float


def _keypoints(keypoints: tuple) -> None:
    if len(keypoints) != KEYPOINT_COUNT:
        raise ValueError(f"a person has {KEYPOINT_COUNT} keypoints, not {len(keypoints)}")


@dataclass(frozen=True)
class PersonPose:
    track_id: TrackId | None
    score: float
    box: PixelBox
    keypoints: tuple[Keypoint, ...]

    def __post_init__(self) -> None:
        _keypoints(self.keypoints)

    def to_row(self) -> dict:
        return {
            "track_id": self.track_id,
            "score": self.score,
            "box": vars(self.box),
            "keypoints": [vars(k) for k in self.keypoints],
        }

    @classmethod
    def from_row(cls, row: dict) -> "PersonPose":
        track_id = row["track_id"]
        return cls(
            track_id=None if track_id is None else TrackId(track_id),
            score=row["score"],
            box=PixelBox(**row["box"]),
            keypoints=tuple(Keypoint(**k) for k in row["keypoints"]),
        )


@dataclass(frozen=True)
class PersonPoses:
    """Every person the pose model found in one colour frame."""

    frame_id: int
    captured_camera_ns: int
    width: int
    height: int
    people: tuple[PersonPose, ...]

    def to_arrow(self) -> pa.Array:
        return one_row(
            {
                "frame_id": self.frame_id,
                "captured_camera_ns": self.captured_camera_ns,
                "width": self.width,
                "height": self.height,
                "people": [person.to_row() for person in self.people],
            },
            PERSON_POSES_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "PersonPoses":
        row = only_row(array, PERSON_POSES_TYPE)
        return cls(
            frame_id=row["frame_id"],
            captured_camera_ns=row["captured_camera_ns"],
            width=row["width"],
            height=row["height"],
            people=tuple(PersonPose.from_row(person) for person in row["people"]),
        )


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    # The tracker has stopped seeing the person; position is where they were last seen.
    LOST = "lost"


class PositionFrame(Enum):
    # The colour camera: x right, y down, z forward.
    CAMERA = "camera"
    # Odometry: origin where the robot started, z up.
    ODOM = "odom"


@dataclass(frozen=True)
class PersonTrack:
    track_id: TrackId
    state: TrackState
    score: float
    box: PixelBox
    keypoints: tuple[Keypoint, ...]
    # The torso centre, when depth gives one.
    position: Point | None
    velocity: Velocity | None
    # In keypoint order; None where depth has no reading or the keypoint is not trusted.
    keypoints_3d: tuple[Point | None, ...]

    def __post_init__(self) -> None:
        _keypoints(self.keypoints)
        _keypoints(self.keypoints_3d)


@dataclass(frozen=True)
class PeopleTracks:
    frame_id: int
    captured_camera_ns: int
    width: int
    height: int
    position_frame: PositionFrame
    gravity_reaction: tuple[float, float, float] | None
    tracks: tuple[PersonTrack, ...]

    def to_arrow(self) -> pa.Array:
        optional = lambda value: None if value is None else vars(value)
        return one_row(
            {
                "frame_id": self.frame_id,
                "captured_camera_ns": self.captured_camera_ns,
                "width": self.width,
                "height": self.height,
                "position_frame": self.position_frame.value,
                "gravity_reaction": None if self.gravity_reaction is None else list(self.gravity_reaction),
                "tracks": [
                    {
                        "track_id": track.track_id,
                        "state": track.state.value,
                        "score": track.score,
                        "box": vars(track.box),
                        "keypoints": [vars(k) for k in track.keypoints],
                        "position": optional(track.position),
                        "velocity": optional(track.velocity),
                        "keypoints_3d": [optional(point) for point in track.keypoints_3d],
                    }
                    for track in self.tracks
                ],
            },
            PEOPLE_TRACKS_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "PeopleTracks":
        row = only_row(array, PEOPLE_TRACKS_TYPE)
        gravity = row["gravity_reaction"]
        if gravity is not None and len(gravity) != 3:
            raise ValueError(f"people tracks gravity reaction has {len(gravity)} values, not 3")
        point = lambda value: None if value is None else Point(**value)
        return cls(
            frame_id=row["frame_id"],
            captured_camera_ns=row["captured_camera_ns"],
            width=row["width"],
            height=row["height"],
            position_frame=PositionFrame(row["position_frame"]),
            gravity_reaction=None if gravity is None else tuple(gravity),
            tracks=tuple(
                PersonTrack(
                    track_id=TrackId(track["track_id"]),
                    state=TrackState(track["state"]),
                    score=track["score"],
                    box=PixelBox(**track["box"]),
                    keypoints=tuple(Keypoint(**k) for k in track["keypoints"]),
                    position=point(track["position"]),
                    velocity=None if track["velocity"] is None else Velocity(**track["velocity"]),
                    keypoints_3d=tuple(point(p) for p in track["keypoints_3d"]),
                )
                for track in row["tracks"]
            ),
        )
