"""Where each keyframe is in the map, and the correction from the robot's odometry frame to the map.

The map frame is z up, gravity aligned, metres. A keyframe pose (x, y, z, yaw) is applied to that keyframe's
gravity-aligned floor frame (x right, y forward, z up, at the camera): a floor-frame point p is at
R_z(yaw) p + (x, y, z) in the map. Yaw is in radians, anticlockwise seen from above.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import NewType

import pyarrow as pa

from .navigation import MapVersion
from .wire import one_row, only_row

FrameId = NewType("FrameId", int)


class Tracking(Enum):
    # Registered against the map around the previous keyframe.
    TRACKING = "tracking"
    # Placed again by matching stored keyframes after tracking was lost.
    RELOCALISED = "relocalised"
    # Not registered: the pose is dead reckoning from the gyro and the last known position.
    LOST = "lost"


KEYFRAME_POSE_TYPE = pa.struct(
    [
        ("frame_id", pa.uint64()),
        ("captured_unix_ns", pa.int64()),
        # The pose graph the pose comes from; it rises whenever optimisation moves past keyframes.
        ("map_version", pa.uint64()),
        ("x_m", pa.float32()),
        ("y_m", pa.float32()),
        ("z_m", pa.float32()),
        ("yaw_rad", pa.float32()),
        ("tracking", pa.string()),
        # Registration RMS of the inliers; meaningful only when inlier_fraction is above zero.
        ("residual_m", pa.float32()),
        # Share of the keyframe's points within the inlier distance of the map surface.
        ("inlier_fraction", pa.float32()),
    ]
)

MAP_TO_ODOM_TYPE = pa.struct(
    [
        ("map_version", pa.uint64()),
        # On the depth camera's own clock: odometry poses from this time on are corrected with this transform.
        ("valid_from_camera_ns", pa.int64()),
        # p_map = R_z(yaw) p_odom + (x, y, z).
        ("x_m", pa.float32()),
        ("y_m", pa.float32()),
        ("z_m", pa.float32()),
        ("yaw_rad", pa.float32()),
        ("sigma_xy_m", pa.float32()),
        ("sigma_z_m", pa.float32()),
        ("sigma_yaw_rad", pa.float32()),
    ]
)


def _finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} is {value}, expected a finite value")


def _finite_not_negative(name: str, value: float) -> None:
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} is {value}, expected a finite value of at least 0")


@dataclass(frozen=True)
class KeyframePose:
    frame_id: FrameId
    captured_unix_ns: int
    map_version: MapVersion
    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float
    tracking: Tracking
    residual_m: float
    inlier_fraction: float

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "z_m", "yaw_rad"):
            _finite(name, getattr(self, name))
        _finite_not_negative("residual_m", self.residual_m)
        if not 0.0 <= self.inlier_fraction <= 1.0:
            raise ValueError(f"inlier_fraction is {self.inlier_fraction}, expected 0 to 1")

    def to_arrow(self) -> pa.Array:
        return one_row(
            {
                "frame_id": self.frame_id,
                "captured_unix_ns": self.captured_unix_ns,
                "map_version": self.map_version,
                "x_m": self.x_m,
                "y_m": self.y_m,
                "z_m": self.z_m,
                "yaw_rad": self.yaw_rad,
                "tracking": self.tracking.value,
                "residual_m": self.residual_m,
                "inlier_fraction": self.inlier_fraction,
            },
            KEYFRAME_POSE_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "KeyframePose":
        row = only_row(array, KEYFRAME_POSE_TYPE)
        return cls(
            frame_id=FrameId(row["frame_id"]),
            captured_unix_ns=row["captured_unix_ns"],
            map_version=MapVersion(row["map_version"]),
            x_m=row["x_m"],
            y_m=row["y_m"],
            z_m=row["z_m"],
            yaw_rad=row["yaw_rad"],
            tracking=Tracking(row["tracking"]),
            residual_m=row["residual_m"],
            inlier_fraction=row["inlier_fraction"],
        )


@dataclass(frozen=True)
class MapToOdom:
    map_version: MapVersion
    valid_from_camera_ns: int
    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float
    sigma_xy_m: float
    sigma_z_m: float
    sigma_yaw_rad: float

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "z_m", "yaw_rad"):
            _finite(name, getattr(self, name))
        for name in ("sigma_xy_m", "sigma_z_m", "sigma_yaw_rad"):
            _finite_not_negative(name, getattr(self, name))

    def to_arrow(self) -> pa.Array:
        return one_row(
            {
                "map_version": self.map_version,
                "valid_from_camera_ns": self.valid_from_camera_ns,
                "x_m": self.x_m,
                "y_m": self.y_m,
                "z_m": self.z_m,
                "yaw_rad": self.yaw_rad,
                "sigma_xy_m": self.sigma_xy_m,
                "sigma_z_m": self.sigma_z_m,
                "sigma_yaw_rad": self.sigma_yaw_rad,
            },
            MAP_TO_ODOM_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "MapToOdom":
        row = only_row(array, MAP_TO_ODOM_TYPE)
        return cls(
            map_version=MapVersion(row["map_version"]),
            valid_from_camera_ns=row["valid_from_camera_ns"],
            x_m=row["x_m"],
            y_m=row["y_m"],
            z_m=row["z_m"],
            yaw_rad=row["yaw_rad"],
            sigma_xy_m=row["sigma_xy_m"],
            sigma_z_m=row["sigma_z_m"],
            sigma_yaw_rad=row["sigma_yaw_rad"],
        )


# Golden examples of this module's messages, merged into fixtures.EXAMPLES. Every float is exact in float32.
FIXTURE_EXAMPLES: dict[str, KeyframePose | MapToOdom] = {
    "keyframe_pose_tracking": KeyframePose(
        frame_id=FrameId(42),
        captured_unix_ns=1_762_560_000_123_456_789,
        map_version=MapVersion(7),
        x_m=1.25,
        y_m=-0.5,
        z_m=0.0625,
        yaw_rad=0.75,
        tracking=Tracking.TRACKING,
        residual_m=0.015625,
        inlier_fraction=0.875,
    ),
    "keyframe_pose_lost": KeyframePose(
        frame_id=FrameId(43),
        captured_unix_ns=1_762_560_000_723_456_789,
        map_version=MapVersion(7),
        x_m=1.25,
        y_m=-0.5,
        z_m=0.0625,
        yaw_rad=1.0,
        tracking=Tracking.LOST,
        residual_m=0.0,
        inlier_fraction=0.0,
    ),
    "map_to_odom": MapToOdom(
        map_version=MapVersion(7),
        valid_from_camera_ns=1_762_560_000_100_000_000,
        x_m=0.5,
        y_m=-0.25,
        z_m=0.0,
        yaw_rad=-0.125,
        sigma_xy_m=0.03125,
        sigma_z_m=0.015625,
        sigma_yaw_rad=0.0078125,
    ),
}
