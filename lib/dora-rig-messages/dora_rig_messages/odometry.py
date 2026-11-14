"""Where odometry says the robot is: one pose per depth frame.

The odom frame is fixed where the IMU rested when odometry started, with z up; it drifts against the real room as
the robot travels. The body frame is the IMU's own axes (on the D455: x right, y down, z forward).

`camera_ns` strictly increases from one pose to the next. A consumer that wants the pose at another camera time
interpolates between the two rows nearest to it on `camera_ns`.

The sigmas say how well the pose is known against the odometry's own map. They do not include that map's drift.
"""

from dataclasses import dataclass

import pyarrow as pa

from .wire import one_row, only_row


def _items(item_type: pa.DataType) -> pa.ListType:
    return pa.list_(pa.field("item", item_type, nullable=False))


ODOM_POSE_TYPE = pa.struct(
    [
        # The time the pose is for, on the camera's own clock in nanoseconds (not a unix time): its depth frame's.
        ("camera_ns", pa.int64()),
        ("depth_frame_id", pa.uint64()),
        # The IMU body's position in the odom frame, x y z.
        ("position_m", _items(pa.float32())),
        # Unit quaternion that turns a body-frame vector into the odom frame.
        ("orientation_xyzw", _items(pa.float32())),
        # In the odom frame.
        ("velocity_m_s", _items(pa.float32())),
        # One sigma along the odom frame's x, y and z.
        ("position_sigma_m", _items(pa.float32())),
        # One sigma of the lean against gravity, about the level axis where it is known worst.
        ("tilt_sigma_rad", pa.float32()),
        # One sigma of the turn about up.
        ("heading_sigma_rad", pa.float32()),
        # Whether the depth frame's cloud corrected the pose. When false the pose is the IMU's prediction.
        ("corrected", pa.bool_()),
        # Cloud points that found a surface in the map; 0 when the cloud did not correct the pose.
        ("matched_points", pa.uint32()),
        # The odom-frame direction of travel the cloud pinned down least, either way round, and the one sigma the
        # cloud alone leaves along it. Both null when the cloud did not correct the pose.
        ("weakest_direction", _items(pa.float32())),
        ("weakest_sigma_m", pa.float32()),
    ]
)


def _vector(values: tuple[float, ...], length: int, name: str) -> tuple[float, ...]:
    if len(values) != length:
        raise ValueError(f"odom pose {name} has {len(values)} values, not {length}")
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class WeakestDirection:
    """The direction of travel a cloud pinned down least, and the one sigma the cloud alone leaves along it."""

    direction: tuple[float, float, float]
    sigma_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", _vector(self.direction, 3, "weakest_direction"))


@dataclass(frozen=True)
class OdomPose:
    """The pose at one depth frame. See the module docstring for the frames, interpolation and what the sigmas mean."""

    camera_ns: int
    depth_frame_id: int
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    velocity_m_s: tuple[float, float, float]
    position_sigma_m: tuple[float, float, float]
    tilt_sigma_rad: float
    heading_sigma_rad: float
    corrected: bool
    matched_points: int
    weakest: WeakestDirection | None

    def __post_init__(self) -> None:
        for name, length in (("position_m", 3), ("orientation_xyzw", 4), ("velocity_m_s", 3), ("position_sigma_m", 3)):
            object.__setattr__(self, name, _vector(getattr(self, name), length, name))

    def to_arrow(self) -> pa.Array:
        return one_row(
            {
                "camera_ns": self.camera_ns,
                "depth_frame_id": self.depth_frame_id,
                "position_m": list(self.position_m),
                "orientation_xyzw": list(self.orientation_xyzw),
                "velocity_m_s": list(self.velocity_m_s),
                "position_sigma_m": list(self.position_sigma_m),
                "tilt_sigma_rad": self.tilt_sigma_rad,
                "heading_sigma_rad": self.heading_sigma_rad,
                "corrected": self.corrected,
                "matched_points": self.matched_points,
                "weakest_direction": None if self.weakest is None else list(self.weakest.direction),
                "weakest_sigma_m": None if self.weakest is None else self.weakest.sigma_m,
            },
            ODOM_POSE_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "OdomPose":
        row = only_row(array, ODOM_POSE_TYPE)
        direction, sigma_m = row.pop("weakest_direction"), row.pop("weakest_sigma_m")
        if (direction is None) != (sigma_m is None):
            raise ValueError("odom pose has one of weakest_direction and weakest_sigma_m without the other")
        missing = sorted(name for name, value in row.items() if value is None)
        if missing:
            raise ValueError(f"odom pose is missing {', '.join(missing)}")
        return cls(**row, weakest=None if direction is None else WeakestDirection(tuple(direction), sigma_m))
