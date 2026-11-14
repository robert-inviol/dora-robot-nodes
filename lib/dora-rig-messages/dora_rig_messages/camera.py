"""How a depth camera's streams relate to each other: each image stream's intrinsics, and the rigid transforms from the
depth stream to the colour stream and to the IMU.

A transform maps a point in the `from` stream's coordinates (x right, y down, z forward, metres) into the `to`
stream's: `to = rotation @ from + translation`. The rotation is a 3x3 matrix stored column major, as librealsense
gives it.
"""

from dataclasses import dataclass

import pyarrow as pa

from .wire import one_row, only_row


def _fixed(item_type: pa.DataType) -> pa.ListType:
    return pa.list_(pa.field("item", item_type, nullable=False))


STREAM_INTRINSICS_TYPE = pa.struct(
    [
        pa.field("width", pa.uint16(), nullable=False),
        pa.field("height", pa.uint16(), nullable=False),
        pa.field("fx", pa.float32(), nullable=False),
        pa.field("fy", pa.float32(), nullable=False),
        pa.field("ppx", pa.float32(), nullable=False),
        pa.field("ppy", pa.float32(), nullable=False),
    ]
)

RIGID_TRANSFORM_TYPE = pa.struct(
    [
        pa.field("rotation", _fixed(pa.float32()), nullable=False),
        pa.field("translation_m", _fixed(pa.float32()), nullable=False),
    ]
)

CAMERA_CALIBRATION_TYPE = pa.struct(
    [
        pa.field("color", STREAM_INTRINSICS_TYPE, nullable=False),
        pa.field("depth", STREAM_INTRINSICS_TYPE, nullable=False),
        pa.field("depth_to_color", RIGID_TRANSFORM_TYPE, nullable=False),
        pa.field("depth_to_imu", RIGID_TRANSFORM_TYPE, nullable=False),
    ]
)


@dataclass(frozen=True)
class StreamIntrinsics:
    """A pinhole model of one image stream at its streamed size; lens distortion is not carried."""

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float


@dataclass(frozen=True)
class RigidTransform:
    rotation: tuple[float, float, float, float, float, float, float, float, float]
    translation_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.rotation) != 9 or len(self.translation_m) != 3:
            raise ValueError(
                f"a rigid transform has 9 rotation and 3 translation values, not {len(self.rotation)} and "
                f"{len(self.translation_m)}"
            )


@dataclass(frozen=True)
class CameraCalibration:
    color: StreamIntrinsics
    depth: StreamIntrinsics
    depth_to_color: RigidTransform
    depth_to_imu: RigidTransform

    def to_arrow(self) -> pa.Array:
        transform = lambda t: {"rotation": list(t.rotation), "translation_m": list(t.translation_m)}
        return one_row(
            {
                "color": vars(self.color),
                "depth": vars(self.depth),
                "depth_to_color": transform(self.depth_to_color),
                "depth_to_imu": transform(self.depth_to_imu),
            },
            CAMERA_CALIBRATION_TYPE,
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "CameraCalibration":
        row = only_row(array, CAMERA_CALIBRATION_TYPE)
        transform = lambda t: RigidTransform(tuple(t["rotation"]), tuple(t["translation_m"]))
        return cls(
            color=StreamIntrinsics(**row["color"]),
            depth=StreamIntrinsics(**row["depth"]),
            depth_to_color=transform(row["depth_to_color"]),
            depth_to_imu=transform(row["depth_to_imu"]),
        )


# Every float is exact in float32, so a fixture reads back equal to its example.
FIXTURE_EXAMPLES: dict[str, CameraCalibration] = {
    "camera_calibration": CameraCalibration(
        color=StreamIntrinsics(width=1280, height=720, fx=640.5, fy=640.25, ppx=639.75, ppy=360.5),
        depth=StreamIntrinsics(width=848, height=480, fx=421.5, fy=421.5, ppx=424.25, ppy=239.75),
        depth_to_color=RigidTransform((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (-0.0625, 0.0, 0.0)),
        depth_to_imu=RigidTransform((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (-0.03125, 0.0078125, -0.015625)),
    ),
}
