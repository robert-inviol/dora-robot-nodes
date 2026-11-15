"""A posed depth frame: one depth frame downsampled for mapping, with the odometry's pose of the camera that took it.

The camera frame is the depth camera's own (x right, y down, z forward, metres). The pose turns camera coordinates
into the odom frame `odom_started_camera_ns` names: a camera point p is at R(orientation) p + position in the odom
frame. The pose is the odometry's at exactly the frame's `camera_ns`, composed with the depth camera's extrinsics on
the body (the IMU); it is never interpolated.
"""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from .camera import STREAM_INTRINSICS_TYPE, StreamIntrinsics
from .localization import CameraNanos, FrameId


def _items(item_type: pa.DataType) -> pa.ListType:
    return pa.list_(pa.field("item", item_type, nullable=False))


POSED_DEPTH_FRAME_TYPE = pa.struct(
    [
        # The depth frame's id, as the camera node numbered it; the same as its odom pose's depth_frame_id.
        pa.field("frame_id", pa.uint64(), nullable=False),
        # The depth frame's capture time on the camera's own clock, and its odom pose's camera_ns.
        pa.field("camera_ns", pa.int64(), nullable=False),
        # The odom frame the pose is in, named by the camera time at which odometry started.
        pa.field("odom_started_camera_ns", pa.int64(), nullable=False),
        # The pinhole model of the downsampled image, at its size.
        pa.field("intrinsics", STREAM_INTRINSICS_TYPE, nullable=False),
        # The depth camera's optical centre in the odom frame, x y z.
        pa.field("camera_position_m", _items(pa.float32()), nullable=False),
        # Unit quaternion that turns a camera-frame vector into the odom frame.
        pa.field("camera_orientation_xyzw", _items(pa.float32()), nullable=False),
        # Row major, intrinsics.width by intrinsics.height, 0 where there is no reading.
        pa.field("depth_mm", _items(pa.uint16()), nullable=False),
    ]
)


def _one_list(values: np.ndarray, item_type: pa.DataType) -> pa.ListArray:
    flat = pa.array(values.ravel(), type=item_type)
    return pa.ListArray.from_arrays(pa.array([0, len(flat)], pa.int32()), flat, type=_items(item_type))


@dataclass(frozen=True, eq=False)
class PosedDepthFrame:
    frame_id: FrameId
    camera_ns: CameraNanos
    odom_started_camera_ns: CameraNanos
    intrinsics: StreamIntrinsics
    camera_position_m: tuple[float, float, float]
    camera_orientation_xyzw: tuple[float, float, float, float]
    # Shape (height, width).
    depth_mm: np.ndarray

    def __post_init__(self) -> None:
        if len(self.camera_position_m) != 3 or len(self.camera_orientation_xyzw) != 4:
            raise ValueError(
                f"a posed depth frame has {len(self.camera_position_m)} position and {len(self.camera_orientation_xyzw)} "
                "orientation values, not 3 and 4"
            )
        if self.depth_mm.shape != (self.intrinsics.height, self.intrinsics.width):
            raise ValueError(
                f"posed depth frame depth is {self.depth_mm.shape}, not the intrinsics' "
                f"({self.intrinsics.height}, {self.intrinsics.width})"
            )
        if self.camera_ns < self.odom_started_camera_ns:
            raise ValueError(f"camera_ns {self.camera_ns} is before its odom frame started, at {self.odom_started_camera_ns}")

    def to_arrow(self) -> pa.Array:
        columns = [
            pa.array([self.frame_id], pa.uint64()),
            pa.array([self.camera_ns], pa.int64()),
            pa.array([self.odom_started_camera_ns], pa.int64()),
            pa.array([vars(self.intrinsics)], STREAM_INTRINSICS_TYPE),
            _one_list(np.array(self.camera_position_m, np.float32), pa.float32()),
            _one_list(np.array(self.camera_orientation_xyzw, np.float32), pa.float32()),
            _one_list(self.depth_mm.astype(np.uint16, copy=False), pa.uint16()),
        ]
        return pa.StructArray.from_arrays(columns, fields=list(POSED_DEPTH_FRAME_TYPE))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PosedDepthFrame)
            and (self.frame_id, self.camera_ns, self.odom_started_camera_ns, self.intrinsics)
            == (other.frame_id, other.camera_ns, other.odom_started_camera_ns, other.intrinsics)
            and (self.camera_position_m, self.camera_orientation_xyzw) == (other.camera_position_m, other.camera_orientation_xyzw)
            and np.array_equal(self.depth_mm, other.depth_mm)
        )

    @classmethod
    def from_arrow(cls, array: pa.Array) -> "PosedDepthFrame":
        if array.type != POSED_DEPTH_FRAME_TYPE:
            raise ValueError(f"expected wire type {POSED_DEPTH_FRAME_TYPE}, received {array.type}")
        if len(array) != 1:
            raise ValueError(f"expected one row of {POSED_DEPTH_FRAME_TYPE}, received {len(array)}")
        if array.null_count:
            raise ValueError("a posed depth frame row is null")
        intrinsics = StreamIntrinsics(**array.field("intrinsics")[0].as_py())
        column = array.field("depth_mm")
        depth = column.values[column.offsets[0].as_py() : column.offsets[1].as_py()].to_numpy(zero_copy_only=False)
        if depth.size != intrinsics.width * intrinsics.height:
            raise ValueError(f"posed depth frame depth has {depth.size} values for {intrinsics.width}x{intrinsics.height}")
        scalar = lambda name: array.field(name)[0].as_py()
        return cls(
            frame_id=FrameId(scalar("frame_id")),
            camera_ns=CameraNanos(scalar("camera_ns")),
            odom_started_camera_ns=CameraNanos(scalar("odom_started_camera_ns")),
            intrinsics=intrinsics,
            camera_position_m=tuple(float(v) for v in scalar("camera_position_m")),
            camera_orientation_xyzw=tuple(float(v) for v in scalar("camera_orientation_xyzw")),
            depth_mm=depth.reshape(intrinsics.height, intrinsics.width),
        )


# Every float is exact in float32, so a fixture reads back equal to its example.
FIXTURE_EXAMPLES: dict[str, PosedDepthFrame] = {
    "posed_depth_frame": PosedDepthFrame(
        frame_id=FrameId(42),
        camera_ns=CameraNanos(1_762_560_000_100_000_000),
        odom_started_camera_ns=CameraNanos(1_762_559_990_000_000_000),
        intrinsics=StreamIntrinsics(width=3, height=2, fx=210.75, fy=210.75, ppx=211.875, ppy=119.625),
        camera_position_m=(1.25, -0.5, 0.25),
        camera_orientation_xyzw=(0.5, -0.5, 0.5, 0.5),
        depth_mm=np.array([[0, 1200, 1210], [2450, 65535, 400]], dtype=np.uint16),
    ),
}
