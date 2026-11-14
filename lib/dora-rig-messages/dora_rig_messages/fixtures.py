"""Golden examples of every message, stored as Arrow IPC files that each language's tests read.

Regenerate with `python -m dora_rig_messages.fixtures` after a deliberate change to a wire type.
"""

from pathlib import Path
from typing import Protocol

import pyarrow as pa

from .camera import CameraCalibration, RigidTransform, StreamIntrinsics
from .drive import DriveDemand
import numpy as np

from .operator import DriveMode, OperatorCommand
from .people import (
    Keypoint, PeopleTracks, PersonPose, PersonPoses, PersonTrack, PixelBox, Point, PositionFrame, TrackState, Velocity,
)
from .perception import (
    CameraIntrinsics, Concepts, Imu, Keyframe, Plane, Room, Scene, SceneObject, Segments, SurfaceKind,
)
from .scan import RangeReading, Scan
from .status import DriverStatus, PilotStatus, TrackId

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class Message(Protocol):
    def to_arrow(self) -> pa.Array: ...


_KEYPOINTS = tuple(Keypoint(400.0 + 8.0 * i, 140.0 + 24.0 * i, 0.75 if i % 4 else 0.25) for i in range(17))

# Every float is exact in float32, so a fixture reads back equal to its example.
EXAMPLES: dict[str, Message] = {
    "drive_demand": DriveDemand(forward=0.5, turn=-0.25),
    "operator_command": OperatorCommand(DriveMode.FOLLOW, throttle=0.75, steer=-0.5),
    "pilot_status": PilotStatus(DriveMode.FOLLOW, locked_id=TrackId(7)),
    "pilot_status_nobody_locked": PilotStatus(DriveMode.MANUAL, locked_id=None),
    "driver_status": DriverStatus(armed=True, left_pct=35, right_pct=-12),
    "scan": Scan(
        spin_rev_per_s=8.0,
        readings=(
            RangeReading(bearing_deg=0.0, range_m=0.5),
            RangeReading(bearing_deg=90.0, range_m=None),
            RangeReading(bearing_deg=180.0, range_m=3.25),
            RangeReading(bearing_deg=292.5, range_m=11.75),
        ),
    ),
    "keyframe": Keyframe(
        frame_id=42,
        captured_unix_ns=1_762_560_000_123_456_789,
        intrinsics=CameraIntrinsics(fx=640.5, fy=640.25, ppx=639.75, ppy=360.5),
        gravity_reaction=(-0.0625, -9.75, -0.375),
        depth_mm=np.array([[0, 1200, 1210], [2450, 65535, 400]], dtype=np.uint16),
        color_jpeg=bytes([0xFF, 0xD8, 0xFF, 0xD9]),
    ),
    "imu": Imu(
        frame_id=42,
        keyframe_camera_ns=1_762_560_000_100_000_000,
        accel_camera_ns=np.array([1_762_560_000_090_000_000, 1_762_560_000_095_000_000], dtype=np.int64),
        accel_m_s2=np.array([[-0.0625, -9.75, -0.375], [0.125, -9.8125, -0.25]], dtype=np.float32),
        gyro_camera_ns=np.array([1_762_560_000_092_500_000, 1_762_560_000_095_000_000, 1_762_560_000_097_500_000], dtype=np.int64),
        gyro_rad_s=np.array([[0.0, 0.5, 0.0], [0.0, 0.25, -0.125], [0.03125, 0.0, 0.0]], dtype=np.float32),
    ),
    "concepts": Concepts(model="claude-opus-5-5", frame_id=42, concepts=("sofa", "red blanket"), complete=False),
    "segments": Segments(
        frame_id=42,
        labels=("sofa", "cushion"),
        scores=(0.875, 0.5),
        segment_ids=np.array([[-1, 0, 0], [1, 0, -1]], dtype=np.int16),
    ),
    "scene": Scene(
        frame_id=42,
        captured_unix_ns=1_762_560_000_123_456_789,
        camera_forward=(0.0, 1.0, 0.0),
        camera_up=(0.0, 0.0, 1.0),
        planes=(
            Plane(SurfaceKind.FLOOR, (0.0, 0.0, 1.0), 0.875, 5.25, ((0.0, 1.0, -0.875), (1.0, 1.0, -0.875), (0.5, 2.0, -0.875))),
            Plane(SurfaceKind.WALL, (0.0, -1.0, 0.0), 4.75, 3.5, ((-1.0, 4.75, 0.0), (1.0, 4.75, 0.0), (0.0, 4.75, 1.5))),
        ),
        room=Room(-0.875, 1.5, ((-1.5, 0.25), (2.0, 0.25), (2.0, 4.75), (-1.5, 4.75)), (False, True, True, True)),
        room_problem=None,
        objects=(SceneObject("sofa", 0, 0.875, 23580, 1.25, 1.75, 0.5, 0.5, -0.375, ((0.0, 1.0), (1.75, 1.0), (1.75, 1.5), (0.0, 1.5))),),
    ),
    "scene_without_room": Scene(
        frame_id=43,
        captured_unix_ns=1_762_560_001_000_000_000,
        camera_forward=(0.0, 1.0, 0.0),
        camera_up=(0.0, 0.0, 1.0),
        planes=(),
        room=None,
        room_problem="no ceiling plane found, cannot bound the room",
        objects=(),
    ),
    "camera_calibration": CameraCalibration(
        color=StreamIntrinsics(width=1280, height=720, fx=640.5, fy=640.25, ppx=639.75, ppy=360.5),
        depth=StreamIntrinsics(width=848, height=480, fx=421.5, fy=421.5, ppx=424.25, ppy=239.75),
        depth_to_color=RigidTransform((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (-0.0625, 0.0, 0.0)),
        depth_to_imu=RigidTransform((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (-0.03125, 0.0078125, -0.015625)),
    ),
    "person_poses": PersonPoses(
        frame_id=42,
        captured_camera_ns=1_762_560_000_100_000_000,
        width=1280,
        height=720,
        people=(
            PersonPose(TrackId(3), 0.875, PixelBox(400.0, 120.0, 180.0, 520.0), _KEYPOINTS),
            PersonPose(None, 0.5, PixelBox(900.5, 200.25, 90.0, 300.0), _KEYPOINTS),
        ),
    ),
    "person_poses_nobody": PersonPoses(frame_id=43, captured_camera_ns=1_762_560_000_133_333_333, width=1280, height=720, people=()),
    "people_tracks": PeopleTracks(
        frame_id=42,
        captured_camera_ns=1_762_560_000_100_000_000,
        width=1280,
        height=720,
        position_frame=PositionFrame.CAMERA,
        gravity_reaction=(-0.0625, -9.75, -0.375),
        tracks=(
            PersonTrack(
                TrackId(3), TrackState.CONFIRMED, 0.875, PixelBox(400.0, 120.0, 180.0, 520.0), _KEYPOINTS,
                position=Point(-0.25, 0.125, 2.5),
                velocity=Velocity(0.5, 0.0, -0.25),
                keypoints_3d=(Point(-0.25, -0.625, 2.5),) * 16 + (None,),
            ),
            PersonTrack(
                TrackId(9), TrackState.LOST, 0.5, PixelBox(900.5, 200.25, 90.0, 300.0), _KEYPOINTS,
                position=None, velocity=None, keypoints_3d=(None,) * 17,
            ),
        ),
    ),
    "people_tracks_in_odom": PeopleTracks(
        frame_id=43, captured_camera_ns=1_762_560_000_133_333_333, width=1280, height=720,
        position_frame=PositionFrame.ODOM, gravity_reaction=None, tracks=(),
    ),
}


def fixture_path(name: str) -> Path:
    return FIXTURE_DIR / f"{name}.arrow"


def read_fixture(name: str) -> pa.Array:
    with pa.ipc.open_file(fixture_path(name)) as reader:
        return reader.read_all().to_batches()[0].to_struct_array()


def write_fixture(name: str, message: Message) -> None:
    batch = pa.RecordBatch.from_struct_array(message.to_arrow())
    with pa.ipc.new_file(fixture_path(name), batch.schema) as writer:
        writer.write_batch(batch)


if __name__ == "__main__":
    for example_name, example in EXAMPLES.items():
        write_fixture(example_name, example)
        print(f"wrote {fixture_path(example_name)}")
