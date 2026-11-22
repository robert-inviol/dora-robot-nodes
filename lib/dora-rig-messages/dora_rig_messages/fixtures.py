"""Golden examples of every message, stored as Arrow IPC files that each language's tests read, and the table of
every wire type's fingerprint, which the Rust wire-schemas crate's tests compare its own definitions with.

Regenerate with `python -m dora_rig_messages.fixtures` after a deliberate change to a wire type.
"""

import json
from pathlib import Path
from typing import Protocol

import pyarrow as pa

import numpy as np

from . import camera, localization, navigation, objects, people, posed_depth, space_map
from .drive import DRIVE_DEMAND_TYPE, DriveDemand
from .navigation import (
    GoalId, GoalKind, GoalReason, GoalState, GoalStatus, MapPose, MapVersion, NavGoal, ObjectId,
)
from .odometry import ODOM_POSE_TYPE, OdomPose, DegenerateDirection
from .operator import OPERATOR_COMMAND_TYPE, DriveMode, OperatorCommand
from .perception import (
    IMU_TYPE, INSTANCE_MASKS_TYPE, KEYFRAME_TYPE, LABELS_TYPE, SCENE_TYPE,
    CameraIntrinsics, Labels, Imu, Keyframe, Plane, Room, Scene, SceneObject, InstanceMasks, SurfaceKind,
)
from .scan import SCAN_TYPE, RangeReading, Scan
from .status import DRIVER_STATUS_TYPE, PILOT_STATUS_TYPE, DriverStatus, PilotStatus, TrackId
from .wire import canonical_encoding, fingerprint

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FINGERPRINT_TABLE = FIXTURE_DIR / "wire_fingerprints.json"

# Every message's wire type under its stable name, the name the Rust wire-schemas crate gives it.
WIRE_TYPES: dict[str, pa.StructType] = {
    "drive_demand": DRIVE_DEMAND_TYPE,
    "operator_command": OPERATOR_COMMAND_TYPE,
    "pilot_status": PILOT_STATUS_TYPE,
    "driver_status": DRIVER_STATUS_TYPE,
    "scan": SCAN_TYPE,
    "keyframe": KEYFRAME_TYPE,
    "imu": IMU_TYPE,
    "labels": LABELS_TYPE,
    "instance_masks": INSTANCE_MASKS_TYPE,
    "scene": SCENE_TYPE,
    "camera_calibration": camera.CAMERA_CALIBRATION_TYPE,
    "keyframe_pose": localization.KEYFRAME_POSE_TYPE,
    "map_to_odom": localization.MAP_TO_ODOM_TYPE,
    "pose_graph_update": localization.KEYFRAME_POSES_MOVED_TYPE,
    "nav_goal": navigation.NAV_GOAL_TYPE,
    "goal_status": navigation.GOAL_STATUS_TYPE,
    "spatial_query": navigation.SPATIAL_QUERY_TYPE,
    "spatial_answer": navigation.SPATIAL_ANSWER_TYPE,
    "object_update": objects.OBJECT_UPDATE_TYPE,
    "odom_pose": ODOM_POSE_TYPE,
    "person_poses": people.PERSON_POSES_TYPE,
    "people_tracks": people.PEOPLE_TRACKS_TYPE,
    "posed_depth_frame": posed_depth.POSED_DEPTH_FRAME_TYPE,
    "occupancy_grid": space_map.OCCUPANCY_GRID_TYPE,
    "topology": space_map.TOPOLOGY_TYPE,
    "keep_out": space_map.KEEP_OUT_TYPE,
    "mesh_blocks": space_map.MESH_BLOCKS_TYPE,
    "mesh_snapshot_request": space_map.MESH_SNAPSHOT_REQUEST_TYPE,
}


class Message(Protocol):
    def to_arrow(self) -> pa.Array: ...


# Every float is exact in float32, so a fixture reads back equal to its example.
EXAMPLES: dict[str, Message] = {
    "drive_demand": DriveDemand(forward=0.5, turn=-0.25),
    "operator_command": OperatorCommand(DriveMode.FOLLOW, throttle=0.75, steer=-0.5),
    "pilot_status": PilotStatus(DriveMode.FOLLOW, locked_id=TrackId(7)),
    "pilot_status_nobody_locked": PilotStatus(DriveMode.MANUAL, locked_id=None),
    "driver_status": DriverStatus(armed=True, left_pct=35, right_pct=-12),
    "scan": Scan(
        spin_rev_per_s=10.0,
        first_reading_unix_ns=1_762_560_000_100_000_000,
        last_reading_unix_ns=1_762_560_000_197_500_000,
        readings=(
            RangeReading(bearing_deg=0.0, range_m=0.5),
            RangeReading(bearing_deg=90.0, range_m=None),
            RangeReading(bearing_deg=180.0, range_m=3.25),
            RangeReading(bearing_deg=292.5, range_m=11.75),
        ),
        intensities=(212, 0, 87, 9),
    ),
    "scan_without_intensity": Scan(
        spin_rev_per_s=8.0,
        first_reading_unix_ns=1_762_560_000_100_000_000,
        last_reading_unix_ns=1_762_560_000_225_000_000,
        readings=(
            RangeReading(bearing_deg=0.0, range_m=0.5),
            RangeReading(bearing_deg=90.0, range_m=None),
        ),
        intensities=None,
    ),
    "keyframe": Keyframe(
        frame_id=42,
        captured_unix_ns=1_762_560_000_123_456_789,
        intrinsics=CameraIntrinsics(fx=640.5, fy=640.25, ppx=639.75, ppy=360.5),
        specific_force=(-0.0625, -9.75, -0.375),
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
    "labels": Labels(model="claude-opus-5-5", frame_id=42, labels=("sofa", "red blanket"), complete=False),
    "instance_masks": InstanceMasks(
        frame_id=42,
        labels=("sofa", "cushion"),
        scores=(0.875, 0.5),
        instance_ids=np.array([[-1, 0, 0], [1, 0, -1]], dtype=np.int16),
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
    "odom_pose": OdomPose(
        camera_ns=1_762_560_000_100_000_000,
        odom_started_camera_ns=1_762_559_990_000_000_000,
        depth_frame_id=42,
        position_m=(1.25, -0.5, 0.25),
        orientation_xyzw=(0.5, -0.5, 0.5, 0.5),
        velocity_m_s=(0.375, 0.0, -0.0625),
        position_sigma_m=(0.015625, 0.03125, 0.0078125),
        tilt_sigma_rad=0.00390625,
        heading_sigma_rad=0.0078125,
        corrected=True,
        matched_points=1234,
        degenerate=DegenerateDirection(direction=(0.0, 1.0, 0.0), sigma_m=0.125),
    ),
    "odom_pose_predicted": OdomPose(
        camera_ns=1_762_560_000_133_000_000,
        odom_started_camera_ns=1_762_559_990_000_000_000,
        depth_frame_id=43,
        position_m=(1.5, -0.5, 0.25),
        orientation_xyzw=(0.0, 0.0, 1.0, 0.0),
        velocity_m_s=(0.375, 0.0, 0.0),
        position_sigma_m=(0.03125, 0.0625, 0.015625),
        tilt_sigma_rad=0.00390625,
        heading_sigma_rad=0.015625,
        corrected=False,
        matched_points=0,
        degenerate=None,
    ),
    "nav_goal_object": NavGoal(
        goal_id=GoalId(7),
        kind=GoalKind.OBJECT,
        target_id=ObjectId(42),
        target_pose=None,
        standoff_m=0.5,
        max_speed_m_s=0.25,
        deadline_unix_ns=1_762_560_060_000_000_000,
        map_version=MapVersion(412),
    ),
    "nav_goal_pose": NavGoal(
        goal_id=GoalId(8),
        kind=GoalKind.POSE,
        target_id=None,
        target_pose=MapPose(x_m=3.25, y_m=-0.75, yaw_rad=1.5),
        standoff_m=0.0,
        max_speed_m_s=0.5,
        deadline_unix_ns=1_762_560_090_000_000_000,
        map_version=MapVersion(412),
    ),
    "nav_goal_stop": NavGoal(
        goal_id=GoalId(9),
        kind=GoalKind.STOP,
        target_id=None,
        target_pose=None,
        standoff_m=0.0,
        max_speed_m_s=0.0,
        deadline_unix_ns=1_762_560_005_000_000_000,
        map_version=MapVersion(412),
    ),
    "goal_status_active": GoalStatus(GoalId(7), GoalState.ACTIVE, reason=None, updated_unix_ns=1_762_560_001_500_000_000),
    "goal_status_blocked": GoalStatus(
        GoalId(7), GoalState.BLOCKED, reason=GoalReason.PERSON_IN_PATH, updated_unix_ns=1_762_560_004_250_000_000
    ),
}

for _area in (camera, localization, navigation, objects, people, posed_depth, space_map):
    EXAMPLES.update(_area.FIXTURE_EXAMPLES)


def fixture_path(name: str) -> Path:
    return FIXTURE_DIR / f"{name}.arrow"


def read_fixture(name: str) -> pa.Array:
    with pa.ipc.open_file(fixture_path(name)) as reader:
        return reader.read_all().to_batches()[0].to_struct_array()


def write_fixture(name: str, message: Message) -> None:
    batch = pa.RecordBatch.from_struct_array(message.to_arrow())
    with pa.ipc.new_file(fixture_path(name), batch.schema) as writer:
        writer.write_batch(batch)


def fingerprint_table() -> dict[str, dict[str, str]]:
    """Every wire type's fingerprint and the canonical encoding it is the hash of, by name."""
    return {
        name: {"fingerprint": fingerprint(wire_type), "canonical": canonical_encoding(wire_type)}
        for name, wire_type in sorted(WIRE_TYPES.items())
    }


def read_fingerprint_table() -> dict[str, dict[str, str]]:
    return json.loads(FINGERPRINT_TABLE.read_text())


def write_fingerprint_table() -> None:
    FINGERPRINT_TABLE.write_text(json.dumps(fingerprint_table(), indent=2) + "\n")


if __name__ == "__main__":
    for example_name, example in EXAMPLES.items():
        write_fixture(example_name, example)
        print(f"wrote {fixture_path(example_name)}")
    write_fingerprint_table()
    print(f"wrote {FINGERPRINT_TABLE}")
