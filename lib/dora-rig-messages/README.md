# dora-rig-messages

The messages a small robot's [dora](https://dora-rs.ai) nodes exchange, each defined once.

Every message is one row of an explicit Arrow struct type, never a schema inferred from a Python
dict, so a node written in another language can produce or read exactly the same thing. A golden
Arrow IPC file of each message lives in `dora_rig_messages/fixtures/`; this package's tests write
and read against those files, and a Rust node's tests can check against the same ones.

| Message | Module | Fields |
|---|---|---|
| drive demand | `drive.py` | `forward`, `turn`: float32 shares of full power in -1..1, magnitudes summing to at most 1. Positive turn steers right |
| operator command | `operator.py` | `mode`: `"manual"` or `"follow"`; `throttle`, `steer`: float32 in -1..1 |
| pilot status | `status.py` | `mode`; `locked_track_id`: int32, -1 when nobody is followed |
| driver status | `status.py` | `armed`: bool; `left_pct`, `right_pct`: int8 signed percent of full power |
| scan | `scan.py` | `spin_rev_per_s`: float32; `bearing_deg`: list of float32; `range_m`: list of float32, null where there was no return |
| keyframe | `perception.py` | `frame_id`; `captured_unix_ns`; `width`, `height`; intrinsics `fx`, `fy`, `ppx`, `ppy`; `gravity_reaction`: 3 float32, the accelerometer at rest in camera coordinates; `depth_mm`: uint16 per pixel aligned to colour, 0 where unknown; `color_jpeg` |
| imu | `perception.py` | `frame_id` of the keyframe the samples lead up to; `keyframe_camera_ns`; `accel_camera_ns` with `accel_m_s2` and `gyro_camera_ns` with `gyro_rad_s`: every sample since the previous keyframe, x y z per sample in the IMU's axes (on the D455 the colour camera's: x right, y down, z forward), timestamps in nanoseconds on the camera's own clock, which all its sensors share (not unix times) |
| concepts | `perception.py` | `model`; `frame_id`; `concepts`: object types named so far; `complete` |
| segments | `perception.py` | `frame_id`; `width`, `height`; `labels`; `scores`; `segment_ids`: int16 per pixel, index into labels or -1, smaller masks winning overlaps |
| scene | `perception.py` | `frame_id`; `captured_unix_ns`; `camera_forward`, `camera_up`; `planes`; `room` (null with a `room_problem` when it cannot be bounded); `objects` with footprints; all in the gravity-aligned floor frame (x right, y forward, z up, metres) |

A drive demand says nothing about the robot underneath. Whatever drives the wheels or tracks
mixes `forward` and `turn` for its own kinematics, and because the magnitudes sum to at most 1 no
wheel is asked for more than full power.

```python
from dora_rig_messages.drive import DriveDemand

node.send_output("drive", DriveDemand(forward=0.2, turn=-0.1).to_arrow())
demand = DriveDemand.from_arrow(event["value"])
```

After a deliberate change to a wire type, regenerate the fixtures with
`python -m dora_rig_messages.fixtures`.
