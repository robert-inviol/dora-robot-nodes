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
| nav goal | `navigation.py` | `goal_id`: uint64; `kind`: `"pose"`, `"object"`, `"place"`, `"follow_track"`, `"look_at"` or `"stop"`; `target_id`: uint64 object, place or track id by kind, null otherwise; `target_pose`: `x_m`, `y_m`, `yaw_rad` in the map frame, null unless the target is a pose (`look_at` takes an object id or a pose, `stop` neither); `standoff_m`, `max_speed_m_s`: float32, at least 0; `deadline_unix_ns`; `map_version` |
| goal status | `navigation.py` | `goal_id`; `state`: `"accepted"`, `"active"`, `"blocked"`, `"succeeded"` or `"failed"`; `reason`: one of `unknown_target`, `no_path`, `target_lost`, `person_in_path`, `stale_map`, `operator_override`, `deadline_passed`, `preempted` when blocked or failed, null otherwise; `updated_unix_ns` |
| space question | `navigation.py` | `question_id`: uint64; `asked_unix_ns`; `asker`: the agent or person asking, for the logs; `text` |
| space answer | `navigation.py` | `question_id`; `revision`: uint32, 0 first and counting up per message; `is_final`: true on the last message for the question; `model`: whose answer it is; `text`; `references`: each `kind` (`"object"`, `"place"`, `"room"`, `"track"` or `"pose"`), `id` (null for a pose) and `x_m`, `y_m` in the map frame; `goal_id`: the nav goal the robot is on because of the answer, or null |
| keyframe pose | `localization.py` | `frame_id`; `captured_unix_ns`; `map_version` of the pose graph, which rises whenever optimisation moves past keyframes; `x_m`, `y_m`, `z_m`, `yaw_rad`: where the keyframe's floor frame sits in the map; `tracking`: `"tracking"`, `"relocalised"` or `"lost"` (dead reckoning, not registered); `residual_m`: registration RMS of the inliers; `inlier_fraction` |
| map to odom | `localization.py` | `map_version`; `valid_from_camera_ns`: odometry poses from this time on the depth camera's clock are corrected with it; `x_m`, `y_m`, `z_m`, `yaw_rad` taking an odometry point into the map; `sigma_xy_m`, `sigma_z_m`, `sigma_yaw_rad`: at least 0 |
| free space | `space_map.py` | `map_version`; `built_unix_ns`; `origin_x_m`, `origin_y_m`: the outer corner of the first cell; `resolution_m`; `width`, `height`; `band_min_m`, `band_max_m`: the height band above the floor the grid describes; `clearance_mm`: uint16 per cell, row major, 0 occupied, 65535 unknown, otherwise millimetres to the nearest occupied cell |
| topology | `space_map.py` | `map_version`; `built_unix_ns`; `rooms` with `room_id`, `polygon`, `floor_m` and `ceiling_m` (null where no ceiling was seen); `doorways` with `doorway_id`, `room_a`, `room_b`, `x_m`, `y_m`, `width_m`; `places` with `place_id`, `room_id`, `x_m`, `y_m`, `clearance_m`; `edges` between places with `place_a`, `place_b`, `length_m`, `min_clearance_m` |
| keep out | `space_map.py` | `map_version`; `zones`, each with `object_id`, `label`, `polygon` and `margin_m`, how far beyond the polygon the robot keeps out; each message replaces the last |
| object update | `objects.py` | `kind`: `"appeared"`, `"updated"`, `"moved"`, `"disappeared"` or `"merged"`; `object_id`; `merged_into`: the surviving object, null unless merged; `label`; `score`: the best segmentation score; `confidence`: sightings over sightings plus misses; `seen_count`; `frame_id` and `captured_unix_ns` of the keyframe that caused the update; `map_version`; `centre_x_m`, `centre_y_m`, `centre_z_m`; `footprint`: the 4 corners of the floor-plan rectangle; `long_m`, `short_m`, `height_m`; `top_m`: the top's height in the map; `crop_jpeg`: the best view so far, null unless it is new to the receiver |

A nav goal names its target by id wherever it can, so the robot resolves where the target is when it acts.
An agent never sends drive demands; whatever executes the goal does, and its own safety limits still apply.

A drive demand says nothing about the robot underneath. Whatever drives the wheels or tracks
mixes `forward` and `turn` for its own kinematics, and because the magnitudes sum to at most 1 no
wheel is asked for more than full power.

```python
from dora_rig_messages.drive import DriveDemand

node.send_output("drive", DriveDemand(forward=0.2, turn=-0.1).to_arrow())
demand = DriveDemand.from_arrow(event["value"])
```

Map coordinates are z up, gravity aligned, metres, with yaw in radians anticlockwise seen from above.

Each module holds its messages. `localization.py`, `navigation.py`, `objects.py` and `space_map.py` also
hold a `FIXTURE_EXAMPLES` dict of golden examples, which `fixtures.py` merges with its own examples of the
other messages and of nav goals and goal statuses. After a deliberate change to a wire type, regenerate the
fixtures with `python -m dora_rig_messages.fixtures`.
