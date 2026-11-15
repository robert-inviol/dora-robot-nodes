# dora-map-localizer

dora node that places every keyframe in the map with the `map-localizer` library.

| Input | Message | Notes |
|---|---|---|
| `keyframe` | `keyframe` | `queue_size: 1` is fine: only the newest keyframe matters |
| `imu` | `imu` | Must not be dropped; every gyro sample is integrated. A keyframe is localized once the imu message leading up to it has arrived. A message with uneven arrays, or samples not later than the ones before, stops the node |
| `calibration` | `camera_calibration` (lib/dora-rig-messages camera), from dora-realsense: where the colour camera sits on the IMU. Needed with an `[odometry]` section, ignored without one |
| `odom_pose` | `odom_pose` (lib/dora-rig-messages odometry) | Only with an `[odometry]` section; without one an `odom_pose` message stops the node. Every pose is needed, in camera-time order |

| Output | Message |
|---|---|
| `keyframe_pose` | `keyframe_pose` (lib/dora-rig-messages localization): one per keyframe, with the map version, tracking state, registration RMS and inlier fraction, whether or not keyframe selection added it to the map. Metadata `localize_ms` is the time taken |
| `pose_graph_update` | `pose_graph_update` (lib/dora-rig-messages localization): after an optimization of the pose graph, every map keyframe already sent that it moved more than 1 cm or 0.23 degrees from the pose last sent for it, at its new pose, followed by the keyframes not selected that follow it. A map keyframe that restarted tracking after loss is sent `restarted` and is among them when it moves. A keyframe's own `keyframe_pose` is never sent again |
| `map_to_odom` | `map_to_odom` (lib/dora-rig-messages localization): with odometry, the transform taking odometry poses into the map, sent after every keyframe and every odometry pose that joined a keyframe |

## Odometry

With an `[odometry]` section the node joins every registered keyframe (tracked or relocalized) to the odometry
pose at the keyframe's camera time, interpolated between the two odometry poses either side, once odometry has
reached that time. The correction averages the latest `window_keyframes` joined keyframes, each at its current
optimized map pose, and its spread gives the sigmas, never below the `min_sigma_*` settings.

Odometry gives the IMU body's pose; the keyframe's floor frame is the colour camera's. The colour camera's
extrinsics on the IMU, where its optical centre sits and how its axes turn into the IMU's, come from the camera's
`calibration` message (its `depth_to_imu` after the inverse of `depth_to_color`). Odometry poses before the first
calibration are left out with a warning; a calibration whose rotations are not rotations, or that moves or turns
the camera against the IMU, stops the node. A pose with the
camera facing too near straight up or down has no heading and is left out with a warning. A keyframe that odometry
has not reached within `retention_s`, or that falls before the odometry kept, is left out of the correction with a
warning. Odometry going back in camera time stops the node.

A pose whose `odom_started_camera_ns` is later than the last pose's is from odometry that started again in a new
odom frame. The localizer warns, drops the old frame's poses and correction, and sends the next correction once a
keyframe joins the new frame. A waiting keyframe from before the new start is left out with a warning. A start
earlier than the frame in use stops the node, as does a camera time earlier than the last pose's across a restart.
Odometry needs the imu input, which gives keyframes their camera time: an `[odometry]` section with
`imu = "absent"` is refused.

## Without a gyro turn

The gyro gives no turn until it has measured its bias on `[gyro] min_stationary_s` of stationary time (hold the camera
still at the start of a walk), nor across a gap in the IMU samples. `[gyro] heading_without_gyro` decides what
happens to such a keyframe:

- `refuse` (the default): the keyframe is reported `lost` and an error is logged naming why, with `gyro_gaps`,
  the count of such keyframes since the node started. Nothing is guessed:
  until the bias is measured the map does not start, and after a gap the heading stays unknown until a later
  keyframe relocalizes against the map.
- `search`: every heading is searched for, and a warning is logged for each such keyframe. Only for recordings
  without a usable IMU.

For a recording made without an IMU, set `[gyro] imu = "absent"` together with `heading_without_gyro = "search"`
and leave the `imu` input unwired: each keyframe is then localized as it arrives, timed by its capture time. Any
other combination is refused when the settings are read, and an imu message arriving with `imu = "absent"` stops
the node.

`RUST_LOG` picks what is logged; without it, info and above.

## Settings

`MAP_LOCALIZER_CONFIG` names a TOML file; without it the library defaults apply. Every key is optional and an
unknown key is an error. `map-localizer.example.toml` lists every key with its default.

## Running

```
cargo build --release
dora start dora-map-localizer/replay-dataflow.yml
cargo test --release -p map-localizer -p dora-map-localizer
```

`replay-dataflow.yml` wires the node to `dora-keyframe-replay` playing a keyframe recording with its IMU
recording.
