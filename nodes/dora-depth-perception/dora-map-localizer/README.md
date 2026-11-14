# dora-map-localizer

dora node that places every keyframe in the map with the `map-localizer` library.

| Input | Message | Notes |
|---|---|---|
| `keyframe` | `keyframe` | `queue_size: 1` is fine: only the newest keyframe matters |
| `imu` | `imu` | Must not be dropped; every gyro sample is integrated. A keyframe is localised once the imu message leading up to it has arrived |
| `odom_pose` | not defined | Reserved for the robot's odometry. There is no `odom_pose` message, so wiring it stops the node with an error |

| Output | Message |
|---|---|
| `keyframe_pose` | `keyframe_pose` (lib/dora-rig-messages localization): one per localised keyframe, with the map version, tracking state, registration RMS and inlier fraction. Metadata `localize_ms` is the time taken |

## Without a gyro turn

The gyro gives no turn until it has measured its bias on `[gyro] min_still_s` of still time (hold the camera
still at the start of a walk), nor across a gap in the IMU samples. `[gyro] heading_without_gyro` decides what
happens to such a keyframe:

- `refuse` (the default): the keyframe is reported `lost` and an error is logged naming why. Nothing is guessed:
  until the bias is measured the map does not start, and after a gap the heading stays unknown until a later
  keyframe relocalises against the map.
- `search`: every heading is searched for, and a warning is logged for each such keyframe. Only for recordings
  without a usable IMU.

For a recording made without an IMU, set `[gyro] imu = "absent"` together with `heading_without_gyro = "search"`
and leave the `imu` input unwired: each keyframe is then localised as it arrives, timed by its capture time. Any
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
