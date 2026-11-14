# dora-depth-inertial-odometry

dora node that tracks the camera's pose from its depth frames and IMU samples with the
[`depth-inertial-odometry`](../depth-inertial-odometry/README.md) library, on the voxel-plane map. It has run on
the simulator's messages only, not yet on a camera.

| Input | Message | Notes |
|---|---|---|
| `calibration` | `camera_calibration` (lib/dora-rig-messages camera) | Its depth intrinsics and depth-to-IMU transform build the tracker. `queue_size: 1` is fine |
| `imu` | `imu` | Must not be dropped; every sample is integrated. Samples are in the IMU's own axes, which are the body frame |
| `depth` | Raw `u16` millimetres, row major, 0 for no reading. Metadata `width`, `height`, `frame_id`, `camera_ns` | As `dora-realsense` sends it. A dropped frame costs one pose and nothing else |

| Output | Message |
|---|---|
| `odom_pose` | `odom_pose` (lib/dora-rig-messages odometry): one for every depth frame tracked once odometry has started. The IMU body's position, orientation and velocity in the odom frame (z up, origin and zero heading where the IMU rested at the start), the sigmas, whether the frame's cloud corrected the pose, and the weakest direction. `camera_ns` is the depth frame's camera time and strictly increases |

## Starting up

1. **Calibration.** Nothing is tracked until the first `calibration` message. Every imu message and depth frame
   before it is dropped, with one warning. A later calibration whose depth intrinsics and depth-to-IMU transform
   equal the first is ignored; one that differs stops the node.
2. **Rest.** The camera must stand still for the rest time. A rest the camera moved through is refused with a
   warning and the rest begins again, so there is one warning for every rest time until the camera holds still.
   No pose is sent before odometry starts.
3. **Tracking.** The first frame's cloud becomes the map, so its pose has `corrected` false. Every later frame
   is corrected against the map; a frame whose cloud cannot correct still gets a pose, the IMU's prediction,
   with `corrected` false and no weakest direction.

## Arrival order

The camera sends the imu message leading up to a frame before the frame, but dora hands a node its inputs input
by input, so a depth frame can come first. Such a frame is held until the imu message for it has arrived. One
frame is held; a newer frame that also comes early takes its place, and a warning names the frame dropped.

## What stops the node

The node does not carry on past an input it cannot use. It stops with the error when:

- a calibration differs from the first, has a depth-to-IMU rotation that is not a rotation, or has depth
  intrinsics that cannot project an image;
- a depth frame's size is not the calibrated depth stream's, or its camera time is not after the last pose's;
- an IMU sample is not newer than the one before it in its stream, an imu message is for a frame no newer than
  the message before it, or a camera time is negative;
- a depth frame is more than `max_imu_lag` ahead of the newest sample of either IMU stream;
- one IMU stream has gone silent, or no depth frame has come while tracking, for more than `max_backlog` of
  camera time;
- an input is not one of the three above, or a message does not decode.

## Settings

The node runs `NodeSettings::d455()`: the library's `TrackerSettings::d455()` and
`PlaneVoxelSettings::indoors()`. It reads no settings file.

| Setting | Value | Meaning |
|---|---|---|
| `rest_for` | 2 s | How long the camera must rest before odometry starts |
| `depth_stride` | 4 | Every fourth pixel of every fourth row of a depth image becomes a cloud point |
| `max_imu_lag` | 10 ms | How far a depth frame may be ahead of the newest sample of either IMU stream |
| `max_backlog` | 1 s | The longest span of samples held for the other IMU stream or for a depth frame |

`RUST_LOG` picks what is logged; without it, info and above. A progress line is logged every 300 poses.

## Running

```
cargo build --release -p dora-depth-inertial-odometry
dora start dora-depth-inertial-odometry/camera-dataflow.yml
cargo test --release -p depth-inertial-odometry -p dora-depth-inertial-odometry
```

`camera-dataflow.yml` wires the node to `dora-realsense` on the camera's machine.
