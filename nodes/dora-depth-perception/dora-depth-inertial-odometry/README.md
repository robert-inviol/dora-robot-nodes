# dora-depth-inertial-odometry

dora node that tracks the camera's pose from its depth frames and IMU samples with the
[`depth-inertial-odometry`](../depth-inertial-odometry/README.md) library, on the voxel-plane map. It has run live
on the Pi with the camera at rest and people in view, on recordings of its inputs, and on the simulator's messages.

| Input | Message | Notes |
|---|---|---|
| `calibration` | `camera_calibration` (lib/dora-rig-messages camera) | Its depth intrinsics and depth-to-IMU transform build the tracker. `queue_size: 1` is fine |
| `imu` | `imu` | Must not be dropped; every sample is integrated. Samples are in the IMU's own axes, which are the body frame |
| `depth` | Raw `u16` millimetres, row major, 0 for no reading. Metadata `width`, `height`, `frame_id`, `camera_ns` | As `dora-realsense` sends it. A dropped frame costs one pose and nothing else |
| `person_poses` | `person_poses` (lib/dora-rig-messages people), from the pose model | Optional. The newest people's boxes are cut out of the depth frames taken up to 0.3 s before or after them, so that people moving neither drag the pose nor stay in the map. `queue_size: 1` |

| Output | Message |
|---|---|
| `odom_pose` | `odom_pose` (lib/dora-rig-messages odometry): one for every depth frame tracked once odometry has started. The IMU body's position, orientation and velocity in the odom frame (z up, origin and zero heading where the IMU was during static initialization), the sigmas, whether the frame's cloud corrected the pose, and the degenerate direction. `camera_ns` is the depth frame's camera time and strictly increases. `odom_started_camera_ns` is the camera time at which odometry started: the same on every pose of one run, and later when the node starts again, which tells a consumer the odom frame is a new one |

## Starting up

1. **Calibration.** Nothing is tracked until the first `calibration` message. Every imu message and depth frame
   before it is dropped, with one warning. A later calibration whose depth intrinsics and depth-to-IMU transform
   equal the first is ignored; one that differs stops the node.
2. **Static initialization.** The camera must be stationary for `stationary_for`. A window the camera moved
   through is refused with a warning and the window begins again, so there is one warning for every window until
   the camera holds still.
   No pose is sent before odometry starts.
3. **Tracking.** The first frame's cloud becomes the map, so its pose has `corrected` false. Every later frame
   is corrected against the map; a frame whose cloud cannot correct still gets a pose, the IMU's prediction,
   with `corrected` false and no degenerate direction.
4. **Dead reckoning.** A frame whose cloud cannot correct the pose (people fill the view, or it sees nothing the
   map holds) is posed by the IMU alone, with `corrected` false, and the filter's position uncertainty grows.
5. **Lost.** On such a frame, once the pose's largest one-sigma position uncertainty against the map passes
   `lost_beyond_position_sigma_m` (4.2 cm, a sixth of the map's 25 cm voxel, so that three sigmas still lie within
   half a voxel and the next clear view can correct it), the pose can no longer be trusted: the node warns that
   odometry is lost, with how the frames of the last second fared, sends no pose for that frame, and starts again
   from step 2, in a new odom frame with a later `odom_started_camera_ns`. With the camera at rest the uncertainty
   takes about 1.7 s to get there.

People are cut out of the depth before tracking. The pose model finds them in the colour image and reports them
later than the depth frame arrives, and the odometry does not wait for it: a depth frame is cut with the newest
boxes there are, if they are from within 0.3 s of it. Each box is grown on every side by 15% of its size and by
as far as a person walking at 1.5 m/s could have gone in the time between the two frames, as a share of a 0.5 m
wide person (30% more for a tenth of a second), then carried into the depth image, which is not aligned to colour,
by bounding where it can fall for anything from 0.4 to 6 m away. A person in the frames before the pose model
first reports them is not cut out.

The progress line, every 300 poses, says how the frames since the last one fared: how many were corrected, how
many had people cut out and the share of the image cut, and for the frames whose cloud was refused the points
given, after downsampling, that found a surface, and within the gate, against the rows needed (smallest, median
and largest of each), the largest position sigma, and the time each frame took to track. The loss warning says
the same of the last second.

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
| `stationary_for` | 2 s | How long the camera must be stationary before odometry starts (static initialization) |
| `depth_stride` | 4 | Every fourth pixel of every fourth row of a depth image becomes a cloud point |
| `max_imu_lag` | 10 ms | How far a depth frame may be ahead of the newest sample of either IMU stream |
| `max_backlog` | 1 s | The longest span of samples held for the other IMU stream or for a depth frame |
| `lost_beyond_position_sigma_m` | 0.0417 m | A frame the cloud did not correct with a larger largest position sigma than this is lost |
| `people.margin` | 0.15 | Each side of a person's box grows by this share of its size |
| `people.walking_speed_m_s`, `people.person_width_m` | 1.5 m/s, 0.5 m | And by the share of its size a person walking this fast covers, over this width, in the box's age |
| `people.max_age_ns` | 0.3 s | Boxes further than this in time from a depth frame cut nothing out of it |

`RUST_LOG` picks what is logged; without it, info and above. A progress line is logged every 300 poses.

## Running

```
cargo build --release -p dora-depth-inertial-odometry
dora start dora-depth-inertial-odometry/camera-dataflow.yml
cargo test --release -p depth-inertial-odometry -p dora-depth-inertial-odometry
```

`camera-dataflow.yml` wires the node to `dora-realsense` on the camera's machine.

`replay-odometry-inputs <recording directory>` runs the node's handling over a recording made by
`dora-odometry-input-recorder` (see the library's README, "Recording and replay"), offline, and prints each loss,
the pose once a second with how far it moved and its sigma, and the counts and timing of the frames.
