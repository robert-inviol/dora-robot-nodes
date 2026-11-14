# dora-realsense

A [dora](https://dora-rs.ai) node that owns an Intel RealSense D400 with an IMU (built against a D455) and publishes
every frame: 1280x720 colour and 848x480 depth at 30 fps, accelerometer at 200 Hz and gyro at 400 Hz. Other nodes
take what they need from it; nothing else should open the camera.

Every stamp is on the camera's own clock, which all its sensors share, in nanoseconds. librealsense's host-synced
global time is turned off, because it drifts apart between the motion and video sensors. Frames go out from the
capture thread as soon as they arrive. An output the dataflow does not list costs nothing, so a dataflow that only
wants depth does no image work.

`rgbd-frames` holds the letterbox and the IMU correction and builds anywhere. The node links librealsense2, so it
builds only on the camera's machine: `cargo build --release -p dora-realsense`.

| Input | What it carries |
|---|---|
| `tick` | Required, any dora timer. It carries nothing; dora closes the events of a node without inputs, and a stop could then never reach it |
| `request_color` | Optional: `u64` frame ids. When given, `color` is sent only for the frames named here, instead of every frame. The node keeps the last 15 colour frames to answer from and fails on a request for an older one |

| Output | What it carries |
|---|---|
| `model_input` | The colour image shrunk by a whole factor (2 for 720p into 640), each pixel the average of its block, centred on a grey (114) square: the input a YOLO model takes. Raw `rgb8` bytes. Metadata: `encoding`, `width`, `height`, `frame_id`, `camera_ns`, `source_width`, `source_height`, `shrink`, `pad_x`, `pad_y`; a point at (x, y) in the square is at ((x - pad_x) * shrink, (y - pad_y) * shrink) in the colour image |
| `depth` | Raw depth, not aligned to colour: `u16` millimetres, row major, 0 where the sensor has no reading. Metadata: `encoding` (`mono16`), `width`, `height`, `frame_id`, `camera_ns` (the depth frame's capture time), and `gravity_reaction` (three floats, the accelerometer smoothed over half a second, in the accelerometer's axes: `calibration`'s `depth_to_imu` rotation takes depth-camera directions into them) once it has settled |
| `color` | The full colour image, raw `rgb8` bytes, with the same metadata as `depth` less gravity |
| `imu` | The `imu` message of `dora-rig-messages`: every accelerometer and gyro sample since the previous frame, each with its own camera time, corrected with the IMU calibration. Its `frame_id` and `keyframe_camera_ns` are the frame these samples lead up to; it goes out before that frame |
| `calibration` | The `camera_calibration` message of `dora-rig-messages`: colour and depth intrinsics, and the depth-to-colour and depth-to-IMU transforms. Sent with the first frame and then once a second |

Images follow dora's image convention (`encoding`, `width` and `height` in the metadata), so nodes from the dora hub
read them. `frame_id` counts frames from 0 and is the same across every output of one frame.

| Variable | Meaning |
|---|---|
| `IMU_CALIBRATION_PATH` | Required: this camera's `calibration.json` from librealsense's `tools/rs-imu-calibration/rs-imu-calibration.py`; every accelerometer and gyro sample is corrected with it |
| `MODEL_INPUT_SIZE` | Optional: the side of the square `model_input`, default 640 |

`dora-keyframe-picker`, in this workspace, picks keyframes from this node's frames without librealsense. It follows
`imu` and `depth`, picks a frame at least once a second and sooner when the camera turns 10 degrees (never closer
than 200 ms), and asks for that frame's colour on `request_color`. It then sends the `imu` and `keyframe` messages of
`dora-rig-messages`: an `imu` message with every sample since the previous keyframe, then the keyframe. The keyframe
holds depth redrawn onto the colour image from the calibration (each depth pixel painted over the colour pixels it
covers, nearest wins, lens distortion not modelled), the colour as JPEG, and the camera's gravity estimate. Frames are
not picked until the camera has a gravity estimate. Keyframe and imu `frame_id`s are this node's frame ids, so they
match `odom_pose`'s `depth_frame_id`; `captured_unix_ns` is when the picker received the depth. `KEYFRAME_RECORD_PATH`
and `IMU_RECORD_PATH` also append every keyframe and imu message it sends to Arrow IPC files, which
`dora-keyframe-replay` plays.

`examples/pose-dataflow.yml` is everything the camera's machine runs for the spatial server: the camera through
YOLOv8 pose on a Hailo-8 (`dora-hailo-perception`'s `dora-hailo-pose`), odometry
(`dora-depth-inertial-odometry`), the people placed in 3D in the odom frame (`dora-people-3d`) and keyframes
(`dora-keyframe-picker`). It publishes `rig/people_tracks`, `rig/odom_pose`, `rig/keyframe` and `rig/imu` over
`dora-zenoh-bridge`.

On a Raspberry Pi 5 the node uses about 40% of one core with `model_input`, `depth`, `imu`, `calibration` and
`color` on request listed; shrinking one frame into the model input takes about 5.3 ms. The keyframe picker uses
about 6%: aligning a keyframe's depth takes about 23 ms and its JPEG 18 ms, once a second.
