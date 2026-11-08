# dora-realsense-keyframes

A [dora](https://dora-rs.ai) node for an Intel RealSense D400 camera with an IMU (built against a D455):
depth aligned to 1280x720 colour, sent as a `keyframe` at least once a second and sooner when the camera
turns, each preceded by an `imu` message with every accelerometer (200 Hz) and gyro (400 Hz) sample since the
previous keyframe. Gravity comes from the accelerometer smoothed over half a second; turning from the gyro. If
dora falls behind and a keyframe is dropped, its samples lead the next `imu` message, so none are lost.

Video and motion stream on two pipelines from one librealsense context: a pipeline that also carries video
hands over only the newest motion sample per video frame. Every timestamp is on the camera's own clock, which
all its sensors share; librealsense's host-synced global time is turned off, because it drifts apart between
the motion and video sensors. Keyframes keep a unix `captured_unix_ns` for latency.

`rgbd-keyframes` holds the keyframe picker and gravity estimate and builds anywhere. The node links
librealsense2, so it builds only on the camera's machine: `cargo build --release -p dora-realsense-keyframes`.

| Input | What it carries |
|---|---|
| `tick` | Hands captured keyframes to dora; the camera runs on its own thread |

| Output | What it carries |
|---|---|
| `imu` | The IMU samples leading up to the keyframe that follows; metadata has `accel_samples` and `gyro_samples` |
| `keyframe` | A keyframe from `dora-rig-messages`; metadata has `align_ms` and `jpeg_ms` |

| Variable | Meaning |
|---|---|
| `KEYFRAME_RECORD_PATH` | Optional: also append every keyframe to this Arrow IPC file, which `dora-keyframe-replay` plays |
| `IMU_RECORD_PATH` | Optional: also append every imu message to this Arrow IPC file, which `dora-keyframe-replay` plays beside the keyframes |

`examples/camera-dataflow.yml` sends keyframes to a perception machine over `dora-zenoh-bridge`.
