# dora-realsense

A [dora](https://dora-rs.ai) node that owns an Intel RealSense D400 with an IMU (built against a D455) and publishes
every frame: 1280x720 colour and 848x480 depth at 30 fps, accelerometer at 200 Hz and gyro at 400 Hz. Other nodes
take what they need from it; nothing else should open the camera.

Every stamp is on the camera's own clock, which all its sensors share, in nanoseconds. That clock is one 32-bit
microsecond counter that counts from the camera's firmware start (not from a stream start) and wraps every 2^32 us,
71 min 35 s: librealsense hands colour and depth the UVC metadata's 32-bit timestamp, and the accelerometer and gyro
the low 32 bits of the HID report's, so all four streams wrap together. One `CameraClock`
(`rgbd_frames::camera_clock`), shared by the video and motion threads, unwraps every stream's readings onto one 64-bit
timeline: it starts at the first depth frame, placed in the counter's second pass (camera time is never near zero),
and places each reading in the pass nearest the latest time of any stream, so a stream that crosses the wrap a little
later than another stays on the same timeline. A stream whose own time does not rise stops the node; a backward step
is never taken for a wrap. librealsense's host-synced
global time is turned off, because it drifts apart between the motion and video sensors. Frames go out from the
capture thread as soon as they arrive, from the first frame after the accelerometer and gyro have both delivered a
sample. An output the dataflow does not list costs nothing, so a dataflow that only wants depth does no image work.

Colour and depth come from a librealsense pipeline, which pairs them. The accelerometer and gyro come straight from
the motion sensor of the device that pipeline opened, not from a pipeline of their own, so every sample arrives and
librealsense holds one device object for the camera. A second pipeline would open a second device object, whose
clock polling tries to claim the USB interface the first one holds ten times a second; librealsense 2.58's RSUSB
backend (the Pi's) leaks a file descriptor on every failed claim, and the process runs out of them within minutes.

`rgbd-frames` holds the rule for which framesets go out, the letterbox and the IMU correction, and builds anywhere.
The node links librealsense2, so it builds only on the camera's machine: `cargo build --release -p dora-realsense`.

| Input | What it carries |
|---|---|
| `tick` | Required, any dora timer. It carries nothing; dora closes the events of a node without inputs, and a stop could then never reach it |
| `request_color` | Optional: `u64` frame ids. When given, `color` is sent only for the frames named here, instead of every frame. The node keeps the last 15 colour frames to answer from and fails on a request for an older one |

| Output | What it carries |
|---|---|
| `model_input` | The colour image shrunk by a whole factor (2 for 720p into 640), each pixel the average of its block, centred on a grey (114) square: the input a YOLO model takes. Raw `rgb8` bytes. Metadata: `encoding`, `width`, `height`, `frame_id`, `camera_ns` (the colour frame's capture time), `source_width`, `source_height`, `shrink`, `pad_x`, `pad_y`; a point at (x, y) in the square is at ((x - pad_x) * shrink, (y - pad_y) * shrink) in the colour image |
| `depth` | Raw depth, not aligned to colour: `u16` millimetres, row major, 0 where the sensor has no reading. Metadata: `encoding` (`mono16`), `width`, `height`, `frame_id`, `camera_ns` (the depth frame's capture time), and `specific_force` (three floats: up in the accelerometer's axes, the specific force the IMU would read at rest, at standard gravity's size; `calibration`'s `depth_to_imu` rotation takes depth-camera directions into those axes) once the IMU has been at rest for a second. It comes from a complementary filter (`rgbd_keyframes::GravityFilter`): every gyro sample turns up, so it follows the camera's turns without lag, and every accelerometer sample pulls it towards the accelerometer's direction with a 3 s time constant (`GRAVITY_TIME_CONSTANT`), which averages away a hand-held or driven camera's pushes and corrects the gyro's drift (its bias is not estimated). The filter starts from the mean accelerometer direction over a second at rest, by the odometry's static-initialization rules. A motion sample not newer than the one before it in its stream, a gyro silent for more than 100 ms, or one stream waiting more than a second for the other stops the node |
| `color` | The full colour image, raw `rgb8` bytes, with the same metadata as `depth` less gravity; `camera_ns` is the colour frame's capture time |
| `imu` | The `imu` message of `dora-rig-messages`: every accelerometer and gyro sample since the previous frame, each with its own camera time, corrected with the IMU calibration. Its `frame_id` and `keyframe_camera_ns` are the frame these samples lead up to; it goes out before that frame |
| `calibration` | The `camera_calibration` message of `dora-rig-messages`: colour and depth intrinsics, and the depth-to-colour and depth-to-IMU transforms. Sent with the first frame and then once a second |

Each frame pairs a new depth frame with a new colour frame. A frame stamped on any clock but the camera's own stops
the node with an error. Every frame that goes out has its colour and depth captured at most 5 ms apart
(`rgbd_frames::frameset`); a frame that would go out further apart stops the node with an error instead. Frames
before the accelerometer and gyro have both delivered are not sent, so they are not held to that limit: the first
frameset after the camera starts pairs frames captured far apart. When
librealsense pairs a new depth frame with the colour frame it already delivered, that depth frame is not sent and the
node logs it.

dora keeps every sample of 4096 bytes or more in a shared-memory region, with a file descriptor, until each receiver
has released it, and a receiving node that stops taking its inputs releases nothing, whatever its `queue_size`. Both
nodes in this workspace send through `shared-memory-outputs`, which stops the sender with an error naming the output
and its receivers once one output has more samples held than its limit (four seconds of frames for the camera, 30
keyframes for the keyframe selector), long before the process runs out of file descriptors.

Images follow dora's image convention (`encoding`, `width` and `height` in the metadata), so nodes from the dora hub
read them. `frame_id` counts frames from 0 and is the same across every output of one frame.

| Variable | Meaning |
|---|---|
| `IMU_CALIBRATION_PATH` | Required: this camera's `calibration.json` from librealsense's `tools/rs-imu-calibration/rs-imu-calibration.py`; every accelerometer and gyro sample is corrected with it |
| `MODEL_INPUT_SIZE` | Optional: the side of the square `model_input`, default 640 |

`dora-keyframe-selector`, in this workspace, selects keyframes from this node's frames without librealsense. It follows
`imu` and `depth`, selects a frame at least once a second and sooner when the camera turns 10 degrees or the
odometry's `odom_pose` says it has travelled 10 cm (never closer than 200 ms), and asks for that frame's colour on `request_color`. It then sends the `imu` and `keyframe` messages of
`dora-rig-messages`: an `imu` message with every sample since the previous keyframe, up to and including the first
gyro sample at or after the keyframe's time (so the gyro covers its whole span; that sample comes with the next
frame's imu, about 33 ms later), then the keyframe. The keyframe
holds depth redrawn onto the colour image from the calibration (each depth pixel painted over the colour pixels it
covers, nearest wins, lens distortion not modelled) with people cut out, the colour as JPEG (people left in), and the
camera's gravity estimate. People are cut out by the odometry's rule (`people_mask::MaskSettings`): the boxes of the
`person_poses` nearest the keyframe's depth in camera time, grown by a margin and by as far as a person walks in
their age, are set to 0 (no reading) in the aligned depth. The boxes are in colour pixels and the depth is aligned to
colour, so a box covers exactly its own pixels at any distance. A keyframe waits for person poses at or after its
time, so none nearer can still come; poses more than 300 ms from it cut nothing, and its progress line says so.
More than 8 keyframes waiting stops the node, naming what they wait for. Frames are not selected until the camera
has a gravity estimate and the pose model has sent its first person poses. Keyframe and imu `frame_id`s are this node's frame ids, so they
match `odom_pose`'s `depth_frame_id`; `captured_unix_ns` is when the keyframe selector received the depth. `KEYFRAME_RECORD_PATH`
and `IMU_RECORD_PATH` also append every keyframe and imu message it sends to Arrow IPC files, which
`dora-keyframe-replay` plays.

`dora-posed-depth`, in this workspace, sends the dense map's fast input: every third depth frame (10 a second;
`PosedDepthSettings::frame_stride`) as a `posed_depth_frame` of `dora-rig-messages`, with people cut out by the
odometry's own rule (`dora-depth-inertial-odometry`'s `people_mask`: the pose model's newest boxes, grown by a margin
and by how far a person walks in their age, read as no reading), so that people, drawn from their skeletons, are not
fused into the map; then downsampled to 424x240 by
keeping the nearest valid reading of each 2x2 block (never an average, so nothing is made up across a depth edge,
and thin near obstacles survive), with the intrinsics halved to match, at the odometry's pose of the depth camera
for exactly that frame. Each valid reading is then rounded to the nearest multiple of `PosedDepthSettings::depth_step`
(2 mm; 0 stays 0, no reading, and the wire unit stays millimetres). That trades depth resolution the dense map cannot
use (its voxels are 5 cm) for a smaller message: the lowest bits of D455 depth are sensor noise, which zstd cannot
compress. On a synthetic room frame with range-dependent D455 noise and holes, the zstd payload is about 130 KB at
1 mm, 110 KB at 2 mm and 92 KB at 4 mm. Its inputs are `depth`, `odom_pose`, `calibration` (the depth intrinsics, and
`depth_to_imu`, the depth camera's extrinsics on the body, which is the IMU) and `person_poses`; its output is `posed_depth`. A frame is
joined only to the odom pose with the same camera time and is dropped when the odometry skipped it; each frame
carries its pose's `odom_started_camera_ns`, so frames follow an odometry restart into the new odom frame. Odometry
whose camera time or start goes back, a pose naming another depth frame, and a calibration that changes stop the
node. It logs every 30 s how many frames went out, how many had people cut out and how many found no pose.

`examples/pose-dataflow.yml` is everything the camera's machine runs for the spatial server: the camera through
YOLOv8 pose on a Hailo-8 (`dora-hailo-perception`'s `dora-hailo-pose`), odometry
(`dora-depth-inertial-odometry`), the people placed in 3D in the odom frame (`dora-people-3d`), keyframes
(`dora-keyframe-selector`) and posed depth frames (`dora-posed-depth`). It publishes `rig/people_tracks`,
`rig/odom_pose`, `rig/keyframe`, `rig/imu`, `rig/calibration` and `rig/posed_depth` over `dora-zenoh-bridge`; only
`rig/posed_depth` and `rig/people_tracks` may lose a message to a newer one. It also runs the LD06 lidar
(`dora-serial-lidar`), whose `scan` stays on the camera's machine.

On a Raspberry Pi 5 the node uses about 40% of one core with `model_input`, `depth`, `imu`, `calibration` and
`color` on request listed; shrinking one frame into the model input takes about 5.3 ms. The keyframe selector uses
about 6%: aligning a keyframe's depth takes about 23 ms and its JPEG 18 ms, once a second.
