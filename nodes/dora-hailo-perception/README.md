# dora-hailo-perception

A [dora](https://dora-rs.ai) node for a Raspberry Pi 5 with a Hailo-8. It runs the camera through
YOLOv8 on the Hailo with `hailotracker`, publishes every frame's detections with persistent track
ids, and serves the video to browsers over WebRTC with a feed of the same detections for drawing
boxes over it.

It came from [`bobtronic/raspberry-pi-hailo`](https://github.com/bobtronic/raspberry-pi-hailo),
which holds the design write-up, the latency measurements, and the tracker-state and rules-engine
nodes that went with it there.

## Needs

- Raspberry Pi 5 with a Hailo-8 (M.2 HAT+) and a camera `rpicam-vid` can open
- From apt: `hailo-all` (HailoRT, the TAPPAS GStreamer elements, the `hailo` Python module and
  `/usr/share/hailo-models/yolov8s_h8.hef`), `python3-gi`, `rpicam-apps`
- A supply that can deliver 5 A

| Input | What it carries |
|---|---|
| `tick` | A timer, about one frame period; queued detections are published on each tick |
| `events` | Optional. Rows with `timestamp_ms`, `rule_name`, `event_type` and `message`, from any node; the messages are listed in the live view |

| Output | What it carries |
|---|---|
| `detections` | One message per frame, even when the frame is empty, so silence means the pipeline has stopped. One row per detection: `frame_id`, `timestamp_ms`, `track_id` (-1 until the tracker has numbered it), `class_id`, `class_name` (COCO), `confidence`, and the box `x`, `y`, `w`, `h` in 0..1 with the origin top left |

| Variable | Meaning |
|---|---|
| `CAM_WIDTH`, `CAM_HEIGHT`, `CAM_FPS` | Camera mode, default 1280x720 at 30 |
| `ENABLE_WEBRTC` | `true` to serve the live view |
| `WEBRTC_PORT` | Port of the live view, default 8080 |

The live view's server answers `GET /` (a page with the video and boxes), `POST /offer` (the
WebRTC handshake), `GET /ws` (a websocket that sends the latest detections about thirty times a
second) and `GET /api/events`.

```yaml
- id: gst-bridge
  path: dora-hailo-perception
  env:
    CAM_WIDTH: "1280"
    CAM_HEIGHT: "720"
    CAM_FPS: "30"
    ENABLE_WEBRTC: "true"
    WEBRTC_PORT: "8080"
  inputs:
    tick: dora/timer/millis/33
  outputs:
    - detections
```

## Pose from another node's frames: `dora-hailo-pose`

The package's second command runs YOLOv8 pose (`yolov8s_pose_h8.hef`) on frames another node sends, such as
`dora-realsense`'s letterboxed `model_input`, instead of reading a camera itself. Frames enter the same kind of
GStreamer pipeline through `appsrc`; Hailo's `libyolov8pose_post.so` decodes the 17 COCO keypoints and
`hailotracker` numbers the people. Each frame's `frame_id` rides through the pipeline as its buffer timestamp, and
the main loop collects finished frames from an `appsink`.

Nothing in the pipeline drops a frame. The node gives it at most three frames at a time, skips a frame that arrives
while three are in flight, and expects every frame back in the order given. It stops with an error when a frame
comes back out of turn, and exits with status 1 when the oldest frame in flight has waited 5 s for its result,
printing every thread's stack first. A native timer (Python's `faulthandler`) also ends the process, with the same
stacks headed `Timeout (0:00:07)!`, when the main loop itself has not come round for 7 s. A status line every 30 s
counts the frames returned and skipped.

| Input | What it carries |
|---|---|
| `model_input` | A square `rgb8` frame with `dora-realsense`'s letterbox metadata (`frame_id`, `camera_ns`, `source_width`, `source_height`, `shrink`, `pad_x`, `pad_y`). Give it `queue_size: 1`: a frame waiting behind a newer one is already late |

| Output | What it carries |
|---|---|
| `person_poses` | The `person_poses` message of `dora-rig-messages`, one per frame even when nobody is in view: each person's box and keypoints in the source image's pixels, and the tracker's id once it has numbered them. Metadata `people` is the count |

| Variable | Meaning |
|---|---|
| `POSE_HEF` | Optional: the model, default `/usr/share/hailo-models/yolov8s_pose_h8.hef` |
| `POSE_POST_PROCESS` | Optional: the post-process library, default TAPPAS's `libyolov8pose_post.so` |

It also needs `dora-rig-messages`. On a Raspberry Pi 5 it keeps up with 30 fps using about 19% of one core.
`dora-realsense`'s `examples/pose-dataflow.yml` runs it.

## Known problems

- The Hailo driver in Raspberry Pi kernel 6.12.25 (in-tree `hailo_pci` 4.20.0) can deadlock the process. Its
  buffer-map ioctl takes the process's memory-map lock for reading around `find_vma` and, still holding it, takes
  it for reading again around `get_user_pages` (Raspberry Pi commit bc0a2321ad55; b5bed2815fc5 narrowed the first
  lock to `find_vma` alone). When another thread of the process asks for the lock for writing in between (any
  `mmap`, `munmap` or `brk`), the second read waits behind the writer and the writer waits for the first read.
  hailonet maps about six user buffers per frame this way. The process then sits in uninterruptible sleep and
  anything that reads its `/proc/<pid>/cmdline`, such as `ps`, hangs too; killing it frees it. `dora-hailo-pose`
  exits on its own 5 s into such a stall. The driver from the `hailo-dkms` package takes the lock once.
- Rarely, stopping while a browser is watching the video makes the node abort inside the
  GStreamer and Hailo teardown (`basic_string: construction from null`).
