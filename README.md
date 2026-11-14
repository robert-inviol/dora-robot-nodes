# dora-robot-nodes

[dora](https://dora-rs.ai) nodes for small robots, built for a toy tank that a Raspberry Pi 5 with
a Hailo-8 drives around the house: it follows a person, or you drive it from your phone with live
video.

Each node is its own package with its own README, which lists the node's inputs, outputs and
settings. They share one small library of message definitions and nothing else, so you can take
one node without the rest.

| Package | What it does |
|---|---|
| [`dora-hailo-perception`](nodes/dora-hailo-perception) | Camera through YOLOv8 on a Hailo-8, detections with persistent track ids, and the video to browsers over WebRTC; or another node's frames through YOLOv8 pose (`dora-hailo-pose`). Raspberry Pi 5 only |
| [`dora-realsense`](nodes/dora-realsense) | A RealSense D455 in Rust: every frame's letterboxed model input, raw depth, IMU samples and calibration, on the camera's clock |
| [`dora-people-lift`](nodes/dora-people-lift) | The pose model's people placed in 3D with depth, smoothed and followed as tracks, in Rust |
| [`dora-person-follower`](nodes/dora-person-follower) | Decides where the robot goes: the operator's joystick, or towards one tracked person. Publishes a drive demand and knows nothing about the robot underneath |
| [`dora-teleop`](nodes/dora-teleop) | A control page for a phone or laptop: joystick, follow toggle, stop button, the live video with the tracker's boxes, and the map |
| [`dora-delta2a-lidar`](nodes/dora-delta2a-lidar) | The 3irobotics Delta-2A lidar, in Rust: decodes its serial stream and publishes each turn of the head as one scan |
| [`dora-scan-view`](nodes/dora-scan-view) | A top-down map page of a lidar scan |
| [`dora-realsense-keyframes`](nodes/dora-realsense-keyframes) | An Intel RealSense D400 camera with an IMU, in Rust: depth aligned to colour as a keyframe at least once a second and sooner when the camera turns, each led by every IMU sample since the last |
| [`dora-zenoh-bridge`](nodes/dora-zenoh-bridge) | Carries messages between dataflows on different machines over Zenoh, as zstd-compressed Arrow IPC. Rust |
| [`dora-scene-concepts`](nodes/dora-scene-concepts) | Asks Claude which object types a keyframe shows, and streams the names out as they come |
| [`dora-sam3-segmenter`](nodes/dora-sam3-segmenter) | Segments each keyframe into the object types named so far with SAM 3 on a CUDA GPU, as one segment map |
| [`dora-depth-perception`](nodes/dora-depth-perception) | A Rust workspace. Room geometry from one keyframe (planes, the room, object footprints) and its node; keyframe replay; `room-map`, one map of a room from a recording; `perception-wire`, the Rust side of the messages; and the mapping nodes: `dora-map-localizer` places every keyframe in one map, `dora-dense-map` builds free space, rooms, doorways, places and keep-out zones from posed depth, and `dora-object-map` keeps persistent objects from SAM 3 masks |
| [`dora-scene-view`](nodes/dora-scene-view) | A live 3D debug page of what perception sees: point cloud, segments, planes, room and object boxes |
| [`dora-world-agent`](nodes/dora-world-agent) | The spatial agent, in Rust: keeps the world model in SQLite, answers `space_question` messages with several Claude Code models at once and a reconciler, and sends `space_answer` revisions and `nav_goal` messages |
| [`dora-rig-messages`](lib/dora-rig-messages) | The messages the nodes exchange, each defined once with an explicit Arrow type and a golden file that tests in any language check against |

## How they fit together

```
 camera -> dora-hailo-perception --- detections ---> dora-person-follower --- drive ---> your robot's driver
                 |  video, boxes                        ^            |
                 v                                      | command    | status
            dora-teleop (control page) -----------------+ <----------+
                 ^  map
                 |
 lidar -> dora-delta2a-lidar --- scan ---> dora-scan-view
```

The follower's output is a drive demand: a forward share and a turn share of full power, whose
magnitudes sum to at most one. Turning that into wheel or track speeds is the one part that
belongs to your robot, so there is no driver here. For a skid-steer tank it is a few lines: the
left track gets `forward + turn`, the right `forward - turn`. Whatever you write should stop the
robot when demands stop arriving; the follower already asks for a stop whenever its own inputs go
quiet, and the control page centres the stick when the browser freezes or closes.

The spatial server runs on a machine of its own. Its `dora-zenoh-bridge` brings in keyframes and IMU
samples from the camera machine (`dora-realsense-keyframes`), segments from a GPU machine running
`dora-sam3-segmenter`, questions, and the robot's goal statuses; it sends back poses, the map, objects,
answers and nav goals.

```
 keyframe, imu                            -> dora-map-localizer -> keyframe_pose
 keyframe, keyframe_pose, segments        -> dora-object-map    -> object_update
 keyframe, keyframe_pose, object_update   -> dora-dense-map     -> free_space, topology, keep_out
 space_question, goal_status, topology,
 free_space, object_update, keyframe_pose -> dora-world-agent   -> space_answer, nav_goal
                                             (world store)
```

The localizer places every keyframe in one map. The dense map builds free space, rooms, doorways and
places from the posed depth, and keep-out zones from objects such as glass. The object map turns SAM 3
masks into persistent objects. The spatial agent keeps all of it in its SQLite world store, answers
questions from it, and sends answers and nav goals back over the bridge.
`nodes/dora-depth-perception/examples/spatial-server-dataflow.yml` is that dataflow.

## Using a node

The Python nodes are installed from their folder in this repository, the messages first:

```bash
pip install "git+https://github.com/robert-inviol/dora-robot-nodes#subdirectory=lib/dora-rig-messages"
pip install "git+https://github.com/robert-inviol/dora-robot-nodes#subdirectory=nodes/dora-teleop"
```

Each installs a command named after the package, which is what a dataflow's `path:` points at;
every node's README has the snippet. `dora-hailo-perception` also needs the Hailo and camera
packages from the Raspberry Pi's apt, listed in its README. The lidar node is built with
`nodes/dora-delta2a-lidar/build-for-pi.sh`, which cross-compiles a static binary for the Pi.

Tested with dora 0.3.13 on a Raspberry Pi 5, Debian 12, HailoRT 4.20.

## Working on them

```bash
uv sync
uv run pytest
cargo test --manifest-path nodes/dora-delta2a-lidar/Cargo.toml
(cd nodes/dora-depth-perception && cargo test --workspace)
(cd nodes/dora-world-agent && cargo test)
cargo test --manifest-path nodes/dora-realsense/Cargo.toml
cargo test --manifest-path nodes/dora-people-lift/Cargo.toml
```

`uv sync` installs the Python packages of the root workspace into one environment.
`dora-sam3-segmenter`, `dora-scene-concepts` and `dora-scene-view` have environments of their own and
run their tests from their own directories. `dora-world-agent` pins its Rust toolchain, so its tests
run from inside its directory. After a deliberate change to
a message's wire type, regenerate its golden file with `python -m dora_rig_messages.fixtures`; the
Rust tests read the same files.

## Licence

MIT.
