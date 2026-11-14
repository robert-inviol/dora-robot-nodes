# Notes for agents

Read this before working in this repo. The READMEs say how each node works; this file says how the work is done and
what is open.

## Working here

- All robot perception work is on the branch `robot-perception`, in one shared worktree. Several sessions edit it at
  once: commit only the files you changed, and message the other sessions before you commit. When your change needs a
  line in another session's file, ask them first.
- Commit and push only when robert says so.
- Build and test on strixen, the server (`ssh -i ~/.ssh/id_ed25519_robert-inviol robert_inviol@192.168.111.34`). Sync
  the worktree to `~/spatial/base` with `rsync -az --exclude target --exclude .git`, then:
  - Rust: `cargo test -q --workspace` and `cargo clippy -q --workspace --all-targets` in each workspace you touched
    (`nodes/dora-depth-perception`, `nodes/dora-world-agent`, `nodes/dora-people-3d`, ...). Every test must pass.
  - Python: `~/.local/bin/uv run --with pytest pytest -q` at the repo root.
  - Some synthetic-walk timing tests in dora-depth-perception fail on a laptop and pass on strixen.
- Messages between machines are single-row Arrow structs, defined in Python in `lib/dora-rig-messages` and mirrored
  in Rust in `nodes/dora-depth-perception/perception-wire`. A field change goes in both, with the fixtures in
  `lib/dora-rig-messages/dora_rig_messages/fixtures` regenerated.
- robert's rules for the code: no fallbacks (work or fail loudly); names, then abstractions, then types over
  primitives, then contracts, then error handling; comments describe the current state only; ASCII punctuation only.
  A design choice needs robert's approval before it is built.

## The pipeline as built

Three machines, joined by `dora-zenoh-bridge` on Zenoh keys `rig/*`.

Camera machine (Pi with the D455 and a Hailo-8), `nodes/dora-realsense/examples/pose-dataflow.yml`:

```
dora-realsense (the only node that opens the D455; every stamp on the camera clock)
 |- model_input ---------------> pose (YOLOv8 pose + hailotracker on the Hailo-8) -> person_poses -> people
 |- depth, imu, calibration ---> odometry (dora-depth-inertial-odometry) -> odom_pose -> people, link
 '- depth, calibration --------> people (dora-people-3d) -> people_tracks (odom frame) -> link
link: listens on :7447, publishes rig/people_tracks and rig/odom_pose
```

The older camera dataflow, `nodes/dora-realsense-keyframes/examples/camera-dataflow.yml`, opens the D455 itself and
publishes `rig/keyframe` and `rig/imu` (depth aligned to 1280x720 colour, a keyframe at least once a second and
sooner on a turn). The two cannot run at once: both open the camera and listen on :7447.

GPU machine, `nodes/dora-depth-perception/examples/live-dataflow.yml`: SAM 3 segments, scene concepts and geometry
from `rig/keyframe`; publishes `rig/segments`.

Server (strixen), `nodes/dora-depth-perception/examples/spatial-server-dataflow.yml`:

```
link: connects to the camera machine; subscribes rig/keyframe, rig/imu, rig/segments, rig/people_tracks,
      rig/space_question, rig/goal_status
 |- keyframe, imu -> localizer (dora-map-localizer) -> keyframe_pose -> dense_map, objects, spatial_agent
 |- keyframe, keyframe_pose, object_update -> dense_map -> free_space -> spatial_agent
 |- keyframe, keyframe_pose, segments -> objects (dora-object-map) -> object_update -> spatial_agent
 '- people_tracks -> spatial_agent (dora-world-agent: SQLite world store, MCP tools, spatial agents)
```

So today a live run gives either people and odometry (pose-dataflow) or localization and mapping
(camera-dataflow), never both, and `rig/odom_pose` has no subscriber on the server.

## People in the world store

People tracks reach the spatial agent's world store this way:

1. On the camera machine (`nodes/dora-realsense/examples/pose-dataflow.yml`), dora-people-3d places each tracked
   person in 3D and publishes `people_tracks` on the Zenoh key `rig/people_tracks`.
2. On the server (`nodes/dora-depth-perception/examples/spatial-server-dataflow.yml`), the Zenoh link subscribes
   `rig/people_tracks` and feeds the `people_tracks` input of `spatial_agent` (dora-world-agent).
3. `nodes/dora-world-agent/dora-world-agent/src/people.rs` (`place_people`) carries confirmed tracks from the odom
   frame into the map with the localizer's latest `map_to_odom` (input `map_to_odom`). Camera-frame tracks, and
   odom-frame tracks before the first correction, are not placed: the node warns once and logs when placing resumes.
4. `world-store` records each track at most once per `TrackBounds.record_every` (250 ms) in the tables `track`,
   `track_pose` (torso, speed, heading) and `track_joint` (one row per seen joint, COCO-17 names). Schema changes are
   migrations in `world-store/src/schema.rs`.
5. The MCP tool `last_seen` returns a track's joints with their height above the floor.

The joints come from the track's `keypoints_3d`, which is in COCO order; the store's joint mapping depends on that
order (`world_model::world::Joint::ALL`). The heading is the walking direction at 0.2 m/s or faster, otherwise the
facing from the shoulders, then the hips (`world-store/src/geometry.rs`).

## Localizer and odometry

dora-map-localizer joins each registered keyframe to the odometry pose at its camera time and sends `map_to_odom`
(see its README). Each `odom_pose` carries `odom_started_camera_ns`, which names its odom frame. A later value means
odometry started again in a new frame: the localizer drops the old frame's poses and correction and re-anchors from
the next keyframe in the new frame. A start that goes back, or a camera time that goes back, stops the node.

## Odometry

dora-depth-inertial-odometry gives the robot's pose at every depth frame from the D455's depth and IMU: the IMU
predicts, each depth cloud is matched point to plane against a local map, and an error-state Kalman filter blends
the two. The library is `nodes/dora-depth-perception/depth-inertial-odometry`; its README explains the method, the
three local maps it can use and how they compare for accuracy and for speed on a Pi 5, and its limits. The node
runs the voxel-plane map.

- It has run on the simulator only: the library on simulated drives, the node on simulated camera messages. It has
  not run in a dora dataflow or on a camera.
- The node publishes nothing until it has the camera's calibration and the robot has rested for 2 s. It stops,
  rather than guess, when a calibration changes, time goes back, the IMU is more than 10 ms behind a depth frame,
  or a silent stream leaves it holding more than 1 s of samples (see the node's README).
- The odom frame is z up, with its origin and zero heading where the IMU rested at the start. The body is the IMU.
- The sigmas on `odom_pose` say how well the pose fits the odometry's own map. They leave out that map's drift and
  are too small as a measure of where the robot is in the room.

## Next

In order; the first two are what it takes for people to reach the world store in a live run.

1. One camera dataflow (approved by robert, with robert-27). Keyframes and their imu messages come from a node fed
   by dora-realsense's outputs inside `pose-dataflow.yml`, and its link also publishes `keyframe=rig/keyframe` and
   `imu=rig/imu`. dora-realsense stays the only node that opens the camera; the server's subscriptions do not
   change. This also puts keyframes and `odom_pose` on one camera clock from one camera session, which
   `map_to_odom` needs. Each of these goes to robert as a proposal before it is built:
   - reusing `rgbd-keyframes` (the picker and gravity estimate, which build without librealsense), and whether
     dora-realsense-keyframes retires;
   - aligning depth to colour from the calibration message's intrinsics and `depth_to_color`;
   - colour through dora-realsense's `request_color` for the picked frame only, holding its depth until it arrives;
   - one imu message per keyframe holding every sample since the previous keyframe, sent before it, keeping
     dora-realsense's `frame_id` and camera time;
   - `KEYFRAME_RECORD_PATH` and `IMU_RECORD_PATH` recordings for `dora-keyframe-replay`.
2. Wire odometry on the server. The `odom_pose` input and `map_to_odom` output of the localizer are commented out in
   `spatial-server-dataflow.yml`, as is the `map_to_odom` input of `spatial_agent`: subscribe
   `rig/odom_pose=odom_pose` on the link, add an `[odometry]` section to the localizer settings, and uncomment them.

Also open:

- dora-people-3d gains `skeleton_3d`: joints fitted to a skeleton with fixed bone lengths. Whether the world store
  records joints from `skeleton_3d` instead of `keypoints_3d` is for robert to decide.
- `camera_in_imu_m` in the localizer's `[odometry]` settings is set by hand. It could be read from the camera's
  calibration message instead, if robert agrees.
- A sighting's time is when the server received it, not when the camera captured it.
- clippy warns on `room-map/src/grid.rs:85` (a loop that only indexes an array).

Odometry, in order:

1. Run dora-depth-inertial-odometry on the Pi from the live D455 with the robot at rest, beside the camera and pose
   nodes. This shows whether the node's dora message handling works, whether each imu message arrives within 10 ms
   of its depth frame, what a frame costs, and whether the 1 s backlog and the queue sizes in its dataflow hold.
2. Record a walk with every depth frame, the imu messages and the calibration, starting and ending at the same
   spot, and replay it through the odometry. The gap between the first and last pose is the first accuracy number
   from a real camera. The same recording can feed map building.
3. Fix what the walk shows. Known before it starts:
   - height drifts when the floor is out of view; a measurement that holds the robot to the floor is not written,
     and whether to add one is for robert to decide;
   - people and other moving things are not masked out of the depth cloud;
   - the lidar is not used: the filter takes point-to-plane rows from any sensor, and none are written for it.
