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
 |- depth, calibration --------> people (dora-people-3d) -> people_tracks (odom frame, with skeleton_3d) -> link
 '- depth, imu, calibration, color (only the frames it names on request_color)
                               -> keyframes (dora-keyframe-picker) -> keyframe, imu (one per keyframe) -> link
link: listens on :7447, publishes rig/people_tracks, rig/odom_pose, rig/keyframe and rig/imu
```

Measured on the Pi 5 in a 75 s run with the camera at rest and one person in view (top over 5 s, % of one core):
dora-realsense 40, Zenoh link 33, pose (python) 19, odometry 13, dora daemon 11, keyframe picker 6, dora-people-3d 5;
about 1.3 of the Pi's 4 cores. Pose at 30 frames/s, every frame matched to its depth; odometry 30 poses/s, 299-300
of every 300 corrected by the depth cloud; a keyframe a second (align 23 ms, JPEG 18 ms); the skeleton fit 0.66 ms
per person per frame. The person kept one id for the whole run.

`nodes/dora-realsense-keyframes` (its own camera-dataflow.yml, which opens the D455 itself) is superseded by the
keyframe picker and can retire.

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

The camera machine now publishes everything the server subscribes to, but the server does not yet subscribe to
`rig/odom_pose` (Next, 1).

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

- It has run live once, on the Pi in `pose-dataflow.yml` with the camera at rest on a bench for about 70 s: it
  started 2 s after the calibration, kept pace at 30 poses/s, and crept about 1 cm in y over 30 s (position sigma
  1-2 mm). It has not yet run on a moving camera.
- The node publishes nothing until it has the camera's calibration and the robot has rested for 2 s. It stops,
  rather than guess, when a calibration changes, time goes back, the IMU is more than 10 ms behind a depth frame,
  or a silent stream leaves it holding more than 1 s of samples (see the node's README).
- The odom frame is z up, with its origin and zero heading where the IMU rested at the start. The body is the IMU.
- The sigmas on `odom_pose` say how well the pose fits the odometry's own map. They leave out that map's drift and
  are too small as a measure of where the robot is in the room.

## Next

In order; the first is what it takes for people to reach the world store in a live run.

1. Wire odometry on the server. The `odom_pose` input and `map_to_odom` output of the localizer are commented out in
   `spatial-server-dataflow.yml`, as is the `map_to_odom` input of `spatial_agent`: subscribe
   `rig/odom_pose=odom_pose` on the link, add an `[odometry]` section to the localizer settings, and uncomment them.
2. Run the camera dataflow and the server together and check people reach the world store.

Also open:

- Whether the world store records joints from `skeleton_3d` instead of `keypoints_3d` is for robert to decide
  (recommended: `skeleton_3d` when filled, no joints otherwise, never a mix).
- `camera_in_imu_m` in the localizer's `[odometry]` settings is set by hand. It could be read from the camera's
  calibration message instead, if robert agrees.
- The Zenoh link costs a third of a core on the Pi, more than the pose node; find out why.
- The keyframe picker's alignment leaves out lens distortion.
- A sighting's time is when the server received it, not when the camera captured it.
- clippy warns on `room-map/src/grid.rs:85` (a loop that only indexes an array).
- 3d-perception's uncommitted IMU calibration change to dora-realsense-keyframes sits in the main clone
  (`~/src/dora-robot-nodes`, branch depth-perception); dora-realsense applies the same correction. Keep or discard
  is robert's call.

Odometry, in order:

1. Find the 1 cm per 30 s creep at rest seen on the Pi's first live run.
2. Record a walk with every depth frame, the imu messages and the calibration, starting and ending at the same
   spot, and replay it through the odometry. The gap between the first and last pose is the first accuracy number
   from a real camera. The same recording can feed map building. No node records every depth frame yet; the
   keyframe picker records only keyframes and their imu messages.
3. Fix what the walk shows. Known before it starts:
   - height drifts when the floor is out of view; a measurement that holds the robot to the floor is not written,
     and whether to add one is for robert to decide;
   - people and other moving things are not masked out of the depth cloud;
   - the lidar is not used: the filter takes point-to-plane rows from any sensor, and none are written for it.
