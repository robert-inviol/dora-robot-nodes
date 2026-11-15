# Notes for agents

Read this before working in this repo. The READMEs say how each node works; this file says how the work is done and
what is open.

## Working here

- All robot perception work is on the branch `robot-perception`, in one shared worktree. Several sessions edit it at
  once: commit only the files you changed, and message the other sessions before you commit. When your change needs a
  line in another session's file, ask them first.
- Commit and push only when robert says so.
- Build and test on strixen, the server (`ssh -i ~/.ssh/id_ed25519_robert-inviol robert_inviol@192.168.111.34`). Sync
  the worktree to `~/dora-robot-perception` there with `rsync -az --delete --exclude target --exclude .git --exclude
  .venv --exclude imu-calibration.json`, then:
  - Rust: `cargo test -q --workspace` and `cargo clippy -q --workspace --all-targets` in each workspace you touched
    (`nodes/dora-depth-perception`, `nodes/dora-world-agent`, `nodes/dora-people-3d`, ...). Every test must pass.
  - Python: `~/.local/bin/uv run --with pytest pytest -q` at the repo root.
  - Some synthetic-walk timing tests in dora-depth-perception fail on a laptop and pass on strixen.
- The Pi (`ssh raspai`, checkout `~/src/dora-robot-nodes-dev`, a plain copy) only runs. Its Rust nodes are
  cross-built on strixen by `./pi-build.sh ~/pi-out` (the script lists what it needs there), then copied:
  `rsync -a strixen:pi-out/ raspai:src/dora-robot-nodes-dev/` from a machine that reaches both (strixen has no key
  for the Pi). Sync the source to the Pi too, for the dataflows and the Python pose node. The Pi's IMU calibration
  is `~/imu-calibration/calibration.json`, copied to `nodes/dora-realsense/examples/imu-calibration.json`.
- Messages between machines are single-row Arrow structs, defined in Python in `lib/dora-rig-messages` and mirrored
  in Rust in `nodes/dora-depth-perception/perception-wire`. A field change goes in both, with the fixtures in
  `lib/dora-rig-messages/dora_rig_messages/fixtures` regenerated.
- robert's rules for the code: no fallbacks (work or fail loudly); names, then abstractions, then types over
  primitives, then contracts, then error handling; comments describe the current state only; ASCII punctuation only.
  A design choice needs robert's approval before it is built.

## The pipeline as built

Three machines, joined by `dora-zenoh-bridge` (YAML node id `bridge`) on Zenoh keys `rig/*`. A published key waits
for the Zenoh connection rather than lose a message; only the keys in `ZENOH_LATEST_ONLY` (`rig/posed_depth` and
`rig/people_tracks` on the camera machine) replace a message still waiting with a newer one.

Camera machine (Pi with the D455 and a Hailo-8), `nodes/dora-realsense/examples/pose-dataflow.yml`:

```
camera: dora-realsense (the only node that opens the D455; every stamp on the camera clock)
 |- model_input ----------------> pose (YOLOv8 pose + hailotracker on the Hailo-8) -> person_poses -> people, odometry
 |- depth, imu, calibration ----> odometry (dora-depth-inertial-odometry) -> odom_pose -> people, posed_depth, bridge
 |                                 (person_poses cut people out of the depth; lost once the position sigma passes 4.2 cm)
 |- depth, calibration, odom_pose -> people (dora-people-3d, POSITION_FRAME=odom)
 |                                 -> people_tracks: 30 a second, in the odom frame named by odom_started_camera_ns,
 |                                    with skeleton_3d -> bridge
 |- depth, calibration, odom_pose -> posed_depth (dora-posed-depth): every third depth frame, 2x2 downsampled, depth
 |                                    rounded to 2 mm, at the odometry pose of exactly that frame -> bridge
 '- depth, imu, calibration, color (only the frames it names on request_color)
                                -> keyframes (dora-keyframe-selector: keyframe selection, at least one a second and
                                   sooner on a 10 degree turn) -> keyframe, imu (one per keyframe) -> bridge
bridge: listens on :7447; publishes rig/people_tracks, rig/odom_pose, rig/keyframe, rig/imu, rig/calibration and
        rig/posed_depth
```

The camera and the pose node stop loudly rather than fall behind: the camera (and the keyframe selector) send
through `shared-memory-outputs`, which stops the sender naming the output and its receivers once a receiver holds
more than its limit (four seconds of frames for the camera, 30 keyframes for the keyframe selector); the pose node
exits with status 1, printing every thread's stack, when a frame has waited 5 s on the Hailo, or when its main loop
has not come round for 7 s.

Measured on the Pi 5 in a 75 s run with the camera at rest and one person in view, before posed depth frames were
added to the dataflow (top over 5 s, % of one core): dora-realsense 40, Zenoh bridge 33, pose (python) 19, odometry
13, dora daemon 11, keyframe selector 6, dora-people-3d 5; about 1.3 of the Pi's 4 cores. Pose at 30 frames/s, every
frame matched to its depth; odometry 30 poses/s, 299-300 of every 300 with a measurement update from the depth cloud;
a keyframe a second (align 23 ms, JPEG 18 ms); the skeleton fit 0.66 ms per person per frame. The person kept one id
for the whole run.

Measured on 2026-11-15 with the current dataflows on the Pi and strixen together, camera at rest on its bench with up
to five people moving in view, no GPU machine. The run was healthy for its full 10 minutes and ended
when robert rebooted the Pi (19:07 NZDT). The numbers are from its first 6.5 minutes:

- Pi CPU (top, % of one core): dora-realsense 38-40, Zenoh bridge 26-43, pose (python) 27-30, dora daemon 13-15,
  odometry 9-12, dora-people-3d 5-8, keyframe selector 6, posed depth 3-4. The camera held 46 file descriptors from
  start to end.
- Uplink, from the bridge's per-key log: 12.2 Mbit/s in all; `rig/posed_depth` 5.4 (9.2 a second, 73 KB each after
  zstd), `rig/keyframe` 3.6 (1 a second, 453 KB), `rig/people_tracks` 2.1 (27 a second, 9.7 KB), `rig/odom_pose`
  1.0 (28 a second, 4.5 KB), `rig/imu` 0.06, `rig/calibration` 0.04. The Pi bridge's encode takes 4.3 ms per odom
  pose, 6.5 ms per people frame, 5.6 ms per posed depth frame and 17 ms per keyframe: 0.37 s of every second, which
  is the bridge's CPU.
- Arrival at the bridge minus capture time, above the fastest `rig/odom_pose` of the run on the same machine
  (median of 10 s medians): on the Pi, odom poses 17 ms, posed depth 22 ms, people 56 ms, imu 99 ms; on strixen,
  odom poses 15 ms, posed depth 26 ms, people 58 ms. The fastest odom pose reached strixen 10.5 ms after it reached
  the Pi's bridge, network and the two machines' NTP clock difference together.
- Odometry was lost 12 times in those 6.5 minutes, 14 in the 10 (no measurement update for 0.5 s, each time after
  a 10 s span in which every frame had one) and started again 2 s later each time. Each new odom frame hides the
  robot and the people on the page until the localizer's `map_to_odom` for it, 2 to 7 s. All 14 losses had, in
  the 3 s before them, a person the pose model tracked whose torso had no depth (people-3d's `Confirmed at None`,
  45 such progress lines in the run, 28 of them in the 3 s before a loss, which is 7% of the run): someone at or
  inside the D455's minimum range, filling the view. The odometry then had nothing left to correct with, and a
  fixed 0.5 s without a measurement update was lost. The odometry now dead reckons through that and is lost only
  once its largest position sigma passes 4.2 cm (about 1.7 s at rest), and says at each loss how the last second's
  clouds fared (see its README).

Measured on 2026-11-15 from 19:50 NZDT for 10 minutes with the same dataflows and that odometry, camera at rest,
people moving in view (98 people-3d track ids; 53 progress lines with a person whose torso had no depth): no
odometry loss; 17,928 of 18,300 poses corrected. The refused clouds came in bursts in which the frames had no depth
readings at all (given 0 points at the median): someone inside the camera's minimum range. Odometry 13% of one Pi 5
core, 3.3 ms median and 6-8 ms worst per frame tracked; its file descriptors (13-14) and memory (8-9 MB) stayed
flat. The pose crept about 6 cm in x over the run. A 2 minute recording made just before (camera at rest, people
passing, up to 86% of a frame cut out) replays on strixen with 3,528 of 3,538 poses corrected and no loss, 1.15 ms
median per frame.
- strixen CPU: localizer 70-73, dense map 31-44, rig view 5-17, plane segmentation 11-12, bridge 3. The localizer
  tracked 391 of 399 keyframes (5 lost, 3 relocalized, at the start and at odometry restarts) and selected 1 into
  the map. rig view's memory grows with the keyframe clouds it keeps (60 MB at start, 550 MB after 580 keyframes).
- Dense map: posed depth frames fused from 1.6 s after the server started, and again 0.4 to 1 s after each odometry
  restart; about 115 mesh blocks a second sent, about 400 blocks held.
- The page received people 27.7 a second (720 B each), the robot 27.8 a second, the surface 4.1 times a second
  (27 mesh blocks and 25 KB each), about 130 KB/s in all, with no gap in the mesh blocks sequence and no snapshot;
  it held 443 mesh blocks at the end.

`rig/calibration` tells the server's localizer the colour camera's extrinsics on the IMU.

GPU machine, `nodes/dora-depth-perception/examples/live-dataflow.yml`: from `rig/keyframe`, open-vocabulary labels
(`labels`, dora-scene-concepts), SAM 3 instance masks (`sam3`, dora-sam3-segmenter) and plane segmentation
(`plane_segmentation`, dora-depth-perception: planes and the room of each keyframe); publishes
`rig/instance_masks`, `rig/room_layout` and `rig/scene`. It runs on robert's laptop (RTX 4050, 6 GB; the SAM 3
environment needs Python 3.12, `uv sync --python 3.12`); `live-laptop.yml` is the labels and SAM 3 part alone. Its
bridge must listen as well as connect: Zenoh on the camera machine does not forward `rig/instance_masks` between the
laptop and strixen when both only connect to it, so strixen reaches the laptop directly.

Server (strixen), `nodes/dora-depth-perception/examples/spatial-server-dataflow.yml`:

```
bridge: connects to the camera machine; subscribes rig/keyframe, rig/imu, rig/instance_masks, rig/spatial_query,
        rig/goal_status, rig/people_tracks, rig/odom_pose, rig/calibration, rig/posed_depth
 |- keyframe, imu, odom_pose, calibration -> localizer (dora-map-localizer)
 |     -> keyframe_pose (tracking, relocalized, restarted or lost) -> dense_map, objects, spatial_agent, rig_view
 |     -> pose_graph_update (past keyframes a pose graph optimization moved) -> dense_map, rig_view
 |     -> map_to_odom -> dense_map, spatial_agent, rig_view
 |- keyframe, keyframe_pose, pose_graph_update, posed_depth, map_to_odom, object_update -> dense_map (dora-dense-map)
 |     (keyframes fused at their map pose and rebuilt when they move; posed depth frames fused at their odometry
 |      pose carried into the map by map_to_odom of their odom frame)
 |     -> occupancy_grid, topology, keep_out -> spatial_agent, bridge
 |     -> mesh_blocks (blocks whose mesh changed, numbered by run and sequence) -> rig_view
 |- keyframe, keyframe_pose, instance_masks -> objects (dora-object-map) -> object_update (with its voxels)
 |                                                                       -> dense_map, spatial_agent, rig_view
 |- spatial_query, goal_status, people_tracks, map_to_odom, ... -> spatial_agent (dora-world-agent: SQLite world
 |     store, MCP tools, spatial agents) -> spatial_answer, nav_goal -> bridge
 |- keyframe -> plane_segmentation (dora-depth-perception) -> room_layout (one keyframe's planes and room) -> rig_view
 '- keyframe, odom_pose, people_tracks, mesh_blocks, topology, ... -> rig_view (dora-rig-view): the 3D page on 8094
       -> mesh_snapshot_request (after a gap in the mesh blocks sequence, or on joining a run late) -> dense_map,
          which answers with a snapshot of every mesh block
```

dora-rig-view serves a page for people to watch the map grow: the dense map's mesh with each object's voxels
painted on it, the robot and its path, people's skeletons, rooms and places, the room layout of the latest keyframe,
and optionally the keyframe clouds, space carved where later registered keyframes saw through them. A page gets the
scene once when it connects, then only changes, over one WebSocket. `cargo run --release --example demo -- 8094` in
nodes/dora-rig-view runs it on a synthetic room fused by the real dense map, without the rig.

The rig view has run with the live rig (measured above, under the camera machine's dataflow).

## People in the world store

People tracks reach the spatial agent's world store this way:

1. On the camera machine (`nodes/dora-realsense/examples/pose-dataflow.yml`), dora-people-3d lifts each tracked
   person to 3D from the depth, transforms them into the odom frame with the odometry pose at the pose frame's camera
   time, and publishes `people_tracks` (30 a second, naming its odom frame in `odom_started_camera_ns`) on the Zenoh
   key `rig/people_tracks`.
2. On the server (`nodes/dora-depth-perception/examples/spatial-server-dataflow.yml`), the bridge subscribes
   `rig/people_tracks` and feeds the `people_tracks` input of `spatial_agent` (dora-world-agent).
3. `nodes/dora-world-agent/dora-world-agent/src/people.rs` (`transform_people_to_map`) carries confirmed tracks from
   the odom frame into the map frame with the localizer's latest `map_to_odom` (input `map_to_odom`), when that
   correction was made in the odom frame the tracks name. Camera-frame tracks, odom-frame tracks before the first
   correction, and tracks of an odom frame the latest correction is not for, are not transformed
   (`NotTransformed`): the node warns once and logs when transforming resumes.
4. `world-store` records each track at most once per `TrackBounds.record_every` (250 ms) in the tables `track`,
   `track_pose` (torso, speed, heading) and `track_joint` (one row per fitted joint, COCO-17 names). Schema changes
   are migrations in `world-store/src/schema.rs`.
5. The MCP tool `last_seen` returns a track's joints with their height above the floor.

The joints come from the track's `skeleton_3d` (the joints dora-people-3d fitted to the person's skeleton; the
detected `keypoints_3d` are not recorded), which is in COCO order; the store's joint mapping depends on that order
(`world_model::world::Joint::ALL`). The heading is the walking direction at 0.2 m/s or faster, otherwise the facing
from the shoulders, then the hips (`world-store/src/geometry.rs`).

## Localizer and odometry

dora-map-localizer registers each keyframe against the map, keeps the map keyframes in a 4-DoF pose graph, and
joins each registered keyframe (tracking or relocalized) to the odometry pose at its camera time to send
`map_to_odom` (see its README). A keyframe that starts a new run after tracking was lost is reported `restarted`:
it is in the map, joined only by a dead-reckoning edge, and later pose graph updates move it; it does not feed
`map_to_odom`. Each `odom_pose` carries `odom_started_camera_ns`, which names its odom frame. A later value means
odometry started again in a new frame: the localizer drops the old frame's poses and correction and re-anchors from
the next keyframe in the new frame. A start that goes back, or a camera time that goes back, stops the node.

Each consumer's rule for a restarted keyframe is in its README: the dense map fuses it (pose graph updates move it
like any map keyframe); the object map and the world store skip it (they take no pose graph updates); the rig view
draws it but does not carve with it.

## Odometry

dora-depth-inertial-odometry gives the robot's pose at every depth frame from the D455's depth and IMU: the IMU
predicts, each depth cloud is downsampled and matched point to plane against a local map, and an error-state Kalman
filter's measurement update blends the two. The library is `nodes/dora-depth-perception/depth-inertial-odometry`; its
README explains the method, the three local maps it can use and how they compare for accuracy and for speed on a Pi
5, and its limits. The node runs the voxel-plane map, and cuts the pose node's people out of the depth.

- It has run live once, on the Pi in `pose-dataflow.yml` with the camera at rest on a bench for about 70 s, before
  people were cut out of the depth: it started 2 s after the calibration, kept pace at 30 poses/s, and crept about
  1 cm in y over 30 s (position sigma 1-2 mm). It has not yet run on a moving camera.
- The node publishes nothing until it has the camera's calibration and static initialization is done (the camera
  stationary for 2 s). It stops, rather than guess, when a calibration changes, time goes back, the IMU is more than
  10 ms behind a depth frame, or a silent stream leaves it holding more than 1 s of samples (see the node's README).
- The odom frame is z up, with its origin and zero heading where the IMU was during static initialization. The body
  is the IMU.
- The sigmas on `odom_pose` say how well the pose fits the odometry's own map. They leave out that map's drift and
  are too small as a measure of where the robot is in the room. `degenerate_direction` is the direction the cloud
  pinned down least.

## Next

People reach the world store in a live run: the Pi's `pose-dataflow.yml` with the server's
`spatial-server-dataflow.yml` on strixen (`ZENOH_CONNECT` set to the Pi, a token file in the agent's settings)
recorded 3 tracks, 74 torso poses and 615 fitted joints in about 90 s with the camera at rest (a run before the
people tracks went to 30 a second and before posed depth frames were added).

1. The localizer flips between tracking and lost every few seconds with the camera still on a bench; find out why.
   In the 2026-11-15 run (camera at rest, people in view) it lost track only at the start and when the odometry
   started again. The odometry's 14 losses in that run came from people filling the view (see the measurements);
   the odometry now dead reckons for up to its 4.2 cm position-sigma bound, and a 10 minute run with people in view
   had no loss. To check again, count the `odometry lost` lines in `log_odometry.txt` of the run (each says how the
   last second's clouds fared). Someone standing within the minimum range for more than about 1.7 s still loses
   it: the filter's accelerometer noise density is set well above the D455's, which makes its position sigma grow
   that fast at rest.

Also open:

- The Zenoh bridge costs a third of a core on the Pi, more than the pose node; on 2026-11-15 its encode (Arrow IPC
  with zstd) took 0.37 s of every second, 4-6 ms for a 4.5 KB odom pose or a 9.7 KB people frame.
- The keyframe selector's alignment leaves out lens distortion.
- A keyframe is 453 KB after zstd, 3.6 of the Pi's 12.2 Mbit/s uplink (2026-11-15), because the keyframe selector
  upscales depth to the colour image's 1280x720. Sending depth at its native 848x480 needs the depth intrinsics and
  the depth-to-colour extrinsics on the keyframe, and a change in every consumer that takes depth as pixel-aligned
  with colour: the object map and the plane segmentation node (instance ids applied per depth pixel),
  dora-scene-view, rig view's coloured keyframe clouds, and the localizer and dense map (colour intrinsics and the
  colour camera's floor frame). robert has deferred it; revisit if the wifi becomes the limit.
- A person observation's time is when the server received it, not when the camera captured it.
- The keyframes tracked after a restart, until a loop closure ties the run to the map, are reported `tracking`
  though their run is not yet tied to the map; whether they should be `restarted` too is for robert to decide.
- The main clone (`~/src/dora-robot-nodes`, branch depth-perception) still holds 3d-perception's uncommitted IMU
  calibration change to the retired dora-realsense-keyframes. dora-realsense applies the same correction. The
  calibration itself, `~/imu-calibration/calibration.json` on the Pi, is still valid and in use.

Odometry, in order:

1. Find the 1 cm per 30 s creep at rest seen on the Pi's first live run.
2. Record a walk (`record-odometry-inputs.yml` on the Pi: every depth frame, the imu messages, the calibration
   and the person poses), starting and ending at the same spot, and replay it through the odometry
   (`replay-odometry-inputs`, see the odometry crate's README). The gap between the first and last pose is the
   first accuracy number from a real camera. The same recording can feed map building. The one recording so far
   (2026-11-15, 2 minutes, camera at rest) is on the Pi in `~/odometry-recordings/1794725183`.
3. Fix what the walk shows. Known before it starts:
   - height drifts when the floor is out of view; a measurement that holds the robot to the floor is not written,
     and whether to add one is for robert to decide;
   - the lidar is not used: the filter takes point-to-plane rows from any sensor, and none are written for it.

## Glossary

- Static initialization: the start of odometry, while the IMU is stationary long enough to find gravity and the gyro
  bias (`StaticInitialization`); stationary detection finds such spans in a recording (`StationaryDetectionSettings`).
- Measurement update: the Kalman filter step that corrects the IMU's prediction with the depth cloud
  (`MeasurementUpdate`).
- Downsampling: cutting a cloud to one point per voxel before registration (`Downsampler`).
- Degenerate direction: the direction of travel a cloud constrains least (`degenerate_direction` on `odom_pose`).
- Extrinsics: where a sensor sits on the body (`SensorExtrinsics`, `CameraExtrinsics`); intrinsics are its lens.
- Specific force: what the accelerometer measures; at rest, the reaction to gravity (`specific_force`).
- Keyframe selection: choosing which frames become keyframes (`dora-keyframe-selector`), and on the server whether a
  registered keyframe joins the map (`KeyframeSelection`).
- Pose graph optimization: the solve that moves map keyframes after loop closures (`PoseGraphOptimization`); a pose
  graph update is the message of keyframes it moved (`pose_graph_update`).
- Yaw prior: a heading registration is held near (`YawPrior`); the relative yaw prior between keyframes comes from
  the gyro (`RelativeYawPrior`), the timing yaw prior from keyframe gaps (`TimingYawPrior`).
- Pose 4-DoF: x, y, z and yaw (`Pose4Dof`).
- Tracking states: `tracking` (registered), `relocalized` (registered again after loss), `restarted` (starts a new run
  after loss, in the map but not registered), `lost` (dead reckoning, not in the map).
- Fusion: folding depth into the TSDF (truncated signed distance field, `Tsdf`).
- Mesh blocks: the dense map's mesh, one per block, sent as changes with a run and sequence, and as a snapshot on a
  mesh snapshot request (`mesh_blocks`, `MeshBlock`, `BlockMesher`).
- Occupancy grid: free, occupied and unknown cells above the floor, with each free cell's clearance
  (`occupancy_grid`).
- Room layout: one keyframe's planes and room from plane segmentation (`room_layout`, `plane_segmentation`).
- Space carving: removing cloud points a later view saw through (rig view).
- Instance masks: SAM 3's per-pixel instance ids with their labels (`instance_masks`, `instance_ids`).
- Open-vocabulary labels: the object type names Claude lists for a keyframe (`labels`).
- Observation: where a person, a joint or an object was seen at one time (`Observation`, `PersonObservation`,
  `JointObservation`).
- Keypoints are what the pose model detected; the skeleton is what dora-people-3d fitted to them.
- Bridge: `dora-zenoh-bridge`, the node joining machines over Zenoh.
- Spatial query and spatial answer: a question to the spatial agent and its answers (`spatial_query`,
  `spatial_answer`).
