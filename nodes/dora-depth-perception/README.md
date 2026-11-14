# dora-depth-perception

Room geometry from one RGB-D frame of a depth camera with an accelerometer (built against an Intel
RealSense D455), in a gravity-aligned floor frame: x right, y forward, z up, metres, origin at the camera.
The workspace also holds the mapping crates and nodes, which place keyframes in one map and build the
house's free space, rooms and objects from them (see Mapping).

The `depth-perception` crate is the geometry, with no dora dependency:

| Stage | What it gives |
|---|---|
| `sample_point_cloud` | Every n-th pixel with a trusted depth, as floor-frame points |
| `find_planes` | Walls and level surfaces (floor, ceiling, table tops), exactly upright or level, with outlines; plane points flattened onto their plane |
| `infer_room` | The room as the convex space between the floor, the ceiling and the outermost wide, ceiling-high walls; sides with no wall seen are marked unseen |
| `object_footprint` | An object's footprint rectangle, height and distance from its segmentation mask |

## Running

```
cargo test --release
cargo bench
cargo run --release --example profile_frame -- <frame dir> [repeats]
```

`profile_frame` times each stage on a recorded frame; its doc comment describes the frame directory.

## Nodes

| Node | Inputs | Outputs |
|---|---|---|
| `dora-depth-perception` | `keyframe`, `segments` | `structure`: planes and room as soon as a keyframe arrives; `scene`: the same plus object footprints when its segments arrive |
| `dora-keyframe-replay` | `tick` | `keyframe`: the next keyframe of the Arrow IPC recording in `KEYFRAME_RECORDING`, restamped; `KEYFRAME_REPLAY_LOOP=true` starts again at the end. `imu`: with `IMU_RECORDING`, the imu message leading up to each keyframe, sent just before it, its camera times moved on by one recording span per loop |
| [`dora-map-localizer`](dora-map-localizer/README.md) | `keyframe`, `imu`, `odom_pose` | `keyframe_pose`: each keyframe's pose in the map, with the map version and tracking state; `map_to_odom`: the correction from odometry to the map |
| [`dora-dense-map`](dora-dense-map/README.md) | `keyframe`, `keyframe_pose`, `object_update`, `tick` | `free_space`, `topology`: rooms, doorways, places and their graph; `keep_out`: zones around glass and mirrors |
| [`dora-object-map`](dora-object-map/README.md) | `keyframe`, `keyframe_pose`, `segments` | `object_update`: an object appeared, updated, moved, disappeared or merged |

`room-map` builds one map of a room from a keyframe recording: gravity-aligned 4-DoF ICP odometry, runs
of trusted keyframes placed against the growing map, loop measurement against the start, and voxel
fusion. `cargo run --release -p room-map --example build_map -- <recording> <out dir> [<imu recording>]`
writes `map.html`, `trajectory.csv` and `report.txt`. How each keyframe's heading is found depends on what
was recorded:

- With an imu recording, the gyro gives the turn between keyframes, with its bias measured on every span
  where the camera is still. Registration starts at the gyro's heading and is held near it; where depth
  cannot register (glass, a blank wall, a person in view), the gyro carries the heading on and tracking
  restarts as soon as depth takes hold; placement then fits each restarted run within a narrow band of
  headings. Keyframes inside a still span are not registered at all. A check that the gyro turns the same
  way as the depth stops the build if the IMU axes do not match the camera.
- Without one, each keyframe's heading is searched for over turns and steps, and a registered turn that
  contradicts the keyframe's gap from the last (the camera node takes a keyframe early only after turning
  past its threshold) is rejected. `examples/keyframe_gap_check.rs` compares tracking with and without that
  check.

`depth-inertial-odometry` is live 6-DoF odometry from the depth cloud and the IMU: the IMU predicts, each
cloud is registered against a local map, and an error-state Kalman filter blends them. It runs on a
simulator; its [README](depth-inertial-odometry/README.md) compares its three maps for accuracy and for
speed on a Raspberry Pi 5.

`perception-wire` is the Rust side of the messages in `lib/dora-rig-messages`: perception, localization,
space map, objects, navigation and odometry; its tests read that package's golden fixtures.

## Mapping

The map frame is z up, gravity aligned, metres. Three crates, with no dora dependency, build the map that
the mapping nodes above wrap:

| Crate | What it does |
|---|---|
| [`map-localizer`](map-localizer/README.md) | Places each keyframe in one map: 4-DoF registration against the map built so far, seeded by the gyro, with loop closures in a 4-DoF pose graph |
| [`dense-map`](dense-map/README.md) | The dense map from posed depth keyframes: a TSDF, a free-space grid, rooms cut at doorways with a graph of places, and keep-out zones |
| [`object-map`](object-map/README.md) | Persistent objects from SAM 3 masks: association, confirmation, absences, moves and merges, and each object's best view |

## Dataflows

- `examples/replay-dataflow.yml`: the whole pipeline on one machine from a recording: replay, Claude
  object types, SAM 3, this node and the debug page on port 8090.
- `examples/live-dataflow.yml`: the same, fed by the camera machine over `dora-zenoh-bridge`, with scenes
  sent back to it.
- `examples/spatial-server-dataflow.yml`: the spatial server on the mapping machine: localizer, dense map,
  object map and `dora-world-agent`, fed over `dora-zenoh-bridge` with keyframes and IMU from the camera
  machine and segments from a GPU machine, sending poses, the map, objects, answers and nav goals back.
- `dora-map-localizer/replay-dataflow.yml`, `dora-dense-map/dense-map-dataflow.yml` and
  `dora-object-map/object-map-dataflow.yml` run one mapping node each.
