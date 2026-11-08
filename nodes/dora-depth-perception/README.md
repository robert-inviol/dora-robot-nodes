# dora-depth-perception

Room geometry from one RGB-D frame of a depth camera with an accelerometer (built against an Intel
RealSense D455), in a gravity-aligned floor frame: x right, y forward, z up, metres, origin at the camera.

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

`perception-wire` is the Rust side of the perception messages in `lib/dora-rig-messages`; its tests read
that package's golden fixtures.

## Dataflows

- `examples/replay-dataflow.yml`: the whole pipeline on one machine from a recording: replay, Claude
  object types, SAM 3, this node and the debug page on port 8090.
- `examples/live-dataflow.yml`: the same, fed by the camera machine over `dora-zenoh-bridge`, with scenes
  sent back to it.
