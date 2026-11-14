# map-localizer

Places each keyframe in one map: 4-DoF registration against the map built so far, and a hand-written 4-DoF
pose graph. The map frame is z up, gravity aligned, metres; a pose (x, y, z, yaw) is applied to a keyframe's
gravity-aligned floor frame from `depth-perception`.

| Module | What it does |
|---|---|
| `cloud` | A keyframe's depth as a thinned floor-frame point cloud |
| `gyro` | Buffers every IMU sample; the turn about the vertical between keyframes, with the gyro bias measured on still spans (`room-map::heading`); no turn until the bias has been measured; the `HeadingHint` for a keyframe |
| `localizer` | Per keyframe: seed from the previous optimised pose plus the gyro turn, register (room-map's `register_near_yaw`) against the map (the keyframes nearest the seed, however old, at their optimised poses), add the odometry edge; match keyframes older than `loops.skip_latest_keyframes` around the new pose as loop closures; relocalise after loss |
| `pose_graph` | Nodes (x, y, z, yaw) per keyframe, odometry, loop-closure and dead-reckoning edges; Levenberg-Marquardt on an envelope Cholesky of the sparse normal equations; Huber loss on loop closures, which are dropped when the optimised graph contradicts them |
| `map_to_odom` | The odometry frame's correction from keyframes with both a map and an odometry pose; `CameraMount` turns the odometry's IMU body pose into the camera's floor frame |
| `synthetic` | Rendered walks through the synthetic living room with IMU messages and true poses, for tests |

## Tracking states

- `tracking`: registered against the map keyframes nearest its seed.
- `relocalised`: tracking was lost and the keyframe matched stored keyframes around its dead-reckoned pose.
- `lost`: not registered; the pose is the last position turned by the gyro and the keyframe stays out of the
  map. After `restart_after_lost` lost keyframes in a row, one with enough points restarts tracking from its
  dead-reckoned pose, joined to the graph by a weak dead-reckoning edge; it is reported `lost`, and later loop
  closures pull its run into place.

## Without a gyro turn

The gyro cannot give a keyframe's turn until it has seen `still.min_still_ns` of still time to measure its bias,
nor across a gap in the samples. `HeadingWithoutGyro` decides what happens then:

- `Refuse` (the default): `HeadingHint::Refused`. The keyframe is `lost` without any registration. Before the
  map has started it stays empty; after, the heading is unknown, dead reckoning stops, and the following
  keyframes are only relocalised against the map, over every heading, until one matches.
- `Search`: `HeadingHint::Search`. Tracking tries headings and shifts around the previous pose, and
  relocalisation the whole circle. For recordings without a usable IMU.

The graph is optimised on each loop closure and at least every `optimise_interval_ns` while edges are added.
An optimisation that moves any keyframe further than `version_move_m` or `version_turn_deg` raises the map
version.

## Running

```
cargo test --release -p map-localizer
cargo run --release -p map-localizer --example localize_recording -- <keyframes.arrow> <trajectory.csv> <refuse|search> [<imu.arrow>]
```

Without an imu recording no keyframe has a gyro turn, so `localize_recording` needs `search`, which is slower
than gyro seeding.
