# map-localizer

Places each keyframe in one map: 4-DoF registration against the map built so far, and a hand-written 4-DoF
pose graph. The map frame is z up, gravity aligned, metres; a pose (x, y, z, yaw) is applied to a keyframe's
gravity-aligned floor frame from `depth-perception`.

| Module | What it does |
|---|---|
| `cloud` | A keyframe's depth as a thinned floor-frame point cloud |
| `gyro` | Buffers every IMU sample; refuses an imu message with uneven arrays or samples out of order (`ImuMessageError`); the turn about the vertical between keyframes, with the gyro bias measured on stationary spans (`room-map::heading`); no turn until the bias has been measured, nor across a gap in the samples; the `RelativeYawPrior` for a keyframe |
| `localizer` | Per keyframe: seed from the previous pose plus the gyro turn, register (room-map's `register_near_yaw`) against the map (the keyframes nearest the seed, however old, at their optimized poses); keyframe selection; for a selected keyframe, add the odometry edge and match keyframes older than `loops.skip_latest_keyframes` around the new pose as loop closures; relocalize after loss |
| `pose_graph` | Nodes (x, y, z, yaw) per keyframe, odometry, loop-closure and dead-reckoning edges; Levenberg-Marquardt on an envelope Cholesky of the sparse normal equations; Huber loss on loop closures, which are dropped when the optimized graph contradicts them |
| `map_to_odom` | The odometry frame's correction from keyframes with both a map and an odometry pose; `CameraExtrinsics`, the colour camera's extrinsics on the IMU (translation and rotation, from the calibration), turns the odometry's IMU body pose into the camera's floor frame |
| `synthetic` | Rendered walks through the synthetic living room with IMU messages and true poses, and a camera at rest with range-dependent depth noise, for tests |

## Keyframe selection

A keyframe that registers (tracking or relocalized) within `keyframe_selection.min_translation_m` and
`min_rotation_deg` of a map keyframe it registered against is not added to the map
(`KeyframeSelection::NotSelected`): its pose is reported, and it keeps its offset from that reference keyframe, so
an optimization that moves the reference keyframe moves it too (`Localizer::followers`). Only a selected keyframe
becomes a pose graph node with a stored cloud, an odometry edge and a loop-closure attempt. A camera at rest
therefore adds nothing to the map and keeps registering against the same map keyframes; the graph grows with the
space explored. Each keyframe not selected costs one small record (its frame id and offset).

## Tracking states

- `tracking`: registered against the map keyframes nearest its seed.
- `relocalized`: tracking was lost and the keyframe matched stored keyframes around its dead-reckoned pose.
- `restarted`: after `restart_after_lost` lost keyframes in a row, one with enough points restarts tracking from
  its dead-reckoned pose, joined to the graph by a weak dead-reckoning edge. It is a map keyframe
  (`KeyframeSelection::Selected`) but not registered against the map, and later loop closures pull its run into
  place, moving its pose. The keyframes tracked after it, until a loop closure ties the run to the map, are
  reported `tracking`.
- `lost`: not registered; the pose is the last position turned by the gyro and the keyframe stays out of the
  map.

Relocalization accepts at `relocalize.min_fitness`, above tracking's `accept_fitness`: it keeps the best of many
seeds spread over a wide search, which match a wrong place more often than tracking's one seed does.

## Without a gyro turn

The gyro cannot give a keyframe's turn until it has seen `still.min_stationary_ns` of stationary time to measure its bias,
nor across a gap in the samples: two consecutive gyro samples around or inside the keyframe's span further apart
than `room-map::heading::MAX_GYRO_GAP_STEPS` times the gyro's median sample step, as a lost imu message leaves. `HeadingWithoutGyro` decides what happens then:

- `Refuse` (the default): `RelativeYawPrior::Refused`. The keyframe is `lost` without any registration. Before the
  map has started it stays empty; after, the heading is unknown, dead reckoning stops, and the following
  keyframes are only relocalized against the map, over every heading, until one matches.
- `Search`: `RelativeYawPrior::Search`. Tracking tries headings and shifts around the previous pose, and
  relocalization the whole circle. For recordings without a usable IMU.

The graph is optimized on each loop closure and at least every `optimize_interval_ns` while edges are added.
An optimization that moves any keyframe further than `version_move_m` or `version_turn_deg` raises the map
version.

## Running

```
cargo test --release -p map-localizer
cargo run --release -p map-localizer --example localize_recording -- <keyframes.arrow> <trajectory.csv> <refuse|search> [<imu.arrow>]
```

Without an imu recording no keyframe has a gyro turn, so `localize_recording` needs `search`, which is slower
than gyro seeding.
