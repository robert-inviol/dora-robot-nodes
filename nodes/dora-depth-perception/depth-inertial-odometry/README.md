# depth-inertial-odometry

Live odometry for a depth camera with an IMU (built for an Intel RealSense D455 on a Raspberry Pi 5):
where the camera is, how it is turned and how fast it moves, updated with every IMU sample and corrected
with every depth cloud. A library with no dora dependency; the node is
[`dora-depth-inertial-odometry`](../dora-depth-inertial-odometry/README.md). It has run on a simulator
only, not yet on recorded or live camera data.

It follows the recipe of [FAST-LIO2](https://github.com/hku-mars/FAST_LIO), adapted from lidar to a
depth camera:

1. **Start.** The body is held still. The accelerometer says which way is up and the gyro's reading is
   its bias (`StillStart`). A start while moving is refused.
2. **Predict.** Every IMU sample moves the estimate on: the gyro turns it, the accelerometer changes its
   velocity and position. Uncertainty grows.
3. **Thin.** Each cloud is cut to one point per 10 cm voxel, the mean of the voxel's readings, kept only
   where those readings lie on one flat patch (`thinning`).
4. **Register.** Each thinned point is measured against the plane the map holds where the point lands
   (`association`, `surfaces`).
5. **Correct.** An error-state Kalman filter moves the whole state to shrink those distances, weighing
   the cloud against the IMU's prediction, repeated up to four times per cloud (`filter`).
6. **Map.** The cloud's points go into the map at the corrected pose; the map keeps a cube around the body.

The state is position, orientation, velocity, gyro bias, accelerometer bias and the direction of up: 17
numbers of error. The world frame starts at the body with z where the resting accelerometer said up was.
That reading is off by the accelerometer's bias, so up is a state the filter keeps learning, and
`State::levelled_pose` gives the pose in a frame whose z is the up learned so far. The body frame is the
IMU's.

## Using it

```rust
let start = StillStart::from_samples(&resting_imu_samples)?;
let mut odometry = DepthInertialOdometry::new(start, PlaneVoxelMap::new(PlaneVoxelSettings::indoors()), OdometrySettings::d455());
odometry.add_imu(sample)?;                              // every IMU sample, in time order
let outcome = odometry.add_cloud(time, &sensor_points)?; // every depth frame, as sensor-frame points
let (position, rotation) = odometry.state().levelled_pose();
```

### From a camera's raw streams

`PoseTracker` (`tracker`) is the same loop fed with what a camera gives: raw depth images and the IMU's
two streams. It holds the pieces in between:

- **Depth image to cloud** (`depth_image`). `DepthProjector` turns a row-major image of `u16`
  millimetres (zero is no reading) and pinhole intrinsics into points in the depth stream's optical
  frame: x right, y down, z forward. It reads every `stride`-th pixel of every `stride`-th row.
- **Mount** (`SensorMount::from_depth_to_imu`). The camera's depth-to-IMU transform, a column-major 3x3
  rotation and a translation, takes those points to the body frame, which is the IMU's.
- **IMU pairing** (`imu_pairing`). The accelerometer and the gyro stamp their own samples at their own
  rates. `ImuPairer` gives one sample per gyro sample with the accelerometer interpolated to its time,
  and holds gyro samples newer than the latest accelerometer sample until the accelerometer passes them.
- **Start.** The tracker collects `rest_for` of samples and starts with `StillStart`. A window in which
  the body moved comes back as `TrackerError::NotAtRest`; collection has then begun again, so feeding on
  tries the next window.

```rust
let mount = SensorMount::from_depth_to_imu(depth_to_imu_rotation, depth_to_imu_translation_m)?;
let mut tracker = PoseTracker::new(depth_intrinsics, mount, PlaneVoxelMap::new(PlaneVoxelSettings::indoors()), TrackerSettings::d455())?;
tracker.add_imu(&accel_readings, &gyro_readings)?;          // every batch, before the frame it leads up to
match tracker.add_depth_frame(time, frame_id, &depth_mm)? { // every depth frame
    DepthFrameOutcome::AwaitingStart => {}
    DepthFrameOutcome::Posed(pose) => { /* a FramePose */ }
}
```

A `FramePose` is the estimate at the frame's time in the level frame: position, orientation, velocity,
one-sigma position along the level axes, one-sigma tilt and heading, whether the cloud corrected the
estimate, how many points matched and the weakest direction. It carries every field of the `odom_pose`
message in `lib/dora-rig-messages`. A frame whose cloud cannot correct still gets a pose, the IMU's
prediction, marked `CloudUse::NotCorrected`. Pose times strictly increase: a frame not newer than the
last posed one is refused.

The sigmas come from the filter's covariance (`State::levelled_uncertainty`). Levelling depends on up,
so a doubt in up reaches the level position (it swings the position about the start) and the tilt; tilt
is the body's orientation error and the error in up taken together, about the level axis where it is
known worst. They are sigmas against the odometry's own map and do not include its drift.

A depth frame is handled at its own time: IMU samples up to it go in first, newer ones wait for the next
frame. When the frame is ahead of the IMU, the gyro samples up to it are paired with the latest
accelerometer reading and the newest reading is held over the rest of the gap; a frame more than
`max_imu_lag` (10 ms) ahead of either stream is refused with `TrackerError::ImuBehind`.

Samples are held while they wait: one IMU stream's for the other stream to pair with, and paired samples
for the next depth frame. A stream that goes silent would let them pile up, so `add_imu` refuses with
`TrackerError::Backlog` once the samples held for one thing span more than `max_backlog` (1 s) of camera
time. They stay held, so the refusal repeats; the tracker is then of no further use.

### The odometry alone

`add_cloud` reports how many points matched and the direction the cloud pinned down least, with the
one-sigma uncertainty the cloud alone leaves along it (`CloudOutcome::Corrected::weakest`). A cloud that
cannot correct the estimate is an error (`OdometryError::NotCorrected`): the estimate keeps the IMU's
prediction and the cloud stays out of the map.

Another sensor joins through `add_cloud_with`: anything that can say "this body point lies on that world
plane" returns a `PoseInformation`, and its measurements enter the same correction as the cloud's. A 2D
lidar's points against wall lines, or "the robot stands on the floor", have this form. Neither is written.

```
cargo test --release -p depth-inertial-odometry
cargo run --release -p depth-inertial-odometry --example simulated_walk -- 60
cargo run --release -p depth-inertial-odometry --example depth_frame_cost
```

`simulated_walk` drives the simulated room for the given number of seconds with each map and prints
accuracy and time per cloud. `depth_frame_cost` times depth-to-points and thinning on one simulated
848 x 480 frame. They are the benchmarks behind the tables below.

## The three maps

A map answers one question: what plane is the surface at this world point, and how well is it known
(`SurfaceMap`). Three maps answer it, and all three stay in the crate so they can be compared on real data.

| Map | What it keeps | How it answers |
|---|---|---|
| `PlaneVoxelMap` (voxel planes) | One plane per 25 cm voxel, a weighted fit to every point that ever fell in it | The plane of the point's own voxel, with a sigma that shrinks as the voxel is seen more; nothing if the voxel's points are not flat |
| `NeighbourPlanes<VoxelHashMap>` (grid neighbours) | Points about 10 cm apart, bucketed in a hashed grid | A plane fitted through the point's five nearest map points, taken as exact |
| `NeighbourPlanes<KdTreeMap>` (tree neighbours) | The same points in an incremental k-d tree that rebalances lopsided subtrees and drops points lazily | The same five-point plane |

The grid and the tree find exactly the same neighbours as a full scan (tested). They differ in which
points they keep: the grid keeps at most 20 per 30 cm voxel no closer than 10 cm; the tree keeps one per
10 cm cube. The five-neighbour plane is FAST-LIO2's design.

**Voxel planes are the map to use.** They are the most accurate, the fastest, need one core, and are
the only map that is honest about what it does not know.

### Accuracy

Simulated room 6 x 5 x 2.4 m with furniture, a D455-like camera (87 x 58 degrees, depth noise 5 mm per
square metre of range, 212 x 120 rays), a consumer IMU with biases (gyro about 0.005 rad/s, accelerometer
about 0.09 m/s2) and noise, clouds at about 31 Hz, IMU at 400 Hz. The drive loops the room while turning;
its length is roughly 25 m in 60 s. "Worst" is the largest error against the truth at any cloud. No loop
closing: this is drift.

| | Voxel planes | Grid neighbours | Tree neighbours |
|---|---|---|---|
| 30 s drive, camera 0.25 m above the floor: worst position | 0.066 m | 0.096 m | 0.088 m |
| 30 s drive: worst orientation | 0.57 deg | 0.94 deg | 1.55 deg |
| 30 s drive: accelerometer bias error at the end | 0.004 m/s2 | 0.052 m/s2 | 0.089 m/s2 |
| 60 s drive, camera 0.25 m above the floor: worst position | 0.073 m | 0.099 m | 0.097 m |
| 60 s drive, camera carried at 1.0 m: worst position | 0.136 m | 0.316 m | 0.371 m |
| Standing still facing a lone wall: uncertainty reported along the wall | 0.035 m | 0.002 m | 0.002 m |

The last row is the honesty test. A wall seen alone says how far away it is and nothing about sliding
along it, so the right answer is "unknown". A plane through five noisy neighbours tilts a few degrees at
random, each tilt looks like a little knowledge of the sideways direction, and hundreds of them add up to
false certainty of 2 mm. A voxel's plane is fitted to hundreds of points over many frames, so it tilts far
less. Across wall distances and noise seeds the voxel planes report 2.5 to 16 cm.

How the filter's stated one-sigma compares with the actual error on the 60 s drive (1 is honest, larger
is overconfident):

| | Voxel planes | Grid neighbours | Tree neighbours |
|---|---|---|---|
| Tilt | 1.8 | 11 | 16 |
| Accelerometer bias | 0.9 | 11 | 14 |
| Level position | 25 | 13 | 35 |
| Height | 9 | 111 | 74 |

Position is overconfident on every map: the filter knows the body's place relative to the map, and the
map's own drift is not in the state. Correcting that is the job of a slower loop that closes loops and
sends back a correction.

### Speed

Time for one `add_cloud` (thin, register, correct, insert) on a cloud of about 25,000 points, 60 s drive,
camera 0.25 m above the floor.

| Raspberry Pi 5 | Voxel planes | Grid neighbours | Tree neighbours |
|---|---|---|---|
| 4 threads: median / 95th percentile / worst | 2.5 / 3.2 / 3.6 ms | 4.2 / 7.2 / 10.1 ms | 4.4 / 7.3 / 9.8 ms |
| 1 thread (`RAYON_NUM_THREADS=1`): median / 95th percentile / worst | 2.6 / 3.4 / 3.8 ms | 8.6 / 17.7 / 23.6 ms | 9.3 / 18.1 / 22.6 ms |
| Map size after 60 s | 731 planes | 14,284 points | 18,644 points |

A frame period at 30 Hz is 33 ms, so voxel planes take about 8% of one core. They gain nothing from more
threads, which leaves the other cores free. On a Ryzen 9 9950X with 32 threads the medians are 0.95, 1.17
and 1.10 ms.

The simulated cloud is a quarter of the D455's 848 x 480 depth resolution each way. On a full frame,
the two steps before registration cost, by the stride the image is read at (Raspberry Pi 5, median of
300 repeats on one simulated frame of the room; neither step uses more than one thread, and the times
with `RAYON_NUM_THREADS=1` are the same):

| Stride | Points | Depth image to points | `Thinner::thin` | Thinned points |
|---|---|---|---|---|
| 1 | 407,040 | 1.62 ms | 21.6 ms | 819 |
| 2 | 101,760 | 0.42 to 0.56 ms | 5.6 ms | 788 |
| 4 | 25,440 | 0.11 to 0.15 ms | 1.6 ms | 721 |

Thinning costs about 55 ns per reading, so a full frame takes two thirds of a 33 ms frame period on one
core and a stride of 4 takes 5%. `TrackerSettings::d455` reads at a stride of 4, which is the density
the simulated drives run at and loses an eighth of the thinned points.

## What a depth camera needs that a lidar does not

Each of these is in the code because the simulated drive failed without it.

- **Noise that grows with range.** Each thinned point carries its own sigma: a floor that averaging does
  not remove plus the range-squared term divided by the root of the readings averaged (`RangeNoise`).
  Matches are gated and weighted in units of that sigma.
- **Flat patches only.** The mean of a voxel that straddles an edge lies on no surface, so a voxel is
  kept only if its readings are as thin as their noise explains (`Thinner`).
- **Surfaces must face the same way.** A turning camera with a narrow view keeps revealing surfaces next
  to ones already mapped. A thinned point that knows which way its patch faces is matched only to a plane
  facing within about 37 degrees of it (`AssociationSettings::min_facing_agreement`).
- **No borrowed planes.** A voxel-plane lookup answers from the point's own voxel only, and a voxel
  thicker than its points' sigmas explain answers nothing (`PlaneVoxelSettings::max_thickness_sigmas`).
  A neighbouring voxel's plane is often a different surface.
- **Neighbours must span a surface.** For the neighbour maps: the five points must spread in two
  directions, and the nearest must be within 10 cm (`NeighbourPlaneSettings`). The edge of the view adds
  points to the map as a thin line, and a plane through a line faces anywhere.
- **Up is a state.** A still start levels with the accelerometer, so its bias becomes tilt. With up
  fixed, that tilt can never be corrected and the filter invents accelerometer bias to explain the floor.
- **The weak direction.** A camera sees a narrow cone, often one wall and the floor. The direction the
  cloud does not pin down is reported with every cloud so a consumer can slow down or distrust it. Only
  the voxel planes report it truthfully.
- **Height.** A camera carried level at chest height sees the floor only beyond about 1.8 m and often
  loses it; height then drifts. At robot height the floor is nearly always in view.

## Limits

- Run on the simulator only. Real D455 depth has holes, stray points at edges and noise that is not
  independent between neighbouring pixels.
- The tests bound drift at 0.10 m and 1 degree over a 30 s drive for voxel planes, and 0.15 m and 2.5
  degrees for the neighbour maps. A drift under 0.05 m over that drive is not reached by any map.
- The full-resolution depth image is only subsampled, not averaged, before thinning.
- People and other moving things are not masked out of the cloud.
- The neighbour maps report their planes as exact, which is the source of their overconfidence.
- Map points carry no memory of how uncertain the pose was when they were inserted.
- The node, [`dora-depth-inertial-odometry`](../dora-depth-inertial-odometry/README.md), has run on the
  simulator's messages only.
