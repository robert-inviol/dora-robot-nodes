//! Runs odometry over the simulated walk with each kind of map and prints accuracy and time per cloud.
//!
//! `cargo run --release -p depth-inertial-odometry --example simulated_walk -- [seconds]`

use std::time::{Duration, Instant};

use depth_inertial_odometry::odometry::{CloudOutcome, StillStart};
use depth_inertial_odometry::simulation::{seeded, DepthCamera, ImuErrors, Scene, Walk};
use depth_inertial_odometry::{DepthInertialOdometry, KdTreeMap, NeighbourPlaneSettings, NeighbourPlanes, OdometrySettings, PlaneVoxelMap, PlaneVoxelSettings, SurfaceMap, VoxelHashMap};

const IMU_PERIOD: Duration = Duration::from_micros(2500);
const IMU_PER_CLOUD: u32 = 13;
const STILL: Duration = Duration::from_secs(2);

fn run<M: SurfaceMap>(name: &str, walk: Walk, map: M, walk_for: Duration) {
    let (scene, camera, errors) = (Scene::living_room(), DepthCamera::d455_quarter(), ImuErrors::consumer());
    let mut rng = seeded(7);
    let still_samples = (STILL.as_micros() / IMU_PERIOD.as_micros()) as u32;
    let resting: Vec<_> = (0..still_samples).map(|i| walk.imu_sample(IMU_PERIOD * i, &errors, &mut rng)).collect();
    let mut odometry = DepthInertialOdometry::new(StillStart::from_samples(&resting).expect("the walk starts at rest"), map, OdometrySettings::d455());

    let (mut worst_level_m, mut worst_height_m) = (0.0f64, 0.0f64);
    let (mut cloud_times, mut worst_m, mut sum_m, mut matched_sum, mut offered_sum, mut iteration_sum) = (Vec::new(), 0.0f64, 0.0f64, 0usize, 0usize, 0usize);
    let last = ((STILL + walk_for).as_micros() / IMU_PERIOD.as_micros()) as u32;
    for i in still_samples..last {
        let time = IMU_PERIOD * i;
        odometry.add_imu(walk.imu_sample(time, &errors, &mut rng)).expect("samples are in order");
        if i % IMU_PER_CLOUD != 0 {
            continue;
        }
        let seconds = time.as_secs_f64();
        let cloud = camera.cloud(&scene, walk.position(seconds), walk.rotation(seconds), &mut rng);
        let started = Instant::now();
        let outcome = odometry.add_cloud(time, &cloud).expect("every cloud registers");
        cloud_times.push(started.elapsed());
        if let CloudOutcome::Corrected { matched, offered, iterations, .. } = outcome {
            matched_sum += matched;
            offered_sum += offered;
            iteration_sum += iterations;
        }
        let offset = odometry.state().levelled_pose().0 - walk.pose_from_start(seconds).0;
        worst_level_m = worst_level_m.max(offset.truncate().length());
        worst_height_m = worst_height_m.max(offset.z.abs());
        let error = offset.length();
        worst_m = worst_m.max(error);
        sum_m += error;
    }
    cloud_times.sort();
    let clouds = cloud_times.len();
    let millis = |d: Duration| d.as_secs_f64() * 1e3;
    println!(
        "{name:<16} clouds {clouds}  per cloud: median {:.2} ms, p95 {:.2} ms, worst {:.2} ms  position error: mean {:.3} m, worst {:.3} m (level {:.3}, height {:.3})  matched {}/{} points, {:.1} iterations  map {} entries",
        millis(cloud_times[clouds / 2]),
        millis(cloud_times[clouds * 95 / 100]),
        millis(cloud_times[clouds - 1]),
        sum_m / clouds as f64,
        worst_m,
        worst_level_m,
        worst_height_m,
        matched_sum / (clouds - 1),
        offered_sum / (clouds - 1),
        iteration_sum as f64 / (clouds - 1) as f64,
        odometry.map().len(),
    );
}

fn main() {
    let seconds = std::env::args().nth(1).map(|text| text.parse::<u64>().expect("the argument is a whole number of seconds")).unwrap_or(60);
    println!("simulated walk, {seconds} s, {} threads", rayon::current_num_threads());
    for (mount, walk) in [("on a robot", Walk::on_a_robot(STILL)), ("carried", Walk::carried(STILL))] {
        println!("camera {mount}, {:.2} m above the floor", walk.height_m);
        let neighbours = NeighbourPlaneSettings::indoors();
        run("grid neighbours", walk, NeighbourPlanes::new(VoxelHashMap::new(0.3, 20, 0.1), neighbours), Duration::from_secs(seconds));
        run("tree neighbours", walk, NeighbourPlanes::new(KdTreeMap::new(0.1), neighbours), Duration::from_secs(seconds));
        run("voxel planes", walk, PlaneVoxelMap::new(PlaneVoxelSettings::indoors()), Duration::from_secs(seconds));
    }
}
