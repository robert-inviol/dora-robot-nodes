use std::time::Duration;

use depth_inertial_odometry::odometry::{CloudOutcome, StillStart};
use depth_inertial_odometry::simulation::{seeded, DepthCamera, ImuErrors, Scene, Walk};
use depth_inertial_odometry::{DepthInertialOdometry, KdTreeMap, NeighbourPlaneSettings, NeighbourPlanes, OdometrySettings, PlaneVoxelMap, PlaneVoxelSettings, SurfaceMap, VoxelHashMap};
use glam::DVec3;

const IMU_PERIOD: Duration = Duration::from_micros(2500);
const IMU_PER_CLOUD: u32 = 13;
const STILL: Duration = Duration::from_secs(2);

struct Flown {
    worst_position_m: f64,
    worst_angle_deg: f64,
    gyro_bias_error_rad_s: f64,
    accel_bias_error_m_s2: f64,
    corrected: usize,
}

/// Runs odometry over a simulated walk and compares it with the truth at every cloud.
fn fly<M: SurfaceMap>(map: M, scene: &Scene, walk_for: Duration) -> Flown {
    let walk = Walk::on_a_robot(STILL);
    let camera = DepthCamera::d455_quarter();
    let errors = ImuErrors::consumer();
    let mut rng = seeded(7);

    let still_samples = (STILL.as_micros() / IMU_PERIOD.as_micros()) as u32;
    let resting: Vec<_> = (0..still_samples).map(|i| walk.imu_sample(IMU_PERIOD * i, &errors, &mut rng)).collect();
    let start = StillStart::from_samples(&resting).expect("the walk starts at rest");
    let mut odometry = DepthInertialOdometry::new(start, map, OdometrySettings::d455());

    let mut flown = Flown { worst_position_m: 0.0, worst_angle_deg: 0.0, gyro_bias_error_rad_s: 0.0, accel_bias_error_m_s2: 0.0, corrected: 0 };
    let last = ((STILL + walk_for).as_micros() / IMU_PERIOD.as_micros()) as u32;
    for i in still_samples..last {
        let time = IMU_PERIOD * i;
        odometry.add_imu(walk.imu_sample(time, &errors, &mut rng)).expect("samples are in order");
        if i % IMU_PER_CLOUD == 0 {
            let seconds = time.as_secs_f64();
            let cloud = camera.cloud(scene, walk.position(seconds), walk.rotation(seconds), &mut rng);
            if let CloudOutcome::Corrected { .. } = odometry.add_cloud(time, &cloud).expect("every cloud registers") {
                flown.corrected += 1;
            }
            let (true_position, true_rotation) = walk.pose_from_start(seconds);
            let (position, rotation) = odometry.state().levelled_pose();
            flown.worst_position_m = flown.worst_position_m.max((position - true_position).length());
            flown.worst_angle_deg = flown.worst_angle_deg.max(rotation.angle_between(true_rotation).to_degrees());
        }
    }
    let state = odometry.state();
    flown.gyro_bias_error_rad_s = (state.gyro_bias - errors.gyro_bias_rad_s).length();
    flown.accel_bias_error_m_s2 = (state.accel_bias - errors.accel_bias_m_s2).length();
    flown
}

/// How far a map may stray from the truth over the drive.
struct Bounds {
    position_m: f64,
    angle_deg: f64,
    gyro_bias_rad_s: f64,
    accel_bias_m_s2: f64,
}

/// Planes fitted through five neighbours carry tilt noise the filter takes for knowledge, which costs
/// accuracy in orientation and in the accelerometer bias.
const NEIGHBOUR_PLANES: Bounds = Bounds { position_m: 0.15, angle_deg: 2.5, gyro_bias_rad_s: 4e-3, accel_bias_m_s2: 0.12 };
const VOXEL_PLANES: Bounds = Bounds { position_m: 0.10, angle_deg: 1.0, gyro_bias_rad_s: 1e-3, accel_bias_m_s2: 0.03 };

fn follows_the_walk(flown: &Flown, bounds: &Bounds, name: &str) {
    println!(
        "{name}: worst {:.3} m, {:.2} deg; bias errors {:.4} rad/s, {:.3} m/s2; {} clouds corrected",
        flown.worst_position_m, flown.worst_angle_deg, flown.gyro_bias_error_rad_s, flown.accel_bias_error_m_s2, flown.corrected
    );
    assert!(flown.worst_position_m < bounds.position_m, "{name}: position strayed {:.3} m", flown.worst_position_m);
    assert!(flown.worst_angle_deg < bounds.angle_deg, "{name}: orientation strayed {:.2} deg", flown.worst_angle_deg);
    assert!(flown.gyro_bias_error_rad_s < bounds.gyro_bias_rad_s, "{name}: gyro bias off by {:.4} rad/s", flown.gyro_bias_error_rad_s);
    assert!(flown.accel_bias_error_m_s2 < bounds.accel_bias_m_s2, "{name}: accelerometer bias off by {:.3} m/s2", flown.accel_bias_error_m_s2);
}

fn grid_neighbours() -> NeighbourPlanes<VoxelHashMap> {
    NeighbourPlanes::new(VoxelHashMap::new(0.3, 20, 0.1), NeighbourPlaneSettings::indoors())
}

fn tree_neighbours() -> NeighbourPlanes<KdTreeMap> {
    NeighbourPlanes::new(KdTreeMap::new(0.1), NeighbourPlaneSettings::indoors())
}

fn voxel_planes() -> PlaneVoxelMap {
    PlaneVoxelMap::new(PlaneVoxelSettings::indoors())
}

#[test]
fn grid_neighbours_follow_a_drive_around_the_room() {
    follows_the_walk(&fly(grid_neighbours(), &Scene::living_room(), Duration::from_secs(30)), &NEIGHBOUR_PLANES, "grid neighbours");
}

#[test]
fn tree_neighbours_follow_a_drive_around_the_room() {
    follows_the_walk(&fly(tree_neighbours(), &Scene::living_room(), Duration::from_secs(30)), &NEIGHBOUR_PLANES, "tree neighbours");
}

#[test]
fn voxel_planes_follow_a_drive_around_the_room() {
    follows_the_walk(&fly(voxel_planes(), &Scene::living_room(), Duration::from_secs(30)), &VOXEL_PLANES, "voxel planes");
}

/// Stands still 2.1 m from an endless wall, looking straight at it, and returns what the last cloud
/// said about the direction it pinned down least.
fn facing_a_lone_wall<M: SurfaceMap>(map: M) -> depth_inertial_odometry::filter::WeakestTranslation {
    let (scene, camera, errors) = (Scene::lone_wall(), DepthCamera::d455_quarter(), ImuErrors::consumer());
    let walk = Walk::carried(Duration::from_secs(60));
    let mut rng = seeded(3);
    let resting: Vec<_> = (0..400).map(|i| walk.imu_sample(IMU_PERIOD * i, &errors, &mut rng)).collect();
    let mut odometry = DepthInertialOdometry::new(StillStart::from_samples(&resting).expect("at rest"), map, OdometrySettings::d455());
    let mut weakest = None;
    for i in 400..1200u32 {
        let time = IMU_PERIOD * i;
        odometry.add_imu(walk.imu_sample(time, &errors, &mut rng)).expect("in order");
        if i % IMU_PER_CLOUD == 0 {
            if let CloudOutcome::Corrected { weakest: found, .. } = odometry.add_cloud(time, &camera.cloud(&scene, DVec3::ZERO, glam::DQuat::IDENTITY, &mut rng)).expect("the wall registers") {
                weakest = Some(found);
            }
        }
    }
    weakest.expect("at least one cloud was corrected")
}

#[test]
fn voxel_planes_report_a_lone_wall_as_leaving_one_direction_free() {
    let weakest = facing_a_lone_wall(voxel_planes());
    println!("voxel planes: facing a lone wall, sigma along it {:.4} m, across it {:.4} m", weakest.sigma_m, weakest.firmest_sigma_m);
    assert!(weakest.direction_world.x.abs() < 0.05, "the free direction lies along the wall: {weakest:?}");
    assert!(weakest.sigma_m > 0.02 && weakest.sigma_m > 20.0 * weakest.firmest_sigma_m, "the wall alone does not place the body along itself: {weakest:?}");
}

#[test]
fn voxel_planes_admit_far_more_doubt_along_a_lone_wall_than_neighbour_planes() {
    let (voxels, neighbours) = (facing_a_lone_wall(voxel_planes()), facing_a_lone_wall(grid_neighbours()));
    println!("facing a lone wall, sigma along it: voxel planes {:.4} m, grid neighbours {:.4} m, tree neighbours {:.4} m", voxels.sigma_m, neighbours.sigma_m, facing_a_lone_wall(tree_neighbours()).sigma_m);
    assert!(voxels.sigma_m > 5.0 * neighbours.sigma_m, "voxel planes {voxels:?}, neighbour planes {neighbours:?}");
}
