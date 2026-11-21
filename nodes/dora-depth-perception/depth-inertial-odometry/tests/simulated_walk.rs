use std::time::Duration;

use depth_inertial_odometry::odometry::{CloudOutcome, OdometryError, StaticInitialization};
use depth_inertial_odometry::simulation::{seeded, DepthCamera, ImuErrors, Scene, Walk};
use depth_inertial_odometry::{DepthInertialOdometry, KdTreeMap, NeighbourPlaneSettings, NeighbourPlanes, OdometrySettings, PlaneVoxelMap, PlaneVoxelSettings, SurfaceMap, VoxelHashMap};
use glam::DVec3;

const IMU_PERIOD: Duration = Duration::from_micros(2500);
const IMU_PER_CLOUD: u32 = 13;
const STATIONARY: Duration = Duration::from_secs(2);

struct Flown {
    worst_position_m: f64,
    worst_angle_deg: f64,
    gyro_bias_error_rad_s: f64,
    accel_bias_error_m_s2: f64,
    corrected: usize,
}

/// Runs odometry over a simulated walk and compares it with the truth at every cloud.
fn fly<M: SurfaceMap>(map: M, scene: &Scene, walk_for: Duration) -> Flown {
    let walk = Walk::on_a_robot(STATIONARY);
    let camera = DepthCamera::d455_quarter();
    let errors = ImuErrors::consumer();
    let mut rng = seeded(7);

    let stationary_samples = (STATIONARY.as_micros() / IMU_PERIOD.as_micros()) as u32;
    let stationary: Vec<_> = (0..stationary_samples).map(|i| walk.imu_sample(IMU_PERIOD * i, &errors, &mut rng)).collect();
    let start = StaticInitialization::from_samples(&stationary, OdometrySettings::d455().gravity_m_s2).expect("the walk starts at rest");
    let mut odometry = DepthInertialOdometry::new(start, map, OdometrySettings::d455());

    let mut flown = Flown { worst_position_m: 0.0, worst_angle_deg: 0.0, gyro_bias_error_rad_s: 0.0, accel_bias_error_m_s2: 0.0, corrected: 0 };
    let last = ((STATIONARY + walk_for).as_micros() / IMU_PERIOD.as_micros()) as u32;
    for i in stationary_samples..last {
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
fn facing_a_lone_wall<M: SurfaceMap>(map: M) -> depth_inertial_odometry::filter::DegenerateTranslation {
    let (scene, camera, errors) = (Scene::lone_wall(), DepthCamera::d455_quarter(), ImuErrors::consumer());
    let walk = Walk::carried(Duration::from_secs(60));
    let mut rng = seeded(3);
    let stationary: Vec<_> = (0..400).map(|i| walk.imu_sample(IMU_PERIOD * i, &errors, &mut rng)).collect();
    let mut odometry = DepthInertialOdometry::new(StaticInitialization::from_samples(&stationary, OdometrySettings::d455().gravity_m_s2).expect("at rest"), map, OdometrySettings::d455());
    let mut degenerate = None;
    for i in 400..1200u32 {
        let time = IMU_PERIOD * i;
        odometry.add_imu(walk.imu_sample(time, &errors, &mut rng)).expect("in order");
        if i % IMU_PER_CLOUD == 0 {
            if let CloudOutcome::Corrected { degenerate: found, .. } = odometry.add_cloud(time, &camera.cloud(&scene, DVec3::ZERO, glam::DQuat::IDENTITY, &mut rng)).expect("the wall registers") {
                degenerate = Some(found);
            }
        }
    }
    degenerate.expect("at least one cloud was corrected")
}

#[test]
fn voxel_planes_report_a_lone_wall_as_leaving_one_direction_free() {
    let degenerate = facing_a_lone_wall(voxel_planes());
    println!("voxel planes: facing a lone wall, sigma along it {:.4} m, across it {:.4} m", degenerate.sigma_m, degenerate.firmest_sigma_m);
    assert!(degenerate.direction_world.x.abs() < 0.05, "the free direction lies along the wall: {degenerate:?}");
    assert!(degenerate.sigma_m > 0.02 && degenerate.sigma_m > 20.0 * degenerate.firmest_sigma_m, "the wall alone does not place the body along itself: {degenerate:?}");
}

#[test]
fn voxel_planes_admit_far_more_doubt_along_a_lone_wall_than_neighbour_planes() {
    let (voxels, neighbours) = (facing_a_lone_wall(voxel_planes()), facing_a_lone_wall(grid_neighbours()));
    println!("facing a lone wall, sigma along it: voxel planes {:.4} m, grid neighbours {:.4} m, tree neighbours {:.4} m", voxels.sigma_m, neighbours.sigma_m, facing_a_lone_wall(tree_neighbours()).sigma_m);
    assert!(voxels.sigma_m > 5.0 * neighbours.sigma_m, "voxel planes {voxels:?}, neighbour planes {neighbours:?}");
}

/// Drives the room with every cloud from `blind_from` for `blind_for` empty, as when fast motion leaves
/// the depth camera nothing to match, so that the IMU alone carries the estimate (dead reckoning). Returns
/// how many clouds after the blind stretch corrected the estimate and how many there were, the dead
/// reckoning time at the end of the stretch, and the pose error at the end of the drive.
struct BlindDrive {
    corrected_after: usize,
    clouds_after: usize,
    /// Of the clouds in the last two seconds, those that corrected the estimate, and all of them.
    corrected_at_the_end: (usize, usize),
    longest_dead_reckoning: Duration,
    final_position_error_m: f64,
    final_angle_error_deg: f64,
}

fn drive_blind(blind_from: Duration, blind_for: Duration, drive_for: Duration) -> BlindDrive {
    let (walk, scene, camera, errors) = (Walk::on_a_robot(STATIONARY), Scene::living_room(), DepthCamera::d455_quarter(), ImuErrors::consumer());
    let settings = OdometrySettings::d455();
    let mut rng = seeded(7);
    let stationary_samples = (STATIONARY.as_micros() / IMU_PERIOD.as_micros()) as u32;
    let stationary: Vec<_> = (0..stationary_samples).map(|i| walk.imu_sample(IMU_PERIOD * i, &errors, &mut rng)).collect();
    let mut odometry = DepthInertialOdometry::new(StaticInitialization::from_samples(&stationary, settings.gravity_m_s2).expect("the walk starts at rest"), voxel_planes(), settings);
    let (sees_again, ends) = (blind_from + blind_for, STATIONARY + drive_for);
    let mut drive = BlindDrive { corrected_after: 0, clouds_after: 0, corrected_at_the_end: (0, 0), longest_dead_reckoning: Duration::ZERO, final_position_error_m: 0.0, final_angle_error_deg: 0.0 };
    for i in stationary_samples..(ends.as_micros() / IMU_PERIOD.as_micros()) as u32 {
        let time = IMU_PERIOD * i;
        odometry.add_imu(walk.imu_sample(time, &errors, &mut rng)).expect("samples are in order");
        if i % IMU_PER_CLOUD != 0 {
            continue;
        }
        let seconds = time.as_secs_f64();
        let blind = time >= blind_from && time < sees_again;
        let cloud = if blind { Vec::new() } else { camera.cloud(&scene, walk.position(seconds), walk.rotation(seconds), &mut rng) };
        let corrected = match odometry.add_cloud(time, &cloud) {
            Ok(outcome) => matches!(outcome, CloudOutcome::Corrected { .. }),
            Err(OdometryError::NotCorrected(_)) => false,
            Err(other) => panic!("{other}"),
        };
        assert!(!(blind && corrected), "an empty cloud corrected the estimate at {time:?}");
        drive.longest_dead_reckoning = drive.longest_dead_reckoning.max(odometry.dead_reckoning_for());
        if time >= sees_again {
            drive.clouds_after += 1;
            drive.corrected_after += usize::from(corrected);
        }
        if time + Duration::from_secs(2) >= ends {
            drive.corrected_at_the_end.0 += usize::from(corrected);
            drive.corrected_at_the_end.1 += 1;
        }
        let (true_position, true_rotation) = walk.pose_from_start(seconds);
        let (position, rotation) = odometry.state().levelled_pose();
        drive.final_position_error_m = (position - true_position).length();
        drive.final_angle_error_deg = rotation.angle_between(true_rotation).to_degrees();
    }
    drive
}

/// The blind stretch ends with the prediction several centimetres off in every direction, beyond what
/// the clouds' own noise explains: the gate must widen with the prediction's covariance to take the
/// clouds again.
#[test]
fn voxel_planes_regain_correction_after_dead_reckoning_on_the_move() {
    let blind_for = Duration::from_millis(2500);
    let drive = drive_blind(Duration::from_secs(4), blind_for, Duration::from_secs(12));
    println!(
        "blind for {blind_for:?} while driving: {} of {} clouds after it corrected, {} of the last {}; dead reckoning up to {:?}; final error {:.3} m, {:.2} deg",
        drive.corrected_after, drive.clouds_after, drive.corrected_at_the_end.0, drive.corrected_at_the_end.1, drive.longest_dead_reckoning, drive.final_position_error_m, drive.final_angle_error_deg
    );
    assert!(drive.longest_dead_reckoning >= blind_for - IMU_PERIOD * IMU_PER_CLOUD, "the IMU alone carried the estimate through the blind stretch: {:?}", drive.longest_dead_reckoning);
    assert!(drive.corrected_after * 10 >= drive.clouds_after * 9, "correction is regained: {} of {} clouds after the blind stretch", drive.corrected_after, drive.clouds_after);
    assert_eq!(drive.corrected_at_the_end.0, drive.corrected_at_the_end.1, "every cloud of the last two seconds corrects");
    assert!(drive.final_position_error_m < VOXEL_PLANES.position_m, "position strayed {:.3} m", drive.final_position_error_m);
    assert!(drive.final_angle_error_deg < VOXEL_PLANES.angle_deg, "orientation strayed {:.2} deg", drive.final_angle_error_deg);
}

/// What a still camera's odometry says of how well it knows its map and its pose.
#[derive(Debug)]
struct RestingConfidence {
    /// The smallest sigma of any surface the cloud's points land on.
    least_surface_sigma_m: f32,
    /// The level position sigma along the axis the pose is known worst on.
    position_sigma_m: f64,
}

/// Stands still in the room and reports the odometry's confidence after each of `checkpoints`, which
/// count from the start and increase.
fn resting_in_the_room(checkpoints: &[Duration]) -> Vec<RestingConfidence> {
    let (walk, scene, camera, errors) = (Walk::on_a_robot(Duration::from_secs(3600)), Scene::living_room(), DepthCamera::d455_quarter(), ImuErrors::consumer());
    let settings = OdometrySettings::d455();
    let mut rng = seeded(5);
    let stationary_samples = (STATIONARY.as_micros() / IMU_PERIOD.as_micros()) as u32;
    let stationary: Vec<_> = (0..stationary_samples).map(|i| walk.imu_sample(IMU_PERIOD * i, &errors, &mut rng)).collect();
    let mut odometry = DepthInertialOdometry::new(StaticInitialization::from_samples(&stationary, settings.gravity_m_s2).expect("at rest"), voxel_planes(), settings);
    let mut found = Vec::new();
    let last = checkpoints.last().expect("a checkpoint");
    for i in stationary_samples..((STATIONARY + *last).as_micros() / IMU_PERIOD.as_micros()) as u32 + IMU_PER_CLOUD {
        let time = IMU_PERIOD * i;
        odometry.add_imu(walk.imu_sample(time, &errors, &mut rng)).expect("in order");
        if i % IMU_PER_CLOUD != 0 {
            continue;
        }
        let seconds = time.as_secs_f64();
        let cloud = camera.cloud(&scene, walk.position(seconds), walk.rotation(seconds), &mut rng);
        odometry.add_cloud(time, &cloud).expect("a still camera's cloud registers");
        if found.len() < checkpoints.len() && time >= STATIONARY + checkpoints[found.len()] {
            let state = odometry.state();
            let least_surface_sigma_m = cloud
                .iter()
                .filter_map(|point| odometry.map().surface_near(state.to_world(point.as_dvec3()).as_vec3()))
                .map(|surface| surface.sigma_m)
                .fold(f32::INFINITY, f32::min);
            found.push(RestingConfidence { least_surface_sigma_m, position_sigma_m: odometry.filter().levelled_uncertainty().position_sigma_m.max_element() });
        }
    }
    found
}

/// How a still camera's position strayed and what the odometry said of it, over one stretch of a long rest.
#[derive(Debug)]
struct RestingStretch {
    /// The largest error of the levelled position along any level axis over the stretch.
    worst_error_m: f64,
    /// The largest error along any level axis in units of the sigma reported along that axis, over the stretch.
    worst_error_sigmas: f64,
    /// The same in units of the sigma against the map alone, which leaves the depth scale out.
    worst_error_map_sigmas: f64,
}

/// Stands still in the room for `stretches` stretches of `stretch_for` each, with the camera's depth scale coming
/// to `1 + depth_scale_change` as a stereo camera's does while it warms up (an exponential approach with
/// `warm_up_for` as its time constant), and reports each stretch.
fn resting_while_the_depth_warms_up(depth_scale_change: f64, warm_up_for: Duration, stretch_for: Duration, stretches: u32) -> Vec<RestingStretch> {
    let (walk, scene, errors) = (Walk::on_a_robot(Duration::from_secs(3600)), Scene::living_room(), ImuErrors::consumer());
    let camera = DepthCamera { columns: 106, rows: 60, ..DepthCamera::d455_quarter() };
    let settings = OdometrySettings::d455();
    let mut rng = seeded(13);
    let stationary_samples = (STATIONARY.as_micros() / IMU_PERIOD.as_micros()) as u32;
    let stationary: Vec<_> = (0..stationary_samples).map(|i| walk.imu_sample(IMU_PERIOD * i, &errors, &mut rng)).collect();
    let mut odometry = DepthInertialOdometry::new(StaticInitialization::from_samples(&stationary, settings.gravity_m_s2).expect("at rest"), voxel_planes(), settings);
    let mut found: Vec<RestingStretch> = Vec::new();
    let clouds_per_stretch = (stretch_for.as_micros() / (IMU_PERIOD * IMU_PER_CLOUD).as_micros()) as u32;
    let mut cloud_index = 0u32;
    for i in stationary_samples.. {
        let time = IMU_PERIOD * i;
        odometry.add_imu(walk.imu_sample(time, &errors, &mut rng)).expect("in order");
        if i % IMU_PER_CLOUD != 0 {
            continue;
        }
        let seconds = time.as_secs_f64();
        let warmed = 1.0 - (-(time - STATIONARY).as_secs_f64() / warm_up_for.as_secs_f64()).exp();
        let warming = DepthCamera { depth_scale: 1.0 + depth_scale_change * warmed, ..camera };
        odometry.add_cloud(time, &warming.cloud(&scene, walk.position(seconds), walk.rotation(seconds), &mut rng)).expect("a still camera's cloud registers");
        let stretch = (cloud_index / clouds_per_stretch) as usize;
        if stretch == stretches as usize {
            return found;
        }
        if stretch == found.len() {
            found.push(RestingStretch { worst_error_m: 0.0, worst_error_sigmas: 0.0, worst_error_map_sigmas: 0.0 });
        }
        let error = (odometry.state().levelled_pose().0 - walk.pose_from_start(seconds).0).abs();
        let sigma = odometry.levelled_uncertainty().position_sigma_m;
        let in_sigmas = error / sigma;
        let in_map_sigmas = error / odometry.filter().levelled_uncertainty().position_sigma_m;
        found[stretch].worst_error_map_sigmas = found[stretch].worst_error_map_sigmas.max(in_map_sigmas.max_element());
        found[stretch].worst_error_m = found[stretch].worst_error_m.max(error.max_element());
        found[stretch].worst_error_sigmas = found[stretch].worst_error_sigmas.max(in_sigmas.max_element());
        cloud_index += 1;
    }
    unreachable!("the rest ends after its last stretch")
}

/// A camera at rest whose depth is steady: its position must not creep with time at rest, and the reported sigma
/// must cover the error.
#[test]
fn a_still_camera_with_steady_depth_does_not_creep() {
    let stretches = resting_while_the_depth_warms_up(0.0, Duration::from_secs(40), Duration::from_secs(30), 5);
    println!("at rest for 150 s with steady depth: {stretches:?}");
    for (index, stretch) in stretches.iter().enumerate() {
        assert!(stretch.worst_error_m < 0.006, "stretch {index}: strayed {:.4} m", stretch.worst_error_m);
        assert!(stretch.worst_error_sigmas < 2.5, "stretch {index}: an error of {:.1} sigmas", stretch.worst_error_sigmas);
    }
    let (first, last) = (&stretches[0], &stretches[stretches.len() - 1]);
    assert!(last.worst_error_m < first.worst_error_m + 0.003, "the error grows with time at rest: {first:?} at first, {last:?} at the end");
}

/// A camera at rest whose depth grows by half a percent as it warms up: the position moves with the depth, which
/// the odometry cannot tell from motion, stops moving once the depth stops changing, and stays within the sigma it
/// reports.
#[test]
fn a_still_camera_whose_depth_warms_up_strays_no_further_than_its_sigma_says() {
    let stretches = resting_while_the_depth_warms_up(0.005, Duration::from_secs(30), Duration::from_secs(30), 5);
    println!("at rest for 150 s while the depth scale grows by 0.5%: {stretches:?}");
    for (index, stretch) in stretches.iter().enumerate() {
        assert!(stretch.worst_error_m < 0.025, "stretch {index}: strayed {:.4} m", stretch.worst_error_m);
        assert!(stretch.worst_error_sigmas < 2.5, "stretch {index}: an error of {:.1} sigmas", stretch.worst_error_sigmas);
    }
    let (settled, last) = (&stretches[3], &stretches[4]);
    assert!(last.worst_error_m < settled.worst_error_m + 0.003, "the error grows once the depth has settled: {settled:?}, then {last:?}");
}

#[test]
fn a_still_cameras_map_and_pose_confidence_stop_growing_at_the_floor() {
    let found = resting_in_the_room(&[Duration::from_secs(10), Duration::from_secs(40)]);
    let floor = PlaneVoxelSettings::indoors().plane_sigma_floor_m;
    println!("at rest: {found:?}, plane sigma floor {floor} m");
    let (early, late) = (&found[0], &found[1]);
    assert!(late.least_surface_sigma_m >= floor, "a surface is known to {} m, under the floor of {floor} m", late.least_surface_sigma_m);
    assert!(late.least_surface_sigma_m > 0.95 * early.least_surface_sigma_m, "surface sigma {early:?} after 10 s, {late:?} after 40 s");
    assert!(late.position_sigma_m > 0.95 * early.position_sigma_m, "position sigma {early:?} after 10 s, {late:?} after 40 s");
}
