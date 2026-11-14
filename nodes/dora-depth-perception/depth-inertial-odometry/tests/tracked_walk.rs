use std::num::NonZeroUsize;
use std::time::Duration;

use depth_inertial_odometry::simulation::{seeded, ImuErrors, ImuStreams, PinholeDepthCamera, Scene, Walk};
use depth_inertial_odometry::thinning::SensorMount;
use depth_inertial_odometry::tracker::{DepthFrameId, DepthFrameOutcome, FramePose, TrackerPhase};
use depth_inertial_odometry::{PlaneVoxelMap, PlaneVoxelSettings, PoseTracker, TrackerSettings};

const FRAME_PERIOD: Duration = Duration::from_micros(32_500);
const STILL: Duration = Duration::from_secs(2);
const WALK_FOR: Duration = Duration::from_secs(30);

/// How far voxel planes may stray from the truth over the drive.
const POSITION_M: f64 = 0.10;
const ANGLE_DEG: f64 = 1.0;
const GYRO_BIAS_RAD_S: f64 = 1e-3;
const ACCEL_BIAS_M_S2: f64 = 0.03;

#[test]
fn a_tracker_fed_raw_depth_images_and_separate_imu_streams_follows_a_drive_around_the_room() {
    let (scene, walk, errors, streams) = (Scene::living_room(), Walk::on_a_robot(STILL), ImuErrors::consumer(), ImuStreams::d455());
    let camera = PinholeDepthCamera::d455(212, 120);
    // The depth sensor looks along the body's x from a few centimetres off the IMU.
    let mount = SensorMount::from_depth_to_imu(PinholeDepthCamera::FACING_FORWARD, [0.03, 0.01, -0.02]).expect("a rotation");
    // The start window ends while the body still rests.
    let settings = TrackerSettings { rest_for: Duration::from_millis(1900), depth_stride: NonZeroUsize::MIN, ..TrackerSettings::d455() };
    let mut tracker = PoseTracker::new(camera.intrinsics, mount, PlaneVoxelMap::new(PlaneVoxelSettings::indoors()), settings).expect("usable intrinsics");
    let mut rng = seeded(7);

    let mut poses: Vec<FramePose> = Vec::new();
    let (mut worst_position_m, mut worst_angle_deg) = (0.0f64, 0.0f64);
    let (mut depth_mm, mut imu_through) = (Vec::new(), None);
    for frame in 0..((STILL + WALK_FOR).as_micros() / FRAME_PERIOD.as_micros()) as u32 {
        let time = FRAME_PERIOD * frame;
        let (accel, gyro) = streams.batch(&walk, &errors, imu_through, time, &mut rng);
        imu_through = Some(time);
        let phase = tracker.add_imu(&accel, &gyro).expect("the walk starts at rest and its samples are in order");
        assert!(phase == TrackerPhase::Tracking || time < STILL, "odometry had not started when the body moved off at {time:?}");

        let seconds = time.as_secs_f64();
        camera.image(&scene, walk.position(seconds), walk.rotation(seconds), &mount, &mut rng, &mut depth_mm);
        let DepthFrameOutcome::Posed(pose) = tracker.add_depth_frame(time, DepthFrameId(frame as u64), &depth_mm).expect("every frame is taken") else {
            assert!(phase != TrackerPhase::Tracking, "a frame after the start got no pose at {time:?}");
            continue;
        };
        let (true_position, true_rotation) = walk.pose_from_start(seconds);
        worst_position_m = worst_position_m.max((pose.position_m - true_position).length());
        worst_angle_deg = worst_angle_deg.max(pose.orientation.angle_between(true_rotation).to_degrees());
        assert_eq!((pose.time, pose.frame_id), (time, DepthFrameId(frame as u64)));
        poses.push(pose);
    }

    let state = tracker.odometry().expect("odometry started").state();
    let (gyro_bias_error, accel_bias_error) = ((state.gyro_bias - errors.gyro_bias_rad_s).length(), (state.accel_bias - errors.accel_bias_m_s2).length());
    let corrected = poses.iter().filter(|pose| pose.corrected()).count();
    println!("tracker: worst {worst_position_m:.3} m, {worst_angle_deg:.2} deg; bias errors {gyro_bias_error:.4} rad/s, {accel_bias_error:.3} m/s2; {corrected} of {} frames corrected", poses.len());
    assert!(worst_position_m < POSITION_M, "position strayed {worst_position_m:.3} m");
    assert!(worst_angle_deg < ANGLE_DEG, "orientation strayed {worst_angle_deg:.2} deg");
    assert!(gyro_bias_error < GYRO_BIAS_RAD_S, "gyro bias off by {gyro_bias_error:.4} rad/s");
    assert!(accel_bias_error < ACCEL_BIAS_M_S2, "accelerometer bias off by {accel_bias_error:.3} m/s2");

    assert_eq!(corrected, poses.len() - 1, "every frame after the one that started the map is corrected");
    assert!(poses.windows(2).all(|pair| pair[0].time < pair[1].time), "pose times strictly increase");
    for pose in &poses[1..] {
        let (sigma, weakest) = (pose.uncertainty, pose.weakest().expect("a corrected frame has a weakest direction"));
        assert!(sigma.position_sigma_m.min_element() > 0.0 && sigma.position_sigma_m.max_element() < 0.05, "{pose:?}");
        assert!(sigma.tilt_sigma_rad > 0.0 && sigma.tilt_sigma_rad < 0.02 && sigma.heading_sigma_rad > 0.0 && sigma.heading_sigma_rad < 0.02, "{pose:?}");
        assert!((weakest.direction.length() - 1.0).abs() < 1e-9 && weakest.sigma_m > 0.0, "{pose:?}");
        assert!(pose.matched_points() >= 50, "{pose:?}");
    }
    // The drive ends with the body moving, and the velocity is the level frame's.
    let last = poses.last().expect("poses");
    let true_velocity = (walk.pose_from_start(last.time.as_secs_f64() + 1e-3).0 - walk.pose_from_start(last.time.as_secs_f64() - 1e-3).0) / 2e-3;
    assert!(true_velocity.length() > 0.2 && (last.velocity_m_s - true_velocity).length() < 0.05, "velocity {} against {true_velocity}", last.velocity_m_s);
}
