use std::num::NonZeroUsize;
use std::time::Duration;

use depth_inertial_odometry::imu_pairing::ImuStream;
use depth_inertial_odometry::simulation::{seeded, ImuErrors, ImuStreams, PinholeDepthCamera, Scene, Walk};
use depth_inertial_odometry::tracker::{Awaited, DepthFrameId, TrackerError};
use depth_inertial_odometry::TrackerSettings;
use dora_depth_inertial_odometry::{CalibrationChange, CalibrationOutcome, DepthCalibration, DepthFrame, DepthOutcome, FrameOutcome, ImuOutcome, NodeSettings, OdometryNode, OdometryNodeError, StartProgress};
use glam::Vec3;
use perception_wire::camera::{decode_camera_calibration, encode_camera_calibration, CameraCalibration, RigidTransform, StreamIntrinsics};
use perception_wire::localization::CameraNanos;
use perception_wire::odometry::{decode_odom_pose, encode_odom_pose, OdomPose};
use perception_wire::{decode_imu, encode_imu, Imu};
use rand::rngs::SmallRng;

const FRAME_PERIOD: Duration = Duration::from_micros(32_500);
/// The camera's clock does not start at zero.
const CLOCK_START: Duration = Duration::from_secs(4_000);
const REST_FOR: Duration = Duration::from_millis(500);
const IDENTITY: RigidTransform = RigidTransform { rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], translation_m: [0.0; 3] };

fn camera_ns(simulated: Duration) -> i64 {
    (CLOCK_START + simulated).as_nanos() as i64
}

/// A simulated camera in the simulated room, giving the messages the camera node sends, and the node fed from it.
struct Rig {
    node: OdometryNode,
    walk: Walk,
    camera: PinholeDepthCamera,
    calibration: CameraCalibration,
    rng: SmallRng,
    imu_through: Option<Duration>,
    depth_mm: Vec<u16>,
}

impl Rig {
    /// A camera that rests for `still` and then drives around the room.
    fn new(still: Duration, rest_for: Duration) -> Rig {
        let camera = PinholeDepthCamera::d455(212, 120);
        let k = camera.intrinsics;
        let depth = StreamIntrinsics { width: k.width as u16, height: k.height as u16, fx: k.fx, fy: k.fy, ppx: k.ppx, ppy: k.ppy };
        let color = StreamIntrinsics { width: 1280, height: 720, fx: 640.0, fy: 640.0, ppx: 640.0, ppy: 360.0 };
        // The depth sensor looks along the body's x from a few centimetres off the IMU.
        let depth_to_imu = RigidTransform { rotation: PinholeDepthCamera::FACING_FORWARD, translation_m: [0.03, 0.01, -0.02] };
        let settings = NodeSettings { tracker: TrackerSettings { rest_for, depth_stride: NonZeroUsize::MIN, ..TrackerSettings::d455() }, ..NodeSettings::d455() };
        Rig {
            node: OdometryNode::new(settings),
            walk: Walk::on_a_robot(still),
            camera,
            calibration: CameraCalibration { color, depth, depth_to_color: IDENTITY, depth_to_imu },
            rng: seeded(7),
            imu_through: None,
            depth_mm: Vec::new(),
        }
    }

    fn standing_still() -> Rig {
        Rig::new(Duration::from_secs(3600), REST_FOR)
    }

    fn time(frame: u64) -> Duration {
        FRAME_PERIOD * frame as u32
    }

    fn send_calibration(&mut self) -> Result<CalibrationOutcome, OdometryNodeError> {
        let calibration = self.calibration;
        self.send_calibration_of(&calibration)
    }

    fn send_calibration_of(&mut self, calibration: &CameraCalibration) -> Result<CalibrationOutcome, OdometryNodeError> {
        self.node.on_calibration(&decode_camera_calibration(&encode_camera_calibration(calibration)).unwrap())
    }

    /// The imu message leading up to a frame: every sample since the message before it.
    fn imu_message(&mut self, frame: u64) -> Imu {
        let through = Rig::time(frame);
        let (accel, gyro) = ImuStreams::d455().batch(&self.walk, &ImuErrors::consumer(), self.imu_through, through, &mut self.rng);
        self.imu_through = Some(through);
        Imu {
            frame_id: frame,
            keyframe_camera_ns: camera_ns(through),
            accel_camera_ns: accel.iter().map(|reading| camera_ns(reading.time)).collect(),
            accel_m_s2: accel.iter().map(|reading| reading.m_s2.as_vec3()).collect(),
            gyro_camera_ns: gyro.iter().map(|reading| camera_ns(reading.time)).collect(),
            gyro_rad_s: gyro.iter().map(|reading| reading.rad_s.as_vec3()).collect(),
        }
    }

    fn send_imu_message(&mut self, imu: &Imu) -> Result<ImuOutcome, OdometryNodeError> {
        self.node.on_imu(&decode_imu(&encode_imu(imu)).unwrap())
    }

    fn send_imu(&mut self, frame: u64) -> Result<ImuOutcome, OdometryNodeError> {
        let imu = self.imu_message(frame);
        self.send_imu_message(&imu)
    }

    fn send_depth(&mut self, frame: u64) -> Result<DepthOutcome, OdometryNodeError> {
        let seconds = Rig::time(frame).as_secs_f64();
        let mount = DepthCalibration::of(&self.calibration).sensor_mount().unwrap();
        self.camera.image(&Scene::living_room(), self.walk.position(seconds), self.walk.rotation(seconds), &mount, &mut self.rng, &mut self.depth_mm);
        let (width, height) = (self.camera.intrinsics.width, self.camera.intrinsics.height);
        self.node.on_depth(DepthFrame { frame_id: DepthFrameId(frame), time: CameraNanos(camera_ns(Rig::time(frame))), width, height, depth_mm: &self.depth_mm })
    }

    /// Sends frames in the camera's order, from `first`, until odometry starts, and returns the frame it started
    /// with.
    fn rest_until_started(&mut self, first: u64) -> u64 {
        for frame in first.. {
            let ImuOutcome::Fed { start, held_frame: None } = self.send_imu(frame).unwrap() else { panic!("frame {frame}'s imu message was not fed") };
            match start {
                StartProgress::Started => return frame,
                StartProgress::Resting { .. } => assert_eq!(self.send_depth(frame).unwrap(), DepthOutcome::Tracked(FrameOutcome::BeforeStart)),
                other => panic!("{other:?} at frame {frame}"),
            }
        }
        unreachable!()
    }

    /// Sends a frame's imu message and then its depth, and returns the pose that comes out.
    fn posed(&mut self, frame: u64) -> OdomPose {
        assert_eq!(self.send_imu(frame).unwrap(), ImuOutcome::Fed { start: StartProgress::Tracking, held_frame: None });
        self.depth_posed(frame)
    }

    /// Sends a frame's depth, its imu message having gone, and returns the pose that comes out.
    fn depth_posed(&mut self, frame: u64) -> OdomPose {
        match self.send_depth(frame).unwrap() {
            DepthOutcome::Tracked(FrameOutcome::Posed(pose)) => decode_odom_pose(&encode_odom_pose(&pose)).unwrap(),
            other => panic!("no pose for frame {frame}: {other:?}"),
        }
    }
}

#[test]
fn a_camera_that_rests_and_then_drives_gives_an_odom_pose_for_every_frame_from_the_start_of_odometry() {
    let (still, drive_for) = (Duration::from_secs(2), Duration::from_secs(8));
    let mut rig = Rig::new(still, Duration::from_millis(1900));
    assert!(matches!(rig.send_calibration(), Ok(CalibrationOutcome::TrackerBuilt(_))));
    let started = rig.rest_until_started(0);
    assert!(Rig::time(started) < still, "odometry had not started when the camera moved off");

    let last = ((still + drive_for).as_micros() / FRAME_PERIOD.as_micros()) as u64;
    // The frame whose imu message started odometry is the first with a pose.
    let mut poses = vec![rig.depth_posed(started)];
    poses.extend((started + 1..=last).map(|frame| rig.posed(frame)));
    let (mut worst_position_m, mut worst_angle_deg) = (0.0f32, 0.0f32);
    for (frame, pose) in (started..=last).zip(&poses) {
        assert_eq!((pose.camera_ns, pose.depth_frame_id), (camera_ns(Rig::time(frame)), frame));
        let (true_position, true_rotation) = rig.walk.pose_from_start(Rig::time(frame).as_secs_f64());
        worst_position_m = worst_position_m.max((pose.position_m - true_position.as_vec3()).length());
        worst_angle_deg = worst_angle_deg.max(pose.orientation.angle_between(true_rotation.as_quat()).to_degrees());
    }
    assert!(poses.windows(2).all(|pair| pair[0].camera_ns < pair[1].camera_ns), "camera_ns strictly increases");
    assert!(worst_position_m < 0.10 && worst_angle_deg < 1.0, "strayed {worst_position_m:.3} m, {worst_angle_deg:.2} deg");

    // The first cloud becomes the map; every later one corrects the pose.
    assert_eq!((poses[0].corrected, poses[0].matched_points, poses[0].weakest), (false, 0, None));
    for pose in &poses[1..] {
        let weakest = pose.weakest.expect("a corrected pose has a weakest direction");
        assert!(pose.corrected && pose.matched_points >= 50 && (weakest.direction.length() - 1.0).abs() < 1e-5 && weakest.sigma_m > 0.0, "{pose:?}");
        assert!(pose.position_sigma_m.min_element() > 0.0 && pose.position_sigma_m.max_element() < 0.05, "{pose:?}");
        assert!(pose.tilt_sigma_rad > 0.0 && pose.tilt_sigma_rad < 0.02 && pose.heading_sigma_rad > 0.0 && pose.heading_sigma_rad < 0.02, "{pose:?}");
    }
    // The drive ends with the camera moving.
    let end = Rig::time(last).as_secs_f64();
    let true_velocity = ((rig.walk.pose_from_start(end + 1e-3).0 - rig.walk.pose_from_start(end - 1e-3).0) / 2e-3).as_vec3();
    let velocity = poses.last().expect("poses").velocity_m_s;
    assert!(true_velocity.length() > 0.2 && (velocity - true_velocity).length() < 0.05, "velocity {velocity} against {true_velocity}");
}

#[test]
fn imu_messages_and_depth_frames_before_the_first_calibration_are_dropped() {
    let mut rig = Rig::standing_still();
    for frame in 0..20 {
        assert_eq!(rig.send_imu(frame).unwrap(), ImuOutcome::BeforeCalibration);
        assert_eq!(rig.send_depth(frame).unwrap(), DepthOutcome::BeforeCalibration);
    }
    assert!(matches!(rig.send_calibration(), Ok(CalibrationOutcome::TrackerBuilt(_))));
    // None of the dropped samples count towards the rest: it runs its whole time from the calibration on.
    let started = rig.rest_until_started(20);
    let rested = Rig::time(started) - Rig::time(20);
    assert!(rested > REST_FOR - FRAME_PERIOD && rested < REST_FOR + FRAME_PERIOD, "started after {rested:?}");
    assert_eq!(rig.posed(started + 1).depth_frame_id, started + 1);
}

#[test]
fn a_calibration_equal_to_the_first_is_ignored_and_one_that_differs_is_refused() {
    let mut rig = Rig::standing_still();
    let first = rig.calibration;
    assert_eq!(rig.send_calibration().unwrap(), CalibrationOutcome::TrackerBuilt(DepthCalibration::of(&first)));
    let started = rig.rest_until_started(0);
    assert_eq!(rig.send_calibration().unwrap(), CalibrationOutcome::Unchanged);
    // Odometry does not use the colour stream.
    let recoloured = CameraCalibration { color: StreamIntrinsics { fx: 700.0, ..first.color }, depth_to_color: RigidTransform { translation_m: [0.059, 0.0, 0.0], ..IDENTITY }, ..first };
    assert_eq!(rig.send_calibration_of(&recoloured).unwrap(), CalibrationOutcome::Unchanged);

    let remounted = CameraCalibration { depth_to_imu: RigidTransform { translation_m: [0.03, 0.02, -0.02], ..first.depth_to_imu }, ..first };
    let refocused = CameraCalibration { depth: StreamIntrinsics { fx: first.depth.fx + 1.0, ..first.depth }, ..first };
    for changed in [remounted, refocused] {
        let refusal = rig.send_calibration_of(&changed).expect_err("a changed calibration");
        let expected = CalibrationChange { started: DepthCalibration::of(&first), now: DepthCalibration::of(&changed) };
        assert!(matches!(&refusal, OdometryNodeError::CalibrationChanged(change) if **change == expected), "{refusal}");
    }
    // The tracker built from the first calibration is as it was.
    assert!(rig.posed(started + 1).camera_ns > 0);
}

#[test]
fn a_calibration_that_cannot_be_tracked_from_is_refused() {
    let mut rig = Rig::standing_still();
    let sheared = CameraCalibration { depth_to_imu: RigidTransform { rotation: [1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0], ..IDENTITY }, ..rig.calibration };
    assert!(matches!(rig.send_calibration_of(&sheared), Err(OdometryNodeError::Mount(_))));
    let unfocused = CameraCalibration { depth: StreamIntrinsics { fx: 0.0, ..rig.calibration.depth }, ..rig.calibration };
    assert!(matches!(rig.send_calibration_of(&unfocused), Err(OdometryNodeError::DepthStream(_))));
    // Neither became the calibration.
    assert!(matches!(rig.send_calibration(), Ok(CalibrationOutcome::TrackerBuilt(_))));
}

#[test]
fn a_camera_that_moves_while_it_should_rest_gets_no_pose_until_it_has_rested() {
    let mut rig = Rig::standing_still();
    rig.send_calibration().unwrap();
    let (mut refusals, mut frame) = (0, 0);
    let started = loop {
        let mut imu = rig.imu_message(frame);
        for (camera_ns, reading) in imu.gyro_camera_ns.iter().zip(&mut imu.gyro_rad_s) {
            let seconds = (Duration::from_nanos(*camera_ns as u64) - CLOCK_START).as_secs_f32();
            if seconds < 0.3 {
                reading.z += 0.5 * (40.0 * seconds).sin();
            }
        }
        let ImuOutcome::Fed { start, held_frame: None } = rig.send_imu_message(&imu).unwrap() else { panic!("frame {frame}'s imu message was not fed") };
        match start {
            StartProgress::Started => break frame,
            StartProgress::NotAtRest(_) => refusals += 1,
            StartProgress::Resting { .. } => {}
            StartProgress::Tracking => panic!("tracking before the start"),
        }
        assert_eq!(rig.send_depth(frame).unwrap(), DepthOutcome::Tracked(FrameOutcome::BeforeStart));
        frame += 1;
    };
    assert_eq!(refusals, 1, "the rest the camera moved through is refused once");
    assert!(Rig::time(started) > 2 * REST_FOR - FRAME_PERIOD, "started at {:?}", Rig::time(started));
    assert!(rig.posed(started + 1).position_m.length() < 0.01);
}

#[test]
fn a_depth_frame_that_arrives_before_its_imu_message_is_posed_when_the_message_arrives() {
    let mut rig = Rig::standing_still();
    rig.send_calibration().unwrap();
    let started = rig.rest_until_started(0);
    let before = rig.posed(started + 1);

    assert_eq!(rig.send_depth(started + 2).unwrap(), DepthOutcome::HeldForImu { replaced: None });
    let ImuOutcome::Fed { start: StartProgress::Tracking, held_frame: Some(FrameOutcome::Posed(late)) } = rig.send_imu(started + 2).unwrap() else { panic!("the held frame got no pose") };
    assert_eq!((late.depth_frame_id, late.camera_ns, late.corrected), (started + 2, camera_ns(Rig::time(started + 2)), true));
    assert!(late.camera_ns > before.camera_ns);

    // A newer frame takes the place of one still held; the imu messages for both then pose the newer one.
    assert_eq!(rig.send_depth(started + 3).unwrap(), DepthOutcome::HeldForImu { replaced: None });
    assert_eq!(rig.send_depth(started + 4).unwrap(), DepthOutcome::HeldForImu { replaced: Some(DepthFrameId(started + 3)) });
    assert_eq!(rig.send_imu(started + 3).unwrap(), ImuOutcome::Fed { start: StartProgress::Tracking, held_frame: None });
    let ImuOutcome::Fed { held_frame: Some(FrameOutcome::Posed(newer)), .. } = rig.send_imu(started + 4).unwrap() else { panic!("the held frame got no pose") };
    assert_eq!((newer.depth_frame_id, newer.corrected), (started + 4, true));
    assert!(rig.posed(started + 5).camera_ns > newer.camera_ns);
}

#[test]
fn a_depth_frame_of_another_size_than_the_calibration_is_refused() {
    let mut rig = Rig::standing_still();
    rig.send_calibration().unwrap();
    // As many readings as the calibrated image, in another shape.
    let frame = DepthFrame { frame_id: DepthFrameId(0), time: CameraNanos(camera_ns(Duration::ZERO)), width: 120, height: 212, depth_mm: &vec![0; 212 * 120] };
    let refusal = rig.node.on_depth(frame).expect_err("a turned image");
    assert!(matches!(refusal, OdometryNodeError::DepthSize { width: 120, height: 212, calibrated_width: 212, calibrated_height: 120, .. }), "{refusal}");
}

#[test]
fn camera_time_that_goes_back_is_refused() {
    let mut rig = Rig::standing_still();
    rig.send_calibration().unwrap();
    let started = rig.rest_until_started(0);
    rig.posed(started + 1);
    rig.posed(started + 2);
    let stale = rig.send_depth(started + 1).expect_err("a frame older than the last pose");
    assert!(matches!(stale, OdometryNodeError::Tracker(TrackerError::FrameOutOfOrder { .. })), "{stale}");

    let mut again = rig.imu_message(started + 3);
    again.gyro_camera_ns[0] = camera_ns(Rig::time(started + 2));
    let unordered = rig.send_imu_message(&again).expect_err("a gyro sample older than the one before it");
    assert!(matches!(unordered, OdometryNodeError::Tracker(TrackerError::Imu(order)) if order.stream == ImuStream::Gyro), "{unordered}");

    let repeated = Imu { frame_id: started + 2, ..rig.imu_message(started + 4) };
    let refusal = rig.send_imu_message(&repeated).expect_err("an imu message for a frame already led up to");
    assert!(matches!(refusal, OdometryNodeError::ImuMessageOutOfOrder { frame_id, previous } if (frame_id.0, previous.0) == (started + 2, started + 2)), "{refusal}");

    let before_zero = Imu { accel_camera_ns: vec![-1], accel_m_s2: vec![Vec3::ZERO], ..rig.imu_message(started + 5) };
    assert!(matches!(rig.send_imu_message(&before_zero), Err(OdometryNodeError::NegativeCameraTime(CameraNanos(-1)))));
}

#[test]
fn a_depth_stream_that_goes_silent_while_tracking_is_refused_once_the_imu_has_run_on_past_the_bound() {
    let mut rig = Rig::standing_still();
    rig.send_calibration().unwrap();
    let started = rig.rest_until_started(0);
    rig.posed(started + 1);
    let max_backlog = TrackerSettings::d455().max_backlog;
    // Imu messages only from here on.
    let (refused_at, refusal) = (started + 2..).find_map(|frame| rig.send_imu(frame).err().map(|refusal| (frame, refusal))).expect("a refusal");
    assert!(matches!(refusal, OdometryNodeError::Tracker(TrackerError::Backlog { waiting_for: Awaited::DepthFrame, span }) if span > max_backlog), "{refusal}");
    let waited = Rig::time(refused_at) - Rig::time(started + 1);
    assert!(waited > max_backlog && waited < max_backlog + 2 * FRAME_PERIOD, "refused after {waited:?}");
}
