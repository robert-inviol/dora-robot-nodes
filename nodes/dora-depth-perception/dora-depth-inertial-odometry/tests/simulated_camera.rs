use std::num::NonZeroUsize;
use std::time::Duration;

use depth_inertial_odometry::imu_pairing::ImuStream;
use depth_inertial_odometry::simulation::{seeded, Cuboid, ImuErrors, ImuStreams, PinholeDepthCamera, Scene, Walk};
use depth_inertial_odometry::SurfaceMap;
use depth_inertial_odometry::tracker::{Awaited, DepthFrameId, TrackerError};
use depth_inertial_odometry::TrackerSettings;
use dora_depth_inertial_odometry::{CalibrationChange, CalibrationOutcome, DepthCalibration, DepthFrame, DepthOutcome, FrameOutcome, ImuOutcome, NodeSettings, OdometryNode, OdometryNodeError, StartProgress};
use glam::{DQuat, DVec3, Vec3};
use perception_wire::camera::{decode_camera_calibration, encode_camera_calibration, CameraCalibration, RigidTransform, StreamIntrinsics};
use perception_wire::localization::CameraNanos;
use perception_wire::odometry::{decode_odom_pose, encode_odom_pose, OdomPose};
use perception_wire::people::{PersonPose, PersonPoses, PixelBox};
use perception_wire::{decode_imu, encode_imu, Imu};
use rand::rngs::SmallRng;

const FRAME_PERIOD: Duration = Duration::from_micros(32_500);
/// The camera's clock does not start at zero.
const CLOCK_START: Duration = Duration::from_secs(4_000);
const STATIONARY_FOR: Duration = Duration::from_millis(500);
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
    /// Someone standing or walking in the room.
    person: Option<Cuboid>,
}

impl Rig {
    /// A camera that rests for `still` and then drives around the room.
    fn new(still: Duration, stationary_for: Duration) -> Rig {
        let camera = PinholeDepthCamera::d455(212, 120);
        let k = camera.intrinsics;
        let depth = StreamIntrinsics { width: k.width as u16, height: k.height as u16, fx: k.fx, fy: k.fy, ppx: k.ppx, ppy: k.ppy };
        let color = StreamIntrinsics { width: 1280, height: 720, fx: 640.0, fy: 640.0, ppx: 640.0, ppy: 360.0 };
        // The depth sensor looks along the body's x from a few centimetres off the IMU.
        let depth_to_imu = RigidTransform { rotation: PinholeDepthCamera::FACING_FORWARD, translation_m: [0.03, 0.01, -0.02] };
        let settings = NodeSettings { tracker: TrackerSettings { stationary_for, depth_stride: NonZeroUsize::MIN, ..TrackerSettings::d455() }, ..NodeSettings::d455() };
        Rig {
            node: OdometryNode::new(settings),
            walk: Walk::on_a_robot(still),
            camera,
            calibration: CameraCalibration { color, depth, depth_to_color: IDENTITY, depth_to_imu },
            rng: seeded(7),
            imu_through: None,
            depth_mm: Vec::new(),
            person: None,
        }
    }

    fn standing_still() -> Rig {
        Rig::new(Duration::from_secs(3600), STATIONARY_FOR)
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
        let extrinsics = DepthCalibration::of(&self.calibration).sensor_extrinsics().unwrap();
        let mut scene = Scene::living_room();
        scene.furniture.extend(self.person);
        self.camera.image(&scene, self.walk.position(seconds), self.walk.rotation(seconds), &extrinsics, &mut self.rng, &mut self.depth_mm);
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
                StartProgress::StaticInitialization { .. } => assert_eq!(self.send_depth(frame).unwrap(), DepthOutcome::Tracked(FrameOutcome::BeforeStart)),
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
    let odom_started = poses[0].odom_started_camera_ns;
    assert!(odom_started > camera_ns(Duration::ZERO) && odom_started <= poses[0].camera_ns, "odometry started at rest, by the first pose");
    assert!(poses.iter().all(|pose| pose.odom_started_camera_ns == odom_started), "one run is one odom frame");
    assert!(worst_position_m < 0.10 && worst_angle_deg < 1.0, "strayed {worst_position_m:.3} m, {worst_angle_deg:.2} deg");

    // The first cloud becomes the map; every later one corrects the pose.
    assert_eq!((poses[0].corrected, poses[0].matched_points, poses[0].degenerate), (false, 0, None));
    for pose in &poses[1..] {
        let degenerate = pose.degenerate.expect("a corrected pose has a degenerate direction");
        assert!(pose.corrected && pose.matched_points >= 50 && (degenerate.direction.length() - 1.0).abs() < 1e-5 && degenerate.sigma_m > 0.0, "{pose:?}");
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
    let initialized_after = Rig::time(started) - Rig::time(20);
    assert!(initialized_after > STATIONARY_FOR - FRAME_PERIOD && initialized_after < STATIONARY_FOR + FRAME_PERIOD, "started after {initialized_after:?}");
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
    assert!(matches!(rig.send_calibration_of(&sheared), Err(OdometryNodeError::Extrinsics(_))));
    let unfocused = CameraCalibration { depth: StreamIntrinsics { fx: 0.0, ..rig.calibration.depth }, ..rig.calibration };
    assert!(matches!(rig.send_calibration_of(&unfocused), Err(OdometryNodeError::DepthStream(_))));
    // Neither became the calibration.
    assert!(matches!(rig.send_calibration(), Ok(CalibrationOutcome::TrackerBuilt(_))));
}

#[test]
fn a_camera_that_moves_while_it_should_be_stationary_gets_no_pose_until_static_initialization() {
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
            StartProgress::NotStationary(_) => refusals += 1,
            StartProgress::StaticInitialization { .. } => {}
            StartProgress::Tracking => panic!("tracking before the start"),
        }
        assert_eq!(rig.send_depth(frame).unwrap(), DepthOutcome::Tracked(FrameOutcome::BeforeStart));
        frame += 1;
    };
    assert_eq!(refusals, 1, "the rest the camera moved through is refused once");
    assert!(Rig::time(started) > 2 * STATIONARY_FOR - FRAME_PERIOD, "started at {:?}", Rig::time(started));
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

#[test]
fn a_view_people_fill_is_dead_reckoned_through_and_lost_once_the_position_uncertainty_passes_the_bound() {
    let mut rig = Rig::standing_still();
    rig.send_calibration().unwrap();
    let started = rig.rest_until_started(0);
    rig.depth_posed(started);
    let mut at_rest = rig.posed(started + 1);
    for frame in started + 2..started + 60 {
        at_rest = rig.posed(frame);
        assert!(at_rest.corrected, "frame {frame}");
    }
    // Someone stands right in front of the camera: the whole colour image is one person, and their box is cut out
    // of every depth frame, which leaves nothing to correct with.
    let filling = |frame: u64| PersonPoses {
        frame_id: frame,
        captured_camera_ns: camera_ns(Rig::time(frame)),
        width: 1280,
        height: 720,
        people: vec![PersonPose { track_id: None, score: 0.9, bbox: PixelBox { x_px: 0.0, y_px: 0.0, w_px: 1280.0, h_px: 720.0 }, keypoints: vec![] }],
    };
    let blocked_from = started + 60;
    let mut lost = None;
    for frame in blocked_from..blocked_from + 300 {
        rig.node.on_person_poses(&filling(frame)).unwrap();
        rig.send_imu(frame).unwrap();
        match rig.send_depth(frame).unwrap() {
            DepthOutcome::Tracked(FrameOutcome::Posed(pose)) => {
                assert!(!pose.corrected, "frame {frame} had nothing left to correct with");
                let strayed_m = (pose.position_m - at_rest.position_m).length();
                assert!(strayed_m < 3.0 * pose.position_sigma_m.max_element(), "the dead-reckoned pose strayed {strayed_m} m, beyond what its sigmas allow: {pose:?}");
            }
            DepthOutcome::Tracked(FrameOutcome::Lost { uncorrected_for, largest_position_sigma_m, last_second }) => {
                lost = Some((frame, uncorrected_for, largest_position_sigma_m, last_second));
                break;
            }
            other => panic!("frame {frame}: {other:?}"),
        }
    }
    let (lost_at, uncorrected_for, largest_position_sigma_m, last_second) = lost.expect("odometry is lost with the view blocked");
    let bound = NodeSettings::d455().lost_beyond_position_sigma_m;
    assert!(largest_position_sigma_m > bound && largest_position_sigma_m < 1.1 * bound, "lost at a sigma of {largest_position_sigma_m} m");
    // At rest the filter's position uncertainty takes well over a second to grow to the bound, in camera time.
    assert!(uncorrected_for > Duration::from_millis(1200) && uncorrected_for < Duration::from_secs(3), "{uncorrected_for:?}");
    let refused = last_second.refused.expect("the last second's clouds were refused");
    assert_eq!((refused.frames, refused.within_gate.max, refused.needed, last_second.with_people), (last_second.frames, 0, 50, last_second.frames), "{last_second}");
    assert!(last_second.cut_fraction.is_some_and(|cut| cut.min == 1.0), "{last_second}");

    // Gone again, the camera rests and odometry starts in a new odom frame.
    let restarted = rig.rest_until_started(lost_at + 1);
    let pose = rig.depth_posed(restarted);
    assert!(pose.odom_started_camera_ns > camera_ns(Rig::time(lost_at)), "a new odom frame");
    assert!(rig.posed(restarted + 1).corrected, "tracking again");
}

/// A person-sized box standing on the floor with its middle at `at`.
fn person_at(at: DVec3) -> Cuboid {
    Cuboid { min: DVec3::new(at.x - 0.22, at.y - 0.15, 0.0), max: DVec3::new(at.x + 0.22, at.y + 0.15, 1.75) }
}

/// The pose model's box around a person in the colour image of a frame taken from a body pose: the box round every
/// corner of the person in front of the camera, within the image. None when the person is out of view.
fn colour_box(rig: &Rig, person: &Cuboid, body_position: DVec3, body_rotation: DQuat) -> Option<PixelBox> {
    let extrinsics = DepthCalibration::of(&rig.calibration).sensor_extrinsics().unwrap();
    // The colour camera is where the depth camera is (the rig's depth_to_color is the identity).
    let (optical_to_world, origin) = (body_rotation * extrinsics.rotation, body_position + body_rotation * extrinsics.translation_m);
    let color = rig.calibration.color;
    let (mut low, mut high) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
    for corner in 0..8 {
        let pick = |bit: usize, axis: usize| if corner & bit == 0 { person.min[axis] } else { person.max[axis] };
        let optical = optical_to_world.conjugate() * (DVec3::new(pick(1, 0), pick(2, 1), pick(4, 2)) - origin);
        assert!(optical.z > 0.1, "the test keeps people in front of the camera");
        let pixel = DVec3::new(f64::from(color.fx) * optical.x / optical.z + f64::from(color.ppx), f64::from(color.fy) * optical.y / optical.z + f64::from(color.ppy), 0.0);
        (low, high) = (low.min(pixel), high.max(pixel));
    }
    let (width, height) = (f64::from(color.width), f64::from(color.height));
    let (x0, y0, x1, y1) = (low.x.clamp(0.0, width), low.y.clamp(0.0, height), high.x.clamp(0.0, width), high.y.clamp(0.0, height));
    (x1 > x0 && y1 > y0).then_some(PixelBox { x_px: x0 as f32, y_px: y0 as f32, w_px: (x1 - x0) as f32, h_px: (y1 - y0) as f32 })
}

/// How the pose model reports a frame's people: as late as inference makes it, so the newest boxes a depth frame
/// can be cut with come from a frame this many frames earlier.
const POSE_LATENCY_FRAMES: u64 = 3;

/// What a resting camera made of someone moving in front of it.
struct PassedBy {
    poses: usize,
    corrected: usize,
    /// The furthest the pose strayed from the pose before the person appeared, and the largest turn from it.
    worst_position_m: f32,
    worst_angle_deg: f32,
    /// Frames with people cut out of them.
    with_people: usize,
    /// The longest run of frames whose cloud did not correct the pose, in camera time.
    longest_uncorrected_s: f64,
}

/// Rests the camera, then has a person follow `path` (where they stand at each time since they appeared) in front
/// of it, the pose model reporting their box late when `seen` holds. A path that starts in view has the person
/// reported from before the start, as if they had walked into view earlier. Panics if odometry is lost.
fn pass_by(path: impl Fn(f64) -> DVec3, for_s: f64, seen: bool) -> (Rig, PassedBy) {
    let mut rig = Rig::standing_still();
    rig.send_calibration().unwrap();
    let started = rig.rest_until_started(0);
    let appeared = started + 60;
    let mut at_rest = rig.depth_posed(started);
    for frame in started + 1..appeared {
        at_rest = rig.posed(frame);
        assert!(at_rest.corrected, "frame {frame}");
    }
    rig.node.take_progress();
    let (rest_position, rest_rotation) = (rig.walk.position(0.0), rig.walk.rotation(0.0));
    let since = |frame: u64| (frame as f64 - appeared as f64) * FRAME_PERIOD.as_secs_f64();
    let mut passed = PassedBy { poses: 0, corrected: 0, worst_position_m: 0.0, worst_angle_deg: 0.0, with_people: 0, longest_uncorrected_s: 0.0 };
    let mut corrected_at = appeared;
    let last = appeared + (for_s / FRAME_PERIOD.as_secs_f64()) as u64;
    for frame in appeared..=last {
        rig.person = Some(person_at(path(since(frame))));
        // The pose model reports where the person was a few frames back, along the same path: a person already in view
        // when the window starts has been reported before it.
        if seen {
            let captured = frame - POSE_LATENCY_FRAMES;
            let people = colour_box(&rig, &person_at(path(since(captured))), rest_position, rest_rotation)
                .map(|bbox| PersonPose { track_id: Some(1), score: 0.9, bbox, keypoints: vec![] })
                .into_iter()
                .collect();
            rig.node.on_person_poses(&PersonPoses { frame_id: captured, captured_camera_ns: camera_ns(Rig::time(captured)), width: 1280, height: 720, people }).unwrap();
        }
        rig.send_imu(frame).unwrap();
        let pose = match rig.send_depth(frame).unwrap() {
            DepthOutcome::Tracked(FrameOutcome::Posed(pose)) => pose,
            other => panic!("frame {frame}, {:.2} s after the person appeared: {other:?}", since(frame)),
        };
        passed.poses += 1;
        passed.corrected += usize::from(pose.corrected);
        if pose.corrected {
            corrected_at = frame;
        }
        passed.longest_uncorrected_s = passed.longest_uncorrected_s.max(since(frame) - since(corrected_at));
        passed.worst_position_m = passed.worst_position_m.max((pose.position_m - at_rest.position_m).length());
        passed.worst_angle_deg = passed.worst_angle_deg.max(pose.orientation.angle_between(at_rest.orientation).to_degrees());
    }
    passed.with_people = rig.node.take_progress().with_people;
    (rig, passed)
}

/// Where a person `ahead_m` in front of the resting camera and `aside_m` to its left stands, in the room.
fn in_front(rig_heading: f64, ahead_m: f64, aside_m: f64) -> DVec3 {
    let camera = Walk::on_a_robot(Duration::from_secs(3600)).position(0.0);
    let (ahead, aside) = (DVec3::new(rig_heading.cos(), rig_heading.sin(), 0.0), DVec3::new(-rig_heading.sin(), rig_heading.cos(), 0.0));
    DVec3::new(camera.x, camera.y, 0.0) + ahead * ahead_m + aside * aside_m
}

/// The resting camera's heading in the room.
const HEADING_RAD: f64 = 0.4;

/// Whether the map holds a surface anywhere on the face a person showed the camera along their path. The map's world
/// frame is the body's at the start, which for a resting camera is the room's moved to the camera and turned by its
/// heading; the levelling is left out, which moves a point a metre away by millimetres.
fn map_holds_the_person(rig: &Rig, path: impl Fn(f64) -> DVec3, for_s: f64) -> Vec<DVec3> {
    let map = rig.node.surface_map().expect("odometry runs");
    let (start, heading) = (rig.walk.position(0.0), DQuat::from_rotation_z(HEADING_RAD));
    let mut held = Vec::new();
    for step in 0..=(for_s * 10.0) as usize {
        let middle = path(step as f64 / 10.0);
        let towards_camera = (DVec3::new(start.x, start.y, 0.0) - DVec3::new(middle.x, middle.y, 0.0)).normalize();
        for height in [0.35, 0.5, 0.65, 0.8] {
            for across in [-0.15, 0.0, 0.15] {
                let on_face = middle + towards_camera * 0.15 + DVec3::new(-towards_camera.y, towards_camera.x, 0.0) * across + DVec3::Z * height;
                let in_map = heading.conjugate() * (on_face - start);
                if map.surface_near(in_map.as_vec3()).is_some() {
                    held.push(on_face);
                }
            }
        }
    }
    held
}

#[test]
fn a_person_crossing_in_front_of_a_resting_camera_is_cut_out_and_neither_drags_the_pose_nor_stays_in_the_map() {
    // From 1.6 m to the right to 1.6 m to the left, 1.2 m in front of the camera, at a walk.
    let crossing = |seconds: f64| in_front(HEADING_RAD, 1.2, -1.6 + 1.2 * seconds);
    let (rig, passed) = pass_by(crossing, 2.6, true);
    assert!(passed.with_people > 40, "the person was cut out of {} frames", passed.with_people);
    assert!(passed.worst_position_m < 0.01 && passed.worst_angle_deg < 0.3, "strayed {:.4} m, {:.3} deg", passed.worst_position_m, passed.worst_angle_deg);
    assert!(passed.corrected > passed.poses * 9 / 10, "{} of {} frames corrected", passed.corrected, passed.poses);
    let held = map_holds_the_person(&rig, crossing, 2.6);
    assert!(held.is_empty(), "the map holds the person at {held:?}");
}

#[test]
fn a_person_walking_up_to_a_resting_camera_until_they_fill_most_of_the_view_is_dead_reckoned_through() {
    // From 3 m away up to 0.25 m in front of the lens, nearer than the camera reads depth, a pause, and away again.
    let approach = |seconds: f64| {
        let ahead_m = if seconds < 2.75 { 3.0 - seconds } else if seconds < 3.25 { 0.25 } else { 0.25 + (seconds - 3.25) };
        in_front(HEADING_RAD, ahead_m + 0.15, 0.0)
    };
    let (rig, passed) = pass_by(approach, 5.9, true);
    assert!(passed.with_people > 100, "the person was cut out of {} frames", passed.with_people);
    assert!(passed.worst_position_m < 0.01 && passed.worst_angle_deg < 0.3, "strayed {:.4} m, {:.3} deg", passed.worst_position_m, passed.worst_angle_deg);
    // Longer than odometry could go uncorrected under a fixed half second.
    assert!(passed.longest_uncorrected_s > 0.6, "the view was blocked for {:.2} s", passed.longest_uncorrected_s);
    let held = map_holds_the_person(&rig, approach, 5.9);
    assert!(held.is_empty(), "the map holds the person at {held:?}");
}

#[test]
fn a_person_the_pose_model_misses_does_not_drag_the_pose() {
    let crossing = |seconds: f64| in_front(HEADING_RAD, 1.2, -1.6 + 1.2 * seconds);
    let (_, passed) = pass_by(crossing, 2.6, false);
    assert_eq!(passed.with_people, 0);
    assert!(passed.worst_position_m < 0.01 && passed.worst_angle_deg < 0.3, "strayed {:.4} m, {:.3} deg", passed.worst_position_m, passed.worst_angle_deg);
}
