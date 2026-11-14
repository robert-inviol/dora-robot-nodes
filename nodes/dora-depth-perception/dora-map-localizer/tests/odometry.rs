use depth_perception::synthetic::SyntheticRoom;
use dora_map_localizer::config::{NodeSettings, OdometrySettings};
use dora_map_localizer::{LocalizerNode, OdometryInputError};
use glam::{Mat3, Quat, Vec3};
use map_localizer::map_to_odom::{CameraMount, MapToOdomSettings, OdomError};
use map_localizer::synthetic::{ellipse_poses, render_walk};
use perception_wire::localization::{decode_map_to_odom, encode_map_to_odom, CameraNanos, FrameId, MapToOdom};
use perception_wire::odometry::{decode_odom_pose, encode_odom_pose, OdomPose};
use perception_wire::{decode_keyframe, encode_keyframe};
use room_map::pose::wrap_angle;
use room_map::Pose;

const MOUNT: CameraMount = CameraMount { camera_in_imu_m: Vec3::new(0.03, -0.005, 0.01) };

fn with_odometry() -> NodeSettings {
    let odometry = OdometrySettings { mount: MOUNT, retention_ns: OdometrySettings::DEFAULT_RETENTION_NS, estimator: MapToOdomSettings::default() };
    NodeSettings { odometry: Some(odometry), ..NodeSettings::default() }
}

/// The odometry message for a level camera whose floor frame is at `floor` in the odometry frame.
fn odom_pose(camera_ns: i64, floor: Pose) -> OdomPose {
    let (sin, cos) = floor.yaw_rad.sin_cos();
    let forward = Vec3::new(-sin, cos, 0.0);
    let orientation = Quat::from_mat3(&Mat3::from_cols(forward.cross(Vec3::Z), Vec3::NEG_Z, forward));
    let pose = OdomPose {
        camera_ns,
        depth_frame_id: 0,
        position_m: floor.translation - orientation * MOUNT.camera_in_imu_m,
        orientation,
        velocity_m_s: Vec3::ZERO,
        position_sigma_m: Vec3::splat(0.01),
        tilt_sigma_rad: 0.001,
        heading_sigma_rad: 0.002,
        corrected: true,
        matched_points: 900,
        weakest: None,
    };
    decode_odom_pose(&encode_odom_pose(&pose)).unwrap()
}

#[test]
fn odometry_in_a_drifted_frame_gives_the_correction_back_to_the_map() {
    let walk = render_walk(&SyntheticRoom::living_room(), &ellipse_poses(24, 1)[..12], 0..0, |_| Vec3::ZERO);
    // map = correction(odom): the odometry frame sits turned and shifted against the map.
    let correction = Pose { yaw_rad: 0.6, translation: Vec3::new(1.5, -0.75, 0.05) };
    let odom_floor = |index: usize| correction.inverse().compose(&walk.truth[index]);
    let mut node = LocalizerNode::new(with_odometry());
    let mut sent: Vec<MapToOdom> = Vec::new();
    for (index, (keyframe, imu)) in walk.keyframes.iter().zip(&walk.imu).enumerate() {
        node.on_imu(imu);
        let handled = node.on_keyframe(decode_keyframe(&encode_keyframe(keyframe)).unwrap());
        // Odometry starts after the first keyframe, and each pose arrives after the keyframe it is for.
        if index == 0 {
            assert!(handled.map_to_odom.is_none() && handled.unjoined.is_empty());
            continue;
        }
        sent.extend(handled.map_to_odom);
        let handled = node.on_odom_pose(&odom_pose(imu.keyframe_camera_ns, odom_floor(index))).unwrap();
        if index == 1 {
            assert_eq!(handled.unjoined.len(), 1, "the first keyframe is before odometry started");
            assert_eq!(handled.unjoined[0].frame_id, FrameId(0));
            assert!(matches!(handled.unjoined[0].problem, OdomError::NotCovered { .. }), "{:?}", handled.unjoined[0].problem);
        } else {
            assert!(handled.unjoined.is_empty(), "{:?}", handled.unjoined);
        }
        let joined = handled.map_to_odom.unwrap_or_else(|| panic!("keyframe {index} joins its odometry"));
        assert_eq!(joined.valid_from, CameraNanos(imu.keyframe_camera_ns));
        sent.push(decode_map_to_odom(&encode_map_to_odom(&joined)).unwrap());
    }
    let last = sent.last().unwrap();
    let c = last.correction;
    let position = (Vec3::new(c.x_m, c.y_m, c.z_m) - correction.translation).length();
    let heading = wrap_angle(c.yaw_rad - correction.yaw_rad).abs().to_degrees();
    println!("correction off by {position:.4} m and {heading:.3} deg; sigmas {last:?}");
    assert!(position < 0.02 && heading < 0.5, "{last:?}");
    assert!(last.sigma_xy_m < 0.03 && last.sigma_yaw_rad.to_degrees() < 1.0, "{last:?}");
}

#[test]
fn keyframes_that_odometry_never_reaches_are_given_up_after_the_retention() {
    let walk = render_walk(&SyntheticRoom::living_room(), &ellipse_poses(24, 1)[..3], 0..0, |_| Vec3::ZERO);
    let mut settings = with_odometry();
    let retention_ns = walk.imu[1].keyframe_camera_ns - walk.imu[0].keyframe_camera_ns - 1;
    settings.odometry.as_mut().unwrap().retention_ns = retention_ns;
    let mut node = LocalizerNode::new(settings);
    let mut unjoined = Vec::new();
    for (keyframe, imu) in walk.keyframes.iter().zip(&walk.imu) {
        node.on_imu(imu);
        let handled = node.on_keyframe(decode_keyframe(&encode_keyframe(keyframe)).unwrap());
        assert!(handled.map_to_odom.is_none());
        unjoined.extend(handled.unjoined);
    }
    assert_eq!(unjoined.iter().map(|u| u.frame_id).collect::<Vec<_>>(), [FrameId(0), FrameId(1)]);
    assert!(unjoined.iter().all(|u| u.problem == OdomError::Empty), "{unjoined:?}");
}

#[test]
fn odometry_without_an_odometry_section_is_refused() {
    let mut node = LocalizerNode::new(NodeSettings::default());
    let refused = node.on_odom_pose(&odom_pose(1_000, Pose::IDENTITY));
    assert!(matches!(refused, Err(OdometryInputError::NotConfigured)), "{refused:?}");
}

#[test]
fn odometry_going_back_in_time_is_refused() {
    let mut node = LocalizerNode::new(with_odometry());
    node.on_odom_pose(&odom_pose(2_000, Pose::IDENTITY)).unwrap();
    let refused = node.on_odom_pose(&odom_pose(1_000, Pose::IDENTITY));
    assert!(matches!(refused, Err(OdometryInputError::Odometry(OdomError::OutOfOrder { .. }))), "{refused:?}");
}
