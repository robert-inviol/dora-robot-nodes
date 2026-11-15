use depth_perception::synthetic::SyntheticRoom;
use dora_map_localizer::config::NodeSettings;
use dora_map_localizer::{GyroGap, LocalizerNode};
use glam::Vec3;
use map_localizer::synthetic::{ellipse_poses, furnished_room, render_walk, turning_poses, SyntheticWalk};
use map_localizer::{HeadingWithoutGyro, KeyframeEstimate, KeyframeSelection};
use perception_wire::localization::{decode_keyframe_pose, decode_pose_graph_update, encode_keyframe_pose, encode_pose_graph_update, FrameId, KeyframePose, Tracking};
use perception_wire::{decode_imu, decode_keyframe, encode_imu, encode_keyframe, Imu};
use room_map::heading::HeadingError;
use room_map::pose::wrap_angle;

#[test]
fn a_walk_sent_as_messages_comes_back_as_keyframe_poses_in_any_arrival_order() {
    let walk = render_walk(&SyntheticRoom::living_room(), &ellipse_poses(24, 1), 0..0, |_| Vec3::new(0.002, -0.003, 0.002));
    let mut node = LocalizerNode::new(NodeSettings::default());
    let mut poses: Vec<KeyframePose> = Vec::new();
    for (index, (keyframe, imu)) in walk.keyframes.iter().zip(&walk.imu).enumerate() {
        let keyframe = decode_keyframe(&encode_keyframe(keyframe)).unwrap();
        let imu = decode_imu(&encode_imu(imu)).unwrap();
        // Every other keyframe arrives before the imu message leading up to it.
        let handled = if index % 2 == 0 {
            assert!(node.on_imu(&imu).unwrap().estimate.is_none());
            node.on_keyframe(keyframe)
        } else {
            let waiting = node.on_keyframe(keyframe);
            assert!(waiting.estimate.is_none() && waiting.dropped.is_none(), "keyframe {index} waits for its imu message");
            node.on_imu(&imu).unwrap()
        };
        assert!(handled.gyro_gap.is_none(), "{:?}", handled.gyro_gap);
        let estimate = handled.estimate.unwrap_or_else(|| panic!("keyframe {index} is localized"));
        poses.push(decode_keyframe_pose(&encode_keyframe_pose(&estimate.to_message())).unwrap());
    }
    let mut worst = (0.0f32, 0.0f32);
    for (pose, truth) in poses.iter().zip(&walk.truth) {
        assert_eq!(pose.tracking, Tracking::Tracking, "{pose:?}");
        assert!(pose.inlier_fraction > 0.5 && pose.residual_m < 0.01, "{pose:?}");
        let position = (Vec3::new(pose.pose.x_m, pose.pose.y_m, pose.pose.z_m) - truth.translation).length();
        let heading = wrap_angle(pose.pose.yaw_rad - truth.yaw_rad).abs().to_degrees();
        worst = (worst.0.max(position), worst.1.max(heading));
    }
    println!("worst error over {} keyframes: {:.4} m {:.3} deg", poses.len(), worst.0, worst.1);
    assert!(worst.0 < 0.01 && worst.1 < 0.3, "{worst:?}");
    assert_eq!(poses.iter().map(|p| p.frame_id).collect::<Vec<_>>(), (0..poses.len() as u64).map(FrameId).collect::<Vec<_>>());
}

#[test]
fn a_keyframe_still_waiting_for_its_imu_message_is_replaced_by_a_newer_one() {
    let walk = render_walk(&SyntheticRoom::living_room(), &ellipse_poses(24, 1)[..3], 0..0, |_| Vec3::ZERO);
    let copy = |index: usize| decode_keyframe(&encode_keyframe(&walk.keyframes[index])).unwrap();
    let mut node = LocalizerNode::new(NodeSettings::default());
    node.on_imu(&walk.imu[0]).unwrap();
    assert!(node.on_keyframe(copy(0)).estimate.is_some());
    assert!(node.on_keyframe(copy(1)).estimate.is_none());
    let handled = node.on_keyframe(copy(2));
    assert_eq!(handled.dropped, Some(FrameId(1)));
    node.on_imu(&walk.imu[1]).unwrap();
    let estimate = node.on_imu(&walk.imu[2]).unwrap().estimate.expect("the newer keyframe is localized once its imu message is in");
    assert_eq!((estimate.frame_id, estimate.tracking), (FrameId(2), Tracking::Tracking));
}

/// Every keyframe through a node with `heading_without_gyro` set as given, imu message first.
fn run(walk: &SyntheticWalk, imu: &[Imu], heading_without_gyro: HeadingWithoutGyro) -> (Vec<KeyframeEstimate>, Vec<GyroGap>) {
    let mut node = LocalizerNode::new(NodeSettings { heading_without_gyro, ..NodeSettings::default() });
    let (mut estimates, mut gaps) = (Vec::new(), Vec::new());
    for (keyframe, imu) in walk.keyframes.iter().zip(imu) {
        node.on_imu(imu).unwrap();
        let handled = node.on_keyframe(decode_keyframe(&encode_keyframe(keyframe)).unwrap());
        estimates.push(handled.estimate.expect("the imu message is in"));
        gaps.extend(handled.gyro_gap);
    }
    (estimates, gaps)
}

/// The walk's imu messages with one message's samples missing.
fn with_samples_missing(walk: &SyntheticWalk, missing: usize) -> Vec<Imu> {
    let mut imu = walk.imu.clone();
    let hole = &mut imu[missing];
    (hole.gyro_camera_ns, hole.gyro_rad_s, hole.accel_camera_ns, hole.accel_m_s2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    imu
}

#[test]
fn a_keyframe_the_gyro_cannot_cover_is_lost_when_refused_and_searched_for_when_asked() {
    let walk = render_walk(&SyntheticRoom::living_room(), &ellipse_poses(24, 1)[..16], 0..0, |_| Vec3::new(0.002, -0.003, 0.002));
    let gap = 10;
    let imu = with_samples_missing(&walk, gap);

    // The keyframe the missing message leads up to has no samples to reach it; the next one's span starts in the
    // hole the missing message leaves.
    let (refused, gaps) = run(&walk, &imu, HeadingWithoutGyro::Refuse);
    assert_eq!(gaps.len(), 2, "{gaps:?}");
    assert_eq!((gaps[0].frame_id, gaps[0].handled_by, gaps[0].count), (FrameId(gap as u64), HeadingWithoutGyro::Refuse, 1));
    assert!(matches!(gaps[0].problem, HeadingError::NotCovered { .. }), "{:?}", gaps[0].problem);
    assert_eq!((gaps[1].frame_id, gaps[1].count), (FrameId(gap as u64 + 1), 2));
    assert!(matches!(gaps[1].problem, HeadingError::SampleGap { .. }), "{:?}", gaps[1].problem);
    for k in [gap, gap + 1] {
        assert_eq!(refused[k].tracking, Tracking::Lost);
        assert!(refused[k].registration.is_none(), "nothing is guessed for a refused keyframe");
    }
    assert_eq!(refused[gap + 2].tracking, Tracking::Relocalized, "the next keyframe finds its heading against the map");
    assert!(refused.iter().enumerate().filter(|(k, _)| !(gap..=gap + 2).contains(k)).all(|(_, e)| e.tracking == Tracking::Tracking));

    let (searched, gaps) = run(&walk, &imu, HeadingWithoutGyro::Search);
    let searched_gaps = vec![(FrameId(gap as u64), HeadingWithoutGyro::Search), (FrameId(gap as u64 + 1), HeadingWithoutGyro::Search)];
    assert_eq!(gaps.iter().map(|g| (g.frame_id, g.handled_by)).collect::<Vec<_>>(), searched_gaps);
    assert!(searched.iter().all(|e| e.tracking == Tracking::Tracking), "{:?}", searched.iter().map(|e| e.tracking).collect::<Vec<_>>());
    let heading = wrap_angle(searched[gap].pose.yaw_rad - walk.truth[gap].yaw_rad).abs().to_degrees();
    assert!(heading < 0.3, "the searched heading is {heading:.2} deg off");
}

#[test]
fn keyframes_are_lost_until_a_stationary_span_has_measured_the_gyro_bias() {
    let walk = render_walk(&furnished_room(), &turning_poses(), 0..0, |_| Vec3::new(0.002, -0.003, 0.002));
    // The walk's IMU starts a tenth of a second before the first keyframe, with the camera about to move; it
    // first stands still at keyframes 21 to 24.
    let mut imu = walk.imu.clone();
    let starts_ns = imu[0].keyframe_camera_ns - 100_000_000;
    let first = &mut imu[0];
    let gyro_from = first.gyro_camera_ns.partition_point(|&t| t < starts_ns);
    let accel_from = first.accel_camera_ns.partition_point(|&t| t < starts_ns);
    first.gyro_camera_ns.drain(..gyro_from);
    first.gyro_rad_s.drain(..gyro_from);
    first.accel_camera_ns.drain(..accel_from);
    first.accel_m_s2.drain(..accel_from);
    let (estimates, gaps) = run(&walk, &imu, HeadingWithoutGyro::Refuse);
    let first_placed = estimates.iter().position(|e| e.tracking != Tracking::Lost).expect("the map starts after the pause");
    println!("first keyframe placed: {first_placed}");
    assert!((21..=24).contains(&first_placed), "{first_placed}");
    assert_eq!(gaps.iter().map(|g| g.frame_id.0 as usize).collect::<Vec<_>>(), (0..first_placed).collect::<Vec<_>>());
    assert!(gaps.iter().all(|g| matches!(g.problem, HeadingError::TooLittleStationaryTime { .. })), "{gaps:?}");
    assert!(estimates[first_placed..].iter().all(|e| e.tracking == Tracking::Tracking));
}

#[test]
fn an_optimization_that_moves_keyframes_already_sent_sends_their_new_poses() {
    // A turn with no depth restarts tracking where dead reckoning puts it; the walk back closes a loop on the start,
    // and optimization moves the keyframes tracked since the restart.
    let bias = |t: i64| Vec3::new(0.002, -0.003 - 0.010 * t as f32 / 25e9, 0.002);
    let walk = render_walk(&furnished_room(), &turning_poses(), 11..16, bias);
    let mut node = LocalizerNode::new(NodeSettings::default());
    let mut sent: std::collections::HashMap<FrameId, Vec3> = std::collections::HashMap::new();
    let mut moved_messages = 0;
    let mut moved_frames = std::collections::HashSet::new();
    for (keyframe, imu) in walk.keyframes.iter().zip(&walk.imu) {
        node.on_imu(&decode_imu(&encode_imu(imu)).unwrap()).unwrap();
        let handled = node.on_keyframe(decode_keyframe(&encode_keyframe(keyframe)).unwrap());
        if let Some(moved) = handled.moved {
            let moved = decode_pose_graph_update(&encode_pose_graph_update(&moved)).unwrap();
            assert_eq!(moved.map_version, node.localizer().map_version());
            for keyframe in &moved.keyframes {
                let was = sent.get(&keyframe.frame_id).unwrap_or_else(|| panic!("keyframe {:?} was sent before it moved", keyframe.frame_id));
                let now = Vec3::new(keyframe.pose.x_m, keyframe.pose.y_m, keyframe.pose.z_m);
                let current = node.localizer().keyframe_pose(keyframe.frame_id).unwrap();
                assert!((now - current.translation).length() < 1e-5, "the moved pose is the graph's current one");
                assert!((now - *was).length() > 0.0099 || wrap_angle(keyframe.pose.yaw_rad - current.yaw_rad).abs() < 1e-6);
                sent.insert(keyframe.frame_id, now);
                moved_frames.insert(keyframe.frame_id);
            }
            moved_messages += 1;
        }
        let estimate = handled.estimate.unwrap();
        if estimate.selection != KeyframeSelection::NotLocalized {
            sent.insert(estimate.frame_id, estimate.pose.translation);
        }
    }
    assert!(moved_messages > 0, "the loop closure moved keyframes already sent");
    assert!(moved_frames.contains(&FrameId(16)), "the keyframe that restarted tracking, sent restarted, is sent again when it moves: {moved_frames:?}");
}
