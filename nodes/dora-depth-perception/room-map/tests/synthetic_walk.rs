use depth_perception::synthetic::SyntheticRoom;
use depth_perception::{sample_point_cloud, DepthImage, FloorFrame, RangeLimits};
use glam::Vec3;
use room_map::grid::voxel_downsample;
use room_map::trajectory::{close_loop, find_loop, track, LoopSettings, TrackingSettings};
use room_map::Pose;

const KEYFRAMES: usize = 24;

/// A level camera walking an ellipse in the room, sweeping its heading, ending where it started.
fn true_poses() -> Vec<Pose> {
    (0..=KEYFRAMES)
        .map(|k| {
            let phase = k as f32 / KEYFRAMES as f32 * std::f32::consts::TAU;
            Pose {
                yaw_rad: 0.6 * phase.sin(),
                translation: Vec3::new(0.25 + 0.5 * phase.cos() - 0.5, 1.0 + 0.4 * phase.sin(), 0.0),
            }
        })
        .collect()
}

fn walk_clouds(poses: &[Pose]) -> Vec<Vec<Vec3>> {
    let scene = SyntheticRoom::living_room();
    poses
        .iter()
        .map(|pose| {
            let frame = scene.render_from(pose.translation, pose.yaw_rad);
            let floor = FloorFrame::from_gravity_reaction(frame.accel_camera);
            let depth = DepthImage::new(scene.width, scene.height, &frame.depth_m);
            let cloud = sample_point_cloud(&depth, &scene.intrinsics, &floor, RangeLimits::D455, 4);
            voxel_downsample(&cloud.points, 0.05)
        })
        .collect()
}

fn relative_to_first(poses: &[Pose]) -> Vec<Pose> {
    poses.iter().map(|p| p.relative_to(&poses[0])).collect()
}

fn error(a: &Pose, b: &Pose) -> (f32, f32) {
    ((a.translation - b.translation).length(), (a.yaw_rad - b.yaw_rad).abs().to_degrees())
}

#[test]
fn tracking_recovers_the_walk_within_a_couple_of_centimetres() {
    let truth = relative_to_first(&true_poses());
    let tracked = track(&walk_clouds(&true_poses()), &TrackingSettings::default());
    assert!(tracked.iter().all(|t| t.registered), "every synthetic keyframe registers");
    let worst = truth.iter().zip(&tracked).map(|(t, k)| error(t, &k.pose)).fold((0.0f32, 0.0f32), |a, e| (a.0.max(e.0), a.1.max(e.1)));
    println!("worst tracking error: {:.4} m, {:.3} deg", worst.0, worst.1);
    assert!(worst.0 < 0.02 && worst.1 < 0.5, "worst error {worst:?}");
}

#[test]
fn closing_the_loop_removes_injected_drift() {
    let truth = relative_to_first(&true_poses());
    let clouds = walk_clouds(&true_poses());
    // Odometry that over-turns and creeps sideways a little on every step.
    let mut drifted = vec![truth[0]];
    for w in truth.windows(2) {
        let step = w[1].relative_to(&w[0]);
        let biased = Pose { yaw_rad: step.yaw_rad + 0.01, translation: step.translation + Vec3::new(0.01, 0.0, 0.0) };
        drifted.push(drifted[drifted.len() - 1].compose(&biased));
    }
    let as_tracked: Vec<_> = drifted
        .iter()
        .map(|&pose| room_map::trajectory::TrackedKeyframe { pose, registration: None, registered: true, accepted: true })
        .collect();
    let closure = find_loop(&clouds, &as_tracked, &LoopSettings::default(), &TrackingSettings::default()).expect("the walk ends at its start");
    let corrected = close_loop(&drifted, closure.measured_last);
    let before = drifted.iter().zip(&truth).map(|(d, t)| error(d, t).0).fold(0.0f32, f32::max);
    let after = corrected.iter().zip(&truth).map(|(c, t)| error(c, t).0).fold(0.0f32, f32::max);
    let end = error(&corrected[KEYFRAMES], &truth[KEYFRAMES]);
    println!("worst drift before closing {before:.3} m, after {after:.3} m; end error {:.4} m {:.3} deg", end.0, end.1);
    assert!(end.0 < 0.02 && end.1 < 0.5, "end error {end:?}");
    assert!(after < before / 3.0, "worst error {before} before, {after} after");
}
