use depth_perception::synthetic::SyntheticRoom;
use depth_perception::{sample_point_cloud, DepthImage, FloorFrame, RangeLimits};
use glam::Vec3;
use room_map::grid::voxel_downsample;
use room_map::trajectory::{close_loop, find_loop, track, LoopSettings, TrackingSettings};
use room_map::keyframe_gaps::GapTurn;
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
    walk_clouds_in(&SyntheticRoom::living_room(), poses)
}

fn walk_clouds_in(scene: &SyntheticRoom, poses: &[Pose]) -> Vec<Vec<Vec3>> {
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
    let clouds = walk_clouds(&true_poses());
    let tracked = track(&clouds, &vec![GapTurn::Unknown; clouds.len() - 1], &TrackingSettings::default());
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
        .map(|&pose| room_map::trajectory::TrackedKeyframe { pose, heading: room_map::trajectory::HeadingSource::Registration, registration: None, registered: true, accepted: true })
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

mod with_gyro {
    use super::*;
    use depth_perception::synthetic::Cuboid;
    use room_map::heading::{keyframe_turns, still_between, GyroBias, ImuStream, StillSettings};
    use room_map::placement::{place_runs, PlacementHeading, PlacementSettings};
    use room_map::trajectory::{check_turn_direction, track_with_gyro, HeadingSource};

    const KEYFRAME_GAP_NS: i64 = 600_000_000;
    const STILL_START_NS: i64 = 2_000_000_000;
    const GYRO_STEP_NS: i64 = 2_500_000;
    const ACCEL_STEP_NS: i64 = 5_000_000;
    /// Keyframes in the middle of the turn that depth cannot register, as when the camera faces glass.
    const BLANK: std::ops::Range<usize> = 11..16;
    /// Keyframes held still after the turn and at the end of the walk.
    const HELD: [std::ops::Range<usize>; 2] = [21..25, 34..38];
    const UP_IN_CAMERA: Vec3 = Vec3::new(0.0, -1.0, 0.0);

    /// The living room with furniture along every wall, so that no view is a bare wall.
    fn furnished_room() -> SyntheticRoom {
        let cuboid = |min: [f32; 3], max: [f32; 3]| Cuboid { min: Vec3::from_array(min), max: Vec3::from_array(max) };
        let mut room = SyntheticRoom::living_room();
        room.objects.extend([
            cuboid([-1.5, 2.6, -1.0], [-0.9, 3.4, -0.2]),
            cuboid([1.4, 0.2, -1.0], [2.0, 0.9, 0.1]),
            cuboid([-0.4, -1.0, -1.0], [0.8, -0.6, -0.4]),
            cuboid([1.2, 3.5, -1.0], [1.8, 4.0, 0.6]),
            cuboid([-1.5, -0.5, -1.0], [-1.1, 0.3, 0.4]),
        ]);
        room
    }

    /// A short walk, a 200 degree turn while drifting 30 cm sideways, a pause, a walk back turning to face
    /// the way it started, and a pause at the end.
    fn true_poses() -> Vec<Pose> {
        let lerp = |a: f32, b: f32, s: f32| a + (b - a) * s;
        let turned = 0.3 + 200f32.to_radians();
        (0..38)
            .map(|k| match k {
                0..=8 => {
                    let s = k as f32 / 8.0;
                    Pose { yaw_rad: lerp(0.0, 0.3, s), translation: Vec3::new(lerp(0.0, 0.3, s), lerp(1.0, 1.4, s), 0.0) }
                }
                9..=20 => {
                    let s = (k - 8) as f32 / 12.0;
                    Pose { yaw_rad: 0.3 + 200f32.to_radians() * s, translation: Vec3::new(lerp(0.3, 0.6, s), 1.4, 0.0) }
                }
                21..=24 => Pose { yaw_rad: turned, translation: Vec3::new(0.6, 1.4, 0.0) },
                25..=33 => {
                    let s = (k - 24) as f32 / 9.0;
                    Pose { yaw_rad: turned + 160f32.to_radians() * s, translation: Vec3::new(lerp(0.6, 0.0, s), lerp(1.4, 1.0, s), 0.0) }
                }
                _ => Pose { yaw_rad: turned + 160f32.to_radians(), translation: Vec3::new(0.0, 1.0, 0.0) },
            })
            .collect()
    }

    fn keyframe_times(count: usize) -> Vec<i64> {
        (0..count as i64).map(|k| STILL_START_NS + k * KEYFRAME_GAP_NS).collect()
    }

    /// The walk's IMU: gyro at 400 Hz with a bias drifting tenfold over the walk and noise, following the
    /// headings linearly between keyframes; accel at 200 Hz reading gravity, plus the hand's shake whenever
    /// the camera moves. Still for the first two seconds.
    fn imu(poses: &[Pose]) -> ImuStream {
        let times = keyframe_times(poses.len());
        let end = times[poses.len() - 1] + KEYFRAME_GAP_NS;
        let bias = |t: i64| Vec3::new(0.002, -0.003 - 0.010 * t as f32 / end as f32, 0.002);
        let mut state: u64 = 0x2545F4914F6CDD1D;
        let mut noise = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f32 / (1u64 << 53) as f32 - 0.5) * 0.007
        };
        let segment = |t: i64| (t >= STILL_START_NS).then(|| ((t - STILL_START_NS) / KEYFRAME_GAP_NS) as usize).filter(|s| s + 1 < poses.len());
        let moving = |t: i64| segment(t).is_some_and(|s| poses[s + 1] != poses[s]);
        let gyro_ns: Vec<i64> = (0..=end / GYRO_STEP_NS).map(|k| k * GYRO_STEP_NS).collect();
        let gyro = gyro_ns
            .iter()
            .map(|&t| {
                let rate = segment(t).map_or(0.0, |s| (poses[s + 1].yaw_rad - poses[s].yaw_rad) / (KEYFRAME_GAP_NS as f32 * 1e-9));
                UP_IN_CAMERA * rate + bias(t) + Vec3::new(noise(), noise(), noise())
            })
            .collect();
        let accel_ns: Vec<i64> = (0..=end / ACCEL_STEP_NS).map(|k| k * ACCEL_STEP_NS).collect();
        let accel = accel_ns
            .iter()
            .map(|&t| {
                let s = t as f32 * 1e-9;
                let shake = if moving(t) { Vec3::new((s * 23.0).sin(), (s * 17.0).cos(), (s * 11.0).sin()) * 0.6 } else { Vec3::ZERO };
                -UP_IN_CAMERA * 9.7 + shake + Vec3::new(noise(), noise(), noise())
            })
            .collect();
        ImuStream::new(gyro_ns, gyro, accel_ns, accel).unwrap()
    }

    struct Walk {
        truth: Vec<Pose>,
        turns: Vec<f32>,
        still: Vec<bool>,
        clouds: Vec<Vec<Vec3>>,
    }

    fn walk() -> Walk {
        let poses = true_poses();
        let stream = imu(&poses);
        let spans = stream.still_spans(&StillSettings::default());
        let bias = GyroBias::from_spans(&spans, &StillSettings::default()).unwrap();
        let times = keyframe_times(poses.len());
        let turns = keyframe_turns(&stream, &bias, &times, &vec![UP_IN_CAMERA * 9.81; poses.len()]).unwrap();
        let mut clouds = walk_clouds_in(&furnished_room(), &poses);
        for k in BLANK {
            clouds[k].clear();
        }
        Walk { truth: relative_to_first(&poses), turns, still: still_between(&spans, &times), clouds }
    }

    fn worst(truth: &[Pose], poses: &[Option<Pose>], keyframes: &[usize]) -> (f32, f32) {
        keyframes.iter().map(|&k| poses[k].map_or((f32::INFINITY, f32::INFINITY), |p| error(&truth[k], &p))).fold((0.0f32, 0.0f32), |a, e| (a.0.max(e.0), a.1.max(e.1)))
    }

    #[test]
    fn a_bias_measured_on_every_still_span_keeps_the_heading_through_a_drifting_bias() {
        let poses = true_poses();
        let stream = imu(&poses);
        let spans = stream.still_spans(&StillSettings::default());
        let times = keyframe_times(poses.len());
        let up = vec![UP_IN_CAMERA * 9.81; poses.len()];
        let heading_error = |bias: &GyroBias| {
            let turned: f32 = keyframe_turns(&stream, bias, &times, &up).unwrap().iter().sum();
            (turned - (poses[poses.len() - 1].yaw_rad - poses[0].yaw_rad)).abs().to_degrees()
        };
        let opening_only = GyroBias::constant(spans[0].gyro_bias_rad_s);
        let every_span = GyroBias::from_spans(&spans, &StillSettings::default()).unwrap();
        let (constant, interpolated) = (heading_error(&opening_only), heading_error(&every_span));
        println!("{} still spans; heading error at the end: {constant:.2} deg with the opening bias, {interpolated:.2} deg with every span", spans.len());
        assert_eq!(spans.len(), 3, "the opening, the pause and the end");
        assert!(interpolated < 0.5 && interpolated < constant / 3.0, "{constant} then {interpolated}");
    }

    #[test]
    fn the_gyro_carries_the_heading_through_keyframes_depth_cannot_register() {
        let Walk { truth, turns, still, clouds } = walk();
        let started = std::time::Instant::now();
        let tracked = track_with_gyro(&clouds, &turns, &still, &TrackingSettings::default());
        let gyro_tracking_s = started.elapsed().as_secs_f64();
        for k in BLANK {
            assert_eq!(tracked[k].heading, HeadingSource::Gyro, "keyframe {k}");
            assert!(error(&truth[k], &tracked[k].pose).1 < 1.0, "keyframe {k} heading off by {:.2} deg", error(&truth[k], &tracked[k].pose).1);
        }
        // Stillness is judged in half-second windows, so a pause's first and last keyframes may not count.
        let inside_pauses: Vec<usize> = HELD.iter().flat_map(|r| (r.start + 1)..(r.end - 1)).collect();
        assert!(inside_pauses.iter().all(|&k| tracked[k].heading == HeadingSource::Still), "keyframes inside a pause are still");
        let moved: Vec<usize> = (1..truth.len()).filter(|&k| truth[k] != truth[k - 1]).collect();
        assert!(moved.iter().all(|&k| tracked[k].heading != HeadingSource::Still), "no keyframe the camera moved to counts as still");
        let with_depth: Vec<usize> = (0..truth.len()).filter(|k| !BLANK.contains(k)).collect();
        let restarts: Vec<usize> = with_depth.iter().copied().filter(|&k| tracked[k].heading == HeadingSource::Gyro).collect();
        assert_eq!(restarts, vec![BLANK.end], "tracking restarts once, on the first keyframe after the blank ones");

        let tracking = TrackingSettings::default();
        let heading = PlacementHeading::NearTracked { extent_deg: 30.0, weight: tracking.gyro_yaw_weight, max_step_m: tracking.max_step_m };
        let placement = PlacementSettings { heading, ..PlacementSettings::default() };
        let started = std::time::Instant::now();
        let (placed, _) = place_runs(&clouds, &tracked, &placement, &tracking.registration);
        let placing_s = started.elapsed().as_secs_f64();
        let placed_worst = worst(&truth, &placed, &with_depth);
        let tracked_worst = worst(&truth, &tracked.iter().map(|t| Some(t.pose)).collect::<Vec<_>>(), &with_depth);
        let started = std::time::Instant::now();
        let search = track(&clouds, &vec![GapTurn::Unknown; clouds.len() - 1], &tracking);
        let search_s = started.elapsed().as_secs_f64();
        let search_worst = worst(&truth, &search.iter().map(|t| Some(t.pose)).collect::<Vec<_>>(), &with_depth);
        println!(
            "gyro tracking alone: worst {:.3} m {:.2} deg; after placing runs: {:.4} m {:.3} deg; depth-only search: {:.3} m {:.2} deg",
            tracked_worst.0, tracked_worst.1, placed_worst.0, placed_worst.1, search_worst.0, search_worst.1
        );
        let per = |seconds: f64| seconds * 1e3 / (truth.len() - 1) as f64;
        println!(
            "time for {} keyframes: gyro tracking {:.0} ms ({:.1} ms each), placing runs {:.0} ms, depth-only search {:.0} ms ({:.1} ms each)",
            truth.len(), gyro_tracking_s * 1e3, per(gyro_tracking_s), placing_s * 1e3, search_s * 1e3, per(search_s)
        );
        assert!(tracked_worst.1 < 1.0, "the gyro keeps every heading right: {tracked_worst:?}");
        assert!(placed_worst.0 < 0.03 && placed_worst.1 < 1.0, "every keyframe with depth placed: {placed_worst:?}");
    }

    #[test]
    fn a_gyro_turning_the_wrong_way_is_caught() {
        let Walk { turns, clouds, .. } = walk();
        let settings = TrackingSettings::default();
        let right = check_turn_direction(&clouds, &turns, 3f32.to_radians(), 12, &settings);
        let reversed: Vec<f32> = turns.iter().map(|t| -t).collect();
        let wrong = check_turn_direction(&clouds, &reversed, 3f32.to_radians(), 12, &settings);
        assert!(right.agreeing > right.reversed, "{right:?}");
        assert!(wrong.reversed > wrong.agreeing, "{wrong:?}");
    }
}
