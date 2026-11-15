use std::time::Instant;

use depth_perception::synthetic::SyntheticRoom;
use glam::Vec3;
use map_localizer::cloud::{registration_cloud, CloudSettings};
use map_localizer::gyro::{GyroHeading, GyroSettings, KeyframeTiming};
use depth_inertial_odometry::odometry::OdometrySettings;
use map_localizer::synthetic::{ellipse_poses, furnished_room, render_still, render_walk, turning_poses, SyntheticWalk};
use map_localizer::{HeadingWithoutGyro, KeyframeEstimate, KeyframeSelection, KeyframeSelectionSettings, Localizer, LocalizerInput, LocalizerSettings};
use perception_wire::localization::{CameraNanos, FrameId, Tracking};
use perception_wire::navigation::UnixNanos;
use room_map::pose::wrap_angle;
use room_map::Pose;

struct Run {
    estimates: Vec<KeyframeEstimate>,
    localizer: Localizer,
    mean_ms: f64,
    worst_ms: f64,
}

/// Every keyframe through the gyro and the localizer, as the node does it.
fn localize(walk: &SyntheticWalk, settings: LocalizerSettings) -> Run {
    let mut gyro = GyroHeading::new(GyroSettings::default());
    let mut localizer = Localizer::new(settings);
    let mut previous: Option<KeyframeTiming> = None;
    let mut estimates = Vec::new();
    let mut times_ms = Vec::new();
    for (keyframe, imu) in walk.keyframes.iter().zip(&walk.imu) {
        gyro.add(imu).unwrap();
        let started = Instant::now();
        let timing = KeyframeTiming { camera_time: CameraNanos(imu.keyframe_camera_ns), specific_force: keyframe.specific_force };
        let heading = gyro.keyframe_heading(previous, timing, HeadingWithoutGyro::Refuse);
        assert_eq!(heading.gyro_problem, None, "keyframe {}", keyframe.frame_id);
        previous = Some(timing);
        let cloud = registration_cloud(keyframe, &CloudSettings::default());
        estimates.push(localizer.localize(LocalizerInput {
            frame_id: FrameId(keyframe.frame_id),
            captured: UnixNanos(keyframe.captured_unix_ns),
            cloud,
            yaw_prior: heading.yaw_prior,
        }));
        times_ms.push(started.elapsed().as_secs_f64() * 1e3);
    }
    let mean_ms = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let worst_ms = times_ms.iter().copied().fold(0.0, f64::max);
    Run { estimates, localizer, mean_ms, worst_ms }
}

fn error(a: &Pose, b: &Pose) -> (f32, f32) {
    ((a.translation - b.translation).length(), wrap_angle(a.yaw_rad - b.yaw_rad).abs().to_degrees())
}

fn worst(pairs: impl Iterator<Item = (f32, f32)>) -> (f32, f32) {
    pairs.fold((0.0f32, 0.0f32), |a, e| (a.0.max(e.0), a.1.max(e.1)))
}

/// Every keyframe registered joins the map.
fn selecting_every_keyframe() -> LocalizerSettings {
    LocalizerSettings { keyframe_selection: KeyframeSelectionSettings { min_translation_m: 0.0, min_rotation_deg: 0.0 }, ..LocalizerSettings::default() }
}

#[test]
fn two_laps_of_the_living_room_track_to_a_centimetre_and_close_loops_on_the_second() {
    let walk = render_walk(&SyntheticRoom::living_room(), &ellipse_poses(24, 2), 0..0, |_| Vec3::new(0.002, -0.003, 0.002));
    // The second lap repeats the first's poses exactly; with every keyframe in the map it closes loops on the first.
    let run = localize(&walk, selecting_every_keyframe());
    assert!(run.estimates.iter().all(|e| e.tracking == Tracking::Tracking), "every keyframe tracks");
    // The final poses, after every optimization.
    let final_worst = worst(walk.keyframes.iter().zip(&walk.truth).map(|(k, t)| error(&run.localizer.keyframe_pose(FrameId(k.frame_id)).unwrap(), t)));
    let as_sent = worst(run.estimates.iter().zip(&walk.truth).map(|(e, t)| error(&e.pose, t)));
    let loops = run.estimates.iter().filter(|e| e.loop_closed).count();
    println!(
        "{} keyframes, {loops} loop closures, map version {}; worst error as sent {:.4} m {:.3} deg, final {:.4} m {:.3} deg; {:.1} ms per keyframe, worst {:.1} ms",
        walk.keyframes.len(), run.localizer.map_version().0, as_sent.0, as_sent.1, final_worst.0, final_worst.1, run.mean_ms, run.worst_ms
    );
    assert!(loops >= 3, "the second lap revisits the first: {loops} loop closures");
    assert!(final_worst.0 < 0.01 && final_worst.1 < 0.3, "{final_worst:?}");
    assert!(as_sent.0 < 0.01 && as_sent.1 < 0.3, "{as_sent:?}");
}

#[test]
fn the_gyro_carries_a_blank_turn_and_a_loop_closure_ties_the_restarted_run_back() {
    let blank = 11..16;
    let bias = |t: i64| Vec3::new(0.002, -0.003 - 0.010 * t as f32 / 25e9, 0.002);
    let walk = render_walk(&furnished_room(), &turning_poses(), blank.clone(), bias);
    let run = localize(&walk, LocalizerSettings::default());
    let states: Vec<Tracking> = run.estimates.iter().map(|e| e.tracking).collect();
    println!("{states:?}");
    for k in blank.clone() {
        assert_eq!(states[k], Tracking::Lost, "keyframe {k}");
        let heading = error(&run.estimates[k].pose, &walk.truth[k]).1;
        assert!(heading < 1.0, "the gyro keeps keyframe {k}'s heading: {heading:.2} deg off");
    }
    // The turn ends facing parts of the room no stored keyframe saw, so the first keyframe with depth restarts
    // tracking where dead reckoning puts it, and the walk back closes the loop on the start.
    assert_eq!(states[blank.end], Tracking::Restarted);
    assert_eq!(run.estimates[blank.end].selection, KeyframeSelection::Selected, "the restarting keyframe joins the map");
    assert!(run.localizer.keyframe_pose(FrameId(blank.end as u64)).is_some());
    let with_depth: Vec<usize> = (0..walk.truth.len()).filter(|k| !blank.contains(k)).collect();
    assert!(with_depth.iter().filter(|&&k| k != blank.end).all(|&k| states[k] == Tracking::Tracking), "every other keyframe with depth tracks");
    assert!(run.estimates[blank.end..].iter().any(|e| e.loop_closed), "the walk back closes a loop");
    let placed = worst(with_depth.iter().map(|&k| error(&run.localizer.keyframe_pose(FrameId(k as u64)).unwrap(), &walk.truth[k])));
    println!(
        "worst error of the keyframes with depth {:.4} m {:.3} deg, map version {}; {:.1} ms per keyframe, worst {:.1} ms",
        placed.0, placed.1, run.localizer.map_version().0, run.mean_ms, run.worst_ms
    );
    assert!(placed.0 < 0.03 && placed.1 < 1.0, "{placed:?}");
}

#[test]
fn a_second_lap_registers_against_the_first_lap_without_loop_closures() {
    let walk = render_walk(&SyntheticRoom::living_room(), &ellipse_poses(48, 2), 0..0, |_| Vec3::new(0.002, -0.003, 0.002));
    let mut settings = LocalizerSettings::default();
    settings.loops.attempt_every_keyframes = usize::MAX;
    let run = localize(&walk, settings);
    assert!(run.estimates.iter().all(|e| e.tracking == Tracking::Tracking), "every keyframe tracks");
    assert!(run.estimates.iter().all(|e| !e.loop_closed), "no loop closures are looked for");
    // On the second lap, the submap holds keyframes the first lap saw at the same place, 48 keyframes before.
    let second_lap = 60..walk.keyframes.len();
    for estimate in &run.estimates[second_lap.clone()] {
        let oldest = estimate.submap.iter().map(|f| f.0).min().expect("tracking registered against a submap");
        assert!(oldest + 40 <= estimate.frame_id.0, "keyframe {} registered only against keyframes from {oldest} on: {:?}", estimate.frame_id.0, estimate.submap);
    }
    let as_sent = worst(run.estimates.iter().zip(&walk.truth).map(|(e, t)| error(&e.pose, t)));
    println!(
        "{} keyframes, worst error as sent {:.4} m {:.3} deg; {:.1} ms per keyframe, worst {:.1} ms",
        walk.keyframes.len(), as_sent.0, as_sent.1, run.mean_ms, run.worst_ms
    );
    assert!(as_sent.0 < 0.01 && as_sent.1 < 0.3, "{as_sent:?}");
}

#[test]
fn a_repeated_lap_registers_against_the_first_and_adds_nothing_to_the_map() {
    let walk = render_walk(&SyntheticRoom::living_room(), &ellipse_poses(24, 2), 0..0, |_| Vec3::new(0.002, -0.003, 0.002));
    let run = localize(&walk, LocalizerSettings::default());
    assert!(run.estimates.iter().all(|e| e.tracking == Tracking::Tracking), "every keyframe tracks");
    let first_lap_nodes = run.estimates[..25].iter().filter(|e| e.selection == KeyframeSelection::Selected).count();
    assert_eq!(run.localizer.graph().len(), first_lap_nodes, "the second lap stands where the first stood");
    let as_sent = worst(run.estimates.iter().zip(&walk.truth).map(|(e, t)| error(&e.pose, t)));
    println!("{} keyframes, {first_lap_nodes} in the map; worst error as sent {:.4} m {:.3} deg", walk.keyframes.len(), as_sent.0, as_sent.1);
    assert!(as_sent.0 < 0.01 && as_sent.1 < 0.3, "{as_sent:?}");
}

#[test]
fn a_camera_at_rest_with_range_noise_never_goes_lost_and_adds_nothing_to_the_map() {
    let noise = OdometrySettings::d455().association.noise;
    let at = Pose { yaw_rad: 0.4, translation: Vec3::new(0.2, 0.8, 0.0) };
    let walk = render_still(&furnished_room(), at, 200, |depth_m| noise.scatter_m(depth_m), |_| Vec3::new(0.002, -0.003, 0.002));
    let run = localize(&walk, LocalizerSettings::default());
    let lost: Vec<u64> = run.estimates.iter().filter(|e| e.tracking != Tracking::Tracking).map(|e| e.frame_id.0).collect();
    let fitness: Vec<f32> = run.estimates.iter().filter_map(|e| e.registration.map(|r| r.fitness)).collect();
    let lowest = fitness.iter().copied().fold(1.0, f32::min);
    let drift = worst(run.estimates.iter().map(|e| error(&e.pose, &Pose::IDENTITY)));
    println!(
        "{} keyframes at rest: {} not tracking, lowest fitness {lowest:.3}, worst drift {:.4} m {:.3} deg; {:.1} ms per keyframe, worst {:.1} ms",
        walk.keyframes.len(), lost.len(), drift.0, drift.1, run.mean_ms, run.worst_ms
    );
    assert!(lost.is_empty(), "keyframes {lost:?} were not tracked");
    assert_eq!(run.localizer.graph().len(), 1, "the map holds only the first keyframe");
    assert!(run.estimates[1..].iter().all(|e| e.selection == KeyframeSelection::NotSelected { reference: FrameId(0) }));
    assert!(drift.0 < 0.01 && drift.1 < 0.3, "{drift:?}");
}

#[test]
fn the_same_walk_gives_the_same_poses_every_run() {
    let walk = render_walk(&SyntheticRoom::living_room(), &ellipse_poses(24, 1), 0..0, |_| Vec3::new(0.002, -0.003, 0.002));
    let poses = |run: Run| run.estimates.iter().map(|e| (e.pose, e.registration.map(|r| r.fitness))).collect::<Vec<_>>();
    let first = poses(localize(&walk, LocalizerSettings::default()));
    assert!((0..2).all(|_| poses(localize(&walk, LocalizerSettings::default())) == first), "registration sums in a fixed order");
}
