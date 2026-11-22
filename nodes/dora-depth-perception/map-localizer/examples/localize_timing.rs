//! Times every stage of localizing the synthetic walks, keyframe by keyframe, and reports their accuracy.
//!
//! Usage: cargo run --release -p map-localizer --example localize_timing -- [<poses.csv>] [<gap_ms>]
//! Runs two laps of the living room with every keyframe selected (the loop-closure walk), two laps with the
//! default keyframe selection, and a camera at rest for 200 keyframes with range noise. For each it prints the
//! median and worst time of every stage, the process CPU time per keyframe (all threads), and the accuracy. With
//! `<poses.csv>`, every keyframe's pose as sent and its final pose are written there, for comparing two builds.
//! With `<gap_ms>`, the example sleeps that long between keyframes, as the node idles between keyframes.

use std::fmt::Write as _;
use std::time::{Duration, Instant};

use depth_inertial_odometry::odometry::OdometrySettings;
use depth_perception::synthetic::SyntheticRoom;
use glam::Vec3;
use map_localizer::cloud::{registration_cloud, CloudSettings};
use map_localizer::gyro::{GyroHeading, GyroSettings, KeyframeTiming};
use map_localizer::synthetic::{ellipse_poses, furnished_room, render_still, render_walk, SyntheticWalk};
use map_localizer::{HeadingWithoutGyro, KeyframeEstimate, KeyframeSelectionSettings, Localizer, LocalizerInput, LocalizerSettings, StageTimings};
use perception_wire::localization::{CameraNanos, FrameId, Tracking};
use perception_wire::navigation::UnixNanos;
use room_map::pose::wrap_angle;
use room_map::Pose;

/// The process's CPU time so far, user and system, over all its threads.
fn process_cpu() -> Duration {
    let stat = std::fs::read_to_string("/proc/self/stat").expect("/proc/self/stat is readable on Linux");
    let after_name = &stat[stat.rfind(')').expect("stat names the command in parentheses") + 2..];
    let fields: Vec<&str> = after_name.split_whitespace().collect();
    // utime and stime are fields 14 and 15 of stat, 12 and 13 after the name; in clock ticks of 1/100 s.
    let ticks: u64 = fields[11].parse::<u64>().expect("utime") + fields[12].parse::<u64>().expect("stime");
    Duration::from_millis(ticks * 10)
}

struct Timed {
    estimate: KeyframeEstimate,
    cloud: Duration,
    cloud_points: usize,
    total: Duration,
}

/// A stage's name and how to read its time from one keyframe.
type Stage = (&'static str, fn(&Timed) -> Duration);

struct Run {
    name: &'static str,
    timed: Vec<Timed>,
    localizer: Localizer,
    cpu: Duration,
}

fn localize(name: &'static str, walk: &SyntheticWalk, settings: LocalizerSettings, gap: Duration) -> Run {
    let mut gyro = GyroHeading::new(GyroSettings::default());
    let mut localizer = Localizer::new(settings);
    let mut previous: Option<KeyframeTiming> = None;
    let mut timed = Vec::new();
    let cpu_before = process_cpu();
    for (keyframe, imu) in walk.keyframes.iter().zip(&walk.imu) {
        gyro.add(imu).expect("the synthetic imu messages are in order");
        let started = Instant::now();
        let timing = KeyframeTiming { camera_time: CameraNanos(imu.keyframe_camera_ns), specific_force: keyframe.specific_force };
        let heading = gyro.keyframe_heading(previous, timing, HeadingWithoutGyro::Refuse);
        previous = Some(timing);
        let cloud = registration_cloud(keyframe, &CloudSettings::default());
        let cloud_time = started.elapsed();
        let cloud_points = cloud.len();
        let estimate = localizer.localize(LocalizerInput {
            frame_id: FrameId(keyframe.frame_id),
            captured: UnixNanos(keyframe.captured_unix_ns),
            cloud,
            yaw_prior: heading.yaw_prior,
            odometry_step: None,
        });
        timed.push(Timed { estimate, cloud: cloud_time, cloud_points, total: started.elapsed() });
        std::thread::sleep(gap);
    }
    Run { name, timed, localizer, cpu: process_cpu() - cpu_before }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn median_and_worst(mut values: Vec<f64>) -> (f64, f64) {
    values.sort_by(f64::total_cmp);
    (values[values.len() / 2], values.last().copied().unwrap_or(0.0))
}

fn report(run: &Run, truth: &[Pose]) {
    let stages: [Stage; 9] = [
        ("cloud", |t| t.cloud),
        ("tracking_target", |t| t.estimate.timings.tracking_target),
        ("tracking_seed_trials", |t| t.estimate.timings.tracking_seed_trials),
        ("tracking_refinement", |t| t.estimate.timings.tracking_refinement),
        ("loop_closure", |t| t.estimate.timings.loop_closure),
        ("relocalization", |t| t.estimate.timings.relocalization),
        ("pose_graph_optimization", |t| t.estimate.timings.pose_graph_optimization),
        ("unaccounted", |t| t.total - t.cloud - sum(&t.estimate.timings)),
        ("total", |t| t.total),
    ];
    let keyframes = run.timed.len();
    println!("== {}: {keyframes} keyframes, {} in the map", run.name, run.localizer.graph().len());
    for (name, stage) in stages {
        let (median, worst) = median_and_worst(run.timed.iter().map(|t| ms(stage(t))).collect());
        println!("  {name:<24} median {median:8.2} ms  worst {worst:8.2} ms");
    }
    let (points, _) = median_and_worst(run.timed.iter().map(|t| t.cloud_points as f64).collect());
    println!("  cloud points (median) {points:.0}");
    println!("  process CPU {:.1} ms per keyframe", ms(run.cpu) / keyframes as f64);
    let error = |a: &Pose, b: &Pose| ((a.translation - b.translation).length(), wrap_angle(a.yaw_rad - b.yaw_rad).abs().to_degrees());
    let worst = |pairs: &mut dyn Iterator<Item = (f32, f32)>| pairs.fold((0.0f32, 0.0f32), |a, e| (a.0.max(e.0), a.1.max(e.1)));
    let as_sent = worst(&mut run.timed.iter().zip(truth).map(|(t, truth)| error(&t.estimate.pose, truth)));
    let placed = |t: &Timed| run.localizer.keyframe_pose(t.estimate.frame_id);
    let final_poses = worst(&mut run.timed.iter().zip(truth).filter_map(|(t, truth)| placed(t).map(|p| error(&p, truth))));
    let loops = run.timed.iter().filter(|t| t.estimate.loop_closed).count();
    let not_tracking = run.timed.iter().filter(|t| t.estimate.tracking != Tracking::Tracking).count();
    println!(
        "  worst error as sent {:.4} m {:.3} deg, final {:.4} m {:.3} deg; {loops} loop closures; {not_tracking} not tracking",
        as_sent.0, as_sent.1, final_poses.0, final_poses.1
    );
}

fn sum(t: &StageTimings) -> Duration {
    t.tracking_target + t.tracking_seed_trials + t.tracking_refinement + t.loop_closure + t.relocalization + t.pose_graph_optimization
}

fn poses_csv(run: &Run, out: &mut String) {
    for t in &run.timed {
        let sent = t.estimate.pose;
        let placed = run.localizer.keyframe_pose(t.estimate.frame_id).unwrap_or(sent);
        writeln!(
            out,
            "{},{},{},{},{},{},{},{},{},{}",
            run.name, t.estimate.frame_id.0, sent.translation.x, sent.translation.y, sent.translation.z, sent.yaw_rad,
            placed.translation.x, placed.translation.y, placed.translation.z, placed.yaw_rad
        )
        .expect("writing to a string");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let gap = Duration::from_millis(args.get(2).map_or(0, |g| g.parse().expect("<gap_ms> is a whole number of milliseconds")));
    let bias = |_| Vec3::new(0.002, -0.003, 0.002);
    let laps = render_walk(&SyntheticRoom::living_room(), &ellipse_poses(24, 2), 0..0, bias);
    let noise = OdometrySettings::d455().association.noise;
    let at = Pose { yaw_rad: 0.4, translation: Vec3::new(0.2, 0.8, 0.0) };
    let still = render_still(&furnished_room(), at, 200, |depth_m| noise.scatter_m(depth_m), bias);
    let every = LocalizerSettings {
        keyframe_selection: KeyframeSelectionSettings { min_translation_m: 0.0, min_rotation_deg: 0.0 },
        ..LocalizerSettings::default()
    };
    let runs = [
        (localize("laps_every_keyframe_selected", &laps, every, gap), &laps.truth),
        (localize("laps", &laps, LocalizerSettings::default(), gap), &laps.truth),
        (localize("still", &still, LocalizerSettings::default(), gap), &still.truth),
    ];
    let mut csv = String::from("run,frame_id,sent_x,sent_y,sent_z,sent_yaw,final_x,final_y,final_z,final_yaw\n");
    for (run, truth) in &runs {
        report(run, truth);
        poses_csv(run, &mut csv);
    }
    if let Some(path) = args.get(1) {
        std::fs::write(path, csv).expect("the poses file is writable");
    }
}
