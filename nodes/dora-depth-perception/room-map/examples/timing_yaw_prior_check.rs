//! Checks depth-only tracking of a recording against what its keyframe gaps say about each turn: how many
//! registered turns agree or contradict, and what rejecting the contradicting ones does to the map.
//!
//! Usage: cargo run --release -p room-map --example timing_yaw_prior_check -- <recording.arrow>

use std::env;
use std::time::Instant;

use depth_perception::{sample_point_cloud, DepthImage, FloorFrame, RangeLimits};
use glam::Vec3;
use rayon::prelude::*;
use room_map::grid::voxel_downsample;
use room_map::timing_yaw_prior::{contradicts, timing_yaw_priors, TimingYawPrior};
use room_map::run_alignment::{align_runs, RunAlignmentSettings};
use room_map::pose::wrap_angle;
use room_map::recording::read_keyframes;
use room_map::trajectory::{track, TrackedKeyframe, TrackingSettings};

const REGISTRATION_STRIDE_PX: usize = 4;
const REGISTRATION_VOXEL_M: f32 = 0.05;

fn summarise(label: &str, tracked: &[TrackedKeyframe], bounds: &[TimingYawPrior], clouds: &[Vec<Vec3>], settings: &TrackingSettings, seconds: f64) {
    let (mut agree, mut contradict, mut unknown) = (0, 0, 0);
    for k in 1..tracked.len() {
        if !tracked[k].accepted {
            continue;
        }
        let turn = wrap_angle(tracked[k].pose.yaw_rad - tracked[k - 1].pose.yaw_rad);
        match bounds[k - 1] {
            TimingYawPrior::Unknown => unknown += 1,
            bound if contradicts(bound, turn, &settings.timing_yaw_prior) => contradict += 1,
            _ => agree += 1,
        }
    }
    let (aligned, alignments) = align_runs(clouds, tracked, &RunAlignmentSettings::default(), &settings.registration);
    let runs_aligned = alignments.iter().filter(|p| p.correction.is_some()).count();
    println!(
        "{label}: {} registered, {} accepted; accepted turns agree {agree}, contradict {contradict}, unknown {unknown}; runs {runs_aligned} of {} aligned, {} keyframes in the map; tracking {seconds:.1} s",
        tracked.iter().filter(|t| t.registered).count(),
        tracked.iter().filter(|t| t.accepted).count(),
        alignments.len(),
        aligned.iter().flatten().count()
    );
    for p in &alignments {
        println!("    keyframes {:>3}..{:<3} {}", p.keyframes.start, p.keyframes.end, if p.correction.is_some() { "aligned" } else { "left out" });
    }
}

fn main() -> eyre::Result<()> {
    let recording = env::args().nth(1).expect("usage: timing_yaw_prior_check <recording>");
    let keyframes = read_keyframes(&recording)?;
    let clouds: Vec<Vec<Vec3>> = keyframes
        .par_iter()
        .map(|k| {
            let floor = FloorFrame::from_specific_force(k.specific_force);
            let depth_m = k.depth_m();
            let depth = DepthImage::new(k.width, k.height, &depth_m);
            voxel_downsample(&sample_point_cloud(&depth, &k.intrinsics, &floor, RangeLimits::D455, REGISTRATION_STRIDE_PX).points, REGISTRATION_VOXEL_M)
        })
        .collect();
    let settings = TrackingSettings::default();
    let bounds = timing_yaw_priors(&keyframes.iter().map(|k| k.captured_unix_ns).collect::<Vec<_>>(), &settings.timing_yaw_prior);
    let early = bounds.iter().filter(|b| matches!(b, TimingYawPrior::AtLeast { .. })).count();
    let full = bounds.iter().filter(|b| matches!(b, TimingYawPrior::AtMost { .. })).count();
    println!("{} keyframe gaps: {early} early, {full} at the full interval, {} unknown", bounds.len(), bounds.len() - early - full);

    let unknown = vec![TimingYawPrior::Unknown; bounds.len()];
    let started = Instant::now();
    let free = track(&clouds, &unknown, &settings);
    summarise("without the gap check", &free, &bounds, &clouds, &settings, started.elapsed().as_secs_f64());
    let started = Instant::now();
    let checked = track(&clouds, &bounds, &settings);
    summarise("rejecting contradicting turns", &checked, &bounds, &clouds, &settings, started.elapsed().as_secs_f64());
    Ok(())
}
