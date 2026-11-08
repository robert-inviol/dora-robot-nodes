//! Checks depth-only tracking of a recording against what its keyframe gaps say about each turn: how many
//! registered turns agree or contradict, and what rejecting the contradicting ones does to the map.
//!
//! Usage: cargo run --release -p room-map --example keyframe_gap_check -- <recording.arrow>

use std::env;
use std::time::Instant;

use depth_perception::{sample_point_cloud, DepthImage, FloorFrame, RangeLimits};
use glam::Vec3;
use rayon::prelude::*;
use room_map::grid::voxel_downsample;
use room_map::keyframe_gaps::{contradicts, gap_turns, GapTurn};
use room_map::placement::{place_runs, PlacementSettings};
use room_map::pose::wrap_angle;
use room_map::recording::read_keyframes;
use room_map::trajectory::{track, TrackedKeyframe, TrackingSettings};

const REGISTRATION_STRIDE_PX: usize = 4;
const REGISTRATION_VOXEL_M: f32 = 0.05;

fn summarise(label: &str, tracked: &[TrackedKeyframe], bounds: &[GapTurn], clouds: &[Vec<Vec3>], settings: &TrackingSettings, seconds: f64) {
    let (mut agree, mut contradict, mut unknown) = (0, 0, 0);
    for k in 1..tracked.len() {
        if !tracked[k].accepted {
            continue;
        }
        let turn = wrap_angle(tracked[k].pose.yaw_rad - tracked[k - 1].pose.yaw_rad);
        match bounds[k - 1] {
            GapTurn::Unknown => unknown += 1,
            bound if contradicts(bound, turn, &settings.gap) => contradict += 1,
            _ => agree += 1,
        }
    }
    let (placed, placements) = place_runs(clouds, tracked, &PlacementSettings::default(), &settings.registration);
    let runs_placed = placements.iter().filter(|p| p.correction.is_some()).count();
    println!(
        "{label}: {} registered, {} accepted; accepted turns agree {agree}, contradict {contradict}, unknown {unknown}; runs {runs_placed} of {} placed, {} keyframes in the map; tracking {seconds:.1} s",
        tracked.iter().filter(|t| t.registered).count(),
        tracked.iter().filter(|t| t.accepted).count(),
        placements.len(),
        placed.iter().flatten().count()
    );
    for p in &placements {
        println!("    keyframes {:>3}..{:<3} {}", p.keyframes.start, p.keyframes.end, if p.correction.is_some() { "placed" } else { "left out" });
    }
}

fn main() -> eyre::Result<()> {
    let recording = env::args().nth(1).expect("usage: keyframe_gap_check <recording>");
    let keyframes = read_keyframes(&recording)?;
    let clouds: Vec<Vec<Vec3>> = keyframes
        .par_iter()
        .map(|k| {
            let floor = FloorFrame::from_gravity_reaction(k.gravity_reaction);
            let depth_m = k.depth_m();
            let depth = DepthImage::new(k.width, k.height, &depth_m);
            voxel_downsample(&sample_point_cloud(&depth, &k.intrinsics, &floor, RangeLimits::D455, REGISTRATION_STRIDE_PX).points, REGISTRATION_VOXEL_M)
        })
        .collect();
    let settings = TrackingSettings::default();
    let bounds = gap_turns(&keyframes.iter().map(|k| k.captured_unix_ns).collect::<Vec<_>>(), &settings.gap);
    let early = bounds.iter().filter(|b| matches!(b, GapTurn::AtLeast { .. })).count();
    let full = bounds.iter().filter(|b| matches!(b, GapTurn::AtMost { .. })).count();
    println!("{} keyframe gaps: {early} early, {full} at the full interval, {} unknown", bounds.len(), bounds.len() - early - full);

    let unknown = vec![GapTurn::Unknown; bounds.len()];
    let started = Instant::now();
    let free = track(&clouds, &unknown, &settings);
    summarise("without the gap check", &free, &bounds, &clouds, &settings, started.elapsed().as_secs_f64());
    let started = Instant::now();
    let checked = track(&clouds, &bounds, &settings);
    summarise("rejecting contradicting turns", &checked, &bounds, &clouds, &settings, started.elapsed().as_secs_f64());
    Ok(())
}
