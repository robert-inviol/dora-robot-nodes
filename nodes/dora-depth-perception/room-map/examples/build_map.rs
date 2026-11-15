//! Builds one room map from a keyframe recording and writes a viewer page and the trajectory.
//!
//! Usage: cargo run --release -p room-map --example build_map -- <recording.arrow> <output dir> [<imu recording.arrow>]
//! With an imu recording the gyro gives each keyframe's turn; without one, headings are searched for.

use std::fmt::Write as _;
use std::time::Instant;
use std::{env, fs, path::Path};

use depth_perception::{find_planes, infer_room, sample_point_cloud, DepthImage, FloorFrame, PlaneSettings, RangeLimits, RoomSettings};
use glam::{Vec2, Vec3};
use rayon::prelude::*;
use room_map::fusion::VoxelMap;
use room_map::grid::voxel_downsample;
use room_map::timing_yaw_prior::{contradicts, timing_yaw_priors, TimingYawPrior};
use room_map::heading::{keyframe_turns, stationary_between, GyroBias, ImuStream, StationaryDetectionSettings};
use room_map::run_alignment::{align_runs, AlignmentYawRange, RunAlignmentSettings};
use room_map::recording::{decode_color, read_imu_for, read_keyframes};
use room_map::trajectory::{check_turn_direction, find_loop, track, track_with_gyro, HeadingSource, LoopSettings, TrackedKeyframe, TrackingSettings};
use room_map::Pose;
use serde_json::json;

/// Every fourth pixel each way is plenty for registration on 5 cm voxels.
const REGISTRATION_STRIDE_PX: usize = 4;
const REGISTRATION_VOXEL_M: f32 = 0.05;
const FUSION_STRIDE_PX: usize = 3;
const MAP_VOXEL_M: f32 = 0.025;
/// Beyond this the D455's depth error is larger than a map voxel by several times.
const FUSION_MAX_RANGE_M: f32 = 5.0;
const MIN_OBSERVATIONS: u32 = 2;
const FUSION_BATCH: usize = 16;
const VIEWER: &str = include_str!("../viewer.html");
/// With a gyro, a run's heading is already right to within the gyro's drift over the break.
const GYRO_ALIGNMENT_YAW_EXTENT_DEG: f32 = 30.0;
/// Pairs that turned at least this much show which way the gyro turns against the depth.
const TURN_CHECK_MIN_DEG: f32 = 3.0;
const TURN_CHECK_PAIRS: usize = 12;

fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1e3
}

fn b64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = (chunk[0] as u32) << 16 | (*chunk.get(1).unwrap_or(&0) as u32) << 8 | *chunk.get(2).unwrap_or(&0) as u32;
        for k in 0..4 {
            out.push(if k <= chunk.len() { TABLE[(n >> (18 - 6 * k) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

fn main() -> eyre::Result<()> {
    let args: Vec<String> = env::args().collect();
    let (recording, out_dir) = (args.get(1).expect("usage: build_map <recording> <out dir> [<imu recording>]"), Path::new(args.get(2).expect("out dir")));
    let imu_recording = args.get(3);
    fs::create_dir_all(out_dir)?;
    let total = Instant::now();

    let started = Instant::now();
    let keyframes = read_keyframes(recording)?;
    let t_read = ms(started);

    let started = Instant::now();
    let floors: Vec<FloorFrame> = keyframes.iter().map(|k| FloorFrame::from_specific_force(k.specific_force)).collect();
    let clouds: Vec<(Vec<Vec3>, f32)> = keyframes
        .par_iter()
        .zip(&floors)
        .map(|(k, floor)| {
            let depth_m = k.depth_m();
            let depth = DepthImage::new(k.width, k.height, &depth_m);
            let cloud = sample_point_cloud(&depth, &k.intrinsics, floor, RangeLimits::D455, REGISTRATION_STRIDE_PX);
            let valid_share = k.depth_mm.iter().filter(|&&d| d > 0).count() as f32 / k.depth_mm.len() as f32;
            (voxel_downsample(&cloud.points, REGISTRATION_VOXEL_M), valid_share)
        })
        .collect();
    let t_clouds = ms(started);
    let registration_clouds: Vec<Vec<Vec3>> = clouds.iter().map(|(c, _)| c.clone()).collect();

    let mut report = String::new();
    let settings = TrackingSettings::default();
    let mut alignment = RunAlignmentSettings::default();
    let (mut t_gyro, mut gyro) = (0.0, None);
    if let Some(imu_path) = imu_recording {
        let started = Instant::now();
        let imu = read_imu_for(imu_path, &keyframes)?;
        let stream = ImuStream::from_imu(&imu)?;
        let spans = stream.stationary_spans(&StationaryDetectionSettings::default());
        let bias = GyroBias::from_spans(&spans, &StationaryDetectionSettings::default())?;
        let times: Vec<i64> = imu.iter().map(|m| m.keyframe_camera_ns).collect();
        let gravity: Vec<Vec3> = keyframes.iter().map(|k| k.specific_force).collect();
        let turns = keyframe_turns(&stream, &bias, &times, &gravity)?;
        let still = stationary_between(&spans, &times);
        let check = check_turn_direction(&registration_clouds, &turns, TURN_CHECK_MIN_DEG.to_radians(), TURN_CHECK_PAIRS, &settings);
        let span_biases: Vec<String> = spans.iter().map(|s| format!("{:.4}", s.gyro_bias_rad_s.length())).collect();
        writeln!(report, "imu: {} stationary spans, {:.1} s still in all, gyro bias magnitude per span [{}] rad/s; {} keyframe pairs still",
            spans.len(), spans.iter().map(|s| s.end_ns - s.start_ns).sum::<i64>() as f64 * 1e-9, span_biases.join(", "), still.iter().filter(|s| **s).count())?;
        writeln!(report, "gyro: turned {:.0} deg in all; turn direction agrees with depth on {} of {} pairs",
            turns.iter().sum::<f32>().to_degrees(), check.agreeing, check.agreeing + check.reversed)?;
        eyre::ensure!(check.reversed <= check.agreeing, "the gyro turns the opposite way from the depth on {} of {} pairs; check the IMU axes against the camera", check.reversed, check.agreeing + check.reversed);
        alignment.yaw_range = AlignmentYawRange::NearTracked { extent_deg: GYRO_ALIGNMENT_YAW_EXTENT_DEG, weight: settings.gyro_yaw_weight, max_step_m: settings.max_step_m };
        t_gyro = ms(started);
        gyro = Some((turns, still));
    }

    let timing_priors = timing_yaw_priors(&keyframes.iter().map(|k| k.captured_unix_ns).collect::<Vec<_>>(), &settings.timing_yaw_prior);
    let started = Instant::now();
    let tracked = match &gyro {
        Some((turns, still)) => track_with_gyro(&registration_clouds, turns, still, &settings),
        // On walk-1, rejecting turns that contradict their gap split good runs and aligned fewer keyframes,
        // so without a gyro the gap check is reported, not enforced.
        None => track(&registration_clouds, &vec![TimingYawPrior::Unknown; timing_priors.len()], &settings),
    };
    let t_track = ms(started);

    let odometry: Vec<Pose> = tracked.iter().map(|t| t.pose).collect();
    let closure = find_loop(&registration_clouds, &tracked, &LoopSettings::default(), &settings);

    let started = Instant::now();
    let (aligned, alignments) = align_runs(&registration_clouds, &tracked, &alignment, &settings.registration);
    let t_align = ms(started);
    let poses: Vec<Pose> = aligned.iter().zip(&odometry).map(|(p, o)| p.unwrap_or(*o)).collect();
    // The end measured against the start again, now with the aligned poses.
    let as_aligned: Vec<TrackedKeyframe> = aligned
        .iter()
        .zip(&tracked)
        .map(|(p, t)| TrackedKeyframe { pose: p.unwrap_or(t.pose), heading: t.heading, registration: t.registration, registered: p.is_some(), accepted: p.is_some() })
        .collect();
    let closure_after = find_loop(&registration_clouds, &as_aligned, &LoopSettings::default(), &settings);

    let started = Instant::now();
    let mut map = VoxelMap::new(MAP_VOXEL_M);
    let fused: Vec<usize> = (0..keyframes.len()).filter(|&i| aligned[i].is_some()).collect();
    for batch in fused.chunks(FUSION_BATCH) {
        let points: Vec<Vec<(Vec3, [u8; 3])>> = batch
            .par_iter()
            .map(|&i| -> eyre::Result<Vec<(Vec3, [u8; 3])>> {
                let k = &keyframes[i];
                let rgb = decode_color(k)?;
                let depth_m = k.depth_m();
                let depth = DepthImage::new(k.width, k.height, &depth_m);
                let range = RangeLimits { min_m: RangeLimits::D455.min_m, max_m: FUSION_MAX_RANGE_M };
                let cloud = sample_point_cloud(&depth, &k.intrinsics, &floors[i], range, FUSION_STRIDE_PX);
                Ok(cloud
                    .points
                    .iter()
                    .zip(&cloud.pixels)
                    .map(|(p, &pixel)| {
                        let c = 3 * pixel as usize;
                        (poses[i].apply(*p), [rgb[c], rgb[c + 1], rgb[c + 2]])
                    })
                    .collect())
            })
            .collect::<eyre::Result<_>>()?;
        for frame_points in &points {
            map.add(frame_points);
        }
    }
    let map_points = map.points(MIN_OBSERVATIONS);
    let t_fusion = ms(started);

    let started = Instant::now();
    let positions: Vec<Vec3> = map_points.iter().map(|p| p.position).collect();
    let fit = find_planes(&positions, &PlaneSettings { max_planes: 16, ..PlaneSettings::default() });
    let room = infer_room(&fit.planes, &positions, &RoomSettings::default());
    let t_planes = ms(started);

    let registered = tracked.iter().filter(|t| t.registered).count();
    let fitness: Vec<f32> = tracked.iter().filter_map(|t| t.registration.map(|r| r.fitness)).collect();
    let rmse: Vec<f32> = tracked.iter().filter_map(|t| t.registration.filter(|r| r.inlier_rmse_m.is_finite()).map(|r| r.inlier_rmse_m)).collect();
    let median = |v: &[f32]| {
        let mut s = v.to_vec();
        s.sort_by(f32::total_cmp);
        s.get(s.len() / 2).copied().unwrap_or(f32::NAN)
    };
    let path_length: f32 = poses.windows(2).map(|w| (w[1].translation - w[0].translation).length()).sum();
    let by_gyro = tracked.iter().filter(|t| t.heading == HeadingSource::Gyro).count();
    let count = |source: HeadingSource| tracked.iter().filter(|t| t.heading == source).count();
    writeln!(report, "headings: {} from registration, {} carried by the gyro, {} still, {} carried over unchanged",
        count(HeadingSource::Registration), by_gyro, count(HeadingSource::Still), count(HeadingSource::CarriedOver))?;
    if let Some((turns, _)) = &gyro {
        let mut disagreement: Vec<f32> = (1..tracked.len())
            .filter(|&k| tracked[k].heading == HeadingSource::Registration)
            .map(|k| room_map::pose::wrap_angle(tracked[k].pose.yaw_rad - tracked[k - 1].pose.yaw_rad - turns[k - 1]).abs().to_degrees())
            .collect();
        disagreement.sort_by(f32::total_cmp);
        if let (Some(mid), Some(top)) = (disagreement.get(disagreement.len() / 2), disagreement.last()) {
            writeln!(report, "registered turn against gyro turn: median {mid:.2} deg, worst {top:.2} deg")?;
        }
    }
    if gyro.is_none() {
        let early = timing_priors.iter().filter(|b| matches!(b, TimingYawPrior::AtLeast { .. })).count();
        let full = timing_priors.iter().filter(|b| matches!(b, TimingYawPrior::AtMost { .. })).count();
        let (mut over, mut under, mut agree) = (0, 0, 0);
        for k in (1..tracked.len()).filter(|&k| tracked[k].accepted) {
            let turn = room_map::pose::wrap_angle(tracked[k].pose.yaw_rad - tracked[k - 1].pose.yaw_rad);
            match timing_priors[k - 1] {
                bound @ TimingYawPrior::AtMost { .. } if contradicts(bound, turn, &settings.timing_yaw_prior) => over += 1,
                bound @ TimingYawPrior::AtLeast { .. } if contradicts(bound, turn, &settings.timing_yaw_prior) => under += 1,
                TimingYawPrior::Unknown => {}
                _ => agree += 1,
            }
        }
        writeln!(report, "keyframe gaps: {early} early (turned past the selector's threshold), {full} at the full interval (turned less), {} unknown; accepted turns agree {agree}, turned more than a full-interval gap allows {over}, less than an early gap needs {under}",
            timing_priors.len() - early - full)?;
    }
    writeln!(report, "{} keyframes, {} registered ({} kept their motion-model guess)", keyframes.len(), registered, keyframes.len() - registered)?;
    writeln!(report, "registration fitness median {:.2}, min {:.2}; inlier RMSE median {:.1} mm", median(&fitness), fitness.iter().copied().fold(1.0, f32::min), median(&rmse) * 1e3)?;
    writeln!(report, "path {:.1} m", path_length)?;
    for (label, c) in [("odometry alone", &closure), ("after aligning runs", &closure_after)] {
        match c {
            Some(c) => writeln!(report, "end against start, {label}: fitness {:.2}, error {:.1} deg yaw, {:.2} m",
                c.registration.fitness, c.yaw_drift_rad.to_degrees(), c.translation_drift_m.length())?,
            None => writeln!(report, "end against start, {label}: no match")?,
        }
    }
    let aligned_runs = alignments.iter().filter(|p| p.correction.is_some()).count();
    writeln!(report, "runs: {} of {} aligned, {} of {} keyframes in the map", aligned_runs, alignments.len(), aligned.iter().flatten().count(), keyframes.len())?;
    for p in &alignments {
        writeln!(report, "  keyframes {:>3}..{:<3} {}", p.keyframes.start, p.keyframes.end, match (&p.correction, &p.registration) {
            (Some(c), Some(r)) => format!("aligned, fitness {:.2}, turned {:.0} deg, moved {:.2} m", r.fitness, c.yaw_rad.to_degrees(), c.translation.length()),
            (Some(_), None) => "anchor".to_owned(),
            (None, r) => format!("left out, best fitness {:.2}", r.map_or(0.0, |r| r.fitness)),
        })?;
    }
    writeln!(report, "map: {} voxels of {} cm, {} seen at least {} times", map.len(), MAP_VOXEL_M * 100.0, map_points.len(), MIN_OBSERVATIONS)?;
    for plane in &fit.planes {
        writeln!(report, "  {:<14} {:>5.1} m2  n=({:+.2},{:+.2},{:+.2}) d={:+.2}", plane.kind.name(), plane.area_m2, plane.normal.x, plane.normal.y, plane.normal.z, plane.offset_m)?;
    }
    match &room {
        Ok(r) => {
            let (lo, hi) = r.footprint_m.iter().fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(l, h), c| (l.min(*c), h.max(*c)));
            writeln!(report, "room {:.2} x {:.2} x {:.2} m, {} edges, {} seen", hi.x - lo.x, hi.y - lo.y, r.ceiling_m - r.floor_m, r.edge_walls.len(), r.edge_walls.iter().flatten().count())?;
        }
        Err(e) => writeln!(report, "room: {e}")?,
    }
    writeln!(report, "timings: read {t_read:.0} ms, clouds {t_clouds:.0} ms, gyro {t_gyro:.0} ms, tracking {t_track:.0} ms ({:.1} ms per keyframe), aligning runs {t_align:.0} ms, fusion {t_fusion:.0} ms, planes+room {t_planes:.0} ms, total {:.0} ms",
        t_track / (keyframes.len() - 1) as f64, ms(total))?;
    print!("{report}");

    let mut trajectory = String::from("keyframe,registered,fitness,rmse_m,odometry_yaw_deg,odometry_x,odometry_y,odometry_z,yaw_deg,x,y,z,valid_depth\n");
    for (i, t) in tracked.iter().enumerate() {
        let (f, r) = t.registration.map_or((1.0, 0.0), |r| (r.fitness, r.inlier_rmse_m));
        let (o, p) = (odometry[i], poses[i]);
        writeln!(trajectory, "{i},{},{f:.3},{r:.4},{:.2},{:.3},{:.3},{:.3},{:.2},{:.3},{:.3},{:.3},{:.3}", t.registered,
            o.yaw_rad.to_degrees(), o.translation.x, o.translation.y, o.translation.z,
            p.yaw_rad.to_degrees(), p.translation.x, p.translation.y, p.translation.z, clouds[i].1)?;
    }
    fs::write(out_dir.join("trajectory.csv"), trajectory)?;
    fs::write(out_dir.join("report.txt"), &report)?;

    let flat_positions: Vec<u8> = map_points.iter().flat_map(|p| p.position.to_array()).flat_map(f32::to_le_bytes).collect();
    let flat_colors: Vec<u8> = map_points.iter().flat_map(|p| p.color).collect();
    let data = json!({
        "title": format!("{} keyframes, {} map points", keyframes.len(), map_points.len()),
        "report": report,
        "positions_b64": b64(&flat_positions),
        "colors_b64": b64(&flat_colors),
        "path": poses.iter().map(|p| p.translation.to_array()).collect::<Vec<_>>(),
        "headings": poses.iter().map(|p| p.yaw_rad).collect::<Vec<_>>(),
        "registered": aligned.iter().map(|p| p.is_some()).collect::<Vec<_>>(),
        "planes": fit.planes.iter().map(|p| json!({"kind": p.kind.name(), "area_m2": p.area_m2, "outline": p.outline_m.iter().map(|c| c.to_array()).collect::<Vec<_>>()})).collect::<Vec<_>>(),
        "room": room.as_ref().ok().map(|r| json!({"floor": r.floor_m, "ceiling": r.ceiling_m,
            "footprint": r.footprint_m.iter().map(|c| c.to_array()).collect::<Vec<_>>(),
            "edge_seen": r.edge_walls.iter().map(|w| w.is_some()).collect::<Vec<_>>()})),
    });
    fs::write(out_dir.join("map.html"), VIEWER.replace("__MAP_JSON__", &data.to_string()))?;
    println!("wrote {}", out_dir.join("map.html").display());
    Ok(())
}
