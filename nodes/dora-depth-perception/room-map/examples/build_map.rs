//! Builds one room map from a keyframe recording and writes a viewer page and the trajectory.
//!
//! Usage: cargo run --release -p room-map --example build_map -- <recording.arrow> <output dir>

use std::fmt::Write as _;
use std::time::Instant;
use std::{env, fs, path::Path};

use depth_perception::{find_planes, infer_room, sample_point_cloud, DepthImage, FloorFrame, PlaneSettings, RangeLimits, RoomSettings};
use glam::{Vec2, Vec3};
use rayon::prelude::*;
use room_map::fusion::VoxelMap;
use room_map::grid::voxel_downsample;
use room_map::recording::{decode_color, read_keyframes};
use room_map::placement::{place_runs, PlacementSettings};
use room_map::trajectory::{find_loop, track, LoopSettings, TrackedKeyframe, TrackingSettings};
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
    let (recording, out_dir) = (args.get(1).expect("usage: build_map <recording> <out dir>"), Path::new(args.get(2).expect("out dir")));
    fs::create_dir_all(out_dir)?;
    let total = Instant::now();

    let started = Instant::now();
    let keyframes = read_keyframes(recording)?;
    let t_read = ms(started);

    let started = Instant::now();
    let floors: Vec<FloorFrame> = keyframes.iter().map(|k| FloorFrame::from_gravity_reaction(k.gravity_reaction)).collect();
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

    let started = Instant::now();
    let settings = TrackingSettings::default();
    let tracked = track(&registration_clouds, &settings);
    let t_track = ms(started);

    let odometry: Vec<Pose> = tracked.iter().map(|t| t.pose).collect();
    let closure = find_loop(&registration_clouds, &tracked, &LoopSettings::default(), &settings);

    let started = Instant::now();
    let (placed, placements) = place_runs(&registration_clouds, &tracked, &PlacementSettings::default(), &settings.registration);
    let t_place = ms(started);
    let poses: Vec<Pose> = placed.iter().zip(&odometry).map(|(p, o)| p.unwrap_or(*o)).collect();
    // The end measured against the start again, now with the placed poses.
    let as_placed: Vec<TrackedKeyframe> = placed
        .iter()
        .zip(&tracked)
        .map(|(p, t)| TrackedKeyframe { pose: p.unwrap_or(t.pose), registration: t.registration, registered: p.is_some(), accepted: p.is_some() })
        .collect();
    let closure_after = find_loop(&registration_clouds, &as_placed, &LoopSettings::default(), &settings);

    let started = Instant::now();
    let mut map = VoxelMap::new(MAP_VOXEL_M);
    let fused: Vec<usize> = (0..keyframes.len()).filter(|&i| placed[i].is_some()).collect();
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
    let mut report = String::new();
    writeln!(report, "{} keyframes, {} registered ({} kept their motion-model guess)", keyframes.len(), registered, keyframes.len() - registered)?;
    writeln!(report, "registration fitness median {:.2}, min {:.2}; inlier RMSE median {:.1} mm", median(&fitness), fitness.iter().copied().fold(1.0, f32::min), median(&rmse) * 1e3)?;
    writeln!(report, "path {:.1} m", path_length)?;
    for (label, c) in [("odometry alone", &closure), ("after placing runs", &closure_after)] {
        match c {
            Some(c) => writeln!(report, "end against start, {label}: fitness {:.2}, error {:.1} deg yaw, {:.2} m",
                c.registration.fitness, c.yaw_drift_rad.to_degrees(), c.translation_drift_m.length())?,
            None => writeln!(report, "end against start, {label}: no match")?,
        }
    }
    let placed_runs = placements.iter().filter(|p| p.correction.is_some()).count();
    writeln!(report, "runs: {} of {} placed, {} of {} keyframes in the map", placed_runs, placements.len(), placed.iter().flatten().count(), keyframes.len())?;
    for p in &placements {
        writeln!(report, "  keyframes {:>3}..{:<3} {}", p.keyframes.start, p.keyframes.end, match (&p.correction, &p.registration) {
            (Some(c), Some(r)) => format!("placed, fitness {:.2}, turned {:.0} deg, moved {:.2} m", r.fitness, c.yaw_rad.to_degrees(), c.translation.length()),
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
    writeln!(report, "timings: read {t_read:.0} ms, clouds {t_clouds:.0} ms, tracking {t_track:.0} ms ({:.1} ms per keyframe), placing runs {t_place:.0} ms, fusion {t_fusion:.0} ms, planes+room {t_planes:.0} ms, total {:.0} ms",
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
        "registered": placed.iter().map(|p| p.is_some()).collect::<Vec<_>>(),
        "planes": fit.planes.iter().map(|p| json!({"kind": p.kind.name(), "area_m2": p.area_m2, "outline": p.outline_m.iter().map(|c| c.to_array()).collect::<Vec<_>>()})).collect::<Vec<_>>(),
        "room": room.as_ref().ok().map(|r| json!({"floor": r.floor_m, "ceiling": r.ceiling_m,
            "footprint": r.footprint_m.iter().map(|c| c.to_array()).collect::<Vec<_>>(),
            "edge_seen": r.edge_walls.iter().map(|w| w.is_some()).collect::<Vec<_>>()})),
    });
    fs::write(out_dir.join("map.html"), VIEWER.replace("__MAP_JSON__", &data.to_string()))?;
    println!("wrote {}", out_dir.join("map.html").display());
    Ok(())
}
