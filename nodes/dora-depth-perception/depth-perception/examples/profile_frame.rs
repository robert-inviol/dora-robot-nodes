//! Times every stage on a recorded frame and prints what it found.
//!
//! Usage: cargo run --release --example profile_frame -- <frame dir> [repeats]
//! A frame dir holds frame.txt ("width height fx fy ppx ppy accel_x accel_y accel_z segment_count"),
//! depth_m.f32 (little-endian, row major) and segment_ids.i16 (-1 where no segment).

use std::time::{Duration, Instant};
use std::{env, fs, path::Path};

use depth_perception::*;
use glam::Vec3;
use rayon::prelude::*;

fn read_le<T, const N: usize>(path: &Path, decode: fn([u8; N]) -> T) -> Vec<T> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    bytes.chunks_exact(N).map(|c| decode(c.try_into().unwrap())).collect()
}

fn median_time<T>(repeats: usize, mut run: impl FnMut() -> T) -> (Duration, T) {
    let mut times = Vec::with_capacity(repeats);
    let mut last = None;
    for _ in 0..repeats {
        let started = Instant::now();
        last = Some(run());
        times.push(started.elapsed());
    }
    times.sort();
    (times[repeats / 2], last.unwrap())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let dir = Path::new(args.get(1).expect("usage: profile_frame <frame dir> [repeats]"));
    let repeats: usize = args.get(2).map_or(20, |r| r.parse().expect("repeats"));
    let header = fs::read_to_string(dir.join("frame.txt")).expect("frame.txt");
    let fields: Vec<f64> = header.split_whitespace().map(|f| f.parse().expect("number in frame.txt")).collect();
    let (width, height) = (fields[0] as usize, fields[1] as usize);
    let intrinsics = CameraIntrinsics { fx: fields[2] as f32, fy: fields[3] as f32, ppx: fields[4] as f32, ppy: fields[5] as f32 };
    let accel = Vec3::new(fields[6] as f32, fields[7] as f32, fields[8] as f32);
    let segment_count = fields[9] as usize;
    let depth_m = read_le(&dir.join("depth_m.f32"), f32::from_le_bytes);
    let segment_ids = read_le(&dir.join("segment_ids.i16"), i16::from_le_bytes);
    let depth = DepthImage::new(width, height, &depth_m);
    let masks: Vec<Vec<bool>> = (0..segment_count).map(|s| segment_ids.iter().map(|&id| id == s as i16).collect()).collect();

    let floor = FloorFrame::from_gravity_reaction(accel);
    let (t_cloud, cloud) = median_time(repeats, || sample_point_cloud(&depth, &intrinsics, &floor, RangeLimits::D455, 2));
    let (t_planes, fit) = median_time(repeats, || find_planes(&cloud.points, &PlaneSettings::default()));
    let (t_room, room) = median_time(repeats, || infer_room(&fit.planes, &cloud.points, &RoomSettings::default()));
    let footprint = |pixels: &Vec<bool>| {
        let mask = ObjectMask { width, height, pixels };
        object_footprint(&mask, &depth, &intrinsics, &floor, RangeLimits::D455, &Default::default())
    };
    let (t_serial, _) = median_time(repeats, || masks.iter().map(footprint).collect::<Vec<_>>());
    let (t_parallel, footprints) = median_time(repeats, || masks.par_iter().map(footprint).collect::<Vec<_>>());

    println!("{}: {} points, {} planes, {} segments ({} with footprints)",
        dir.display(), cloud.points.len(), fit.planes.len(), segment_count, footprints.iter().flatten().count());
    println!("  point cloud        {:>8.2} ms", t_cloud.as_secs_f64() * 1e3);
    println!("  planes             {:>8.2} ms", t_planes.as_secs_f64() * 1e3);
    println!("  room               {:>8.2} ms", t_room.as_secs_f64() * 1e3);
    println!("  footprints serial  {:>8.2} ms", t_serial.as_secs_f64() * 1e3);
    println!("  footprints rayon   {:>8.2} ms", t_parallel.as_secs_f64() * 1e3);
    for plane in &fit.planes {
        println!("    {:<14} {:>5.1} m2  n=({:+.2},{:+.2},{:+.2}) d={:+.2}", plane.kind.name(), plane.area_m2,
            plane.normal.x, plane.normal.y, plane.normal.z, plane.offset_m);
    }
    match room {
        Ok(room) => {
            let (lo, hi) = room.footprint_m.iter().fold((glam::Vec2::INFINITY, glam::Vec2::NEG_INFINITY), |(l, h), c| (l.min(*c), h.max(*c)));
            println!("    room {:.2} x {:.2} x {:.2} m, {} edges ({} seen)", hi.x - lo.x, hi.y - lo.y, room.ceiling_m - room.floor_m,
                room.edge_walls.len(), room.edge_walls.iter().flatten().count());
        }
        Err(error) => println!("    room: {error}"),
    }
    let summary: Vec<String> = footprints.iter().enumerate().filter_map(|(s, f)| f.as_ref().map(|f|
        format!("[{s},{:.3},{:.3},{:.3},{:.3},{:.3}]", f.distance_m, f.footprint_m.0, f.footprint_m.1, f.height_m, f.top_above_camera_m))).collect();
    fs::write(dir.join("rust_footprints.json"), format!("[{}]", summary.join(","))).expect("writing rust_footprints.json");
}
