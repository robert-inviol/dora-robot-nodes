//! Fuses the depth frames of an odometry inputs recording (see dora-depth-inertial-odometry's `recording`) into the
//! dense map at one fixed pose, as dora-posed-depth would send them (every third frame, halved each way by the nearest
//! reading of each 2x2 block, rounded to 2 mm), ticking the map every 50 ms of camera time, and prints how many mesh
//! blocks went out and their bytes every ten seconds. For a recording made with the camera at rest, every send after
//! the surface settles is noise or something in the room changing.
//!
//! `cargo run --release -p dense-map --example replay_depth_at_rest -- <recording directory> [--times n] [--from s]
//! [--to s] [--settings settings.toml]` replays the frames from `--from` to `--to` seconds into the recording (all of
//! it by default) `--times` over (once by default), camera time running on, with the dense map's settings from the
//! file when one is named (as `dora-dense-map/dense-map-settings.toml`), the defaults otherwise.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::{Duration, Instant};

use dense_map::{BlockTsdf, DenseMapSettings, DenseMapper, MapOutput};
use dora_node_api::arrow::array::{AsArray, StructArray};
use dora_node_api::arrow::datatypes::{Fields, Int64Type, UInt16Type, UInt32Type, UInt64Type};
use dora_node_api::arrow::ipc::reader::StreamReader;
use glam::{Mat3, Quat, Vec3};
use perception_wire::camera::{decode_camera_calibration, StreamIntrinsics};
use perception_wire::localization::{CameraNanos, FrameId, MapToOdom, Pose4Dof};
use perception_wire::navigation::{MapVersion, UnixNanos};
use perception_wire::posed_depth::PosedDepthFrame;

const FRAME_STRIDE: u64 = 3;
const DEPTH_STEP_MM: u32 = 2;
const TICK: Duration = Duration::from_millis(50);
const REPORT_EVERY_S: f64 = 10.0;

fn read_stream(path: &Path) -> StreamReader<BufReader<File>> {
    let file = File::open(path).unwrap_or_else(|e| panic!("opening {}: {e}", path.display()));
    StreamReader::try_new(BufReader::new(file), None).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

fn halved(depth_mm: &[u16], width: usize, height: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(width / 2 * height / 2);
    for row in 0..height / 2 {
        for column in 0..width / 2 {
            let at = |r: usize, c: usize| depth_mm[(2 * row + r) * width + 2 * column + c];
            let nearest = [at(0, 0), at(0, 1), at(1, 0), at(1, 1)].into_iter().filter(|&mm| mm != 0).min().unwrap_or(0);
            let step = DEPTH_STEP_MM;
            out.push(if nearest == 0 { 0 } else { (((nearest as u32 + step / 2) / step * step).clamp(step, u16::MAX as u32 / step * step)) as u16 });
        }
    }
    out
}

#[derive(Default)]
struct Window {
    sends: usize,
    emptied: usize,
    bytes: usize,
    blocks: BTreeMap<[i32; 3], usize>,
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let directory = arguments.next().expect("the recording directory");
    let (mut times, mut from_s, mut to_s, mut settings) = (1usize, 0.0f64, f64::INFINITY, DenseMapSettings::default());
    while let Some(flag) = arguments.next() {
        let value = arguments.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--times" => times = value.parse().expect("--times, a count"),
            "--from" => from_s = value.parse().expect("--from, seconds"),
            "--to" => to_s = value.parse().expect("--to, seconds"),
            "--settings" => settings = toml::from_str(&std::fs::read_to_string(&value).expect("a readable settings file")).expect("settings"),
            other => panic!("unknown flag {other}"),
        }
    }
    settings.validate().expect("valid settings");
    let directory = Path::new(&directory);
    let calibration_batch = read_stream(&directory.join("calibration.arrows")).next().expect("a calibration").expect("a readable calibration");
    let fields: Fields = calibration_batch.schema().fields().iter().skip(1).cloned().collect();
    let calibration = decode_camera_calibration(&StructArray::try_new(fields, calibration_batch.columns()[1..].to_vec(), None).unwrap()).unwrap();
    let k = calibration.depth;
    let intrinsics = StreamIntrinsics { width: k.width / 2, height: k.height / 2, fx: k.fx / 2.0, fy: k.fy / 2.0, ppx: (k.ppx - 0.5) / 2.0, ppy: (k.ppy - 0.5) / 2.0 };

    let mut mapper = DenseMapper::new(settings.clone(), BlockTsdf::new(settings.tsdf), UnixNanos(0));
    let odom_started = CameraNanos(0);
    let zero = Pose4Dof { x_m: 0.0, y_m: 0.0, z_m: 0.0, yaw_rad: 0.0 };
    mapper.on_map_to_odom(MapToOdom { map_version: MapVersion(1), valid_from: CameraNanos(0), odom_started, correction: zero, sigma_xy_m: 0.01, sigma_z_m: 0.01, sigma_yaw_rad: 0.01 });
    // A level camera at 0.3 m looking along +y.
    let orientation = Quat::from_mat3(&Mat3::from_cols(Vec3::X, Vec3::NEG_Z, Vec3::Y));

    let start = Instant::now();
    let mut next_tick = Duration::ZERO;
    let mut window = Window::default();
    let mut window_start = 0.0f64;
    let (mut total_sends, mut total_bytes, mut frames) = (0usize, 0usize, 0usize);
    let mut first_ns = None;
    let mut recording_start_ns = None;
    let mut all_sends: BTreeMap<[i32; 3], usize> = BTreeMap::new();
    let mut offset_ns = 0i64;
    let mut last_ns = 0i64;
    println!("seconds  sends  emptied  blocks  kB  held_blocks  held_vertices");
    for _ in 0..times {
        for batch in read_stream(&directory.join("depth.arrows")) {
            let batch = batch.expect("a readable depth batch");
            let frame_id = batch.column(1).as_primitive::<UInt64Type>().value(0);
            if frame_id % FRAME_STRIDE != 0 {
                continue;
            }
            let camera_ns = batch.column(2).as_primitive::<Int64Type>().value(0);
            let width = batch.column(3).as_primitive::<UInt32Type>().value(0) as usize;
            let height = batch.column(4).as_primitive::<UInt32Type>().value(0) as usize;
            let lists = batch.column(5).as_list::<i32>();
            let values = lists.value(0);
            let depth_mm = values.as_primitive::<UInt16Type>().values();
            let recording_start = *recording_start_ns.get_or_insert(camera_ns);
            let into_recording_s = (camera_ns - recording_start) as f64 * 1e-9;
            if into_recording_s < from_s || into_recording_s > to_s {
                continue;
            }
            let first = *first_ns.get_or_insert(camera_ns);
            let time_ns = camera_ns - first + offset_ns;
            last_ns = time_ns;
            let at = Duration::from_nanos(time_ns as u64);
            while next_tick <= at {
                let (outputs, _) = mapper.on_tick_settled(start + next_tick, UnixNanos(next_tick.as_nanos() as i64)).unwrap();
                for output in outputs {
                    if let MapOutput::MeshBlocks(message) = output {
                        for block in &message.blocks {
                            let bytes = 12 + 12 * block.positions_m.len() + 12 * block.triangles.len();
                            window.sends += 1;
                            window.emptied += usize::from(block.triangles.is_empty());
                            window.bytes += bytes;
                            *window.blocks.entry(block.block.to_array()).or_default() += 1;
                            total_sends += 1;
                            *all_sends.entry(block.block.to_array()).or_default() += 1;
                            total_bytes += bytes;
                        }
                    }
                }
                let seconds = next_tick.as_secs_f64();
                if seconds - window_start >= REPORT_EVERY_S {
                    let held: Vec<_> = mapper.sent_surface().collect();
                    let vertices: usize = held.iter().map(|b| b.positions_m.len()).sum();
                    println!("{seconds:7.1} {:6} {:8} {:7} {:5} {:12} {:14}", window.sends, window.emptied, window.blocks.len(), window.bytes / 1000, held.len(), vertices);
                    window = Window::default();
                    window_start = seconds;
                }
                next_tick += TICK;
            }
            let frame = PosedDepthFrame {
                frame_id: FrameId(frame_id),
                camera_ns: CameraNanos(time_ns),
                odom_started,
                intrinsics,
                camera_position_m: Vec3::new(0.0, 0.0, 0.3),
                camera_orientation: orientation,
                depth_mm: halved(depth_mm, width, height).into(),
            };
            mapper.on_posed_depth(&frame);
            frames += 1;
        }
        offset_ns = last_ns + 100_000_000;
        first_ns = None;
    }
    let mut busiest: Vec<([i32; 3], usize)> = all_sends.into_iter().collect();
    busiest.sort_by_key(|(_, sends)| std::cmp::Reverse(*sends));
    let block_m = settings.tsdf.voxel_m * dense_map::BLOCK_EDGE_VOXELS as f32;
    for (block, sends) in busiest.iter().take(15) {
        let centre = Vec3::from_array(block.map(|b| (b as f32 + 0.5) * block_m));
        println!("block {block:?} at {centre:.2} m, {:.1} m from the camera: sent {sends} times", centre.distance(Vec3::new(0.0, 0.0, 0.3)));
    }
    let minutes = last_ns as f64 / 60e9;
    println!(
        "{frames} frames over {:.1} min: {total_sends} block sends ({:.0} a minute), {:.0} kB a minute",
        minutes,
        total_sends as f64 / minutes,
        total_bytes as f64 / 1000.0 / minutes
    );
}
