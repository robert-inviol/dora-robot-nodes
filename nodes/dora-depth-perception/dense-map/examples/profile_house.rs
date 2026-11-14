//! Times the dense map on a rendered 15 m x 15 m house of six rooms: integrating each 848x480 keyframe, and
//! slicing free space and topology from the whole map.
//!
//! `cargo run --release -p dense-map --example profile_house`; `OPEN_PLAN=1` leaves out the inner walls, so every
//! keyframe sees out to the range limit; `SHOW_PLAN=1` prints the rooms.

use std::time::{Duration, Instant};


use dense_map::synthetic::{house, keyframe_at, look_around, solid, wall_along_x, wall_along_y};
use dense_map::{BlockTsdf, DenseMapSettings, DenseMapper, MapOutput};
use glam::Vec2;
use perception_wire::navigation::{MapVersion, UnixNanos};
use rayon::prelude::*;

fn main() {
    let mut solids = Vec::new();
    let open_plan = std::env::var_os("OPEN_PLAN").is_some();
    solids.extend(wall_along_y(5.0, 0.0, 15.0, &[(2.0, 2.9), (11.0, 11.9)]));
    solids.extend(wall_along_y(10.0, 0.0, 15.0, &[(3.0, 3.9), (12.0, 12.9)]));
    solids.extend(wall_along_x(7.5, 0.0, 15.0, &[(2.0, 2.9), (7.0, 7.9), (12.0, 12.9)]));
    solids.push(solid(Vec2::new(1.0, 5.0), Vec2::new(3.0, 5.9), 0.8));
    solids.push(solid(Vec2::new(12.0, 2.0), Vec2::new(13.2, 3.0), 0.75));
    if open_plan {
        solids.retain(|s| s.max.z - s.min.z < 1.0);
    }
    let house = house(Vec2::new(15.0, 15.0), solids);

    let spots: Vec<Vec2> = (0..10)
        .flat_map(|i| (0..10).map(move |j| Vec2::new(0.75 + 1.5 * i as f32, 0.75 + 1.5 * j as f32)))
        .filter(|p| [5.05, 10.05].iter().all(|w| (p.x - w).abs() > 0.4) && (p.y - 7.55).abs() > 0.4)
        .collect();
    let walk = look_around(&spots, 6);
    let started = Instant::now();
    let keyframes: Vec<_> = walk
        .par_iter()
        .enumerate()
        .map(|(i, (position, heading))| keyframe_at(&house, i as u64, *position, *heading, MapVersion(1)))
        .collect();
    println!("rendered {} keyframes in {:.1} s", keyframes.len(), started.elapsed().as_secs_f32());

    let mut mapper = DenseMapper::new(DenseMapSettings::default(), BlockTsdf::new(Default::default()));
    let mut times: Vec<Duration> = Vec::new();
    for (keyframe, pose) in keyframes {
        mapper.on_keyframe(keyframe);
        let started = Instant::now();
        mapper.on_keyframe_pose(pose);
        times.push(started.elapsed());
    }
    times.sort();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    println!(
        "integration per keyframe: median {:.1} ms, 90th {:.1} ms, max {:.1} ms; {} blocks",
        ms(times[times.len() / 2]),
        ms(times[times.len() * 9 / 10]),
        ms(times[times.len() - 1]),
        mapper.field().block_count()
    );

    for round in 0..3 {
        let started = Instant::now();
        let outputs = mapper.slice_now(UnixNanos(round)).expect("a floor");
        let elapsed = started.elapsed();
        let mut summary = String::new();
        for output in &outputs {
            match output {
                MapOutput::FreeSpace(f) => summary += &format!("free space {}x{}; ", f.width, f.height),
                MapOutput::Topology(t) => {
                    summary += &format!("{} rooms, {} doorways, {} places, {} edges", t.rooms.len(), t.doorways.len(), t.places.len(), t.edges.len())
                }
            }
        }
        println!("free space and topology {round}: {:.1} ms; {summary}", ms(elapsed));
        if round == 0 && std::env::var_os("SHOW_PLAN").is_some() {
            show_plan(&outputs);
        }
    }
}

/// The plan as text, one character per 20 cm: the room's last id digit, '#' occupied, '.' free outside rooms,
/// ' ' unknown, 'D' a doorway.
fn show_plan(outputs: &[MapOutput]) {
    let (Some(MapOutput::FreeSpace(free)), Some(MapOutput::Topology(topology))) = (outputs.first(), outputs.get(1)) else { return };
    let step = 4;
    for row in (0..free.height).step_by(step).rev() {
        let line: String = (0..free.width)
            .step_by(step)
            .map(|column| {
                let centre = free.origin + (Vec2::new(column as f32, row as f32) + 0.5) * free.resolution_m;
                if topology.doorways.iter().any(|d| d.position.distance(centre) < 0.15) {
                    return 'D';
                }
                match free.cell(column, row) {
                    perception_wire::space_map::Clearance::Occupied => '#',
                    perception_wire::space_map::Clearance::Unknown => ' ',
                    perception_wire::space_map::Clearance::Free { .. } => topology
                        .rooms
                        .iter()
                        .find(|r| inside(&r.polygon, centre))
                        .map_or('.', |r| char::from_digit((r.room_id.0 % 10) as u32, 10).expect("a digit")),
                }
            })
            .collect();
        println!("{line}");
    }
}

fn inside(polygon: &[Vec2], point: Vec2) -> bool {
    let mut inside = false;
    for (a, b) in polygon.iter().zip(polygon.iter().cycle().skip(1)) {
        if (a.y > point.y) != (b.y > point.y) && point.x < a.x + (point.y - a.y) / (b.y - a.y) * (b.x - a.x) {
            inside = !inside;
        }
    }
    inside
}
