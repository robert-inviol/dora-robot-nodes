//! Posed depth frames with a D455's noise fused at rest, ten a second, through the mapper: the surface settles and
//! stops being sent, real changes are sent, and a person who walked off is carved away.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use dense_map::{BlockTsdf, DenseMapSettings, DenseMapper, MapOutput, PosedDepthArrival};
use depth_perception::synthetic::{Cuboid, SyntheticRoom};
use depth_perception::CameraIntrinsics;
use dora_node_api::arrow::buffer::ScalarBuffer;
use glam::{Mat3, Quat, Vec3};
use perception_wire::camera::StreamIntrinsics;
use perception_wire::localization::{CameraNanos, FrameId, MapToOdom, Pose4Dof};
use perception_wire::navigation::{MapVersion, UnixNanos};
use perception_wire::posed_depth::PosedDepthFrame;
use perception_wire::space_map::{MeshBlock, MeshBlocks, MeshBlocksKind, MeshSequence};

/// The D455's depth noise, one sigma, at a range of 1 m; it grows with the square of the range.
const NOISE_AT_1_M: f32 = 0.005;
const HOLES: f32 = 0.04;
const DEPTH_STEP_MM: u32 = 2;
const FRAME: Duration = Duration::from_millis(100);
const TICK: Duration = Duration::from_millis(50);

/// A 4 m wide room with its back wall `depth_m` ahead of a level camera 0.3 m above the floor at the origin, which
/// looks along +y, seen at the posed depth frames' 424x240.
fn room(depth_m: f32, objects: Vec<Cuboid>) -> SyntheticRoom {
    SyntheticRoom {
        room: Cuboid { min: Vec3::new(-2.0, -0.5, -0.3), max: Vec3::new(2.0, depth_m, 2.1) },
        objects,
        width: 424,
        height: 240,
        intrinsics: CameraIntrinsics { fx: 212.0, fy: 212.0, ppx: 211.5, ppy: 119.5 },
    }
}

/// A person standing 2 m ahead: a box 0.4 m wide, 0.3 m deep and 1.7 m tall.
fn person() -> Cuboid {
    Cuboid { min: Vec3::new(-0.2, 2.0, -0.3), max: Vec3::new(0.2, 2.3, 1.4) }
}

/// xorshift64*, so that the noise is the same on every run.
struct Noise(u64);

impl Noise {
    fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32
    }

    fn gaussian(&mut self) -> f32 {
        let (u, v) = (self.uniform().max(1e-7), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

/// The mapper fed as the node feeds it, and the surface a receiver of every mesh blocks message holds.
struct Rig {
    mapper: DenseMapper<BlockTsdf>,
    noise: Noise,
    start: Instant,
    elapsed: Duration,
    frames: u64,
    received: BTreeMap<[i32; 3], MeshBlock>,
    next_sequence: MeshSequence,
    /// Every mesh block sent, by frame: (frame, block, whether it was empty).
    sends: Vec<(u64, [i32; 3], bool)>,
}

impl Rig {
    fn new(settings: DenseMapSettings) -> Rig {
        let tsdf = settings.tsdf;
        let mut mapper = DenseMapper::new(settings, BlockTsdf::new(tsdf), UnixNanos(0));
        let zero = Pose4Dof { x_m: 0.0, y_m: 0.0, z_m: 0.0, yaw_rad: 0.0 };
        mapper.on_map_to_odom(MapToOdom { map_version: MapVersion(1), valid_from: CameraNanos(0), odom_started: CameraNanos(0), correction: zero, sigma_xy_m: 0.01, sigma_z_m: 0.01, sigma_yaw_rad: 0.01 });
        Rig { mapper, noise: Noise(0x9e37_79b9_7f4a_7c15), start: Instant::now(), elapsed: Duration::ZERO, frames: 0, received: BTreeMap::new(), next_sequence: MeshSequence(0), sends: Vec::new() }
    }

    /// A frame of `room` with noise, holes and the depth step, fused, then the ticks up to the next frame.
    fn frame(&mut self, room: &SyntheticRoom) {
        let rendered = room.render();
        let depth_mm: Vec<u16> = rendered
            .depth_m
            .iter()
            .map(|&depth_m| {
                if self.noise.uniform() < HOLES {
                    return 0;
                }
                let noisy_mm = (depth_m + NOISE_AT_1_M * depth_m * depth_m * self.noise.gaussian()) * 1000.0;
                let step = DEPTH_STEP_MM as f32;
                ((noisy_mm / step).round() * step).clamp(step, u16::MAX as f32) as u16
            })
            .collect();
        self.fuse(room, depth_mm);
    }

    fn fuse(&mut self, room: &SyntheticRoom, depth_mm: Vec<u16>) {
        let k = room.intrinsics;
        let frame = PosedDepthFrame {
            frame_id: FrameId(self.frames),
            camera_ns: CameraNanos(self.elapsed.as_nanos() as i64),
            odom_started: CameraNanos(0),
            intrinsics: StreamIntrinsics { width: room.width as u16, height: room.height as u16, fx: k.fx, fy: k.fy, ppx: k.ppx, ppy: k.ppy },
            camera_position_m: Vec3::ZERO,
            camera_orientation: Quat::from_mat3(&Mat3::from_cols(Vec3::X, Vec3::NEG_Z, Vec3::Y)),
            depth_mm: ScalarBuffer::from(depth_mm),
        };
        assert_eq!(self.mapper.on_posed_depth(&frame), PosedDepthArrival::Fused);
        let next_frame = self.elapsed + FRAME;
        while self.elapsed < next_frame {
            let (outputs, _) = self.mapper.on_tick_settled(self.start + self.elapsed, UnixNanos(self.elapsed.as_nanos() as i64)).unwrap();
            for output in outputs {
                if let MapOutput::MeshBlocks(message) = output {
                    self.receive(message);
                }
            }
            self.elapsed += TICK;
        }
        self.frames += 1;
    }

    fn receive(&mut self, message: MeshBlocks) {
        assert_eq!(message.sequence, self.next_sequence, "no gap in the sequence");
        self.next_sequence = message.sequence.next();
        assert_eq!(message.kind, MeshBlocksKind::Changes);
        for block in message.blocks {
            self.sends.push((self.frames, block.block.to_array(), block.triangles.is_empty()));
            match block.triangles.is_empty() {
                true => self.received.remove(&block.block.to_array()),
                false => self.received.insert(block.block.to_array(), block),
            };
        }
    }

    fn sends_from(&self, frame: u64) -> usize {
        self.sends.iter().filter(|(at, _, _)| *at >= frame).count()
    }

    /// Mesh vertices the receiver holds inside a box.
    fn vertices_within(&self, within: &Cuboid) -> usize {
        let inside = |p: &Vec3| p.cmpge(within.min).all() && p.cmple(within.max).all();
        self.received.values().map(|block| block.positions_m.iter().filter(|p| inside(p)).count()).sum()
    }

    /// The frames, from `from`, until the receiver holds no vertex inside the box; None if it still does.
    fn frames_until_gone(&mut self, room: &SyntheticRoom, within: &Cuboid, from: u64, limit: u64) -> Option<u64> {
        while self.frames < from + limit {
            if self.vertices_within(within) == 0 {
                return Some(self.frames - from);
            }
            self.frame(room);
        }
        None
    }

    /// The frames, from `from`, until the receiver holds a vertex inside the box; None if it never does.
    fn frames_until_seen(&mut self, room: &SyntheticRoom, within: &Cuboid, from: u64, limit: u64) -> Option<u64> {
        while self.frames < from + limit {
            if self.vertices_within(within) > 0 {
                return Some(self.frames - from);
            }
            self.frame(room);
        }
        None
    }
}

/// Where an object's surface would be meshed: its box grown by a voxel, from 15 cm above the floor so that the floor
/// under it is left out.
fn around(cuboid: &Cuboid) -> Cuboid {
    let min = cuboid.min - 0.05;
    Cuboid { min: Vec3::new(min.x, min.y, min.z.max(-0.15)), max: cuboid.max + 0.05 }
}

#[test]
fn a_static_room_with_noise_is_sent_a_few_times_a_block_then_not_at_all_and_never_flickers() {
    let mut rig = Rig::new(DenseMapSettings::default());
    let room = room(4.5, vec![Cuboid { min: Vec3::new(0.8, 1.5, -0.3), max: Vec3::new(1.4, 2.0, 0.5) }]);
    for _ in 0..600 {
        rig.frame(&room);
    }
    let blocks: std::collections::BTreeSet<[i32; 3]> = rig.sends.iter().map(|(_, block, _)| *block).collect();
    let mut per_block: BTreeMap<[i32; 3], usize> = BTreeMap::new();
    for (_, block, _) in &rig.sends {
        *per_block.entry(*block).or_default() += 1;
    }
    let most = per_block.values().max().copied().unwrap_or(0);
    println!("{} sends of {} blocks, at most {most} for one; {} in the last 30 s", rig.sends.len(), blocks.len(), rig.sends_from(300));
    assert!(blocks.len() > 100, "the room is meshed: {} blocks", blocks.len());
    assert!(rig.sends.len() <= 10 * blocks.len(), "a few sends a block while the surface settles: {} for {} blocks", rig.sends.len(), blocks.len());
    assert!(most <= 25, "no block is sent more than 25 times in a minute: {most}");
    assert!(rig.sends_from(300) <= blocks.len() / 2, "after 30 s the surface has settled: {} sends", rig.sends_from(300));
    // The first frames' noise makes surface in front of the far wall that averaging takes away again, once; a few
    // slivers of the side walls, seen at a slant, go later.
    let emptied: Vec<&(u64, [i32; 3], bool)> = rig.sends.iter().filter(|(_, _, emptied)| *emptied).collect();
    let late = emptied.iter().filter(|(at, _, _)| *at >= 50).count();
    assert!(late <= blocks.len() / 20, "{late} blocks emptied once the first 5 s have settled");
    for (at, block, _) in &emptied {
        let times = emptied.iter().filter(|(_, other, _)| other == block).count();
        assert_eq!(times, 1, "block {block:?}, emptied at frame {at}, is emptied once: it does not flicker");
    }
    let snapshot = rig.mapper.mesh_snapshot(UnixNanos(0));
    assert_eq!(snapshot.sequence, rig.next_sequence, "the snapshot is numbered next");
    let held: Vec<&MeshBlock> = rig.received.values().collect();
    assert_eq!(snapshot.blocks.iter().collect::<Vec<_>>(), held, "a snapshot is the surface as sent");
}

#[test]
fn a_person_who_walked_off_is_carved_away_while_the_room_stays() {
    for back_wall_m in [4.5, 7.0] {
        let mut rig = Rig::new(DenseMapSettings::default());
        let empty = room(back_wall_m, vec![]);
        let with_person = room(back_wall_m, vec![person()]);
        for _ in 0..300 {
            rig.frame(&with_person);
        }
        let standing = rig.vertices_within(&around(&person()));
        assert!(standing > 50, "the person who stood 30 s is in the mesh: {standing} vertices");
        let left = rig.frames;
        let gone = rig.frames_until_gone(&empty, &around(&person()), left, 300);
        println!("back wall at {back_wall_m} m: the person is gone {gone:?} frames after they left");
        assert!(gone.is_some_and(|frames| frames <= 60), "back wall at {back_wall_m} m: carved within 6 s, not {gone:?}");
        let floor = Cuboid { min: Vec3::new(-2.0, 1.0, -0.35), max: Vec3::new(2.0, 3.0, -0.25) };
        assert!(rig.vertices_within(&floor) > 100, "the floor they stood on stays");
    }
}

#[test]
fn a_thin_post_in_front_of_a_wall_stays_through_noise_and_holes() {
    let mut rig = Rig::new(DenseMapSettings::default());
    // A post 5 cm square, 1 m in front of the back wall: about 3 pixels wide at 3 m.
    let post = Cuboid { min: Vec3::new(0.3, 3.0, -0.3), max: Vec3::new(0.35, 3.05, 1.5) };
    let room = room(4.0, vec![post]);
    for _ in 0..600 {
        rig.frame(&room);
    }
    let at_post = Cuboid { min: post.min - 0.05, max: post.max + 0.05 };
    assert!(rig.vertices_within(&at_post) > 20, "the post is in the mesh after 60 s: {} vertices", rig.vertices_within(&at_post));
}

#[test]
fn depth_holes_carve_nothing() {
    let mut rig = Rig::new(DenseMapSettings::default());
    let with_person = room(4.5, vec![person()]);
    for _ in 0..100 {
        rig.frame(&with_person);
    }
    let standing = rig.vertices_within(&around(&person()));
    // Frames with no reading at all: the person is neither seen nor seen through.
    for _ in 0..300 {
        rig.fuse(&with_person, vec![0; with_person.width * with_person.height]);
    }
    assert_eq!(rig.vertices_within(&around(&person())), standing, "holes leave the surface as it was");
}

#[test]
fn something_new_is_sent_within_one_and_a_half_seconds_and_a_wall_moved_by_a_voxel_within_eight() {
    let mut rig = Rig::new(DenseMapSettings::default());
    let empty = room(4.0, vec![]);
    for _ in 0..300 {
        rig.frame(&empty);
    }
    // A box set down 2 m ahead, in space seen free for 30 s.
    let new_box = Cuboid { min: Vec3::new(-0.6, 2.0, -0.3), max: Vec3::new(-0.2, 2.4, 0.3) };
    let with_box = room(4.0, vec![new_box]);
    let from = rig.frames;
    let seen = rig.frames_until_seen(&with_box, &around(&new_box), from, 100);
    println!("a box set down is sent after {seen:?} frames");
    assert!(seen.is_some_and(|frames| frames <= 15), "the box is sent within 1.5 s, not {seen:?}");

    // A back wall 2 m ahead moves 5 cm nearer, less than a reading contradicts: the running average carries it, and
    // the vertices sent at its old place go once it has moved by the surface tolerance.
    let mut rig = Rig::new(DenseMapSettings::default());
    let near = room(2.0, vec![]);
    for _ in 0..300 {
        rig.frame(&near);
    }
    let old_wall = Cuboid { min: Vec3::new(-1.0, 1.99, 0.0), max: Vec3::new(1.0, 2.01, 1.0) };
    let moved = room(1.95, vec![]);
    let from = rig.frames;
    let gone = rig.frames_until_gone(&moved, &old_wall, from, 200);
    println!("a wall moved 5 cm is sent after {gone:?} frames");
    assert!(gone.is_some_and(|frames| frames <= 80), "the moved wall is sent within eight seconds, not {gone:?}");
}

#[test]
fn surface_seen_for_the_first_time_is_sent_by_the_next_pass() {
    let mut rig = Rig::new(DenseMapSettings::default());
    rig.frame(&room(4.0, vec![]));
    let wall = Cuboid { min: Vec3::new(-1.0, 3.9, 0.0), max: Vec3::new(1.0, 4.1, 1.0) };
    assert!(rig.vertices_within(&wall) > 0, "one frame's surface is sent within its 0.1 s");
}
