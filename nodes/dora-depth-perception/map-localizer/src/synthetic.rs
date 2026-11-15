//! Walks through a rendered room as keyframe and imu messages with their true poses, for tests and benchmarks.
//!
//! The camera is level, so a keyframe's floor frame turns with it about the vertical. The IMU runs from two
//! seconds before the first keyframe, with the camera still, so the gyro's bias can be measured. Depth is exact,
//! or for a camera at rest (`render_still`), noisy by range as a stereo camera's is.

use std::ops::Range;

use depth_perception::synthetic::{Cuboid, SyntheticRoom};
use glam::Vec3;
use perception_wire::{Imu, Keyframe};
use rayon::prelude::*;
use room_map::Pose;

pub const KEYFRAME_GAP_NS: i64 = 600_000_000;
pub const STATIONARY_START_NS: i64 = 2_000_000_000;
const GYRO_STEP_NS: i64 = 2_500_000;
const ACCEL_STEP_NS: i64 = 5_000_000;
const UNIX_START_NS: i64 = 1_762_560_000_000_000_000;
/// The camera's up direction in its own axes (x right, y down, z forward).
pub const UP_IN_CAMERA: Vec3 = Vec3::new(0.0, -1.0, 0.0);

pub struct SyntheticWalk {
    /// Each keyframe's pose relative to the first: the map frame the localizer builds.
    pub truth: Vec<Pose>,
    pub keyframes: Vec<Keyframe>,
    /// For each keyframe, the samples since the one before (for the first, since the IMU started).
    pub imu: Vec<Imu>,
}

/// The living room with furniture along every wall, so that no view is a bare wall.
pub fn furnished_room() -> SyntheticRoom {
    let cuboid = |min: [f32; 3], max: [f32; 3]| Cuboid { min: Vec3::from_array(min), max: Vec3::from_array(max) };
    let mut room = SyntheticRoom::living_room();
    room.objects.extend([
        cuboid([-1.5, 2.6, -1.0], [-0.9, 3.4, -0.2]),
        cuboid([1.4, 0.2, -1.0], [2.0, 0.9, 0.1]),
        cuboid([-0.4, -1.0, -1.0], [0.8, -0.6, -0.4]),
        cuboid([1.2, 3.5, -1.0], [1.8, 4.0, 0.6]),
        cuboid([-1.5, -0.5, -1.0], [-1.1, 0.3, 0.4]),
    ]);
    room
}

/// Laps of an ellipse, sweeping the heading, each lap ending where it started.
pub fn ellipse_poses(keyframes_per_lap: usize, laps: usize) -> Vec<Pose> {
    (0..=keyframes_per_lap * laps)
        .map(|k| {
            let phase = k as f32 / keyframes_per_lap as f32 * std::f32::consts::TAU;
            Pose { yaw_rad: 0.6 * phase.sin(), translation: Vec3::new(0.25 + 0.5 * phase.cos() - 0.5, 1.0 + 0.4 * phase.sin(), 0.0) }
        })
        .collect()
}

/// A short walk, a 200 degree turn while drifting 30 cm sideways, a pause, a walk back turning to face the way
/// it started, and a pause at the end.
pub fn turning_poses() -> Vec<Pose> {
    let lerp = |a: f32, b: f32, s: f32| a + (b - a) * s;
    let turned = 0.3 + 200f32.to_radians();
    (0..38)
        .map(|k| match k {
            0..=8 => {
                let s = k as f32 / 8.0;
                Pose { yaw_rad: lerp(0.0, 0.3, s), translation: Vec3::new(lerp(0.0, 0.3, s), lerp(1.0, 1.4, s), 0.0) }
            }
            9..=20 => {
                let s = (k - 8) as f32 / 12.0;
                Pose { yaw_rad: 0.3 + 200f32.to_radians() * s, translation: Vec3::new(lerp(0.3, 0.6, s), 1.4, 0.0) }
            }
            21..=24 => Pose { yaw_rad: turned, translation: Vec3::new(0.6, 1.4, 0.0) },
            25..=33 => {
                let s = (k - 24) as f32 / 9.0;
                Pose { yaw_rad: turned + 160f32.to_radians() * s, translation: Vec3::new(lerp(0.6, 0.0, s), lerp(1.4, 1.0, s), 0.0) }
            }
            _ => Pose { yaw_rad: turned + 160f32.to_radians(), translation: Vec3::new(0.0, 1.0, 0.0) },
        })
        .collect()
}

pub fn keyframe_camera_ns(index: usize) -> i64 {
    STATIONARY_START_NS + index as i64 * KEYFRAME_GAP_NS
}

fn depth_mm(depth_m: f32) -> u16 {
    (depth_m * 1000.0).round().clamp(0.0, u16::MAX as f32) as u16
}

fn keyframe(room: &SyntheticRoom, index: usize, specific_force: Vec3, depth_mm: Vec<u16>) -> Keyframe {
    Keyframe {
        frame_id: index as u64,
        captured_unix_ns: UNIX_START_NS + keyframe_camera_ns(index),
        width: room.width,
        height: room.height,
        intrinsics: room.intrinsics,
        specific_force,
        depth_mm: depth_mm.into(),
        color_jpeg: Vec::new(),
    }
}

/// Renders a walk. Keyframes in `blank` have no depth, as when the camera faces glass. The gyro reads the turn
/// between keyframes spread evenly over the gap, plus `gyro_bias` at each camera time and a little noise.
pub fn render_walk(room: &SyntheticRoom, poses: &[Pose], blank: Range<usize>, gyro_bias: impl Fn(i64) -> Vec3) -> SyntheticWalk {
    let keyframes: Vec<Keyframe> = poses
        .par_iter()
        .enumerate()
        .map(|(index, pose)| {
            let frame = room.render_from(pose.translation, pose.yaw_rad);
            let depth: Vec<u16> = frame.depth_m.iter().map(|&d| if blank.contains(&index) { 0 } else { depth_mm(d) }).collect();
            keyframe(room, index, frame.accel_camera, depth)
        })
        .collect();
    let imu = imu_messages(poses, gyro_bias);
    let truth = poses.iter().map(|p| p.relative_to(&poses[0])).collect();
    SyntheticWalk { truth, keyframes, imu }
}

/// A camera at rest at `pose` for `keyframes` keyframes. Each keyframe's depth is the room's with fresh noise of
/// `depth_sigma_m(depth)` standard deviation at each pixel; the gyro reads `gyro_bias` and a little noise.
pub fn render_still(room: &SyntheticRoom, pose: Pose, keyframes: usize, depth_sigma_m: impl Fn(f32) -> f32 + Sync, gyro_bias: impl Fn(i64) -> Vec3) -> SyntheticWalk {
    let frame = room.render_from(pose.translation, pose.yaw_rad);
    let rendered: Vec<Keyframe> = (0..keyframes)
        .into_par_iter()
        .map(|index| {
            let mut normal = StandardNormal::seeded(index as u64);
            let depth: Vec<u16> =
                frame.depth_m.iter().map(|&d| if d > 0.0 { depth_mm(d + depth_sigma_m(d) * normal.next()) } else { 0 }).collect();
            keyframe(room, index, frame.accel_camera, depth)
        })
        .collect();
    let poses = vec![pose; keyframes];
    let imu = imu_messages(&poses, gyro_bias);
    SyntheticWalk { truth: vec![Pose::IDENTITY; keyframes], keyframes: rendered, imu }
}

/// Standard normal draws from a xorshift generator, by the Box-Muller transform.
struct StandardNormal {
    state: u64,
}

impl StandardNormal {
    fn seeded(seed: u64) -> StandardNormal {
        StandardNormal { state: (0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0xBF58_476D_1CE4_E5B9)) | 1 }
    }

    fn uniform(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        ((self.state >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }

    fn next(&mut self) -> f32 {
        let (a, b) = (self.uniform(), self.uniform());
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }
}

/// One imu message per keyframe: the samples since the keyframe before, or for the first, since the IMU started.
fn imu_messages(poses: &[Pose], gyro_bias: impl Fn(i64) -> Vec3) -> Vec<Imu> {
    let mut state: u64 = 0x2545F4914F6CDD1D;
    let mut noise = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f32 / (1u64 << 53) as f32 - 0.5) * 0.007
    };
    let segment = |t: i64| (t >= STATIONARY_START_NS).then(|| ((t - STATIONARY_START_NS) / KEYFRAME_GAP_NS) as usize).filter(|s| s + 1 < poses.len());
    let rate = |t: i64| segment(t).map_or(0.0, |s| (poses[s + 1].yaw_rad - poses[s].yaw_rad) / (KEYFRAME_GAP_NS as f32 * 1e-9));
    let moving = |t: i64| segment(t).is_some_and(|s| poses[s + 1] != poses[s]);
    (0..poses.len())
        .map(|index| {
            let to = keyframe_camera_ns(index);
            let from = if index == 0 { 0 } else { keyframe_camera_ns(index - 1) };
            let gyro_camera_ns: Vec<i64> = (from / GYRO_STEP_NS + 1..=to / GYRO_STEP_NS).map(|k| k * GYRO_STEP_NS).collect();
            let accel_camera_ns: Vec<i64> = (from / ACCEL_STEP_NS + 1..=to / ACCEL_STEP_NS).map(|k| k * ACCEL_STEP_NS).collect();
            let gyro_rad_s = gyro_camera_ns.iter().map(|&t| UP_IN_CAMERA * rate(t) + gyro_bias(t) + Vec3::new(noise(), noise(), noise())).collect();
            let accel_m_s2 = accel_camera_ns
                .iter()
                .map(|&t| {
                    let s = t as f32 * 1e-9;
                    let shake = if moving(t) { Vec3::new((s * 23.0).sin(), (s * 17.0).cos(), (s * 11.0).sin()) * 0.6 } else { Vec3::ZERO };
                    UP_IN_CAMERA * 9.7 + shake + Vec3::new(noise(), noise(), noise())
                })
                .collect();
            Imu { frame_id: index as u64, keyframe_camera_ns: to, accel_camera_ns, accel_m_s2, gyro_camera_ns, gyro_rad_s }
        })
        .collect()
}
