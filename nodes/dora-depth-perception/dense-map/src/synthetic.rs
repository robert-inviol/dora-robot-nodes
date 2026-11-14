//! Rendered keyframes of box-shaped houses, for tests, examples and profiling.

use std::sync::Arc;

use depth_perception::synthetic::{Cuboid, SyntheticRoom};
use depth_perception::CameraIntrinsics;
use glam::{Vec2, Vec3};
use perception_wire::navigation::{MapVersion, UnixNanos};

use crate::keyframe::{DepthKeyframe, FrameId, KeyframePose, Tracking};

/// The camera sits this far above the floor, as on the robot.
pub const CAMERA_HEIGHT_M: f32 = 0.3;
pub const CEILING_HEIGHT_M: f32 = 2.4;
pub const WALL_THICKNESS_M: f32 = 0.1;

/// A D455-like 848x480 depth camera with a 90 degree horizontal field of view.
pub fn d455_intrinsics() -> CameraIntrinsics {
    CameraIntrinsics { fx: 424.0, fy: 424.0, ppx: 423.5, ppy: 239.5 }
}

/// A house whose floor plan spans `size`, in map coordinates with the floor at `-CAMERA_HEIGHT_M`, holding the
/// given walls and furniture.
pub fn house(size: Vec2, solids: Vec<Cuboid>) -> SyntheticRoom {
    SyntheticRoom {
        room: Cuboid { min: Vec3::new(0.0, 0.0, -CAMERA_HEIGHT_M), max: size.extend(CEILING_HEIGHT_M - CAMERA_HEIGHT_M) },
        objects: solids,
        width: 848,
        height: 480,
        intrinsics: d455_intrinsics(),
    }
}

/// A full-height wall along x at `y`, from `x0` to `x1`, with doorway gaps `(from, to)` along it.
pub fn wall_along_x(y: f32, x0: f32, x1: f32, doors: &[(f32, f32)]) -> Vec<Cuboid> {
    wall_pieces(x0, x1, doors)
        .into_iter()
        .map(|(a, b)| solid(Vec2::new(a, y), Vec2::new(b, y + WALL_THICKNESS_M), CEILING_HEIGHT_M))
        .collect()
}

/// A full-height wall along y at `x`, from `y0` to `y1`, with doorway gaps `(from, to)` along it.
pub fn wall_along_y(x: f32, y0: f32, y1: f32, doors: &[(f32, f32)]) -> Vec<Cuboid> {
    wall_pieces(y0, y1, doors)
        .into_iter()
        .map(|(a, b)| solid(Vec2::new(x, a), Vec2::new(x + WALL_THICKNESS_M, b), CEILING_HEIGHT_M))
        .collect()
}

fn wall_pieces(from: f32, to: f32, doors: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut pieces = Vec::new();
    let mut start = from;
    for &(door_from, door_to) in doors {
        pieces.push((start, door_from));
        start = door_to;
    }
    pieces.push((start, to));
    pieces.into_iter().filter(|(a, b)| b > a).collect()
}

/// A box standing on the floor.
pub fn solid(low: Vec2, high: Vec2, height_m: f32) -> Cuboid {
    Cuboid { min: low.extend(-CAMERA_HEIGHT_M), max: high.extend(height_m - CAMERA_HEIGHT_M) }
}

/// Two 4 m square rooms side by side, a wall at x = 4 with a door 0.8 m wide from y = 1.6 to 2.4.
pub fn two_rooms() -> SyntheticRoom {
    house(Vec2::new(8.1, 4.0), wall_along_y(4.0, 0.0, 4.0, &[(1.6, 2.4)]))
}

/// A keyframe rendered by a level camera at a floor-plan position, facing `heading_rad` anticlockwise from +y, and
/// the pose the localizer would give it.
pub fn keyframe_at(house: &SyntheticRoom, frame_id: u64, position: Vec2, heading_rad: f32, map_version: MapVersion) -> (DepthKeyframe, KeyframePose) {
    let camera = position.extend(0.0);
    let frame = house.render_from(camera, heading_rad);
    let depth_mm: Vec<u16> = frame.depth_m.iter().map(|d| (d * 1000.0).round().min(u16::MAX as f32) as u16).collect();
    let keyframe = DepthKeyframe {
        frame_id: FrameId(frame_id),
        captured: UnixNanos(1_762_560_000_000_000_000 + frame_id as i64 * 100_000_000),
        width: house.width,
        height: house.height,
        intrinsics: house.intrinsics,
        gravity_reaction: frame.accel_camera,
        depth_mm: Arc::from(depth_mm),
    };
    let pose = KeyframePose { frame_id: FrameId(frame_id), map_version, translation: camera, yaw_rad: heading_rad, tracking: Tracking::Tracking };
    (keyframe, pose)
}

/// A look around at each position: `turns` headings evenly round the circle.
pub fn look_around(positions: &[Vec2], turns: usize) -> Vec<(Vec2, f32)> {
    positions
        .iter()
        .flat_map(|&p| (0..turns).map(move |t| (p, t as f32 * std::f32::consts::TAU / turns as f32)))
        .collect()
}
