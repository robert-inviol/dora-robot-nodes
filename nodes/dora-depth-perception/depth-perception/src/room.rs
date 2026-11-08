//! The room as the space between the floor, the ceiling and the outermost walls.
//!
//! The room is modelled as convex: each bounding wall is a line on the floor plan, and the room is the
//! side of every such line that the camera is on. Where no wall was seen, the room stops just past the
//! observed points and that edge is marked unseen.

use std::fmt;

use glam::{Vec2, Vec3};

use crate::geometry2d::median;
use crate::planes::{Plane, SurfaceKind};

#[derive(Clone, Debug)]
pub struct RoomSettings {
    /// A room wall reaches the ceiling; furniture stops short. The lower wall is often hidden, so only its top is tested.
    pub max_gap_below_ceiling_m: f32,
    /// Narrower vertical planes are door frames and furniture sides, not the room's walls.
    pub min_wall_width_m: f32,
    /// Walls whose outward normals are within this angle face the same way; only the farthest bounds the room.
    pub same_direction_deg: f32,
    /// Unseen sides stop this far past the furthest observed point.
    pub unseen_edge_margin_m: f32,
    /// Points this close to the floor or ceiling are left out when finding the observed extent.
    pub floor_and_ceiling_clearance_m: f32,
}

impl Default for RoomSettings {
    fn default() -> Self {
        RoomSettings {
            max_gap_below_ceiling_m: 0.3,
            min_wall_width_m: 2.0,
            same_direction_deg: 20.0,
            unseen_edge_margin_m: 0.2,
            floor_and_ceiling_clearance_m: 0.1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Room {
    pub floor_m: f32,
    pub ceiling_m: f32,
    /// Floor-plan corners (right, forward), counter-clockwise.
    pub footprint_m: Vec<Vec2>,
    /// For each edge ending at the matching corner, the index of the wall plane it lies on, or `None` if unseen.
    pub edge_walls: Vec<Option<usize>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoomError {
    NoFloor,
    NoCeiling,
    NoPointsBetweenFloorAndCeiling,
}

impl fmt::Display for RoomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            RoomError::NoFloor => "no floor plane found, cannot bound the room",
            RoomError::NoCeiling => "no ceiling plane found, cannot bound the room",
            RoomError::NoPointsBetweenFloorAndCeiling => "no points between the floor and the ceiling",
        })
    }
}

impl std::error::Error for RoomError {}

fn level_height_m(planes: &[Plane], kind: SurfaceKind) -> Option<f32> {
    let mut heights: Vec<f32> = planes.iter().filter(|p| p.kind == kind).map(|p| -p.offset_m).collect();
    (!heights.is_empty()).then(|| median(&mut heights))
}

fn wall_width_m(wall: &Plane) -> f32 {
    let along_wall = Vec2::new(-wall.normal.y, wall.normal.x);
    let (low, high) = wall
        .outline_m
        .iter()
        .map(|p| p.truncate().dot(along_wall))
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)));
    high - low
}

/// Indices of the wide, ceiling-high walls that are the farthest in their facing direction.
fn bounding_walls(planes: &[Plane], ceiling_m: f32, settings: &RoomSettings) -> Vec<usize> {
    let tall: Vec<usize> = (0..planes.len())
        .filter(|&i| {
            let p = &planes[i];
            p.kind == SurfaceKind::Wall
                && p.outline_m.iter().map(|q| q.z).fold(f32::NEG_INFINITY, f32::max) >= ceiling_m - settings.max_gap_below_ceiling_m
                && wall_width_m(p) >= settings.min_wall_width_m
        })
        .collect();
    // The facing direction is the normal pointed away from the camera, so opposite walls never match.
    let outward = |i: usize| planes[i].normal.truncate() * -planes[i].offset_m.signum();
    let same_direction = settings.same_direction_deg.to_radians().cos();
    tall.iter()
        .copied()
        .filter(|&i| {
            tall.iter().all(|&j| outward(i).dot(outward(j)) <= same_direction || planes[i].offset_m.abs() >= planes[j].offset_m.abs())
        })
        .collect()
}

/// Keeps the part of the polygon on the camera side of `normal . p + offset = 0` (Sutherland-Hodgman),
/// labelling the new edge along the line with `wall`.
fn clip(polygon: &[(Vec2, Option<usize>)], normal: Vec2, offset_m: f32, wall: usize) -> Vec<(Vec2, Option<usize>)> {
    let camera_side = offset_m.signum();
    let mut clipped = Vec::with_capacity(polygon.len() + 1);
    for k in 0..polygon.len() {
        let (current, _) = polygon[k];
        let (following, following_edge) = polygon[(k + 1) % polygon.len()];
        let (current_distance, following_distance) = (normal.dot(current) + offset_m, normal.dot(following) + offset_m);
        let (current_inside, following_inside) = (current_distance * camera_side >= 0.0, following_distance * camera_side >= 0.0);
        if current_inside != following_inside {
            let crossing = current + (following - current) * current_distance / (current_distance - following_distance);
            clipped.push((crossing, if current_inside { following_edge } else { Some(wall) }));
            if following_inside {
                clipped.push((following, following_edge));
            }
        } else if following_inside {
            clipped.push((following, following_edge));
        }
    }
    clipped
}

pub fn infer_room(planes: &[Plane], points: &[Vec3], settings: &RoomSettings) -> Result<Room, RoomError> {
    let floor_m = level_height_m(planes, SurfaceKind::Floor).ok_or(RoomError::NoFloor)?;
    let ceiling_m = level_height_m(planes, SurfaceKind::Ceiling).ok_or(RoomError::NoCeiling)?;
    let clearance = settings.floor_and_ceiling_clearance_m;
    let (low, high) = points
        .iter()
        .filter(|p| p.z > floor_m + clearance && p.z < ceiling_m - clearance)
        .fold(None, |extent: Option<(Vec2, Vec2)>, p| {
            let xy = p.truncate();
            Some(extent.map_or((xy, xy), |(lo, hi)| (lo.min(xy), hi.max(xy))))
        })
        .ok_or(RoomError::NoPointsBetweenFloorAndCeiling)?;
    let (low, high) = (low - settings.unseen_edge_margin_m, high + settings.unseen_edge_margin_m);
    let mut polygon: Vec<(Vec2, Option<usize>)> =
        [low, Vec2::new(high.x, low.y), high, Vec2::new(low.x, high.y)].into_iter().map(|c| (c, None)).collect();
    for wall in bounding_walls(planes, ceiling_m, settings) {
        polygon = clip(&polygon, planes[wall].normal.truncate(), planes[wall].offset_m, wall);
    }
    Ok(Room {
        floor_m,
        ceiling_m,
        footprint_m: polygon.iter().map(|(corner, _)| *corner).collect(),
        edge_walls: polygon.iter().map(|(_, edge)| *edge).collect(),
    })
}
