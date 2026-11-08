//! Large walls and level surfaces, found with RANSAC and constrained to be exactly upright or level.

use glam::{Vec2, Vec3};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;

use crate::geometry2d::{convex_hull_indices, median};

#[derive(Clone, Debug)]
pub struct PlaneSettings {
    /// Generous enough to absorb D455 noise at the far wall and small relief such as skirting.
    pub inlier_distance_m: f32,
    pub ransac_iterations: usize,
    /// RANSAC scores candidates on an evenly strided subset of this many points.
    pub ransac_sample_points: usize,
    pub min_area_m2: f32,
    pub max_planes: usize,
    /// Inliers this far apart in the plane are separate surfaces that happen to be coplanar.
    pub connectivity_cell_m: f32,
    /// A candidate within this angle of upright or level is snapped to it; anything else is rejected.
    pub orientation_tolerance_deg: f32,
    /// Level planes this close in height to the lowest (highest) one are also floor (ceiling).
    pub same_level_m: f32,
    /// A plane through the camera centre catches a whole band of image rows with no surface behind it.
    pub min_camera_distance_m: f32,
    pub seed: u64,
}

impl Default for PlaneSettings {
    fn default() -> Self {
        PlaneSettings {
            inlier_distance_m: 0.10,
            ransac_iterations: 400,
            ransac_sample_points: 60_000,
            min_area_m2: 1.0,
            max_planes: 10,
            connectivity_cell_m: 0.10,
            orientation_tolerance_deg: 15.0,
            same_level_m: 0.15,
            min_camera_distance_m: 0.3,
            seed: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Orientation {
    Vertical,
    Horizontal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceKind {
    Floor,
    Ceiling,
    Wall,
    LevelSurface,
}

impl SurfaceKind {
    pub fn name(self) -> &'static str {
        match self {
            SurfaceKind::Floor => "floor",
            SurfaceKind::Ceiling => "ceiling",
            SurfaceKind::Wall => "wall",
            SurfaceKind::LevelSurface => "level surface",
        }
    }
}

/// A plane `normal . p + offset_m = 0`; for a level plane the normal is exactly up, so its height is `-offset_m`.
#[derive(Clone, Debug)]
pub struct Plane {
    pub kind: SurfaceKind,
    pub normal: Vec3,
    pub offset_m: f32,
    pub area_m2: f32,
    /// Convex outline of the plane's points, projected onto it.
    pub outline_m: Vec<Vec3>,
}

impl Plane {
    #[inline]
    pub fn signed_distance_m(&self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.offset_m
    }
}

#[derive(Clone, Debug)]
pub struct PlaneFit {
    pub planes: Vec<Plane>,
    /// For each input point, the index into `planes` of the plane it lies on.
    pub point_planes: Vec<Option<u8>>,
    /// The input points with every plane member projected onto its plane.
    pub flattened: Vec<Vec3>,
}

struct Candidate {
    normal: Vec3,
    offset_m: f32,
    orientation: Orientation,
}

fn snapped_orientation(unit_normal: Vec3, tolerance_deg: f32) -> Option<Orientation> {
    let tilt_deg = unit_normal.z.abs().min(1.0).acos().to_degrees();
    if tilt_deg < tolerance_deg {
        Some(Orientation::Horizontal)
    } else if tilt_deg > 90.0 - tolerance_deg {
        Some(Orientation::Vertical)
    } else {
        None
    }
}

fn snapped_normal(orientation: Orientation, rough_normal: Vec3) -> Vec3 {
    match orientation {
        Orientation::Horizontal => Vec3::Z,
        Orientation::Vertical => Vec3::new(rough_normal.x, rough_normal.y, 0.0).normalize(),
    }
}

/// Offset placing the plane through the median of the points along the normal.
fn median_offset(normal: Vec3, points: impl Iterator<Item = Vec3>) -> f32 {
    let mut along: Vec<f32> = points.map(|p| normal.dot(p)).collect();
    -median(&mut along)
}

fn ransac_plane(points: &[Vec3], candidates: &[u32], settings: &PlaneSettings, round: u64) -> Option<Candidate> {
    let stride = candidates.len().div_ceil(settings.ransac_sample_points).max(1);
    let sample: Vec<Vec3> = candidates.iter().step_by(stride).map(|&i| points[i as usize]).collect();
    let inlier = settings.inlier_distance_m;
    let best = (0..settings.ransac_iterations)
        .into_par_iter()
        .filter_map(|iteration| {
            let mut rng = SmallRng::seed_from_u64(settings.seed ^ (round << 32) ^ iteration as u64);
            let [a, b, c] = [0; 3].map(|_| sample[rng.gen_range(0..sample.len())]);
            let rough = (b - a).cross(c - a);
            if rough.length_squared() < 1e-12 {
                return None;
            }
            let orientation = snapped_orientation(rough.normalize(), settings.orientation_tolerance_deg)?;
            let normal = snapped_normal(orientation, rough);
            let offset_m = median_offset(normal, [a, b, c].into_iter());
            let count = sample.iter().filter(|p| (normal.dot(**p) + offset_m).abs() < inlier).count();
            Some((count, normal, offset_m, orientation))
        })
        .max_by_key(|(count, ..)| *count)?;
    let (_, rough_normal, rough_offset, orientation) = best;
    let inliers = candidates
        .iter()
        .map(|&i| points[i as usize])
        .filter(|p| (rough_normal.dot(*p) + rough_offset).abs() < inlier);
    let normal = snapped_normal(orientation, rough_normal);
    Some(Candidate { normal, offset_m: median_offset(normal, inliers), orientation })
}

fn in_plane_axes(normal: Vec3) -> (Vec3, Vec3) {
    let helper = if normal.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
    let first = normal.cross(helper).normalize();
    (first, normal.cross(first))
}

/// Which points belong to the largest 8-connected group of occupied grid cells, and how many cells it covers.
fn largest_connected_patch(coords: &[Vec2], cell_m: f32) -> (Vec<bool>, usize) {
    let low = coords.iter().fold(Vec2::splat(f32::INFINITY), |m, c| m.min(*c));
    let cells: Vec<(usize, usize)> = coords
        .iter()
        .map(|c| {
            let cell = ((*c - low) / cell_m).floor();
            (cell.x as usize, cell.y as usize)
        })
        .collect();
    let columns = cells.iter().map(|c| c.0).max().unwrap_or(0) + 1;
    let rows = cells.iter().map(|c| c.1).max().unwrap_or(0) + 1;
    let mut occupied = vec![false; columns * rows];
    for &(x, y) in &cells {
        occupied[y * columns + x] = true;
    }
    const UNLABELLED: u32 = u32::MAX;
    let mut labels = vec![UNLABELLED; columns * rows];
    let mut component_cells = Vec::new();
    let mut stack = Vec::new();
    for start in 0..occupied.len() {
        if !occupied[start] || labels[start] != UNLABELLED {
            continue;
        }
        let label = component_cells.len() as u32;
        let mut size = 0;
        labels[start] = label;
        stack.push(start);
        while let Some(cell) = stack.pop() {
            size += 1;
            let (x, y) = ((cell % columns) as isize, (cell / columns) as isize);
            for (dx, dy) in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= columns as isize || ny >= rows as isize {
                    continue;
                }
                let neighbour = ny as usize * columns + nx as usize;
                if occupied[neighbour] && labels[neighbour] == UNLABELLED {
                    labels[neighbour] = label;
                    stack.push(neighbour);
                }
            }
        }
        component_cells.push(size);
    }
    let mut component_points = vec![0usize; component_cells.len()];
    for &(x, y) in &cells {
        component_points[labels[y * columns + x] as usize] += 1;
    }
    let Some(largest) = (0..component_points.len()).max_by_key(|&c| component_points[c]) else {
        return (vec![false; coords.len()], 0);
    };
    let members = cells.iter().map(|&(x, y)| labels[y * columns + x] == largest as u32).collect();
    (members, component_cells[largest])
}

fn name_surfaces(found: &[(Candidate, f32, Vec<Vec3>)], same_level_m: f32) -> Vec<SurfaceKind> {
    let level_heights = || found.iter().filter(|(c, ..)| c.orientation == Orientation::Horizontal).map(|(c, ..)| -c.offset_m);
    let floor_m = level_heights().filter(|h| *h < 0.0).reduce(f32::min);
    let ceiling_m = level_heights().filter(|h| *h > 0.0).reduce(f32::max);
    found
        .iter()
        .map(|(candidate, ..)| {
            let height_m = -candidate.offset_m;
            let near = |level: Option<f32>| level.is_some_and(|l| (height_m - l).abs() < same_level_m);
            match candidate.orientation {
                Orientation::Vertical => SurfaceKind::Wall,
                Orientation::Horizontal if near(floor_m) => SurfaceKind::Floor,
                Orientation::Horizontal if near(ceiling_m) => SurfaceKind::Ceiling,
                Orientation::Horizontal => SurfaceKind::LevelSurface,
            }
        })
        .collect()
}

pub fn find_planes(points: &[Vec3], settings: &PlaneSettings) -> PlaneFit {
    assert!(settings.max_planes <= u8::MAX as usize, "plane indices are stored as u8");
    let mut point_planes = vec![None; points.len()];
    let mut flattened = points.to_vec();
    let mut found: Vec<(Candidate, f32, Vec<Vec3>)> = Vec::new();
    let mut remaining: Vec<u32> = (0..points.len() as u32).collect();
    for round in 0..(settings.max_planes * 2) as u64 {
        if found.len() == settings.max_planes || remaining.len() < settings.ransac_sample_points / 20 {
            break;
        }
        let Some(candidate) = ransac_plane(points, &remaining, settings, round) else {
            break;
        };
        let distance = |i: u32| candidate.normal.dot(points[i as usize]) + candidate.offset_m;
        let (on_plane, rest): (Vec<u32>, Vec<u32>) =
            remaining.iter().partition(|&&i| distance(i).abs() < settings.inlier_distance_m);
        // The whole inlier set leaves the pool either way, so a small coplanar patch is not found again.
        remaining = rest;
        if on_plane.is_empty() {
            continue;
        }
        let (axis_a, axis_b) = in_plane_axes(candidate.normal);
        let coords: Vec<Vec2> =
            on_plane.iter().map(|&i| Vec2::new(axis_a.dot(points[i as usize]), axis_b.dot(points[i as usize]))).collect();
        let (in_patch, patch_cells) = largest_connected_patch(&coords, settings.connectivity_cell_m);
        let area_m2 = patch_cells as f32 * settings.connectivity_cell_m.powi(2);
        if area_m2 < settings.min_area_m2 || candidate.offset_m.abs() < settings.min_camera_distance_m {
            continue;
        }
        let plane_index = found.len() as u8;
        let mut patch_coords = Vec::new();
        let mut members = Vec::new();
        for (k, &i) in on_plane.iter().enumerate() {
            if in_patch[k] {
                flattened[i as usize] -= distance(i) * candidate.normal;
                point_planes[i as usize] = Some(plane_index);
                patch_coords.push(coords[k]);
                members.push(i);
            }
        }
        let outline = convex_hull_indices(&patch_coords).into_iter().map(|k| flattened[members[k] as usize]).collect();
        found.push((candidate, area_m2, outline));
    }
    let kinds = name_surfaces(&found, settings.same_level_m);
    let planes = found
        .into_iter()
        .zip(kinds)
        .map(|((candidate, area_m2, outline_m), kind)| Plane {
            kind,
            normal: candidate.normal,
            offset_m: candidate.offset_m,
            area_m2,
            outline_m,
        })
        .collect();
    PlaneFit { planes, point_planes, flattened }
}
