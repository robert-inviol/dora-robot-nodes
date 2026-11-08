//! Voxel downsampling, nearest-neighbour search on a hashed grid, and surface normals.

use std::collections::HashMap;

use glam::{IVec3, Vec3};
use rayon::prelude::*;

#[inline]
fn cell_of(point: Vec3, cell_m: f32) -> IVec3 {
    (point / cell_m).floor().as_ivec3()
}

/// The centroid of the points in each occupied voxel.
pub fn voxel_downsample(points: &[Vec3], cell_m: f32) -> Vec<Vec3> {
    let mut cells: HashMap<IVec3, (Vec3, u32)> = HashMap::with_capacity(points.len() / 4);
    for &point in points {
        let entry = cells.entry(cell_of(point, cell_m)).or_insert((Vec3::ZERO, 0));
        entry.0 += point;
        entry.1 += 1;
    }
    cells.into_values().map(|(sum, count)| sum / count as f32).collect()
}

/// Points bucketed by cell, for finding neighbours within one cell's distance.
pub struct NeighbourGrid {
    cell_m: f32,
    cells: HashMap<IVec3, Vec<u32>>,
    pub points: Vec<Vec3>,
}

impl NeighbourGrid {
    pub fn new(points: Vec<Vec3>, cell_m: f32) -> NeighbourGrid {
        let mut cells: HashMap<IVec3, Vec<u32>> = HashMap::with_capacity(points.len() / 2);
        for (index, &point) in points.iter().enumerate() {
            cells.entry(cell_of(point, cell_m)).or_default().push(index as u32);
        }
        NeighbourGrid { cell_m, cells, points }
    }

    fn around(&self, query: Vec3) -> impl Iterator<Item = u32> + '_ {
        let centre = cell_of(query, self.cell_m);
        (-1..=1).flat_map(move |dx| {
            (-1..=1).flat_map(move |dy| {
                (-1..=1).flat_map(move |dz| self.cells.get(&(centre + IVec3::new(dx, dy, dz))).into_iter().flatten().copied())
            })
        })
    }

    /// The nearest point within `max_distance_m`, which must not exceed the cell size.
    pub fn nearest(&self, query: Vec3, max_distance_m: f32) -> Option<(u32, f32)> {
        let limit = max_distance_m * max_distance_m;
        self.around(query)
            .map(|index| (index, self.points[index as usize].distance_squared(query)))
            .filter(|(_, d2)| *d2 <= limit)
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    pub fn within(&self, query: Vec3, radius_m: f32) -> impl Iterator<Item = u32> + '_ {
        let limit = radius_m * radius_m;
        self.around(query).filter(move |&index| self.points[index as usize].distance_squared(query) <= limit)
    }

    pub fn cell_m(&self) -> f32 {
        self.cell_m
    }
}

/// Eigenvalues (ascending) and the eigenvector of the smallest, for a symmetric 3x3 matrix (Jacobi).
fn smallest_eigenvector(mut a: [[f32; 3]; 3]) -> ([f32; 3], Vec3) {
    let mut v = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..12 {
        let (mut p, mut q) = (0, 1);
        for (i, j) in [(0, 2), (1, 2)] {
            if a[i][j].abs() > a[p][q].abs() {
                (p, q) = (i, j);
            }
        }
        if a[p][q].abs() < 1e-12 {
            break;
        }
        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let t = if theta == 0.0 { 1.0 } else { t };
        let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
        for k in 0..3 {
            let (akp, akq) = (a[k][p], a[k][q]);
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
        }
        for k in 0..3 {
            let (apk, aqk) = (a[p][k], a[q][k]);
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
        }
        for row in &mut v {
            let (vkp, vkq) = (row[p], row[q]);
            row[p] = c * vkp - s * vkq;
            row[q] = s * vkp + c * vkq;
        }
    }
    let mut order = [0, 1, 2];
    order.sort_by(|&i, &j| a[i][i].total_cmp(&a[j][j]));
    let smallest = order[0];
    ([a[order[0]][order[0]], a[order[1]][order[1]], a[order[2]][order[2]]], Vec3::new(v[0][smallest], v[1][smallest], v[2][smallest]))
}

/// A unit normal for each point whose neighbourhood within `radius_m` is flat enough, else `None`.
pub fn surface_normals(grid: &NeighbourGrid, radius_m: f32, min_neighbours: usize, max_flatness: f32) -> Vec<Option<Vec3>> {
    grid.points
        .par_iter()
        .map(|&point| {
            let mut count = 0usize;
            let mut mean = Vec3::ZERO;
            let neighbours: Vec<Vec3> = grid.within(point, radius_m).map(|i| grid.points[i as usize]).collect();
            for &n in &neighbours {
                mean += n;
                count += 1;
            }
            if count < min_neighbours {
                return None;
            }
            mean /= count as f32;
            let mut c = [[0.0f32; 3]; 3];
            for n in &neighbours {
                let d = *n - mean;
                for i in 0..3 {
                    for j in 0..3 {
                        c[i][j] += d[i] * d[j];
                    }
                }
            }
            let (values, normal) = smallest_eigenvector(c);
            (values[0] <= max_flatness * values[1]).then(|| normal.normalize())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_normal_of_a_flat_patch_is_perpendicular_to_it() {
        let points: Vec<Vec3> = (0..20).flat_map(|i| (0..20).map(move |j| Vec3::new(i as f32 * 0.02, 1.0, j as f32 * 0.02))).collect();
        let grid = NeighbourGrid::new(points, 0.1);
        let normals = surface_normals(&grid, 0.1, 5, 0.3);
        let normal = normals[210].expect("flat patch has a normal");
        assert!(normal.y.abs() > 0.999, "{normal}");
    }

    #[test]
    fn nearest_finds_the_closest_point_within_range() {
        let grid = NeighbourGrid::new(vec![Vec3::ZERO, Vec3::new(0.05, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0)], 0.1);
        assert_eq!(grid.nearest(Vec3::new(0.04, 0.0, 0.0), 0.1).map(|n| n.0), Some(1));
        assert_eq!(grid.nearest(Vec3::new(0.3, 0.0, 0.0), 0.1), None);
    }
}
