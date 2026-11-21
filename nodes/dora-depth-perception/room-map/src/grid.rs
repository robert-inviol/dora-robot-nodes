//! Voxel downsampling, nearest-neighbour search on a hashed grid, and surface normals.
//!
//! The grids hash cells with a fixed hasher, so the same points in the same order give the same output order in
//! every run.

use glam::{IVec3, Vec3};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

#[inline]
fn cell_of(point: Vec3, cell_m: f32) -> IVec3 {
    (point / cell_m).floor().as_ivec3()
}

/// The centroid of the points in each occupied voxel, in the same order for the same input.
pub fn voxel_downsample(points: &[Vec3], cell_m: f32) -> Vec<Vec3> {
    let mut cells: FxHashMap<IVec3, (Vec3, u32)> = FxHashMap::default();
    cells.reserve(points.len() / 4);
    for &point in points {
        let entry = cells.entry(cell_of(point, cell_m)).or_insert((Vec3::ZERO, 0));
        entry.0 += point;
        entry.1 += 1;
    }
    cells.into_values().map(|(sum, count)| sum / count as f32).collect()
}

/// The 27 cells around a query's cell, by x, then y, then z offset.
const AROUND: [IVec3; 27] = {
    let mut offsets = [IVec3::ZERO; 27];
    let mut k = 0;
    while k < 27 {
        offsets[k] = IVec3::new(k as i32 / 9 - 1, (k as i32 / 3) % 3 - 1, k as i32 % 3 - 1);
        k += 1;
    }
    offsets
};

/// `AROUND` with the query's own cell first, so that a nearest neighbour found there rules out most of the others.
const NEAREST_FIRST: [usize; 27] = [13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26];

/// Rounding slack on the distance from a query to a cell, so that a cell is never passed over for a point exactly
/// at the search distance.
const CELL_DISTANCE_SLACK_M: f32 = 1e-5;

/// Indices of a set of points bucketed by cubic cell, for finding a point's neighbours within one cell's distance.
pub struct CellIndex {
    cell_m: f32,
    cells: FxHashMap<IVec3, Vec<u32>>,
}

impl CellIndex {
    pub fn new(points: &[Vec3], cell_m: f32) -> CellIndex {
        let mut cells: FxHashMap<IVec3, Vec<u32>> = FxHashMap::default();
        cells.reserve(points.len() / 2);
        for (index, &point) in points.iter().enumerate() {
            cells.entry(cell_of(point, cell_m)).or_default().push(index as u32);
        }
        CellIndex { cell_m, cells }
    }

    pub fn cell_m(&self) -> f32 {
        self.cell_m
    }

    /// The query's cell, and the squared distance from the query to each of the 27 cells around it (in `AROUND`
    /// order), less the rounding slack.
    fn cell_distances(&self, query: Vec3) -> (IVec3, [f32; 27]) {
        let centre = cell_of(query, self.cell_m);
        let inside = query - centre.as_vec3() * self.cell_m;
        let below = (inside - CELL_DISTANCE_SLACK_M).max(Vec3::ZERO);
        let above = (Vec3::splat(self.cell_m) - inside - CELL_DISTANCE_SLACK_M).max(Vec3::ZERO);
        let gap = |offset: i32, below: f32, above: f32| match offset {
            -1 => below * below,
            1 => above * above,
            _ => 0.0,
        };
        let distances = AROUND.map(|o| gap(o.x, below.x, above.x) + gap(o.y, below.y, above.y) + gap(o.z, below.z, above.z));
        (centre, distances)
    }

    /// The nearest of `points` to `query` within `max_distance_m`, which must not exceed the cell size, with its
    /// squared distance; of points equally near, the lowest index. `points` are the ones the index was built from.
    pub fn nearest(&self, points: &[Vec3], query: Vec3, max_distance_m: f32) -> Option<(u32, f32)> {
        debug_assert!(max_distance_m <= self.cell_m, "a search of {max_distance_m} m reaches past cells of {} m", self.cell_m);
        let (centre, distances) = self.cell_distances(query);
        let mut best: Option<(u32, f32)> = None;
        let mut limit = max_distance_m * max_distance_m;
        for k in NEAREST_FIRST {
            if distances[k] > limit {
                continue;
            }
            let Some(indices) = self.cells.get(&(centre + AROUND[k])) else { continue };
            for &index in indices {
                let d2 = points[index as usize].distance_squared(query);
                let better = match best {
                    None => d2 <= limit,
                    Some((best_index, best_d2)) => d2 < best_d2 || (d2 == best_d2 && index < best_index),
                };
                if better {
                    best = Some((index, d2));
                    limit = d2;
                }
            }
        }
        best
    }

    /// Every one of `points` within `radius_m` of `query`, which must not exceed the cell size, in cell order
    /// (`AROUND`) and within a cell in index order.
    pub fn within<'a>(&'a self, points: &'a [Vec3], query: Vec3, radius_m: f32) -> impl Iterator<Item = u32> + 'a {
        debug_assert!(radius_m <= self.cell_m, "a search of {radius_m} m reaches past cells of {} m", self.cell_m);
        let limit = radius_m * radius_m;
        let (centre, distances) = self.cell_distances(query);
        AROUND
            .iter()
            .zip(distances)
            .filter(move |(_, cell_distance)| *cell_distance <= limit)
            .filter_map(move |(offset, _)| self.cells.get(&(centre + *offset)))
            .flatten()
            .copied()
            .filter(move |&index| points[index as usize].distance_squared(query) <= limit)
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
        for row in &mut a {
            let (akp, akq) = (row[p], row[q]);
            row[p] = c * akp - s * akq;
            row[q] = s * akp + c * akq;
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

/// Normals are found in parallel at least this many points to a task, so that a few threads share the work rather
/// than every thread waking for a sliver of it.
const NORMALS_PER_TASK: usize = 2048;

/// A unit normal for each point whose neighbourhood within `radius_m` is flat enough, else `None`.
/// `index` is over `points`, with cells at least `radius_m` wide.
pub fn surface_normals(points: &[Vec3], index: &CellIndex, radius_m: f32, min_neighbours: usize, max_flatness: f32) -> Vec<Option<Vec3>> {
    points
        .par_iter()
        .with_min_len(NORMALS_PER_TASK)
        .map(|&point| {
            let mut count = 0usize;
            let mut mean = Vec3::ZERO;
            let neighbours: Vec<Vec3> = index.within(points, point, radius_m).map(|i| points[i as usize]).collect();
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
        let index = CellIndex::new(&points, 0.1);
        let normals = surface_normals(&points, &index, 0.1, 5, 0.3);
        let normal = normals[210].expect("flat patch has a normal");
        assert!(normal.y.abs() > 0.999, "{normal}");
    }

    #[test]
    fn nearest_finds_the_closest_point_within_range() {
        let points = vec![Vec3::ZERO, Vec3::new(0.05, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0)];
        let index = CellIndex::new(&points, 0.1);
        assert_eq!(index.nearest(&points, Vec3::new(0.04, 0.0, 0.0), 0.1).map(|n| n.0), Some(1));
        assert_eq!(index.nearest(&points, Vec3::new(0.3, 0.0, 0.0), 0.1), None);
    }

    /// Every point of `points` within `radius_m` of `query`, by brute force.
    fn brute_within(points: &[Vec3], query: Vec3, radius_m: f32) -> Vec<u32> {
        (0..points.len() as u32).filter(|&i| points[i as usize].distance_squared(query) <= radius_m * radius_m).collect()
    }

    #[test]
    fn the_index_finds_what_a_brute_force_search_finds_across_cell_boundaries() {
        let points: Vec<Vec3> = (0..3000).map(|k| Vec3::new((k % 17) as f32 * 0.031 - 0.2, (k % 23) as f32 * 0.027 - 0.3, (k / 391) as f32 * 0.043)).collect();
        let queries: Vec<Vec3> = (0..500).map(|k| Vec3::new((k % 13) as f32 * 0.041 - 0.25, (k % 7) as f32 * 0.063 - 0.2, (k % 11) as f32 * 0.029)).collect();
        for cell_m in [0.08, 0.15, 0.3] {
            let index = CellIndex::new(&points, cell_m);
            for &query in &queries {
                let mut found: Vec<u32> = index.within(&points, query, cell_m).collect();
                found.sort_unstable();
                let expected = brute_within(&points, query, cell_m);
                assert_eq!(found, expected, "within {cell_m} m of {query}");
                let nearest = expected
                    .iter()
                    .map(|&i| (i, points[i as usize].distance_squared(query)))
                    .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
                assert_eq!(index.nearest(&points, query, cell_m), nearest, "nearest within {cell_m} m of {query}");
            }
        }
    }

    #[test]
    fn downsampling_gives_the_same_order_every_time() {
        let points: Vec<Vec3> = (0..2000).map(|k| Vec3::new((k % 37) as f32 * 0.07, (k % 11) as f32 * 0.13, (k / 50) as f32 * 0.05)).collect();
        let first = voxel_downsample(&points, 0.1);
        // A fixed hasher gives the same order in another process too; within one, every call agrees.
        assert!((0..5).all(|_| voxel_downsample(&points, 0.1) == first));
    }
}
