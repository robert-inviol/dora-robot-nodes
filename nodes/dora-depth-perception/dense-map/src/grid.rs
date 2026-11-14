//! The floor plan grid: which cells of a height band above the floor are free, occupied or unknown, and how far
//! each cell is from the nearest cell of a kind (an exact Euclidean distance transform).

use glam::{IVec2, Vec2};
use rayon::prelude::*;

use crate::tsdf::{DistanceField, Sample};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    Unknown,
    Free,
    Occupied,
}

/// Where a grid sits in the map: cell (column, row) spans `origin + [column, column + 1) * resolution` in x and the
/// same in y with row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridGeometry {
    pub origin: Vec2,
    pub resolution_m: f32,
    pub width: usize,
    pub height: usize,
}

impl GridGeometry {
    pub fn len(&self) -> usize {
        self.width * self.height
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn index(&self, cell: IVec2) -> Option<usize> {
        (cell.x >= 0 && cell.y >= 0 && (cell.x as usize) < self.width && (cell.y as usize) < self.height)
            .then(|| cell.y as usize * self.width + cell.x as usize)
    }

    #[inline]
    pub fn cell(&self, index: usize) -> IVec2 {
        IVec2::new((index % self.width) as i32, (index / self.width) as i32)
    }

    #[inline]
    pub fn cell_at(&self, point: Vec2) -> IVec2 {
        ((point - self.origin) / self.resolution_m).floor().as_ivec2()
    }

    #[inline]
    pub fn centre(&self, cell: IVec2) -> Vec2 {
        self.origin + (cell.as_vec2() + 0.5) * self.resolution_m
    }

    /// A grid corner, as polygons use them.
    #[inline]
    pub fn corner(&self, corner: IVec2) -> Vec2 {
        self.origin + corner.as_vec2() * self.resolution_m
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CellGrid {
    pub geometry: GridGeometry,
    pub cells: Vec<Cell>,
}

/// The height band, in the map frame, that the grid slices through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slice {
    pub min_z_m: f32,
    pub max_z_m: f32,
    /// A voxel whose signed distance is below this is occupied.
    pub occupied_below_m: f32,
}

/// The field sliced into floor cells, one per voxel column: occupied where any voxel of the band is near or behind
/// a surface, free where every voxel of the band was seen and is clear, unknown otherwise. The grid covers every
/// voxel the field holds with a ring of unknown cells around it, or is None while the field is empty.
pub fn slice_field(field: &dyn DistanceField, slice: Slice) -> Option<CellGrid> {
    let voxel_m = field.voxel_m();
    let bounds = field.bounds()?;
    let low = (bounds.min.truncate() / voxel_m).round().as_ivec2() - 1;
    let high = (bounds.max.truncate() / voxel_m).round().as_ivec2() + 1;
    let geometry = GridGeometry {
        origin: low.as_vec2() * voxel_m,
        resolution_m: voxel_m,
        width: (high.x - low.x) as usize,
        height: (high.y - low.y) as usize,
    };
    let first_layer = (slice.min_z_m / voxel_m - 0.5).ceil() as i32;
    let last_layer = (slice.max_z_m / voxel_m - 0.5).floor() as i32;
    let mut cells = vec![Cell::Unknown; geometry.len()];
    cells.par_chunks_mut(geometry.width).enumerate().for_each(|(row, cells)| {
        for (column, cell) in cells.iter_mut().enumerate() {
            let column_voxel = low + IVec2::new(column as i32, row as i32);
            let mut all_free = first_layer <= last_layer;
            for layer in first_layer..=last_layer {
                match field.voxel(column_voxel.extend(layer)) {
                    Sample::Observed { distance_m } if distance_m < slice.occupied_below_m => {
                        *cell = Cell::Occupied;
                        all_free = false;
                        break;
                    }
                    Sample::Observed { .. } => {}
                    Sample::Unobserved => all_free = false,
                }
            }
            if all_free {
                *cell = Cell::Free;
            }
        }
    });
    Some(CellGrid { geometry, cells })
}

/// Squared distance in cells from each cell to the nearest source cell, by Felzenszwalb and Huttenlocher's lower
/// envelope of parabolas along rows, then along columns. Infinite where there is no source at all.
pub fn squared_distance_transform(geometry: &GridGeometry, is_source: impl Fn(usize) -> bool + Sync) -> Vec<f64> {
    let (width, height) = (geometry.width, geometry.height);
    let mut along_rows: Vec<f64> = (0..geometry.len()).into_par_iter().map(|i| if is_source(i) { 0.0 } else { f64::INFINITY }).collect();
    along_rows.par_chunks_mut(width).for_each(transform_line);
    let mut transposed = vec![0.0; geometry.len()];
    transposed.par_chunks_mut(height).enumerate().for_each(|(column, line)| {
        for (row, value) in line.iter_mut().enumerate() {
            *value = along_rows[row * width + column];
        }
        transform_line(line);
    });
    along_rows.par_chunks_mut(width).enumerate().for_each(|(row, line)| {
        for (column, value) in line.iter_mut().enumerate() {
            *value = transposed[column * height + row];
        }
    });
    along_rows
}

/// The one-dimensional squared distance transform of sampled function `f`, in place.
fn transform_line(f: &mut [f64]) {
    let n = f.len();
    let Some(first) = f.iter().position(|v| v.is_finite()) else { return };
    let mut parabolas: Vec<usize> = Vec::with_capacity(n);
    let mut boundaries: Vec<f64> = Vec::with_capacity(n + 1);
    parabolas.push(first);
    boundaries.push(f64::NEG_INFINITY);
    boundaries.push(f64::INFINITY);
    let intersection = |q: usize, p: usize, f: &[f64]| {
        let (q, p, fq, fp) = (q as f64, p as f64, f[q], f[p]);
        ((fq + q * q) - (fp + p * p)) / (2.0 * q - 2.0 * p)
    };
    for q in first + 1..n {
        if !f[q].is_finite() {
            continue;
        }
        let mut s = intersection(q, *parabolas.last().expect("one parabola"), f);
        while s <= boundaries[parabolas.len() - 1] {
            parabolas.pop();
            boundaries.pop();
            s = intersection(q, *parabolas.last().expect("the first parabola is never removed"), f);
        }
        parabolas.push(q);
        *boundaries.last_mut().expect("boundaries outnumber parabolas") = s;
        boundaries.push(f64::INFINITY);
    }
    let input: Vec<f64> = f.to_vec();
    let mut k = 0;
    for (q, value) in f.iter_mut().enumerate() {
        while boundaries[k + 1] < q as f64 {
            k += 1;
        }
        let p = parabolas[k];
        *value = (q as f64 - p as f64).powi(2) + input[p];
    }
}

#[cfg(test)]
mod tests {
    use glam::{IVec3, Vec3};

    use super::*;
    use crate::tsdf::{Bounds, SurfacePoint};

    fn geometry(width: usize, height: usize) -> GridGeometry {
        GridGeometry { origin: Vec2::ZERO, resolution_m: 1.0, width, height }
    }

    fn brute_force(geometry: &GridGeometry, sources: &[IVec2]) -> Vec<f64> {
        (0..geometry.len())
            .map(|i| {
                let cell = geometry.cell(i);
                sources.iter().map(|s| (*s - cell).length_squared() as f64).fold(f64::INFINITY, f64::min)
            })
            .collect()
    }

    #[test]
    fn the_distance_transform_is_exact_against_brute_force() {
        let geometry = geometry(13, 9);
        let sources = [IVec2::new(0, 0), IVec2::new(7, 3), IVec2::new(12, 8), IVec2::new(3, 7), IVec2::new(4, 7)];
        let indices: Vec<usize> = sources.iter().map(|s| geometry.index(*s).unwrap()).collect();
        let transform = squared_distance_transform(&geometry, |i| indices.contains(&i));
        assert_eq!(transform, brute_force(&geometry, &sources));
    }

    #[test]
    fn a_single_source_gives_squared_euclidean_distance() {
        let geometry = geometry(5, 4);
        let transform = squared_distance_transform(&geometry, |i| i == geometry.index(IVec2::new(1, 1)).unwrap());
        assert_eq!(transform[geometry.index(IVec2::new(4, 3)).unwrap()], 13.0);
        assert_eq!(transform[geometry.index(IVec2::new(1, 1)).unwrap()], 0.0);
    }

    #[test]
    fn no_source_leaves_every_cell_infinitely_far() {
        let geometry = geometry(3, 2);
        assert!(squared_distance_transform(&geometry, |_| false).iter().all(|d| d.is_infinite()));
    }

    /// A field that is free everywhere inside a box except one occupied column, and unseen above a height.
    struct Fake;

    impl DistanceField for Fake {
        fn voxel_m(&self) -> f32 {
            0.5
        }
        fn cleared(&self) -> Self {
            Fake
        }
        fn integrate(&mut self, _: &crate::keyframe::PosedDepth) {}
        fn voxel(&self, index: IVec3) -> Sample {
            match (index.x, index.y, index.z) {
                (_, _, z) if z > 1 => Sample::Unobserved,
                (0..=3, 3, _) => Sample::Unobserved,
                (2, 1, 1) => Sample::Observed { distance_m: -0.1 },
                (0..=3, 0..=3, _) => Sample::Observed { distance_m: 0.4 },
                _ => Sample::Unobserved,
            }
        }
        fn surface(&self) -> Vec<SurfacePoint> {
            vec![]
        }
        fn bounds(&self) -> Option<Bounds> {
            Some(Bounds { min: Vec3::ZERO, max: Vec3::new(2.0, 2.0, 1.0) })
        }
    }

    #[test]
    fn a_column_is_occupied_by_any_surface_in_the_band_and_free_only_if_all_of_it_is_seen_clear() {
        let grid = slice_field(&Fake, Slice { min_z_m: 0.0, max_z_m: 1.0, occupied_below_m: 0.05 }).unwrap();
        assert_eq!(grid.geometry.origin, Vec2::new(-0.5, -0.5));
        assert_eq!((grid.geometry.width, grid.geometry.height), (6, 6));
        let at = |x: i32, y: i32| grid.cells[grid.geometry.index(IVec2::new(x + 1, y + 1)).unwrap()];
        assert_eq!(at(2, 1), Cell::Occupied);
        assert_eq!(at(0, 0), Cell::Free);
        assert_eq!(at(3, 2), Cell::Free);
        assert_eq!(at(1, 3), Cell::Unknown, "unseen row");
        assert_eq!(at(-1, 0), Cell::Unknown, "the ring outside the field");
        let band_above_what_was_seen = slice_field(&Fake, Slice { min_z_m: 0.5, max_z_m: 1.5, occupied_below_m: 0.05 }).unwrap();
        assert_eq!(band_above_what_was_seen.cells[band_above_what_was_seen.geometry.index(IVec2::new(1, 1)).unwrap()], Cell::Unknown);
    }
}
