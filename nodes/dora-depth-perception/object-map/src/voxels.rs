//! Object points counted in map-frame cubes, how much two such sets overlap, and the extent they span.

use std::collections::HashSet;

use depth_perception::geometry2d::{convex_hull_indices, min_area_rectangle};
use glam::{IVec3, Vec2, Vec3};
use perception_wire::objects::ObjectExtent;

/// The cube edge that map points are counted in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelGrid {
    pub size_m: f32,
}

impl VoxelGrid {
    pub fn cell(&self, point: Vec3) -> IVec3 {
        (point / self.size_m).floor().as_ivec3()
    }

    pub fn centre(&self, cell: IVec3) -> Vec3 {
        (cell.as_vec3() + 0.5) * self.size_m
    }
}

/// Occupied cells of one object, with their bounding box for quick rejection.
#[derive(Clone, Debug, Default)]
pub struct VoxelSet {
    cells: HashSet<IVec3>,
    min: IVec3,
    max: IVec3,
}

impl VoxelSet {
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub fn cells(&self) -> impl Iterator<Item = IVec3> + '_ {
        self.cells.iter().copied()
    }

    pub fn insert(&mut self, cell: IVec3) {
        if self.cells.is_empty() {
            (self.min, self.max) = (cell, cell);
        } else {
            (self.min, self.max) = (self.min.min(cell), self.max.max(cell));
        }
        self.cells.insert(cell);
    }

    pub fn absorb(&mut self, other: &VoxelSet) {
        other.cells().for_each(|cell| self.insert(cell));
    }

    /// Whether the bounding boxes, each grown by one cell, meet.
    fn may_touch(&self, other: &VoxelSet) -> bool {
        !self.is_empty()
            && !other.is_empty()
            && (self.min - IVec3::ONE).cmple(other.max).all()
            && (other.min - IVec3::ONE).cmple(self.max).all()
    }

    /// The fraction of this set's cells that have a cell of `other` in their 3x3x3 neighbourhood. One cell of
    /// slack absorbs the pose error between the keyframes the two sets were seen from.
    pub fn fraction_touching(&self, other: &VoxelSet) -> f32 {
        if !self.may_touch(other) {
            return 0.0;
        }
        let touching = self.cells().filter(|&cell| neighbourhood(cell).any(|n| other.cells.contains(&n))).count();
        touching as f32 / self.len() as f32
    }

    /// The minimum-area floor-plan rectangle around every cell, and the height band the cells span.
    pub fn extent(&self, grid: VoxelGrid) -> ObjectExtent {
        assert!(!self.is_empty(), "an empty voxel set has no extent");
        let columns: HashSet<(i32, i32)> = self.cells().map(|c| (c.x, c.y)).collect();
        let corners: Vec<Vec2> = columns
            .into_iter()
            .flat_map(|(x, y)| [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)])
            .map(|(x, y)| Vec2::new(x as f32, y as f32) * grid.size_m)
            .collect();
        let hull: Vec<Vec2> = convex_hull_indices(&corners).into_iter().map(|k| corners[k]).collect();
        let footprint_m = min_area_rectangle(&hull);
        let sides = ((footprint_m[1] - footprint_m[0]).length(), (footprint_m[2] - footprint_m[1]).length());
        let bottom_m = self.min.z as f32 * grid.size_m;
        let top_m = (self.max.z + 1) as f32 * grid.size_m;
        let plan_centre = footprint_m.iter().copied().sum::<Vec2>() / 4.0;
        ObjectExtent {
            centre_m: plan_centre.extend((bottom_m + top_m) / 2.0),
            footprint_m,
            long_m: sides.0.max(sides.1),
            short_m: sides.0.min(sides.1),
            height_m: top_m - bottom_m,
            top_m,
        }
    }
}

impl FromIterator<IVec3> for VoxelSet {
    fn from_iter<I: IntoIterator<Item = IVec3>>(cells: I) -> Self {
        let mut set = VoxelSet::default();
        cells.into_iter().for_each(|cell| set.insert(cell));
        set
    }
}

fn neighbourhood(cell: IVec3) -> impl Iterator<Item = IVec3> {
    (-1..=1).flat_map(move |x| (-1..=1).flat_map(move |y| (-1..=1).map(move |z| cell + IVec3::new(x, y, z))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(min: IVec3, max: IVec3) -> VoxelSet {
        (min.x..=max.x)
            .flat_map(|x| (min.y..=max.y).flat_map(move |y| (min.z..=max.z).map(move |z| IVec3::new(x, y, z))))
            .collect()
    }

    #[test]
    fn a_set_shifted_by_one_cell_still_touches_everywhere() {
        let a = block(IVec3::ZERO, IVec3::splat(3));
        let b = block(IVec3::X, IVec3::splat(3) + IVec3::X);
        assert_eq!(a.fraction_touching(&b), 1.0);
    }

    #[test]
    fn far_apart_sets_do_not_touch() {
        let a = block(IVec3::ZERO, IVec3::splat(3));
        let b = block(IVec3::splat(10), IVec3::splat(12));
        assert_eq!(a.fraction_touching(&b), 0.0);
    }

    #[test]
    fn a_small_set_inside_a_large_one_touches_fully_but_not_the_other_way() {
        let large = block(IVec3::ZERO, IVec3::new(9, 9, 0));
        let small = block(IVec3::ZERO, IVec3::new(1, 1, 0));
        assert_eq!(small.fraction_touching(&large), 1.0);
        assert!(large.fraction_touching(&small) < 0.1);
    }

    #[test]
    fn the_extent_of_a_block_is_its_cells_outer_faces() {
        let grid = VoxelGrid { size_m: 0.05 };
        let extent = block(IVec3::new(0, 0, 0), IVec3::new(19, 9, 3)).extent(grid);
        assert!((extent.long_m - 1.0).abs() < 1e-5 && (extent.short_m - 0.5).abs() < 1e-5, "{extent:?}");
        assert!((extent.height_m - 0.2).abs() < 1e-5 && (extent.top_m - 0.2).abs() < 1e-5, "{extent:?}");
        assert!((extent.centre_m - Vec3::new(0.5, 0.25, 0.1)).length() < 1e-5, "{extent:?}");
    }
}
