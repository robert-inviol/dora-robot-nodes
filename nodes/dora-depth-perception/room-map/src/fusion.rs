//! Every keyframe's coloured points in one voxel map.

use std::collections::HashMap;

use glam::{IVec3, Vec3};

#[derive(Clone, Copy, Default)]
struct Voxel {
    sum: Vec3,
    color_sum: [u32; 3],
    count: u32,
}

/// Coloured points accumulated per voxel: the map is each voxel's mean position and colour.
pub struct VoxelMap {
    cell_m: f32,
    voxels: HashMap<IVec3, Voxel>,
}

pub struct MapPoint {
    pub position: Vec3,
    pub color: [u8; 3],
    pub observations: u32,
}

impl VoxelMap {
    pub fn new(cell_m: f32) -> VoxelMap {
        VoxelMap { cell_m, voxels: HashMap::new() }
    }

    pub fn add(&mut self, points: &[(Vec3, [u8; 3])]) {
        for &(point, color) in points {
            let voxel = self.voxels.entry((point / self.cell_m).floor().as_ivec3()).or_default();
            voxel.sum += point;
            for c in 0..3 {
                voxel.color_sum[c] += color[c] as u32;
            }
            voxel.count += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.voxels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }

    /// Voxels seen at least `min_observations` times; single stray readings are mostly depth noise.
    pub fn points(&self, min_observations: u32) -> Vec<MapPoint> {
        self.voxels
            .values()
            .filter(|v| v.count >= min_observations)
            .map(|v| MapPoint {
                position: v.sum / v.count as f32,
                color: v.color_sum.map(|c| (c / v.count) as u8),
                observations: v.count,
            })
            .collect()
    }
}
