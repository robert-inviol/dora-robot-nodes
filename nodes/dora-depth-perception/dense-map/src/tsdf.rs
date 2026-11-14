//! A truncated signed distance field: the distance from each voxel to the nearest surface along the camera's view,
//! averaged over every keyframe that saw it.

use glam::{IVec3, Vec3};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::keyframe::PosedDepth;
use crate::settings::TsdfSettings;

/// What the field knows about one voxel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sample {
    Unobserved,
    /// Positive in front of a surface, negative behind it, clamped to the truncation distance.
    Observed { distance_m: f32 },
}

/// A point on a surface and the way it faces, out of the solid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfacePoint {
    pub position: Vec3,
    pub normal: Vec3,
}

/// An axis-aligned box in the map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub min: Vec3,
    pub max: Vec3,
}

/// A distance field on a regular voxel grid whose voxel `i` spans `[i, i + 1) * voxel_m` on each axis.
pub trait DistanceField: Send + Sync + 'static {
    fn voxel_m(&self) -> f32;

    /// An empty field with the same settings.
    fn cleared(&self) -> Self
    where
        Self: Sized;

    fn integrate(&mut self, depth: &PosedDepth);

    fn voxel(&self, index: IVec3) -> Sample;

    fn sample(&self, point: Vec3) -> Sample {
        self.voxel((point / self.voxel_m()).floor().as_ivec3())
    }

    /// Points where the distance crosses zero between neighbouring voxels, with the field's gradient as their normal.
    fn surface(&self) -> Vec<SurfacePoint>;

    /// A box around every voxel the field holds, or None while it holds none.
    fn bounds(&self) -> Option<Bounds>;
}

const BLOCK_EDGE: i32 = 8;
const BLOCK_VOXELS: usize = (BLOCK_EDGE * BLOCK_EDGE * BLOCK_EDGE) as usize;

#[derive(Clone)]
struct Block {
    distance_m: [f32; BLOCK_VOXELS],
    weight: [f32; BLOCK_VOXELS],
}

impl Block {
    fn empty() -> Box<Block> {
        Box::new(Block { distance_m: [0.0; BLOCK_VOXELS], weight: [0.0; BLOCK_VOXELS] })
    }
}

#[inline]
fn local_index(local: IVec3) -> usize {
    (local.x + BLOCK_EDGE * (local.y + BLOCK_EDGE * local.z)) as usize
}

#[inline]
fn split(voxel: IVec3) -> (IVec3, usize) {
    let edge = IVec3::splat(BLOCK_EDGE);
    (voxel.div_euclid(edge), local_index(voxel.rem_euclid(edge)))
}

/// A TSDF stored in 8x8x8-voxel blocks, allocated where keyframes look.
#[derive(Clone)]
pub struct BlockTsdf {
    settings: TsdfSettings,
    blocks: FxHashMap<IVec3, Box<Block>>,
}

impl BlockTsdf {
    pub fn new(settings: TsdfSettings) -> BlockTsdf {
        BlockTsdf { settings, blocks: FxHashMap::default() }
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    fn block_m(&self) -> f32 {
        self.settings.voxel_m * BLOCK_EDGE as f32
    }

    #[inline]
    fn depth_m(&self, depth: &PosedDepth, pixel: usize) -> Option<f32> {
        let depth_m = depth.depth_mm[pixel] as f32 * 0.001;
        (depth_m > self.settings.min_range_m && depth_m < self.settings.max_range_m).then_some(depth_m)
    }

    /// Every block a ray through a sampled pixel crosses, from the camera to just behind the surface it hits.
    fn blocks_in_view(&self, depth: &PosedDepth) -> FxHashSet<IVec3> {
        let stride = self.settings.allocation_stride_px;
        let block_m = self.block_m();
        let rows: Vec<usize> = (0..depth.height).step_by(stride).collect();
        rows.par_iter()
            .fold(FxHashSet::default, |mut blocks, &row| {
                for column in (0..depth.width).step_by(stride) {
                    let Some(depth_m) = self.depth_m(depth, row * depth.width + column) else { continue };
                    let ray = depth.pose.camera_to_map * depth.intrinsics.deproject(column as f32, row as f32, 1.0);
                    let end = depth_m + self.settings.truncation_m;
                    let step = 0.5 * block_m / ray.length();
                    let steps = (end / step).ceil() as usize;
                    for i in 0..=steps {
                        let point = depth.pose.position + ray * (i as f32 * step).min(end);
                        blocks.insert((point / block_m).floor().as_ivec3());
                    }
                }
                blocks
            })
            .reduce(FxHashSet::default, |mut a, b| {
                a.extend(b);
                a
            })
    }

    fn integrate_block(&self, index: IVec3, block: &mut Block, depth: &PosedDepth) {
        let voxel_m = self.settings.voxel_m;
        let (truncation, weight) = (self.settings.truncation_m, self.settings.observation_weight);
        let map_to_camera = depth.pose.camera_to_map.transpose();
        let k = depth.intrinsics;
        let first_voxel = index * BLOCK_EDGE;
        for z in 0..BLOCK_EDGE {
            for y in 0..BLOCK_EDGE {
                for x in 0..BLOCK_EDGE {
                    let local = IVec3::new(x, y, z);
                    let centre = ((first_voxel + local).as_vec3() + 0.5) * voxel_m;
                    let camera = map_to_camera * (centre - depth.pose.position);
                    if camera.z <= self.settings.min_range_m {
                        continue;
                    }
                    let column = (k.fx * camera.x / camera.z + k.ppx + 0.5).floor();
                    let row = (k.fy * camera.y / camera.z + k.ppy + 0.5).floor();
                    if column < 0.0 || row < 0.0 || column >= depth.width as f32 || row >= depth.height as f32 {
                        continue;
                    }
                    let Some(depth_m) = self.depth_m(depth, row as usize * depth.width + column as usize) else { continue };
                    let signed = depth_m - camera.z;
                    if signed < -truncation {
                        continue;
                    }
                    // Behind a surface the reading says less and less about the voxel, so its weight falls away
                    // to nothing at the truncation distance.
                    let weight = if signed < -voxel_m { weight * (truncation + signed) / (truncation - voxel_m) } else { weight };
                    if weight <= 0.0 {
                        continue;
                    }
                    let i = local_index(local);
                    let previous = block.weight[i];
                    let total = previous + weight;
                    block.distance_m[i] = (block.distance_m[i] * previous + signed.min(truncation) * weight) / total;
                    block.weight[i] = total.min(self.settings.max_weight);
                }
            }
        }
    }

    fn voxel_parts(&self, block: &Block, i: usize) -> Sample {
        if block.weight[i] < self.settings.min_observed_weight {
            Sample::Unobserved
        } else {
            Sample::Observed { distance_m: block.distance_m[i] }
        }
    }
}

impl DistanceField for BlockTsdf {
    fn voxel_m(&self) -> f32 {
        self.settings.voxel_m
    }

    fn cleared(&self) -> BlockTsdf {
        BlockTsdf::new(self.settings)
    }

    fn integrate(&mut self, depth: &PosedDepth) {
        let in_view = self.blocks_in_view(depth);
        for index in &in_view {
            self.blocks.entry(*index).or_insert_with(Block::empty);
        }
        let mut targets: Vec<(IVec3, Box<Block>)> =
            in_view.iter().map(|index| (*index, self.blocks.remove(index).expect("allocated above"))).collect();
        targets.par_iter_mut().for_each(|(index, block)| self.integrate_block(*index, block, depth));
        self.blocks.extend(targets);
    }

    fn voxel(&self, index: IVec3) -> Sample {
        let (block, i) = split(index);
        match self.blocks.get(&block) {
            Some(block) => self.voxel_parts(block, i),
            None => Sample::Unobserved,
        }
    }

    fn surface(&self) -> Vec<SurfacePoint> {
        let (voxel_m, truncation_m) = (self.settings.voxel_m, self.settings.truncation_m);
        let distance = |voxel: IVec3| match self.voxel(voxel) {
            Sample::Observed { distance_m } => Some(distance_m),
            Sample::Unobserved => None,
        };
        self.blocks
            .par_iter()
            .flat_map_iter(|(index, block)| {
                let first_voxel = *index * BLOCK_EDGE;
                (0..BLOCK_VOXELS as i32).filter_map(move |i| {
                    let local = IVec3::new(i % BLOCK_EDGE, (i / BLOCK_EDGE) % BLOCK_EDGE, i / (BLOCK_EDGE * BLOCK_EDGE));
                    let Sample::Observed { distance_m } = self.voxel_parts(block, i as usize) else { return None };
                    if distance_m < 0.0 || distance_m >= truncation_m {
                        return None;
                    }
                    let voxel = first_voxel + local;
                    let mut crosses = false;
                    let mut gradient = Vec3::ZERO;
                    for axis in [IVec3::X, IVec3::Y, IVec3::Z] {
                        let (ahead, behind) = (distance(voxel + axis)?, distance(voxel - axis)?);
                        crosses |= ahead < 0.0 || behind < 0.0;
                        gradient += axis.as_vec3() * (ahead - behind) / (2.0 * voxel_m);
                    }
                    let steepness = gradient.length();
                    if !crosses || steepness == 0.0 {
                        return None;
                    }
                    let normal = gradient / steepness;
                    // Distances along the view overstate the distance to a surface seen at a slant by the steepness.
                    let offset = distance_m / steepness;
                    let centre = (voxel.as_vec3() + 0.5) * voxel_m;
                    (offset <= voxel_m).then(|| SurfacePoint { position: centre - normal * offset, normal })
                })
            })
            .collect()
    }

    fn bounds(&self) -> Option<Bounds> {
        let block_m = self.block_m();
        let mut indices = self.blocks.keys();
        let first = *indices.next()?;
        let (low, high) = indices.fold((first, first), |(low, high), index| (low.min(*index), high.max(*index)));
        Some(Bounds { min: low.as_vec3() * block_m, max: (high + 1).as_vec3() * block_m })
    }
}

#[cfg(test)]
mod tests {
    use depth_perception::CameraIntrinsics;
    use glam::Mat3;

    use super::*;
    use crate::keyframe::CameraPose;

    /// A camera at the origin looking along +y at a flat wall `wall_m` away.
    fn facing_wall(wall_m: f32, depth_mm: &mut Vec<u16>) -> PosedDepth<'_> {
        let (width, height) = (64, 48);
        *depth_mm = vec![(wall_m * 1000.0).round() as u16; width * height];
        let camera_to_map = Mat3::from_cols(Vec3::X, Vec3::NEG_Z, Vec3::Y);
        PosedDepth {
            width,
            height,
            intrinsics: CameraIntrinsics { fx: 40.0, fy: 40.0, ppx: 31.5, ppy: 23.5 },
            depth_mm,
            pose: CameraPose { camera_to_map, position: Vec3::ZERO },
        }
    }

    #[test]
    fn a_wall_ahead_is_a_zero_crossing_with_free_space_before_it() {
        let mut depth_mm = Vec::new();
        let depth = facing_wall(2.0, &mut depth_mm);
        let mut tsdf = BlockTsdf::new(TsdfSettings::default());
        tsdf.integrate(&depth);
        let Sample::Observed { distance_m } = tsdf.sample(Vec3::new(0.0, 1.0, 0.0)) else { panic!("unobserved") };
        assert_eq!(distance_m, 0.2, "truncated in front");
        let Sample::Observed { distance_m } = tsdf.sample(Vec3::new(0.01, 1.92, 0.01)) else { panic!("unobserved") };
        assert!((distance_m - 0.075).abs() < 1e-4, "{distance_m}");
        let Sample::Observed { distance_m } = tsdf.sample(Vec3::new(0.01, 2.07, 0.01)) else { panic!("unobserved") };
        assert!((distance_m + 0.075).abs() < 1e-4, "{distance_m}");
        assert_eq!(tsdf.sample(Vec3::new(0.0, 2.5, 0.0)), Sample::Unobserved, "beyond the truncation behind the wall");
        assert_eq!(tsdf.sample(Vec3::new(0.0, -1.0, 0.0)), Sample::Unobserved, "behind the camera");
    }

    #[test]
    fn surface_points_lie_on_the_wall_and_face_the_camera() {
        let mut depth_mm = Vec::new();
        let depth = facing_wall(2.0, &mut depth_mm);
        let mut tsdf = BlockTsdf::new(TsdfSettings::default());
        tsdf.integrate(&depth);
        let surface = tsdf.surface();
        assert!(surface.len() > 100, "{}", surface.len());
        for point in &surface {
            assert!((point.position.y - 2.0).abs() < 0.01, "{point:?}");
            assert!(point.normal.dot(Vec3::NEG_Y) > 0.99, "{point:?}");
        }
    }

    #[test]
    fn repeated_keyframes_average_towards_the_latest_surface() {
        let mut near_mm = Vec::new();
        let mut far_mm = Vec::new();
        let mut tsdf = BlockTsdf::new(TsdfSettings::default());
        tsdf.integrate(&facing_wall(2.0, &mut near_mm));
        tsdf.integrate(&facing_wall(2.1, &mut far_mm));
        let Sample::Observed { distance_m } = tsdf.sample(Vec3::new(0.01, 2.02, 0.01)) else { panic!("unobserved") };
        assert!((distance_m - 0.025).abs() < 1e-4, "halfway between the two walls: {distance_m}");
    }

    #[test]
    fn a_cleared_field_holds_nothing() {
        let mut depth_mm = Vec::new();
        let mut tsdf = BlockTsdf::new(TsdfSettings::default());
        tsdf.integrate(&facing_wall(2.0, &mut depth_mm));
        assert!(tsdf.bounds().is_some());
        assert_eq!(tsdf.cleared().bounds(), None);
    }
}
