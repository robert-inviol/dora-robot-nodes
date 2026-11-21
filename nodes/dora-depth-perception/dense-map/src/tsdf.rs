//! A truncated signed distance field: the distance from each voxel to the nearest surface along the camera's view,
//! averaged over every keyframe that saw it.

use std::sync::Arc;

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

/// A truncated signed distance field (TSDF) on a regular voxel grid whose voxel `i` spans `[i, i + 1) * voxel_m` on each axis.
pub trait Tsdf: Send + Sync + 'static {
    fn voxel_m(&self) -> f32;

    /// An empty field with the same settings.
    fn cleared(&self) -> Self
    where
        Self: Sized;

    fn fuse(&mut self, depth: &PosedDepth);

    fn voxel(&self, index: IVec3) -> Sample;

    fn sample(&self, point: Vec3) -> Sample {
        self.voxel((point / self.voxel_m()).floor().as_ivec3())
    }

    /// Points where the distance crosses zero between neighbouring voxels, with the field's gradient as their normal.
    fn surface(&self) -> Vec<SurfacePoint>;

    /// A box around every voxel the field holds, or None while it holds none.
    fn bounds(&self) -> Option<Bounds>;

    /// The blocks of `BLOCK_EDGE_VOXELS` voxels a side that the field holds: block `b` spans voxels
    /// `b * BLOCK_EDGE_VOXELS` up to the next block's.
    fn blocks(&self) -> Vec<IVec3>;

    /// The blocks fusion changed since the last call; the call empties the record.
    fn take_changed_blocks(&mut self) -> FxHashSet<IVec3>;
}

/// A distance field held in blocks, which depth is fused into block by block, so that part of it can be rebuilt.
pub trait BlockField: Tsdf + Sized {
    /// The blocks a depth frame touches: every block a ray through a sampled pixel crosses, from the camera to
    /// just behind the surface it hits.
    fn footprint(&self, depth: &PosedDepth) -> FxHashSet<IVec3>;

    /// Fuses a depth frame into its footprint, which it returns.
    fn fuse_into_footprint(&mut self, depth: &PosedDepth) -> FxHashSet<IVec3>;

    /// Fuses a depth frame into the blocks of its footprint that are in `within`, leaving every other block alone.
    fn fuse_within(&mut self, depth: &PosedDepth, within: &FxHashSet<IVec3>);

    /// Makes each of `blocks` what it is in `rebuilt`: a block `rebuilt` does not hold is removed.
    fn replace_blocks(&mut self, rebuilt: &Self, blocks: &FxHashSet<IVec3>);

    /// A copy of the field as it stands, for reading on another thread while this one changes; cheap, as blocks are
    /// shared until either side changes them.
    fn snapshot(&self) -> Self;
}

/// The edge of a block, in voxels.
pub const BLOCK_EDGE_VOXELS: i32 = BLOCK_EDGE;

const BLOCK_EDGE: i32 = 8;
const BLOCK_VOXELS: usize = (BLOCK_EDGE * BLOCK_EDGE * BLOCK_EDGE) as usize;

#[derive(Clone)]
struct Block {
    distance_m: [f32; BLOCK_VOXELS],
    weight: [f32; BLOCK_VOXELS],
}

impl Block {
    fn empty() -> Arc<Block> {
        Arc::new(Block { distance_m: [0.0; BLOCK_VOXELS], weight: [0.0; BLOCK_VOXELS] })
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

/// What one depth pixel says.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Reading {
    /// No reading, or one nearer than the min range: nothing is known along the pixel's ray.
    Nothing,
    /// A surface at this depth, within the range fused.
    Surface { depth_m: f32 },
    /// Nothing nearer than this depth, which is at or beyond the max range.
    Beyond { depth_m: f32 },
}

/// What a reading says about one voxel.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Observation {
    /// The ray passed the voxel and ended `margin_m` beyond it, at least the truncation distance.
    Free { margin_m: f32, depth_m: f32 },
    /// The voxel is `signed_m` in front of the reading's surface, within the truncation distance either side.
    Near { signed_m: f32, depth_m: f32 },
}

/// A TSDF stored in 8x8x8-voxel blocks, allocated where keyframes look.
#[derive(Clone)]
pub struct BlockTsdf {
    settings: TsdfSettings,
    /// Shared with snapshots until fused into.
    blocks: FxHashMap<IVec3, Arc<Block>>,
    changed: FxHashSet<IVec3>,
}

impl BlockTsdf {
    pub fn new(settings: TsdfSettings) -> BlockTsdf {
        BlockTsdf { settings, blocks: FxHashMap::default(), changed: FxHashSet::default() }
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    fn block_m(&self) -> f32 {
        self.settings.voxel_m * BLOCK_EDGE as f32
    }

    /// The share of the observation weight a reading at `depth_m` adds: its inverse variance relative to a reading at
    /// the full weight range. Stereo depth noise grows with the square of the range, so its variance with the fourth
    /// power; nearer readings add the full weight. A voxel at its weight cap then wanders with the same noise at any
    /// range, and a near view of it outweighs a far one.
    #[inline]
    fn range_weight(&self, depth_m: f32) -> f32 {
        let ratio = self.settings.full_weight_range_m / depth_m;
        (ratio * ratio * ratio * ratio).min(1.0)
    }

    /// What the reading at a pixel says.
    #[inline]
    fn reading(&self, depth: &PosedDepth, pixel: usize) -> Reading {
        let depth_m = depth.depth_mm[pixel] as f32 * 0.001;
        if depth_m <= self.settings.min_range_m {
            Reading::Nothing
        } else if depth_m < self.settings.max_range_m {
            Reading::Surface { depth_m }
        } else {
            Reading::Beyond { depth_m }
        }
    }

    /// Every block a ray through a sampled pixel crosses: from the camera to just behind the surface it hits, or to
    /// the max range for a reading beyond it.
    fn blocks_in_view(&self, depth: &PosedDepth) -> FxHashSet<IVec3> {
        let stride = self.settings.allocation_stride_px;
        let block_m = self.block_m();
        let rows: Vec<usize> = (0..depth.height).step_by(stride).collect();
        rows.par_iter()
            .fold(FxHashSet::default, |mut blocks, &row| {
                for column in (0..depth.width).step_by(stride) {
                    let ray = depth.pose.camera_to_map * depth.intrinsics.deproject(column as f32, row as f32, 1.0);
                    // Lengths along the ray in metres of depth along the optical axis, as `ray` has unit depth.
                    let end = match self.reading(depth, row * depth.width + column) {
                        Reading::Nothing => continue,
                        Reading::Surface { depth_m } => depth_m + self.settings.truncation_m,
                        Reading::Beyond { .. } => self.settings.max_range_m,
                    };
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

    /// Folds one reading into every voxel of the block that projects onto a pixel with one, with the reading's range
    /// weight. A voxel at least the truncation distance in front of the reading is seen free and takes the truncation
    /// distance; a reading beyond the max range carves the voxels within it so and is not a surface. A voxel near the
    /// reading takes its signed distance, with a weight falling away to nothing behind it at the truncation distance.
    /// A pixel without a reading says nothing, about free space or surface. Free and near readings weigh the same at
    /// one range, so that noise carrying a reading either side of the truncation distance does not pull a surface.
    ///
    /// A reading that contradicts what a voxel holds halves the voxel's weight before it is averaged in, so that a
    /// surface that has gone (a person who walked off) or come (a box set down) takes over within a few frames, while
    /// a lone wrong reading moves a surface seen many times by little; see [`BlockTsdf::contradicts`].
    fn fuse_block(&self, index: IVec3, block: &mut Block, depth: &PosedDepth) {
        let voxel_m = self.settings.voxel_m;
        let (truncation, observation_weight) = (self.settings.truncation_m, self.settings.observation_weight);
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
                    let observation = match self.reading(depth, row as usize * depth.width + column as usize) {
                        Reading::Nothing => continue,
                        Reading::Surface { depth_m } if depth_m - camera.z >= truncation => Observation::Free { margin_m: depth_m - camera.z, depth_m },
                        Reading::Beyond { depth_m } if depth_m - camera.z >= truncation && camera.z < self.settings.max_range_m => {
                            Observation::Free { margin_m: depth_m - camera.z, depth_m }
                        }
                        Reading::Beyond { .. } => continue,
                        Reading::Surface { depth_m } => {
                            let signed_m = depth_m - camera.z;
                            if signed_m < -truncation {
                                continue;
                            }
                            Observation::Near { signed_m, depth_m }
                        }
                    };
                    let (signed, weight) = match observation {
                        Observation::Free { depth_m, .. } => (truncation, observation_weight * self.range_weight(depth_m)),
                        Observation::Near { signed_m, depth_m } => {
                            let weight = observation_weight * self.range_weight(depth_m);
                            // Behind a surface the reading says less and less about the voxel, so its weight falls
                            // away to nothing at the truncation distance.
                            let weight = if signed_m < -voxel_m { weight * (truncation + signed_m) / (truncation - voxel_m) } else { weight };
                            (signed_m, weight)
                        }
                    };
                    if weight <= 0.0 {
                        continue;
                    }
                    let i = local_index(local);
                    let mut previous = block.weight[i];
                    if previous > 0.0 && self.contradicts(observation, block.distance_m[i]) {
                        previous *= 0.5;
                    }
                    let total = previous + weight;
                    block.distance_m[i] = (block.distance_m[i] * previous + signed * weight) / total;
                    block.weight[i] = total.min(self.settings.max_weight);
                }
            }
        }
    }

    /// Whether an observation says the voxel holding `distance_m` has changed, beyond what the reading's noise
    /// (`depth_noise_at_1_m` times the square of its range) explains: a ray that surely passed the voxel, ending
    /// `contradiction_sigmas` of its noise beyond the truncation distance behind it, where the voxel holds a surface
    /// or what lies behind one (less than `min_contradiction_m` short of the truncation distance); or a reading near
    /// the voxel, in front of or at its surface, whose signed distance differs from the voxel's by more than
    /// `contradiction_sigmas` of its noise and by more than `min_contradiction_m`. Readings well behind a surface,
    /// whose weight is falling away, contradict nothing.
    fn contradicts(&self, observation: Observation, distance_m: f32) -> bool {
        let t = &self.settings;
        let noise_m = |depth_m: f32| t.depth_noise_at_1_m * depth_m * depth_m;
        match observation {
            Observation::Free { margin_m, depth_m } => {
                margin_m >= t.truncation_m + t.contradiction_sigmas * noise_m(depth_m) && distance_m < t.truncation_m - t.min_contradiction_m
            }
            Observation::Near { signed_m, depth_m } => {
                signed_m >= -t.voxel_m && (signed_m - distance_m).abs() > (t.contradiction_sigmas * noise_m(depth_m)).max(t.min_contradiction_m)
            }
        }
    }

    /// Fuses a depth frame into each of `targets`, allocating the blocks not held yet.
    fn fuse_blocks(&mut self, depth: &PosedDepth, targets: impl Iterator<Item = IVec3>) {
        let mut taken: Vec<(IVec3, Arc<Block>)> =
            targets.map(|index| (index, self.blocks.remove(&index).unwrap_or_else(Block::empty))).collect();
        taken.par_iter_mut().for_each(|(index, block)| self.fuse_block(*index, Arc::make_mut(block), depth));
        self.changed.extend(taken.iter().map(|(index, _)| *index));
        self.blocks.extend(taken);
    }

    fn voxel_parts(&self, block: &Block, i: usize) -> Sample {
        if block.weight[i] < self.settings.min_observed_weight {
            Sample::Unobserved
        } else {
            Sample::Observed { distance_m: block.distance_m[i] }
        }
    }
}

impl Tsdf for BlockTsdf {
    fn voxel_m(&self) -> f32 {
        self.settings.voxel_m
    }

    fn cleared(&self) -> BlockTsdf {
        BlockTsdf::new(self.settings)
    }

    fn fuse(&mut self, depth: &PosedDepth) {
        self.fuse_into_footprint(depth);
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

    fn blocks(&self) -> Vec<IVec3> {
        self.blocks.keys().copied().collect()
    }

    fn take_changed_blocks(&mut self) -> FxHashSet<IVec3> {
        std::mem::take(&mut self.changed)
    }
}

impl BlockField for BlockTsdf {
    fn footprint(&self, depth: &PosedDepth) -> FxHashSet<IVec3> {
        self.blocks_in_view(depth)
    }

    fn fuse_into_footprint(&mut self, depth: &PosedDepth) -> FxHashSet<IVec3> {
        let footprint = self.blocks_in_view(depth);
        self.fuse_blocks(depth, footprint.iter().copied());
        footprint
    }

    fn fuse_within(&mut self, depth: &PosedDepth, within: &FxHashSet<IVec3>) {
        let footprint = self.blocks_in_view(depth);
        self.fuse_blocks(depth, footprint.into_iter().filter(|index| within.contains(index)));
    }

    fn replace_blocks(&mut self, rebuilt: &BlockTsdf, blocks: &FxHashSet<IVec3>) {
        for index in blocks {
            match rebuilt.blocks.get(index) {
                Some(block) => self.blocks.insert(*index, block.clone()),
                None => self.blocks.remove(index),
            };
        }
        self.changed.extend(blocks.iter().copied());
    }

    fn snapshot(&self) -> BlockTsdf {
        self.clone()
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
    fn a_wall_ahead_is_a_zero_crossing_with_occupancy_grid_before_it() {
        let mut depth_mm = Vec::new();
        let depth = facing_wall(2.0, &mut depth_mm);
        let mut tsdf = BlockTsdf::new(TsdfSettings::default());
        tsdf.fuse(&depth);
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
        tsdf.fuse(&depth);
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
        tsdf.fuse(&facing_wall(2.0, &mut near_mm));
        tsdf.fuse(&facing_wall(2.06, &mut far_mm));
        let Sample::Observed { distance_m } = tsdf.sample(Vec3::new(0.01, 2.02, 0.01)) else { panic!("unobserved") };
        assert!((distance_m - 0.005).abs() < 1e-4, "halfway between the two walls, which do not contradict: {distance_m}");
    }

    #[test]
    fn a_surface_seen_through_is_carved_free_within_a_few_frames_and_one_reading_moves_it_little() {
        let (mut wall_mm, mut beyond_mm) = (Vec::new(), Vec::new());
        let mut tsdf = BlockTsdf::new(TsdfSettings::default());
        for _ in 0..100 {
            tsdf.fuse(&facing_wall(2.0, &mut wall_mm));
        }
        let at_wall = Vec3::new(0.01, 2.02, 0.01);
        tsdf.fuse(&facing_wall(3.0, &mut beyond_mm));
        let Sample::Observed { distance_m } = tsdf.sample(at_wall) else { panic!("unobserved") };
        assert!(distance_m < 0.0, "one reading through the wall leaves it: {distance_m}");
        let frames = (1..=20)
            .find(|_| {
                tsdf.fuse(&facing_wall(3.0, &mut beyond_mm));
                matches!(tsdf.sample(at_wall), Sample::Observed { distance_m } if distance_m > 0.0)
            })
            .expect("carved within 20 frames");
        assert!(frames <= 6, "a wall seen 100 times is carved in {frames} more frames");
    }

    #[test]
    fn fusion_records_the_blocks_it_changed_until_they_are_taken() {
        let mut depth_mm = Vec::new();
        let mut tsdf = BlockTsdf::new(TsdfSettings::default());
        tsdf.fuse(&facing_wall(2.0, &mut depth_mm));
        let changed = tsdf.take_changed_blocks();
        assert!(!changed.is_empty());
        assert_eq!(changed.len(), tsdf.blocks().len(), "every block was new");
        assert!(tsdf.take_changed_blocks().is_empty(), "taken once");
    }

    #[test]
    fn a_cleared_field_holds_nothing() {
        let mut depth_mm = Vec::new();
        let mut tsdf = BlockTsdf::new(TsdfSettings::default());
        tsdf.fuse(&facing_wall(2.0, &mut depth_mm));
        assert!(tsdf.bounds().is_some());
        assert_eq!(tsdf.cleared().bounds(), None);
    }
}
