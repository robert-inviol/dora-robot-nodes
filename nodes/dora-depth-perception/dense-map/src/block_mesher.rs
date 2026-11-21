//! The field's surface as triangles, one mesh block per block, by surface nets.
//!
//! A cell is the cube between eight neighbouring voxel centres. Each cell whose corners are all observed and do not
//! all share a sign holds one vertex: the mean of the points where the distance crosses zero along its edges. Each
//! voxel edge the distance crosses zero along joins the vertices of the four cells around it in a quad. A block owns
//! the quads of the edges that start at its voxels, so it reads its neighbours' voxels one deep, and the vertices it
//! shares with them come out the same in both.

use glam::{IVec3, Vec3};
use perception_wire::space_map::MeshBlock;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::tsdf::{Tsdf, Sample, BLOCK_EDGE_VOXELS};

const EDGE: i32 = BLOCK_EDGE_VOXELS;
/// Voxels read along each axis: one before the block to one after it.
const SPAN: i32 = EDGE + 2;
/// Cells along each axis whose vertices a block's quads use: from the one before the block to its last.
const CELLS: i32 = EDGE + 1;
const AXES: [IVec3; 3] = [IVec3::X, IVec3::Y, IVec3::Z];

/// The block's surface: every quad of an edge that starts at one of its voxels, as two triangles anticlockwise seen
/// from the free side. A block without surface gives a mesh block without triangles.
pub fn mesh_block(field: &dyn Tsdf, block: IVec3) -> MeshBlock {
    BlockSamples::of(field, block).mesh()
}

/// The vertex of a cell inside it, from its lowest corner, in voxels; None unless every corner is observed and the
/// distance crosses zero inside.
fn cell_vertex(distance: impl Fn(IVec3) -> Option<f32>) -> Option<Vec3> {
    let mut corners = [0.0f32; 8];
    for (i, corner) in corners.iter_mut().enumerate() {
        *corner = distance(IVec3::new(i as i32 & 1, (i as i32 >> 1) & 1, (i as i32 >> 2) & 1))?;
    }
    let inside = corners.iter().filter(|d| **d < 0.0).count();
    if inside == 0 || inside == 8 {
        return None;
    }
    let mut sum = Vec3::ZERO;
    let mut crossings = 0;
    for i in 0..8usize {
        for bit in [1usize, 2, 4] {
            let j = i | bit;
            if j == i {
                continue;
            }
            let (d0, d1) = (corners[i], corners[j]);
            if (d0 < 0.0) == (d1 < 0.0) {
                continue;
            }
            let start = Vec3::new((i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32);
            let along = Vec3::new((bit & 1) as f32, ((bit >> 1) & 1) as f32, ((bit >> 2) & 1) as f32);
            sum += start + along * (d0 / (d0 - d1));
            crossings += 1;
        }
    }
    Some(sum / crossings as f32)
}

/// The distances a block's surface is meshed from: its own voxels and its neighbours' one deep.
#[derive(Clone)]
pub struct BlockSamples {
    block: IVec3,
    voxel_m: f32,
    /// SPAN^3 distances from the voxel before the block to the one after it on each axis; NaN where unobserved.
    distance_m: Box<[f32]>,
}

impl BlockSamples {
    pub fn of(field: &dyn Tsdf, block: IVec3) -> BlockSamples {
        let first = block * EDGE;
        let mut distance_m = vec![f32::NAN; (SPAN * SPAN * SPAN) as usize].into_boxed_slice();
        for z in -1..=EDGE {
            for y in -1..=EDGE {
                for x in -1..=EDGE {
                    let local = IVec3::new(x, y, z);
                    if let Sample::Observed { distance_m: d } = field.voxel(first + local) {
                        distance_m[Self::index(local)] = d;
                    }
                }
            }
        }
        BlockSamples { block, voxel_m: field.voxel_m(), distance_m }
    }

    fn index(local: IVec3) -> usize {
        ((local.x + 1) + SPAN * ((local.y + 1) + SPAN * (local.z + 1))) as usize
    }

    /// The distance at a voxel from the one before the block to the one after it, None where unobserved.
    fn at(&self, local: IVec3) -> Option<f32> {
        let d = self.distance_m[Self::index(local)];
        (!d.is_nan()).then_some(d)
    }

    /// The block's surface: every quad of an edge that starts at one of its voxels.
    pub fn mesh(&self) -> MeshBlock {
        let (block, voxel_m) = (self.block, self.voxel_m);
        let first = block * EDGE;
        let distance = |local: IVec3| self.at(local);

        let cell_at = |cell: IVec3| ((cell.x + 1) + CELLS * ((cell.y + 1) + CELLS * (cell.z + 1))) as usize;
        let mut vertex_of = vec![None; (CELLS * CELLS * CELLS) as usize];
        let mut positions_m = Vec::new();
        for z in -1..EDGE {
            for y in -1..EDGE {
                for x in -1..EDGE {
                    let cell = IVec3::new(x, y, z);
                    if let Some(offset) = cell_vertex(|corner| distance(cell + corner)) {
                        vertex_of[cell_at(cell)] = Some(positions_m.len() as u32);
                        positions_m.push(((first + cell).as_vec3() + 0.5 + offset) * voxel_m);
                    }
                }
            }
        }

        let mut triangles = Vec::new();
        for z in 0..EDGE {
            for y in 0..EDGE {
                for x in 0..EDGE {
                    let voxel = IVec3::new(x, y, z);
                    for (a, &along) in AXES.iter().enumerate() {
                        let (Some(start), Some(end)) = (distance(voxel), distance(voxel + along)) else { continue };
                        if (start < 0.0) == (end < 0.0) {
                            continue;
                        }
                        // b and c follow a cyclically, so b x c = a.
                        let (b, c) = (AXES[(a + 1) % 3], AXES[(a + 2) % 3]);
                        let around = [voxel - b - c, voxel - c, voxel, voxel - b].map(|cell| vertex_of[cell_at(cell)]);
                        let [Some(v0), Some(v1), Some(v2), Some(v3)] = around else { continue };
                        // Anticlockwise in (b, c) faces +a; the free side is where the distance is positive.
                        if start < 0.0 {
                            triangles.extend([[v0, v1, v2], [v0, v2, v3]]);
                        } else {
                            triangles.extend([[v0, v2, v1], [v0, v3, v2]]);
                        }
                    }
                }
            }
        }
        if triangles.is_empty() {
            positions_m.clear();
        }
        MeshBlock { block, positions_m, triangles }
    }

    /// The distance interpolated between the eight voxel centres around a vertex of this block's surface, and the
    /// steepness of the field there; None when a voxel around it is unobserved.
    fn interpolate(&self, vertex_m: Vec3) -> Option<FieldAt> {
        let voxels = vertex_m / self.voxel_m - 0.5 - (self.block * EDGE).as_vec3();
        // A vertex lies in one of the cells from the one before the block to its last, or on that cell's far face.
        let corner = voxels.floor().as_ivec3().clamp(IVec3::splat(-1), IVec3::splat(EDGE - 1));
        let t = (voxels - corner.as_vec3()).clamp(Vec3::ZERO, Vec3::ONE);
        let mut d = [0.0f32; 8];
        for (i, value) in d.iter_mut().enumerate() {
            *value = self.at(corner + IVec3::new(i as i32 & 1, (i as i32 >> 1) & 1, (i as i32 >> 2) & 1))?;
        }
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        // Along x first, then y, then z; the gradient from the differences across each axis.
        let x = [lerp(d[0], d[1], t.x), lerp(d[2], d[3], t.x), lerp(d[4], d[5], t.x), lerp(d[6], d[7], t.x)];
        let xy = [lerp(x[0], x[1], t.y), lerp(x[2], x[3], t.y)];
        let across_x = [d[1] - d[0], d[3] - d[2], d[5] - d[4], d[7] - d[6]];
        let across_y = [x[1] - x[0], x[3] - x[2]];
        let gradient = Vec3::new(
            lerp(lerp(across_x[0], across_x[1], t.y), lerp(across_x[2], across_x[3], t.y), t.z),
            lerp(across_y[0], across_y[1], t.z),
            xy[1] - xy[0],
        ) / self.voxel_m;
        Some(FieldAt { distance_m: lerp(xy[0], xy[1], t.z), steepness: gradient.length() })
    }
}

/// The field interpolated at a point.
#[derive(Clone, Copy, Debug, PartialEq)]
struct FieldAt {
    distance_m: f32,
    /// The length of the field's gradient: 1 across a surface seen square on, more at a slant, as distances are
    /// measured along the view.
    steepness: f32,
}

/// How far the surface moved at a vertex of either surface: how much the distance there changed between the field the
/// surface was sent from and the field now, divided by the field's steepness when that is above 1. Measuring the
/// change of the field, not its value, leaves out where surface nets puts a vertex off the interpolated surface.
/// None when either field has an unobserved voxel there.
fn moved_by(sent: &BlockSamples, now: &BlockSamples, vertex_m: Vec3) -> Option<f32> {
    let (before, after) = (sent.interpolate(vertex_m)?, now.interpolate(vertex_m)?);
    let steepness = (0.5 * (before.steepness + after.steepness)).max(1.0);
    Some((after.distance_m - before.distance_m).abs() / steepness)
}

/// When a block's surface is sent again: once enough of the vertices of the surface last sent and of the surface now
/// have moved further than a tolerance ([`moved_by`]). A vertex that appears in or goes from a cell while the field
/// around it stays (a sign flipping near a voxel corner, or the surface passing into the neighbouring block) has not
/// moved; one where the field was or is unobserved, as where a surface is seen for the first time, has.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResendRule {
    pub surface_tolerance_m: f32,
    pub min_moved_vertices: usize,
}

impl ResendRule {
    /// Whether the surface `now`, meshed from `samples`, has moved from the one sent.
    fn moved(&self, sent: &SentBlock, samples: &BlockSamples, now: &MeshBlock) -> bool {
        let compared = now.positions_m.len() + sent.positions_m.len();
        let enough = self.min_moved_vertices.min(compared);
        if enough == 0 {
            return false;
        }
        let moved = |vertex: &&Vec3| moved_by(&sent.samples, samples, **vertex).is_none_or(|d| d > self.surface_tolerance_m);
        now.positions_m.iter().chain(&sent.positions_m).filter(moved).nth(enough - 1).is_some()
    }
}

/// A block's surface as last sent, and the distances it was meshed from.
struct SentBlock {
    samples: BlockSamples,
    /// Empty when the block was sent without surface.
    positions_m: Vec<Vec3>,
}

/// A meshing pass: the mesh blocks to send, and how many blocks were meshed and kept as sent.
#[derive(Debug, Default)]
pub struct MeshPass {
    pub blocks: Vec<MeshBlock>,
    /// Blocks meshed whose surface is within the resend rule of the one last sent, which stands.
    pub kept_as_sent: usize,
}

/// The surfaces last sent, so that a block is sent again only when its surface has moved.
pub struct BlockMesher {
    rule: ResendRule,
    sent: FxHashMap<IVec3, SentBlock>,
}

impl BlockMesher {
    pub fn new(rule: ResendRule) -> BlockMesher {
        BlockMesher { rule, sent: FxHashMap::default() }
    }

    /// Meshes every block whose surface the `changed` blocks can reach (they and their neighbours). A block goes in
    /// the pass's mesh blocks when it has surface and none was sent before, or when the resend rule finds its surface
    /// moved from the one sent, including to none (a mesh block without triangles).
    pub fn mesh_changed_blocks(&mut self, field: &dyn Tsdf, changed: &FxHashSet<IVec3>) -> MeshPass {
        let reach: FxHashSet<IVec3> = changed
            .iter()
            .flat_map(|&block| (-1..=1).flat_map(move |z| (-1..=1).flat_map(move |y| (-1..=1).map(move |x| block + IVec3::new(x, y, z)))))
            .collect();
        let meshed: Vec<(BlockSamples, MeshBlock)> = reach
            .into_par_iter()
            .map(|block| {
                let samples = BlockSamples::of(field, block);
                let mesh_block = samples.mesh();
                (samples, mesh_block)
            })
            .collect();
        let mut pass = MeshPass::default();
        for (samples, mesh_block) in meshed {
            let send = match self.sent.get(&mesh_block.block) {
                None => !mesh_block.triangles.is_empty(),
                Some(sent) => self.rule.moved(sent, &samples, &mesh_block),
            };
            if !send {
                pass.kept_as_sent += usize::from(self.sent.contains_key(&mesh_block.block));
                continue;
            }
            self.sent.insert(mesh_block.block, SentBlock { samples, positions_m: mesh_block.positions_m.clone() });
            pass.blocks.push(mesh_block);
        }
        pass.blocks.sort_by_key(|mesh_block| mesh_block.block.to_array());
        pass
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyframe::PosedDepth;
    use crate::tsdf::{Bounds, SurfacePoint};

    const VOXEL_M: f32 = 0.05;
    const TRUNCATION_M: f32 = 0.2;

    /// A wall at y = `wall_m` seen from y below it, observed through the box of voxels from `low` to `high`.
    struct Wall {
        wall_m: f32,
        low: IVec3,
        high: IVec3,
    }

    impl Tsdf for Wall {
        fn voxel_m(&self) -> f32 {
            VOXEL_M
        }
        fn cleared(&self) -> Wall {
            Wall { wall_m: self.wall_m, low: IVec3::ZERO, high: IVec3::NEG_ONE }
        }
        fn fuse(&mut self, _: &PosedDepth) {}
        fn voxel(&self, index: IVec3) -> Sample {
            if index.cmplt(self.low).any() || index.cmpgt(self.high).any() {
                return Sample::Unobserved;
            }
            let centre_y = (index.y as f32 + 0.5) * VOXEL_M;
            Sample::Observed { distance_m: (self.wall_m - centre_y).clamp(-TRUNCATION_M, TRUNCATION_M) }
        }
        fn surface(&self) -> Vec<SurfacePoint> {
            vec![]
        }
        fn bounds(&self) -> Option<Bounds> {
            None
        }
        fn blocks(&self) -> Vec<IVec3> {
            vec![]
        }
        fn take_changed_blocks(&mut self) -> FxHashSet<IVec3> {
            FxHashSet::default()
        }
    }

    fn wall(wall_m: f32) -> Wall {
        Wall { wall_m, low: IVec3::splat(-40), high: IVec3::splat(40) }
    }

    fn normal(mesh_block: &MeshBlock, triangle: [u32; 3]) -> Vec3 {
        let [a, b, c] = triangle.map(|i| mesh_block.positions_m[i as usize]);
        (b - a).cross(c - a).normalize()
    }

    #[test]
    fn a_wall_meshes_into_triangles_on_it_facing_the_free_side() {
        // The wall at y = 0.5 m crosses block y = 1 (0.4 to 0.8 m).
        let mesh_block = mesh_block(&wall(0.51), IVec3::new(0, 1, 0));
        assert_eq!(mesh_block.triangles.len(), 2 * EDGE as usize * EDGE as usize, "one quad per voxel edge across the wall");
        for p in &mesh_block.positions_m {
            assert!((p.y - 0.51).abs() < 1e-4, "{p}");
        }
        for &triangle in &mesh_block.triangles {
            assert!(normal(&mesh_block, triangle).dot(Vec3::NEG_Y) > 0.999, "faces the camera's side");
        }
    }

    #[test]
    fn a_block_the_wall_does_not_cross_has_no_surface() {
        let mesh_block = mesh_block(&wall(0.51), IVec3::new(0, 3, 0));
        assert!(mesh_block.triangles.is_empty() && mesh_block.positions_m.is_empty());
    }

    #[test]
    fn neighbouring_blocks_meet_without_a_gap() {
        let field = wall(0.51);
        let left = mesh_block(&field, IVec3::new(0, 1, 0));
        let right = mesh_block(&field, IVec3::new(1, 1, 0));
        // The cell between voxel columns 7 and 8 belongs to both; its vertices sit on x = 0.4 m.
        let seam_x = |p: &&Vec3| (p.x - 0.4).abs() < 1e-5;
        let mut from_left: Vec<[u32; 3]> = left.positions_m.iter().filter(seam_x).map(|p| p.to_array().map(f32::to_bits)).collect();
        let mut from_right: Vec<[u32; 3]> = right.positions_m.iter().filter(seam_x).map(|p| p.to_array().map(f32::to_bits)).collect();
        from_left.sort();
        from_right.sort();
        assert!(!from_left.is_empty());
        assert_eq!(from_left, from_right, "the column of vertices where they meet is the same in both");
    }

    #[test]
    fn surface_next_to_unobserved_voxels_is_left_out() {
        let half = Wall { wall_m: 0.51, low: IVec3::splat(-40), high: IVec3::new(3, 40, 40) };
        let mesh_block = mesh_block(&half, IVec3::new(0, 1, 0));
        // Voxels up to x = 3 are observed, so cells up to x = 2 have vertices and edges at x = 0, 1, 2 have quads.
        assert_eq!(mesh_block.triangles.len(), 2 * 3 * EDGE as usize, "no quad uses a cell with an unobserved corner");
        assert!(mesh_block.positions_m.iter().all(|p| p.x <= 0.15 + 1e-6));
    }

    const RULE: ResendRule = ResendRule { surface_tolerance_m: 0.02, min_moved_vertices: 3 };

    fn blocks(blocks: &[[i32; 3]]) -> FxHashSet<IVec3> {
        blocks.iter().map(|&b| IVec3::from_array(b)).collect()
    }

    #[test]
    fn only_moved_surfaces_are_sent_again_and_a_vanished_one_once_as_empty() {
        let mut mesher = BlockMesher::new(RULE);
        let changed = blocks(&[[0, 1, 0]]);
        let first = mesher.mesh_changed_blocks(&wall(0.51), &changed).blocks;
        assert_eq!(first.len(), 9, "the block and its eight neighbours across the wall's plane");
        assert!(first.iter().all(|p| p.block.y == 1 && !p.triangles.is_empty()));
        let again = mesher.mesh_changed_blocks(&wall(0.51), &changed);
        assert!(again.blocks.is_empty(), "nothing changed");
        assert_eq!(again.kept_as_sent, 9);
        assert!(mesher.mesh_changed_blocks(&wall(0.525), &changed).blocks.is_empty(), "a 1.5 cm move is within the tolerance");
        let moved = mesher.mesh_changed_blocks(&wall(0.54), &changed).blocks;
        assert_eq!(moved.len(), 9, "the wall moved 3 cm from what was sent, within the same blocks");
        let gone = mesher.mesh_changed_blocks(&wall(1.51), &changed).blocks;
        assert_eq!(gone.len(), 9);
        assert!(gone.iter().all(|p| p.triangles.is_empty()), "the wall left these blocks");
        assert!(mesher.mesh_changed_blocks(&wall(1.51), &changed).blocks.is_empty(), "an emptied block is sent once");
        let back = mesher.mesh_changed_blocks(&wall(0.54), &changed).blocks;
        assert_eq!(back.len(), 9, "surface where the block was sent empty is new");
    }

    #[test]
    fn a_surface_crossing_into_the_next_block_without_moving_is_not_sent_as_emptied() {
        // Block y = 1 holds voxel centres up to 0.775 m, block y = 2 from 0.825 m: a wall at 0.82 m crosses the edge
        // between them, owned by block 1, and at 0.83 m the next edge, owned by block 2.
        let mut mesher = BlockMesher::new(RULE);
        let changed = blocks(&[[0, 1, 0], [0, 2, 0]]);
        let first = mesher.mesh_changed_blocks(&wall(0.82), &changed).blocks;
        assert!(!first.is_empty() && first.iter().all(|p| p.block.y == 1));
        let crossed = mesher.mesh_changed_blocks(&wall(0.83), &changed).blocks;
        assert!(crossed.iter().all(|p| p.block.y == 2 && !p.triangles.is_empty()), "block 2's surface is new; block 1's stands: {crossed:?}");
        for _ in 0..3 {
            assert!(mesher.mesh_changed_blocks(&wall(0.82), &changed).blocks.is_empty(), "back by 1 cm: within the tolerance of both");
            assert!(mesher.mesh_changed_blocks(&wall(0.83), &changed).blocks.is_empty());
        }
    }

    #[test]
    fn surface_where_nothing_was_seen_before_is_a_move() {
        let mut mesher = BlockMesher::new(RULE);
        let changed = blocks(&[[0, 1, 0]]);
        let half = Wall { wall_m: 0.51, low: IVec3::splat(-40), high: IVec3::new(3, 40, 40) };
        let first = mesher.mesh_changed_blocks(&half, &changed).blocks;
        assert!(first.iter().any(|p| p.block == IVec3::new(0, 1, 0)));
        let whole = mesher.mesh_changed_blocks(&wall(0.51), &changed).blocks;
        assert!(whole.iter().any(|p| p.block == IVec3::new(0, 1, 0)), "the rest of the block's wall came into view");
    }

    #[test]
    fn a_move_is_measured_across_the_surface_and_unknown_where_nothing_was_seen() {
        let block = IVec3::new(0, 1, 0);
        let (sent, now) = (BlockSamples::of(&wall(0.51), block), BlockSamples::of(&wall(0.535), block));
        let moved = moved_by(&sent, &now, Vec3::new(0.12, 0.51, 0.07)).unwrap();
        assert!((moved - 0.025).abs() < 1e-5, "{moved}");
        assert!(moved_by(&sent, &sent, Vec3::new(0.12, 0.49, 0.07)).unwrap() < 1e-6);
        let half = BlockSamples::of(&Wall { wall_m: 0.51, low: IVec3::splat(-40), high: IVec3::new(3, 40, 40) }, block);
        assert_eq!(moved_by(&half, &sent, Vec3::new(0.3, 0.51, 0.07)), None, "nothing seen there before");
    }
}
