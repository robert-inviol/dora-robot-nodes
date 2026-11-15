//! The field's surface as triangles, one mesh block per block, by surface nets.
//!
//! A cell is the cube between eight neighbouring voxel centres. Each cell whose corners are all observed and do not
//! all share a sign holds one vertex: the mean of the points where the distance crosses zero along its edges. Each
//! voxel edge the distance crosses zero along joins the vertices of the four cells around it in a quad. A block owns
//! the quads of the edges that start at its voxels, so it reads its neighbours' voxels one deep, and the vertices it
//! shares with them come out the same in both.

use std::time::{Duration, Instant};

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
    let voxel_m = field.voxel_m();
    let first = block * EDGE;
    let sample_at = |local: IVec3| ((local.x + 1) + SPAN * ((local.y + 1) + SPAN * (local.z + 1))) as usize;
    let mut samples = vec![None; (SPAN * SPAN * SPAN) as usize];
    for z in -1..=EDGE {
        for y in -1..=EDGE {
            for x in -1..=EDGE {
                let local = IVec3::new(x, y, z);
                if let Sample::Observed { distance_m } = field.voxel(first + local) {
                    samples[sample_at(local)] = Some(distance_m);
                }
            }
        }
    }
    let distance = |local: IVec3| samples[sample_at(local)];

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

/// When a block's surface is sent again: only once it has changed enough to see, and not sooner than a period after
/// it was last sent, so that noise at the edge of what the camera sees does not resend it every pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResendRule {
    /// A vertex in a cell that has one both times has moved further than this.
    pub min_vertex_move_m: f32,
    /// Or more than this fraction of the cells holding a vertex appeared or went.
    pub min_changed_cells: f32,
    pub min_block_period: Duration,
}

/// The cell each vertex lies in, keyed by cell: surface nets put at most one vertex in a cell, inside it.
fn vertex_cells(mesh_block: &MeshBlock, voxel_m: f32) -> FxHashMap<IVec3, Vec3> {
    mesh_block.positions_m.iter().map(|&p| (((p / voxel_m) - 0.5).floor().as_ivec3(), p)).collect()
}

fn changed_enough(sent: &FxHashMap<IVec3, Vec3>, now: &FxHashMap<IVec3, Vec3>, rule: &ResendRule) -> bool {
    let differing = now.keys().filter(|cell| !sent.contains_key(cell)).count() + sent.keys().filter(|cell| !now.contains_key(cell)).count();
    if differing as f32 > rule.min_changed_cells * sent.len().max(now.len()) as f32 {
        return true;
    }
    let limit = rule.min_vertex_move_m * rule.min_vertex_move_m;
    now.iter().any(|(cell, p)| sent.get(cell).is_some_and(|q| p.distance_squared(*q) > limit))
}

/// A meshing pass: the mesh blocks to send, and the blocks whose change waits for their period to pass.
#[derive(Debug, Default)]
pub struct MeshPass {
    pub blocks: Vec<MeshBlock>,
    pub deferred: Vec<IVec3>,
}

/// The surfaces last sent and when, so that a block is sent again only when its surface changed enough to see.
pub struct BlockMesher {
    rule: ResendRule,
    /// The vertex cells of every block whose surface was last sent with triangles.
    sent: FxHashMap<IVec3, FxHashMap<IVec3, Vec3>>,
    sent_at: FxHashMap<IVec3, Instant>,
}

impl BlockMesher {
    pub fn new(rule: ResendRule) -> BlockMesher {
        BlockMesher { rule, sent: FxHashMap::default(), sent_at: FxHashMap::default() }
    }

    /// Meshes every block whose surface the `changed` blocks can reach (they and their neighbours). A block goes in
    /// the pass's mesh blocks when its surface is new, changed enough by the rule, or gone (a mesh block without triangles),
    /// and its period has passed; a block that changed within its period is deferred.
    pub fn mesh_changed_blocks(&mut self, field: &dyn Tsdf, changed: &FxHashSet<IVec3>, now: Instant) -> MeshPass {
        let voxel_m = field.voxel_m();
        let reach: FxHashSet<IVec3> = changed
            .iter()
            .flat_map(|&block| (-1..=1).flat_map(move |z| (-1..=1).flat_map(move |y| (-1..=1).map(move |x| block + IVec3::new(x, y, z)))))
            .collect();
        let meshed: Vec<MeshBlock> = reach.into_par_iter().map(|block| mesh_block(field, block)).collect();
        let mut pass = MeshPass::default();
        for mesh_block in meshed {
            let cells = (!mesh_block.triangles.is_empty()).then(|| vertex_cells(&mesh_block, voxel_m));
            let differs = match (self.sent.get(&mesh_block.block), &cells) {
                (None, None) => false,
                (Some(_), None) | (None, Some(_)) => true,
                (Some(sent), Some(now)) => changed_enough(sent, now, &self.rule),
            };
            if !differs {
                continue;
            }
            let waiting = self.sent_at.get(&mesh_block.block).is_some_and(|at| now.duration_since(*at) < self.rule.min_block_period);
            if waiting {
                pass.deferred.push(mesh_block.block);
                continue;
            }
            self.sent_at.insert(mesh_block.block, now);
            match cells {
                Some(cells) => self.sent.insert(mesh_block.block, cells),
                None => self.sent.remove(&mesh_block.block),
            };
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

    const EVERY_PASS: ResendRule = ResendRule { min_vertex_move_m: 0.005, min_changed_cells: 0.1, min_block_period: Duration::ZERO };

    #[test]
    fn only_changed_surfaces_are_sent_again_and_a_vanished_one_once_as_empty() {
        let mut mesher = BlockMesher::new(EVERY_PASS);
        let changed: FxHashSet<IVec3> = [IVec3::new(0, 1, 0)].into_iter().collect();
        let now = Instant::now();
        let first = mesher.mesh_changed_blocks(&wall(0.51), &changed, now).blocks;
        assert_eq!(first.len(), 9, "the block and its eight neighbours across the wall's plane");
        assert!(first.iter().all(|p| p.block.y == 1 && !p.triangles.is_empty()));
        assert!(mesher.mesh_changed_blocks(&wall(0.51), &changed, now).blocks.is_empty(), "nothing changed");
        assert!(mesher.mesh_changed_blocks(&wall(0.513), &changed, now).blocks.is_empty(), "a 3 mm move is below the minimum");
        let moved = mesher.mesh_changed_blocks(&wall(0.53), &changed, now).blocks;
        assert_eq!(moved.len(), 9, "the wall moved 2 cm from what was sent, within the same blocks");
        let gone = mesher.mesh_changed_blocks(&wall(1.51), &changed, now).blocks;
        assert_eq!(gone.len(), 9);
        assert!(gone.iter().all(|p| p.triangles.is_empty()), "the wall left these blocks");
        assert!(mesher.mesh_changed_blocks(&wall(1.51), &changed, now).blocks.is_empty(), "an emptied block is sent once");
    }

    #[test]
    fn a_block_that_changes_again_within_its_period_waits_for_it() {
        let rule = ResendRule { min_block_period: Duration::from_secs(5), ..EVERY_PASS };
        let mut mesher = BlockMesher::new(rule);
        let changed: FxHashSet<IVec3> = [IVec3::new(0, 1, 0)].into_iter().collect();
        let start = Instant::now();
        assert_eq!(mesher.mesh_changed_blocks(&wall(0.51), &changed, start).blocks.len(), 9);
        let early = mesher.mesh_changed_blocks(&wall(0.53), &changed, start + Duration::from_secs(1));
        assert!(early.blocks.is_empty());
        assert_eq!(early.deferred.len(), 9, "the change waits");
        let later = mesher.mesh_changed_blocks(&wall(0.53), &changed, start + Duration::from_secs(5));
        assert_eq!(later.blocks.len(), 9, "and goes once the period has passed");
    }

    #[test]
    fn a_few_cells_coming_and_going_or_small_moves_are_not_a_change() {
        let cells = |count: i32, shift_m: f32| -> FxHashMap<IVec3, Vec3> {
            (0..count).map(|i| (IVec3::new(i, 0, 0), Vec3::new(i as f32 * 0.05 + shift_m, 0.0, 0.0))).collect()
        };
        let rule = ResendRule { min_vertex_move_m: 0.02, min_changed_cells: 0.1, min_block_period: Duration::ZERO };
        let sent = cells(100, 0.0);
        assert!(!changed_enough(&sent, &cells(95, 0.01), &rule), "5 of 100 cells gone, the rest moved 1 cm");
        assert!(changed_enough(&sent, &cells(85, 0.0), &rule), "15 of 100 cells gone");
        assert!(changed_enough(&sent, &cells(100, 0.03), &rule), "every vertex moved 3 cm");
    }
}
