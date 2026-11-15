//! The dense map's mesh blocks, encoded for the page: each vertex as millimetres from its block's corner, which
//! fits in an i16 because a mesh block's vertices lie within a voxel of its block.

use std::collections::BTreeMap;
use std::io::Write;

use bytes::{BufMut, Bytes, BytesMut};
use flate2::write::GzEncoder;
use flate2::Compression;
use perception_wire::space_map::{MeshBlock, MeshBlocks};

/// The first byte of a binary message to the page that carries mesh blocks.
pub const SURFACE_MESSAGE: u8 = 2;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SurfaceError {
    #[error("block {block:?} has {vertices} vertices; a mesh block for the page holds at most 65535")]
    TooManyVertices { block: [i32; 3], vertices: usize },
    #[error("block {block:?} has a vertex {offset_m} m from its corner, beyond what an i16 of millimetres holds")]
    VertexOutsideBlock { block: [i32; 3], offset_m: f32 },
    #[error("the dense map's blocks changed from {was:?} to {now:?}; the page's surface would no longer fit together")]
    GridChanged { was: BlockGrid, now: BlockGrid },
}

/// How the dense map cuts space into blocks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockGrid {
    pub voxel_m: f32,
    pub block_voxels: u32,
}

impl BlockGrid {
    pub fn of(surface: &MeshBlocks) -> BlockGrid {
        BlockGrid { voxel_m: surface.voxel_m, block_voxels: surface.block_voxels }
    }

    pub fn block_m(&self) -> f32 {
        self.voxel_m * self.block_voxels as f32
    }
}

/// The latest surface of every mesh block that has one, with the whole surface encoded once per change, so a page
/// that connects gets it without encoding it again.
#[derive(Default)]
pub struct MeshCache {
    grid: Option<BlockGrid>,
    blocks: BTreeMap<[i32; 3], MeshBlock>,
}

/// What one `mesh_blocks` message changed: the message every connected page gets, and the whole surface for
/// pages that connect from now on.
pub struct MeshCacheChange {
    pub update: Bytes,
    pub snapshot: Bytes,
}

impl MeshCache {
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Takes one `mesh_blocks` message's grid and mesh blocks.
    pub fn apply(&mut self, grid: BlockGrid, mesh_blocks: &[MeshBlock]) -> Result<MeshCacheChange, SurfaceError> {
        if let Some(kept) = self.grid.filter(|kept| *kept != grid) {
            return Err(SurfaceError::GridChanged { was: kept, now: grid });
        }
        let update = encode_mesh_blocks(grid, mesh_blocks.iter())?;
        self.grid = Some(grid);
        for mesh_block in mesh_blocks {
            let block = mesh_block.block.to_array();
            if mesh_block.triangles.is_empty() {
                self.blocks.remove(&block);
            } else {
                self.blocks.insert(block, mesh_block.clone());
            }
        }
        let snapshot = encode_mesh_blocks(grid, self.blocks.values()).expect("every kept mesh block was encoded when it came");
        Ok(MeshCacheChange { update, snapshot })
    }

    /// Takes a message whose mesh blocks replace every block held: a snapshot, or the first message of a new run of
    /// the dense map, whose grid may differ from the last run's. The update is the whole surface.
    pub fn replace(&mut self, grid: BlockGrid, mesh_blocks: &[MeshBlock]) -> Result<MeshCacheChange, SurfaceError> {
        let blocks: BTreeMap<[i32; 3], MeshBlock> =
            mesh_blocks.iter().filter(|mesh_block| !mesh_block.triangles.is_empty()).map(|mesh_block| (mesh_block.block.to_array(), mesh_block.clone())).collect();
        let snapshot = encode_mesh_blocks(grid, blocks.values())?;
        (self.grid, self.blocks) = (Some(grid), blocks);
        Ok(MeshCacheChange { update: snapshot.clone(), snapshot })
    }

    /// The surface of every block held, in block order.
    pub fn blocks(&self) -> impl ExactSizeIterator<Item = &MeshBlock> {
        self.blocks.values()
    }
}

/// `SURFACE_MESSAGE`, then gzip of: voxel size (f32 m), block edge (u32 voxels), mesh block count (u32), and for each
/// mesh block its block (3 x i32), vertex count (u32), triangle count (u32), every vertex (3 x i16 mm from the block's
/// corner) and every triangle (3 x u16). Little endian. A mesh block without triangles removes its block.
pub fn encode_mesh_blocks<'a>(grid: BlockGrid, mesh_blocks: impl ExactSizeIterator<Item = &'a MeshBlock>) -> Result<Bytes, SurfaceError> {
    let mut body = BytesMut::new();
    body.put_f32_le(grid.voxel_m);
    body.put_u32_le(grid.block_voxels);
    body.put_u32_le(mesh_blocks.len() as u32);
    for mesh_block in mesh_blocks {
        let block = mesh_block.block.to_array();
        if mesh_block.positions_m.len() > u16::MAX as usize {
            return Err(SurfaceError::TooManyVertices { block, vertices: mesh_block.positions_m.len() });
        }
        block.iter().for_each(|&axis| body.put_i32_le(axis));
        body.put_u32_le(mesh_block.positions_m.len() as u32);
        body.put_u32_le(mesh_block.triangles.len() as u32);
        let corner = mesh_block.block.as_vec3() * grid.block_m();
        for position in &mesh_block.positions_m {
            for offset_m in (*position - corner).to_array() {
                let mm = (offset_m * 1000.0).round();
                if mm.abs() > i16::MAX as f32 {
                    return Err(SurfaceError::VertexOutsideBlock { block, offset_m });
                }
                body.put_i16_le(mm as i16);
            }
        }
        for triangle in &mesh_block.triangles {
            triangle.iter().for_each(|&corner| body.put_u16_le(corner as u16));
        }
    }
    let mut encoder = GzEncoder::new(vec![SURFACE_MESSAGE], Compression::fast());
    encoder.write_all(&body).expect("writing to memory");
    Ok(Bytes::from(encoder.finish().expect("writing to memory")))
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use flate2::read::GzDecoder;
    use glam::{IVec3, Vec3};

    use super::*;

    const GRID: BlockGrid = BlockGrid { voxel_m: 0.05, block_voxels: 8 };

    #[test]
    fn a_patch_is_encoded_from_its_block_corner_in_millimetres() {
        let mesh_block = MeshBlock {
            block: IVec3::new(1, -1, 0),
            positions_m: vec![Vec3::new(0.45, -0.1, 0.0), Vec3::new(0.4, -0.4, 0.05), Vec3::new(0.8, -0.4, 0.0)],
            triangles: vec![[0, 1, 2]],
        };
        let encoded = encode_mesh_blocks(GRID, [&mesh_block].into_iter()).unwrap();
        assert_eq!(encoded[0], SURFACE_MESSAGE);
        let mut body = Vec::new();
        GzDecoder::new(&encoded[1..]).read_to_end(&mut body).unwrap();
        let mut expected = 0.05f32.to_le_bytes().to_vec();
        expected.extend(8u32.to_le_bytes());
        expected.extend(1u32.to_le_bytes());
        [1i32, -1, 0].iter().for_each(|v| expected.extend(v.to_le_bytes()));
        expected.extend(3u32.to_le_bytes());
        expected.extend(1u32.to_le_bytes());
        [50i16, 300, 0, 0, 0, 50, 400, 0, 0].iter().for_each(|v| expected.extend(v.to_le_bytes()));
        [0u16, 1, 2].iter().for_each(|v| expected.extend(v.to_le_bytes()));
        assert_eq!(body, expected);
    }

    #[test]
    fn a_vertex_far_outside_its_block_is_refused() {
        let mesh_block = MeshBlock { block: IVec3::ZERO, positions_m: vec![Vec3::new(40.0, 0.0, 0.0)], triangles: vec![[0, 0, 0]] };
        let refused = encode_mesh_blocks(GRID, [&mesh_block].into_iter());
        assert!(matches!(refused, Err(SurfaceError::VertexOutsideBlock { block: [0, 0, 0], .. })), "{refused:?}");
    }

    fn mesh_blocks(blocks: &[(i32, bool)]) -> Vec<MeshBlock> {
        let mesh_block = |x: i32, surface: bool| MeshBlock {
            block: IVec3::new(x, 0, 0),
            positions_m: if surface { vec![Vec3::new(0.4 * x as f32, 0.1, 0.1), Vec3::new(0.4 * x as f32 + 0.1, 0.1, 0.1), Vec3::new(0.4 * x as f32, 0.2, 0.1)] } else { vec![] },
            triangles: if surface { vec![[0, 1, 2]] } else { vec![] },
        };
        blocks.iter().map(|&(x, surface)| mesh_block(x, surface)).collect()
    }

    fn patch_count(encoded: &Bytes) -> u32 {
        let mut body = Vec::new();
        GzDecoder::new(&encoded[1..]).read_to_end(&mut body).unwrap();
        u32::from_le_bytes(body[8..12].try_into().unwrap())
    }

    #[test]
    fn mesh_cache_keeps_blocks_for_late_pages_until_a_patch_empties_them() {
        let mut blocks = MeshCache::default();
        let first = blocks.apply(GRID, &mesh_blocks(&[(0, true), (1, true)])).unwrap();
        assert_eq!((patch_count(&first.update), patch_count(&first.snapshot)), (2, 2));
        let second = blocks.apply(GRID, &mesh_blocks(&[(1, false), (2, true)])).unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(patch_count(&second.update), 2, "the change: one block emptied, one added");
        assert_eq!(patch_count(&second.snapshot), 2, "the whole surface: blocks 0 and 2");
    }

    #[test]
    fn a_replacement_drops_every_block_held_and_may_change_the_grid() {
        let mut blocks = MeshCache::default();
        blocks.apply(GRID, &mesh_blocks(&[(0, true), (1, true)])).unwrap();
        let finer = BlockGrid { voxel_m: 0.04, ..GRID };
        let replaced = blocks.replace(finer, &mesh_blocks(&[(2, true), (3, false)])).unwrap();
        assert_eq!(blocks.blocks().map(|p| p.block.x).collect::<Vec<_>>(), [2]);
        assert_eq!((patch_count(&replaced.update), patch_count(&replaced.snapshot)), (1, 1));
        assert!(blocks.apply(GRID, &mesh_blocks(&[(0, true)])).is_err(), "the replacement's grid holds from now on");
    }

    #[test]
    fn mesh_blocks_on_another_grid_are_refused() {
        let mut blocks = MeshCache::default();
        blocks.apply(GRID, &mesh_blocks(&[(0, true)])).unwrap();
        let finer = BlockGrid { voxel_m: 0.04, ..GRID };
        assert!(matches!(blocks.apply(finer, &mesh_blocks(&[(0, true)])), Err(SurfaceError::GridChanged { .. })));
    }
}
