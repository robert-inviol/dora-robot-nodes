//! The dense map's surface patches, encoded for the page: each vertex as millimetres from its block's corner, which
//! fits in an i16 because a patch's vertices lie within a voxel of its block.

use std::io::Write;

use bytes::{BufMut, Bytes, BytesMut};
use flate2::write::GzEncoder;
use flate2::Compression;
use perception_wire::space_map::SurfacePatch;

/// The first byte of a binary message to the page that carries surface patches.
pub const SURFACE_MESSAGE: u8 = 2;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SurfaceError {
    #[error("block {block:?} has {vertices} vertices; a patch for the page holds at most 65535")]
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
    pub fn block_m(&self) -> f32 {
        self.voxel_m * self.block_voxels as f32
    }
}

/// `SURFACE_MESSAGE`, then gzip of: voxel size (f32 m), block edge (u32 voxels), patch count (u32), and for each
/// patch its block (3 x i32), vertex count (u32), triangle count (u32), every vertex (3 x i16 mm from the block's
/// corner) and every triangle (3 x u16). Little endian. A patch without triangles removes its block.
pub fn encode_patches<'a>(grid: BlockGrid, patches: impl ExactSizeIterator<Item = &'a SurfacePatch>) -> Result<Bytes, SurfaceError> {
    let mut body = BytesMut::new();
    body.put_f32_le(grid.voxel_m);
    body.put_u32_le(grid.block_voxels);
    body.put_u32_le(patches.len() as u32);
    for patch in patches {
        let block = patch.block.to_array();
        if patch.positions_m.len() > u16::MAX as usize {
            return Err(SurfaceError::TooManyVertices { block, vertices: patch.positions_m.len() });
        }
        block.iter().for_each(|&axis| body.put_i32_le(axis));
        body.put_u32_le(patch.positions_m.len() as u32);
        body.put_u32_le(patch.triangles.len() as u32);
        let corner = patch.block.as_vec3() * grid.block_m();
        for position in &patch.positions_m {
            for offset_m in (*position - corner).to_array() {
                let mm = (offset_m * 1000.0).round();
                if mm.abs() > i16::MAX as f32 {
                    return Err(SurfaceError::VertexOutsideBlock { block, offset_m });
                }
                body.put_i16_le(mm as i16);
            }
        }
        for triangle in &patch.triangles {
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
        let patch = SurfacePatch {
            block: IVec3::new(1, -1, 0),
            positions_m: vec![Vec3::new(0.45, -0.1, 0.0), Vec3::new(0.4, -0.4, 0.05), Vec3::new(0.8, -0.4, 0.0)],
            triangles: vec![[0, 1, 2]],
        };
        let encoded = encode_patches(GRID, [&patch].into_iter()).unwrap();
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
        let patch = SurfacePatch { block: IVec3::ZERO, positions_m: vec![Vec3::new(40.0, 0.0, 0.0)], triangles: vec![[0, 0, 0]] };
        let refused = encode_patches(GRID, [&patch].into_iter());
        assert!(matches!(refused, Err(SurfaceError::VertexOutsideBlock { block: [0, 0, 0], .. })), "{refused:?}");
    }
}
