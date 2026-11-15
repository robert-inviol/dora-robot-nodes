//! The planes and room of a scene or room_layout message, read back: one keyframe's layout in its floor frame
//! (gravity aligned, origin at the camera, x right, y forward, z up), without the scene's objects.

use dora_node_api::arrow::array::{Array, AsArray, StructArray};
use dora_node_api::arrow::datatypes::{Float32Type, UInt64Type};
use depth_perception::SurfaceKind;
use eyre::{bail, ensure, Result};
use glam::{Vec2, Vec3};

use crate::{checked, scene_fields};

#[derive(Clone, Debug, PartialEq)]
pub struct RoomLayoutPlane {
    pub kind: SurfaceKind,
    pub area_m2: f32,
    /// Convex outline of the plane's points, projected onto it.
    pub outline_m: Vec<Vec3>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RoomBox {
    pub floor_m: f32,
    pub ceiling_m: f32,
    /// Floor-plan corners, anticlockwise seen from above.
    pub footprint_m: Vec<Vec2>,
    /// For each edge ending at the matching corner, whether a wall was seen along it.
    pub edge_seen: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RoomLayout {
    pub frame_id: u64,
    pub planes: Vec<RoomLayoutPlane>,
    /// The room, or why there is none.
    pub room: Result<RoomBox, String>,
}

fn floats_at(list: &dyn Array, row: usize) -> Vec<f32> {
    list.as_list::<i32>().value(row).as_primitive::<Float32Type>().values().to_vec()
}

fn in_threes(values: &[f32], name: &str) -> Result<Vec<Vec3>> {
    ensure!(values.len() % 3 == 0, "{name} holds {} values, not a multiple of 3", values.len());
    Ok(values.chunks_exact(3).map(Vec3::from_slice).collect())
}

pub fn decode_room_layout(array: &dyn Array) -> Result<RoomLayout> {
    let row = checked(array, &scene_fields(), "scene")?;
    let column = |name: &str| row.column_by_name(name).expect("checked wire type");
    let frame_id = column("frame_id").as_primitive::<UInt64Type>().value(0);
    let planes_list = column("planes").as_list::<i32>().value(0);
    let planes_rows: &StructArray = planes_list.as_struct();
    let plane_column = |name: &str| planes_rows.column_by_name(name).expect("checked wire type");
    let mut planes = Vec::with_capacity(planes_rows.len());
    for i in 0..planes_rows.len() {
        let name = plane_column("kind").as_string::<i32>().value(i);
        let Some(kind) = SurfaceKind::from_name(name) else { bail!("frame {frame_id} has a plane of unknown kind {name:?}") };
        planes.push(RoomLayoutPlane {
            kind,
            area_m2: plane_column("area_m2").as_primitive::<Float32Type>().value(i),
            outline_m: in_threes(&floats_at(plane_column("outline_m").as_ref(), i), "a plane outline")?,
        });
    }
    let room_rows = column("room").as_struct();
    let room = if room_rows.is_valid(0) {
        let room_column = |name: &str| room_rows.column_by_name(name).expect("checked wire type");
        let footprint = floats_at(room_column("footprint_m").as_ref(), 0);
        ensure!(footprint.len() % 2 == 0, "frame {frame_id}'s room footprint holds {} values, not pairs", footprint.len());
        let edge_seen: Vec<bool> = room_column("edge_seen").as_list::<i32>().value(0).as_boolean().iter().map(|seen| seen == Some(true)).collect();
        let footprint_m: Vec<Vec2> = footprint.chunks_exact(2).map(Vec2::from_slice).collect();
        ensure!(edge_seen.len() == footprint_m.len(), "frame {frame_id}'s room has {} corners and {} edges", footprint_m.len(), edge_seen.len());
        Ok(RoomBox {
            floor_m: room_column("floor_m").as_primitive::<Float32Type>().value(0),
            ceiling_m: room_column("ceiling_m").as_primitive::<Float32Type>().value(0),
            footprint_m,
            edge_seen,
        })
    } else {
        let problem = column("room_problem").as_string::<i32>();
        ensure!(problem.is_valid(0), "frame {frame_id} has neither a room nor a reason for none");
        Err(problem.value(0).to_owned())
    };
    Ok(RoomLayout { frame_id, planes, room })
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use dora_node_api::arrow::ipc::reader::FileReader;

    use super::*;

    fn fixture(name: &str) -> StructArray {
        let path = format!("{}/../../../lib/dora-rig-messages/dora_rig_messages/fixtures/{name}.arrow", env!("CARGO_MANIFEST_DIR"));
        let mut reader = FileReader::try_new(File::open(&path).unwrap_or_else(|e| panic!("{path}: {e}")), None).unwrap();
        StructArray::from(reader.next().unwrap().unwrap())
    }

    #[test]
    fn a_scene_reads_back_its_planes_and_room() {
        let layout = decode_room_layout(&fixture("scene")).unwrap();
        assert_eq!(layout.frame_id, 42);
        assert_eq!(layout.planes.iter().map(|p| p.kind).collect::<Vec<_>>(), vec![SurfaceKind::Floor, SurfaceKind::Wall]);
        assert_eq!(layout.planes[1].outline_m[2], Vec3::new(0.0, 4.75, 1.5));
        let room = layout.room.unwrap();
        assert_eq!((room.floor_m, room.ceiling_m), (-0.875, 1.5));
        assert_eq!(room.footprint_m[1], Vec2::new(2.0, 0.25));
        assert_eq!(room.edge_seen, vec![false, true, true, true]);
    }

    #[test]
    fn a_scene_without_a_room_reads_back_the_reason() {
        let layout = decode_room_layout(&fixture("scene_without_room")).unwrap();
        assert!(layout.planes.is_empty());
        assert_eq!(layout.room, Err("no ceiling plane found, cannot bound the room".to_owned()));
    }
}
