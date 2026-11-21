//! The space_map messages of space_map.py: occupancy grid, topology, keep out zones and mesh blocks.

use arrow_schema::{DataType, Fields};

use crate::{field, items};

pub fn point_fields() -> Fields {
    Fields::from(vec![field("x_m", DataType::Float32), field("y_m", DataType::Float32)])
}

pub fn occupancy_grid_fields() -> Fields {
    Fields::from(vec![
        field("map_version", DataType::UInt64),
        field("built_unix_ns", DataType::Int64),
        field("origin_x_m", DataType::Float32),
        field("origin_y_m", DataType::Float32),
        field("resolution_m", DataType::Float32),
        field("width", DataType::UInt32),
        field("height", DataType::UInt32),
        field("band_min_m", DataType::Float32),
        field("band_max_m", DataType::Float32),
        field("clearance_mm", items(DataType::UInt16)),
    ])
}

pub fn room_fields() -> Fields {
    Fields::from(vec![
        field("room_id", DataType::UInt64),
        field("polygon", items(DataType::Struct(point_fields()))),
        field("floor_m", DataType::Float32),
        field("ceiling_m", DataType::Float32),
    ])
}

pub fn doorway_fields() -> Fields {
    Fields::from(vec![
        field("doorway_id", DataType::UInt64),
        field("room_a", DataType::UInt64),
        field("room_b", DataType::UInt64),
        field("x_m", DataType::Float32),
        field("y_m", DataType::Float32),
        field("width_m", DataType::Float32),
    ])
}

pub fn place_fields() -> Fields {
    Fields::from(vec![
        field("place_id", DataType::UInt64),
        field("room_id", DataType::UInt64),
        field("x_m", DataType::Float32),
        field("y_m", DataType::Float32),
        field("clearance_m", DataType::Float32),
    ])
}

pub fn edge_fields() -> Fields {
    Fields::from(vec![
        field("place_a", DataType::UInt64),
        field("place_b", DataType::UInt64),
        field("length_m", DataType::Float32),
        field("min_clearance_m", DataType::Float32),
    ])
}

pub fn topology_fields() -> Fields {
    Fields::from(vec![
        field("map_version", DataType::UInt64),
        field("built_unix_ns", DataType::Int64),
        field("rooms", items(DataType::Struct(room_fields()))),
        field("doorways", items(DataType::Struct(doorway_fields()))),
        field("places", items(DataType::Struct(place_fields()))),
        field("edges", items(DataType::Struct(edge_fields()))),
    ])
}

pub fn zone_fields() -> Fields {
    Fields::from(vec![
        field("object_id", DataType::UInt64),
        field("label", DataType::Utf8),
        field("polygon", items(DataType::Struct(point_fields()))),
        field("margin_m", DataType::Float32),
    ])
}

pub fn keep_out_fields() -> Fields {
    Fields::from(vec![field("map_version", DataType::UInt64), field("zones", items(DataType::Struct(zone_fields())))])
}

pub fn mesh_block_fields() -> Fields {
    Fields::from(vec![
        field("block_x", DataType::Int32),
        field("block_y", DataType::Int32),
        field("block_z", DataType::Int32),
        field("positions_m", items(DataType::Float32)),
        field("triangles", items(DataType::UInt32)),
    ])
}

pub fn mesh_blocks_fields() -> Fields {
    Fields::from(vec![
        field("map_version", DataType::UInt64),
        field("built_unix_ns", DataType::Int64),
        field("run_started_unix_ns", DataType::Int64),
        field("sequence", DataType::UInt64),
        field("kind", DataType::Utf8),
        field("voxel_m", DataType::Float32),
        field("block_voxels", DataType::UInt32),
        field("blocks", items(DataType::Struct(mesh_block_fields()))),
    ])
}

pub fn mesh_snapshot_request_fields() -> Fields {
    Fields::from(vec![field("requested_unix_ns", DataType::Int64)])
}
