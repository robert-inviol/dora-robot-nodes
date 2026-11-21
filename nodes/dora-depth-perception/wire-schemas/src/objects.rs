//! The object update message of objects.py.

use arrow_schema::{DataType, Fields};

use crate::{field, item};

pub fn map_point_fields() -> Fields {
    Fields::from(vec![field("x_m", DataType::Float32), field("y_m", DataType::Float32)])
}

pub fn voxel_fields() -> Fields {
    Fields::from(vec![field("x", DataType::Int32), field("y", DataType::Int32), field("z", DataType::Int32)])
}

pub fn object_update_fields() -> Fields {
    Fields::from(vec![
        field("kind", DataType::Utf8),
        field("object_id", DataType::UInt64),
        field("merged_into", DataType::UInt64),
        field("label", DataType::Utf8),
        field("score", DataType::Float32),
        field("confidence", DataType::Float32),
        field("seen_count", DataType::UInt32),
        field("frame_id", DataType::UInt64),
        field("captured_unix_ns", DataType::Int64),
        field("map_version", DataType::UInt64),
        field("centre_x_m", DataType::Float32),
        field("centre_y_m", DataType::Float32),
        field("centre_z_m", DataType::Float32),
        field("footprint", DataType::List(item(DataType::Struct(map_point_fields())))),
        field("long_m", DataType::Float32),
        field("short_m", DataType::Float32),
        field("height_m", DataType::Float32),
        field("top_m", DataType::Float32),
        field("crop_jpeg", DataType::Binary),
        field("voxel_m", DataType::Float32),
        field("voxels", DataType::List(item(DataType::Struct(voxel_fields())))),
    ])
}
