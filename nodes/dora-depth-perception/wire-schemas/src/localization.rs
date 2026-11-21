//! The localization messages of localization.py: keyframe poses, pose graph updates and the map to odom
//! correction.

use arrow_schema::{DataType, Fields};

use crate::{field, list_of, required};

pub const POSE_FIELDS: [&str; 4] = ["x_m", "y_m", "z_m", "yaw_rad"];

pub fn keyframe_pose_fields() -> Fields {
    let mut fields = vec![
        field("frame_id", DataType::UInt64),
        field("captured_unix_ns", DataType::Int64),
        field("map_version", DataType::UInt64),
    ];
    fields.extend(POSE_FIELDS.map(|name| field(name, DataType::Float32)));
    fields.extend([
        field("tracking", DataType::Utf8),
        field("selection", DataType::Utf8),
        field("reference_frame_id", DataType::UInt64),
        field("residual_m", DataType::Float32),
        field("inlier_fraction", DataType::Float32),
    ]);
    Fields::from(fields)
}

pub fn map_to_odom_fields() -> Fields {
    let mut fields = vec![
        field("map_version", DataType::UInt64),
        field("valid_from_camera_ns", DataType::Int64),
        field("odom_started_camera_ns", DataType::Int64),
    ];
    fields.extend(POSE_FIELDS.map(|name| field(name, DataType::Float32)));
    fields.extend(["sigma_xy_m", "sigma_z_m", "sigma_yaw_rad"].map(|name| field(name, DataType::Float32)));
    Fields::from(fields)
}

pub fn updated_keyframe_pose_type() -> DataType {
    DataType::Struct(Fields::from(vec![
        required("frame_id", DataType::UInt64),
        required("x_m", DataType::Float32),
        required("y_m", DataType::Float32),
        required("z_m", DataType::Float32),
        required("yaw_rad", DataType::Float32),
    ]))
}

pub fn pose_graph_update_fields() -> Fields {
    Fields::from(vec![required("map_version", DataType::UInt64), required("keyframes", list_of(updated_keyframe_pose_type(), false))])
}
