//! The perception messages of perception.py: keyframes, their imu samples, labels, instance masks and scenes.

use arrow_schema::{DataType, Fields};

use crate::{field, item, items};

pub fn keyframe_fields() -> Fields {
    Fields::from(vec![
        field("frame_id", DataType::UInt64),
        field("captured_unix_ns", DataType::Int64),
        field("width", DataType::UInt16),
        field("height", DataType::UInt16),
        field("fx", DataType::Float32),
        field("fy", DataType::Float32),
        field("ppx", DataType::Float32),
        field("ppy", DataType::Float32),
        field("specific_force", items(DataType::Float32)),
        field("depth_mm", items(DataType::UInt16)),
        field("color_jpeg", DataType::Binary),
    ])
}

pub fn imu_fields() -> Fields {
    Fields::from(vec![
        field("frame_id", DataType::UInt64),
        field("keyframe_camera_ns", DataType::Int64),
        field("accel_camera_ns", items(DataType::Int64)),
        field("accel_m_s2", items(DataType::Float32)),
        field("gyro_camera_ns", items(DataType::Int64)),
        field("gyro_rad_s", items(DataType::Float32)),
    ])
}

pub fn labels_fields() -> Fields {
    Fields::from(vec![
        field("model", DataType::Utf8),
        field("frame_id", DataType::UInt64),
        field("labels", items(DataType::Utf8)),
        field("complete", DataType::Boolean),
    ])
}

pub fn instance_masks_fields() -> Fields {
    Fields::from(vec![
        field("frame_id", DataType::UInt64),
        field("width", DataType::UInt16),
        field("height", DataType::UInt16),
        field("labels", items(DataType::Utf8)),
        field("scores", items(DataType::Float32)),
        field("instance_ids", items(DataType::Int16)),
    ])
}

pub fn plane_fields() -> Fields {
    Fields::from(vec![
        field("kind", DataType::Utf8),
        field("normal", items(DataType::Float32)),
        field("offset_m", DataType::Float32),
        field("area_m2", DataType::Float32),
        field("outline_m", items(DataType::Float32)),
    ])
}

pub fn room_fields() -> Fields {
    Fields::from(vec![
        field("floor_m", DataType::Float32),
        field("ceiling_m", DataType::Float32),
        field("footprint_m", items(DataType::Float32)),
        field("edge_seen", items(DataType::Boolean)),
    ])
}

pub fn object_fields() -> Fields {
    Fields::from(vec![
        field("label", DataType::Utf8),
        field("instance_index", DataType::UInt16),
        field("score", DataType::Float32),
        field("point_count", DataType::UInt32),
        field("distance_m", DataType::Float32),
        field("footprint_long_m", DataType::Float32),
        field("footprint_short_m", DataType::Float32),
        field("height_m", DataType::Float32),
        field("top_m", DataType::Float32),
        field("corners_m", items(DataType::Float32)),
    ])
}

/// A scene: one keyframe's planes, room and objects. A scene without objects is a room layout.
pub fn scene_fields() -> Fields {
    Fields::from(vec![
        field("frame_id", DataType::UInt64),
        field("captured_unix_ns", DataType::Int64),
        field("camera_forward", items(DataType::Float32)),
        field("camera_up", items(DataType::Float32)),
        field("planes", DataType::List(item(DataType::Struct(plane_fields())))),
        field("room", DataType::Struct(room_fields())),
        field("room_problem", DataType::Utf8),
        field("objects", DataType::List(item(DataType::Struct(object_fields())))),
    ])
}
