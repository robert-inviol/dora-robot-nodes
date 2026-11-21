//! The navigation messages of navigation.py: nav goals, goal status, spatial queries and answers.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Fields};

use crate::field;

pub fn map_pose_fields() -> Fields {
    Fields::from(vec![field("x_m", DataType::Float32), field("y_m", DataType::Float32), field("yaw_rad", DataType::Float32)])
}

pub fn nav_goal_fields() -> Fields {
    Fields::from(vec![
        field("goal_id", DataType::UInt64),
        field("kind", DataType::Utf8),
        field("target_id", DataType::UInt64),
        field("target_pose", DataType::Struct(map_pose_fields())),
        field("standoff_m", DataType::Float32),
        field("max_speed_m_s", DataType::Float32),
        field("deadline_unix_ns", DataType::Int64),
        field("map_version", DataType::UInt64),
    ])
}

pub fn goal_status_fields() -> Fields {
    Fields::from(vec![
        field("goal_id", DataType::UInt64),
        field("state", DataType::Utf8),
        field("reason", DataType::Utf8),
        field("updated_unix_ns", DataType::Int64),
    ])
}

pub fn spatial_query_fields() -> Fields {
    Fields::from(vec![
        field("question_id", DataType::UInt64),
        field("asked_unix_ns", DataType::Int64),
        field("asker", DataType::Utf8),
        field("text", DataType::Utf8),
    ])
}

pub fn reference_fields() -> Fields {
    Fields::from(vec![
        field("kind", DataType::Utf8),
        field("id", DataType::UInt64),
        field("x_m", DataType::Float32),
        field("y_m", DataType::Float32),
    ])
}

/// The item field of a spatial answer's references.
pub fn reference_item() -> Arc<Field> {
    Arc::new(Field::new("item", DataType::Struct(reference_fields()), false))
}

pub fn spatial_answer_fields() -> Fields {
    Fields::from(vec![
        field("question_id", DataType::UInt64),
        field("revision", DataType::UInt32),
        field("is_final", DataType::Boolean),
        field("model", DataType::Utf8),
        field("text", DataType::Utf8),
        field("references", DataType::List(reference_item())),
        field("goal_id", DataType::UInt64),
    ])
}
