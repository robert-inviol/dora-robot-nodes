//! The odom pose message of odometry.py.

use arrow_schema::{DataType, Fields};

use crate::{field, items};

pub const DEGENERATE_DIRECTION: &str = "degenerate_direction";
pub const DEGENERATE_SIGMA_M: &str = "degenerate_sigma_m";

pub fn odom_pose_fields() -> Fields {
    Fields::from(vec![
        field("camera_ns", DataType::Int64),
        field("odom_started_camera_ns", DataType::Int64),
        field("depth_frame_id", DataType::UInt64),
        field("position_m", items(DataType::Float32)),
        field("orientation_xyzw", items(DataType::Float32)),
        field("velocity_m_s", items(DataType::Float32)),
        field("position_sigma_m", items(DataType::Float32)),
        field("tilt_sigma_rad", DataType::Float32),
        field("heading_sigma_rad", DataType::Float32),
        field("corrected", DataType::Boolean),
        field("matched_points", DataType::UInt32),
        field(DEGENERATE_DIRECTION, items(DataType::Float32)),
        field(DEGENERATE_SIGMA_M, DataType::Float32),
    ])
}
