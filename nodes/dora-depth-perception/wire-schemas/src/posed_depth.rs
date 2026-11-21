//! The posed depth frame of posed_depth.py.

use arrow_schema::{DataType, Fields};

use crate::camera::stream_intrinsics_type;
use crate::{list_of, required};

pub fn posed_depth_frame_fields() -> Fields {
    Fields::from(vec![
        required("frame_id", DataType::UInt64),
        required("camera_ns", DataType::Int64),
        required("odom_started_camera_ns", DataType::Int64),
        required("intrinsics", stream_intrinsics_type()),
        required("camera_position_m", list_of(DataType::Float32, false)),
        required("camera_orientation_xyzw", list_of(DataType::Float32, false)),
        required("depth_mm", list_of(DataType::UInt16, false)),
    ])
}
