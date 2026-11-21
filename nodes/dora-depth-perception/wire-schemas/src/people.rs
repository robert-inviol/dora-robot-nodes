//! The people messages of people.py: 2D poses from the pose model, and tracks placed in 3D.

use arrow_schema::{DataType, Fields};

use crate::{field, floats_struct, list_of, required};

pub fn pixel_box_type() -> DataType {
    floats_struct(&["x_px", "y_px", "w_px", "h_px"])
}

pub fn keypoints_type() -> DataType {
    list_of(floats_struct(&["x_px", "y_px", "score"]), false)
}

pub fn point_type() -> DataType {
    floats_struct(&["x_m", "y_m", "z_m"])
}

pub fn person_poses_fields() -> Fields {
    let person = DataType::Struct(Fields::from(vec![
        field("track_id", DataType::UInt64),
        required("score", DataType::Float32),
        required("box", pixel_box_type()),
        required("keypoints", keypoints_type()),
    ]));
    Fields::from(vec![
        required("frame_id", DataType::UInt64),
        required("captured_camera_ns", DataType::Int64),
        required("width", DataType::UInt16),
        required("height", DataType::UInt16),
        required("people", list_of(person, false)),
    ])
}

pub fn people_tracks_fields() -> Fields {
    let track = DataType::Struct(Fields::from(vec![
        required("track_id", DataType::UInt64),
        required("state", DataType::Utf8),
        required("score", DataType::Float32),
        required("box", pixel_box_type()),
        required("keypoints", keypoints_type()),
        field("position", point_type()),
        field("velocity", floats_struct(&["vx_m_s", "vy_m_s", "vz_m_s"])),
        required("keypoints_3d", list_of(point_type(), true)),
        required("skeleton_3d", list_of(point_type(), true)),
    ]));
    Fields::from(vec![
        required("frame_id", DataType::UInt64),
        required("captured_camera_ns", DataType::Int64),
        required("width", DataType::UInt16),
        required("height", DataType::UInt16),
        required("position_frame", DataType::Utf8),
        field("odom_started_camera_ns", DataType::Int64),
        field("specific_force", list_of(DataType::Float32, false)),
        required("tracks", list_of(track, false)),
    ])
}
