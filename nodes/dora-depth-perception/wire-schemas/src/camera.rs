//! The camera calibration message of camera.py.

use arrow_schema::{DataType, Fields};

use crate::{floats_struct, list_of, required};

pub fn stream_intrinsics_type() -> DataType {
    let mut fields = vec![required("width", DataType::UInt16), required("height", DataType::UInt16)];
    let DataType::Struct(floats) = floats_struct(&["fx", "fy", "ppx", "ppy"]) else { unreachable!("a struct type") };
    fields.extend(floats.iter().map(|f| f.as_ref().clone()));
    DataType::Struct(fields.into())
}

pub fn rigid_transform_type() -> DataType {
    DataType::Struct(Fields::from(vec![
        required("rotation", list_of(DataType::Float32, false)),
        required("translation_m", list_of(DataType::Float32, false)),
    ]))
}

pub fn camera_calibration_fields() -> Fields {
    Fields::from(vec![
        required("color", stream_intrinsics_type()),
        required("depth", stream_intrinsics_type()),
        required("depth_to_color", rigid_transform_type()),
        required("depth_to_imu", rigid_transform_type()),
    ])
}
