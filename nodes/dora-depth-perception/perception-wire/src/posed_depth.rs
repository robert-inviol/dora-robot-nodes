//! The posed depth frame of lib/dora-rig-messages/dora_rig_messages/posed_depth.py, as a single-row Arrow struct.
//! Field names, types and nullability must match that module exactly; the tests read its golden fixture.
//!
//! A posed depth frame is one depth frame downsampled for mapping, with the odometry's pose of the depth camera that
//! took it. The camera frame is the depth camera's own (x right, y down, z forward, metres); the pose turns camera
//! coordinates into the odom frame `odom_started` names. It is the odometry's pose at exactly the frame's camera time,
//! composed with the depth camera's extrinsics on the body (the IMU), never interpolated.

use std::sync::Arc;

use dora_node_api::arrow::array::{Array, ArrayRef, AsArray, Float32Array, Int64Array, StructArray, UInt16Array, UInt64Array};
use dora_node_api::arrow::buffer::ScalarBuffer;
use dora_node_api::arrow::datatypes::{DataType, Float32Type, Int64Type, UInt16Type, UInt64Type};
use eyre::{bail, Result};
use glam::{Quat, Vec3};

use crate::camera::{stream_intrinsics_type, StreamIntrinsics};
use crate::localization::{CameraNanos, FrameId};
pub use wire_schemas::posed_depth::posed_depth_frame_fields;
use crate::{checked, list_values, one_list, scalar};

#[derive(Clone, Debug, PartialEq)]
pub struct PosedDepthFrame {
    /// The depth frame's id, as the camera node numbered it; the same as its odom pose's `depth_frame_id`.
    pub frame_id: FrameId,
    /// The depth frame's capture time, and its odom pose's `camera_ns`.
    pub camera_ns: CameraNanos,
    /// The odom frame the pose is in, named by the camera time at which odometry started.
    pub odom_started: CameraNanos,
    /// The pinhole model of the downsampled image, at its size.
    pub intrinsics: StreamIntrinsics,
    /// The depth camera's optical centre in the odom frame.
    pub camera_position_m: Vec3,
    /// Turns a camera-frame vector into the odom frame.
    pub camera_orientation: Quat,
    /// Row major, `intrinsics.width` by `intrinsics.height`, 0 where there is no reading.
    pub depth_mm: ScalarBuffer<u16>,
}

fn floats<const N: usize>(row: &StructArray, name: &str) -> Result<[f32; N]> {
    let values = list_values::<Float32Type>(row, name);
    match <[f32; N]>::try_from(&values[..]) {
        Ok(values) => Ok(values),
        Err(_) => bail!("posed depth frame {name} has {} values, not {N}", values.len()),
    }
}

pub fn decode_posed_depth_frame(array: &dyn Array) -> Result<PosedDepthFrame> {
    let row = checked(array, &posed_depth_frame_fields(), "posed depth frame")?;
    if let Some(null) = row.fields().iter().zip(row.columns()).find(|(_, column)| column.is_null(0)) {
        bail!("posed depth frame {} is null", null.0.name());
    }
    let intrinsics_row = row.column_by_name("intrinsics").expect("checked wire type").as_struct();
    let intrinsics = StreamIntrinsics {
        width: scalar::<UInt16Type>(intrinsics_row, "width"),
        height: scalar::<UInt16Type>(intrinsics_row, "height"),
        fx: scalar::<Float32Type>(intrinsics_row, "fx"),
        fy: scalar::<Float32Type>(intrinsics_row, "fy"),
        ppx: scalar::<Float32Type>(intrinsics_row, "ppx"),
        ppy: scalar::<Float32Type>(intrinsics_row, "ppy"),
    };
    let depth_mm = list_values::<UInt16Type>(row, "depth_mm");
    let pixels = intrinsics.width as usize * intrinsics.height as usize;
    if depth_mm.len() != pixels {
        bail!("posed depth frame depth has {} values for {}x{}", depth_mm.len(), intrinsics.width, intrinsics.height);
    }
    let frame = PosedDepthFrame {
        frame_id: FrameId(scalar::<UInt64Type>(row, "frame_id")),
        camera_ns: CameraNanos(scalar::<Int64Type>(row, "camera_ns")),
        odom_started: CameraNanos(scalar::<Int64Type>(row, "odom_started_camera_ns")),
        intrinsics,
        camera_position_m: Vec3::from_array(floats(row, "camera_position_m")?),
        camera_orientation: Quat::from_array(floats(row, "camera_orientation_xyzw")?),
        depth_mm,
    };
    if frame.camera_ns < frame.odom_started {
        bail!("posed depth frame at camera time {} ns is before its odom frame started, at {} ns", frame.camera_ns.0, frame.odom_started.0);
    }
    Ok(frame)
}

pub fn encode_posed_depth_frame(frame: &PosedDepthFrame) -> StructArray {
    let k = frame.intrinsics;
    let DataType::Struct(intrinsics_fields) = stream_intrinsics_type() else { unreachable!("a struct type") };
    let intrinsics_columns: Vec<ArrayRef> = vec![
        Arc::new(UInt16Array::from(vec![k.width])),
        Arc::new(UInt16Array::from(vec![k.height])),
        Arc::new(Float32Array::from(vec![k.fx])),
        Arc::new(Float32Array::from(vec![k.fy])),
        Arc::new(Float32Array::from(vec![k.ppx])),
        Arc::new(Float32Array::from(vec![k.ppy])),
    ];
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(vec![frame.frame_id.0])),
        Arc::new(Int64Array::from(vec![frame.camera_ns.0])),
        Arc::new(Int64Array::from(vec![frame.odom_started.0])),
        Arc::new(StructArray::new(intrinsics_fields, intrinsics_columns, None)),
        one_list(Arc::new(Float32Array::from(frame.camera_position_m.to_array().to_vec()))),
        one_list(Arc::new(Float32Array::from(frame.camera_orientation.to_array().to_vec()))),
        one_list(Arc::new(UInt16Array::new(frame.depth_mm.clone(), None))),
    ];
    StructArray::new(posed_depth_frame_fields(), columns, None)
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
    fn the_posed_depth_frame_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("posed_depth_frame");
        let frame = decode_posed_depth_frame(&golden).unwrap();
        let expected = PosedDepthFrame {
            frame_id: FrameId(42),
            camera_ns: CameraNanos(1_762_560_000_100_000_000),
            odom_started: CameraNanos(1_762_559_990_000_000_000),
            intrinsics: StreamIntrinsics { width: 3, height: 2, fx: 210.75, fy: 210.75, ppx: 211.875, ppy: 119.625 },
            camera_position_m: Vec3::new(1.25, -0.5, 0.25),
            camera_orientation: Quat::from_xyzw(0.5, -0.5, 0.5, 0.5),
            depth_mm: ScalarBuffer::from(vec![0, 1200, 1210, 2450, 65535, 400]),
        };
        assert_eq!(frame, expected);
        assert_eq!(encode_posed_depth_frame(&frame), golden);
    }

    #[test]
    fn a_posed_depth_frame_whose_depth_does_not_fill_its_image_is_rejected() {
        let mut frame = decode_posed_depth_frame(&fixture("posed_depth_frame")).unwrap();
        frame.intrinsics.width = 4;
        let error = decode_posed_depth_frame(&encode_posed_depth_frame(&frame)).unwrap_err();
        assert!(error.to_string().contains("6 values for 4x2"), "{error}");
    }

    #[test]
    fn a_posed_depth_frame_from_before_its_odom_frame_started_is_rejected() {
        let mut frame = decode_posed_depth_frame(&fixture("posed_depth_frame")).unwrap();
        frame.odom_started = CameraNanos(frame.camera_ns.0 + 1);
        let error = decode_posed_depth_frame(&encode_posed_depth_frame(&frame)).unwrap_err();
        assert!(error.to_string().contains("before its odom frame started"), "{error}");
    }

    #[test]
    fn a_map_to_odom_is_not_read_as_a_posed_depth_frame() {
        let error = decode_posed_depth_frame(&fixture("map_to_odom")).unwrap_err();
        assert!(error.to_string().contains("posed depth frame wire type"), "{error}");
    }
}
