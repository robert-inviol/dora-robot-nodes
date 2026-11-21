//! The camera calibration message of lib/dora-rig-messages/dora_rig_messages/camera.py.

use dora_node_api::arrow::array::{Array, StructArray};
use eyre::Result;
use glam::{Mat3, Vec3};
use serde::{Deserialize, Serialize};

use crate::serde_wire::{decode_row, encode_row};
pub(crate) use wire_schemas::camera::stream_intrinsics_type;
pub use wire_schemas::camera::camera_calibration_fields;

/// A pinhole model of one image stream at its streamed size; lens distortion is not carried.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamIntrinsics {
    pub width: u16,
    pub height: u16,
    pub fx: f32,
    pub fy: f32,
    pub ppx: f32,
    pub ppy: f32,
}

impl StreamIntrinsics {
    /// The point at `depth_m` along the ray through pixel (`x_px`, `y_px`), in this stream's coordinates.
    pub fn deproject(&self, x_px: f32, y_px: f32, depth_m: f32) -> Vec3 {
        Vec3::new((x_px - self.ppx) / self.fx * depth_m, (y_px - self.ppy) / self.fy * depth_m, depth_m)
    }

    /// The pixel a point in this stream's coordinates lands on, or None when it is not in front of the camera.
    pub fn project(&self, point: Vec3) -> Option<(f32, f32)> {
        (point.z > 0.0).then(|| (point.x / point.z * self.fx + self.ppx, point.y / point.z * self.fy + self.ppy))
    }
}

/// Maps a point from one stream's coordinates into another's: `rotation * point + translation_m`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigidTransform {
    /// Column major, as librealsense gives it.
    pub rotation: [f32; 9],
    pub translation_m: [f32; 3],
}

impl RigidTransform {
    pub fn apply(&self, point: Vec3) -> Vec3 {
        Mat3::from_cols_array(&self.rotation) * point + Vec3::from_array(self.translation_m)
    }

    pub fn inverse(&self) -> RigidTransform {
        let rotation = Mat3::from_cols_array(&self.rotation).transpose();
        RigidTransform {
            rotation: rotation.to_cols_array(),
            translation_m: (-(rotation * Vec3::from_array(self.translation_m))).to_array(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraCalibration {
    pub color: StreamIntrinsics,
    pub depth: StreamIntrinsics,
    pub depth_to_color: RigidTransform,
    pub depth_to_imu: RigidTransform,
}

pub fn encode_camera_calibration(calibration: &CameraCalibration) -> StructArray {
    encode_row(calibration, camera_calibration_fields())
}

pub fn decode_camera_calibration(array: &dyn Array) -> Result<CameraCalibration> {
    decode_row(array, &camera_calibration_fields(), "camera calibration")
}
