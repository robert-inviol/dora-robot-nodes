//! Keyframes as the dense map keeps them: depth, the camera that took it, and where the localizer put it.

use std::sync::Arc;

use depth_perception::{CameraIntrinsics, FloorFrame};
use glam::{Mat3, Vec3};
pub use perception_wire::localization::{FrameId, Tracking};
use perception_wire::localization::KeyframePose as WireKeyframePose;
use perception_wire::navigation::{MapVersion, UnixNanos};
use perception_wire::Keyframe as WireKeyframe;

/// One keyframe's depth, kept so the map can be rebuilt when the localizer moves it.
#[derive(Clone, Debug)]
pub struct DepthKeyframe {
    pub frame_id: FrameId,
    pub captured: UnixNanos,
    pub width: usize,
    pub height: usize,
    pub intrinsics: CameraIntrinsics,
    /// The accelerometer at rest, in camera coordinates: the reaction to gravity.
    pub gravity_reaction: Vec3,
    /// Row major, 0 where the sensor has no reading.
    pub depth_mm: Arc<[u16]>,
}

impl DepthKeyframe {
    /// The depth and camera of a keyframe message, copied out of the message's buffers.
    pub fn from_wire(keyframe: &WireKeyframe) -> DepthKeyframe {
        DepthKeyframe {
            frame_id: FrameId(keyframe.frame_id),
            captured: UnixNanos(keyframe.captured_unix_ns),
            width: keyframe.width,
            height: keyframe.height,
            intrinsics: keyframe.intrinsics,
            gravity_reaction: keyframe.gravity_reaction,
            depth_mm: Arc::from(&keyframe.depth_mm[..]),
        }
    }
}

/// Where the localizer put a keyframe: a turn about the vertical, then a translation, applied to the keyframe's
/// gravity-aligned floor frame (origin at the camera). A lost pose is a guess, and its keyframe is not integrated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyframePose {
    pub frame_id: FrameId,
    pub map_version: MapVersion,
    pub translation: Vec3,
    /// Anticlockwise seen from above.
    pub yaw_rad: f32,
    pub tracking: Tracking,
}

impl From<&WireKeyframePose> for KeyframePose {
    fn from(message: &WireKeyframePose) -> KeyframePose {
        let pose = message.pose;
        KeyframePose {
            frame_id: message.frame_id,
            map_version: message.map_version,
            translation: Vec3::new(pose.x_m, pose.y_m, pose.z_m),
            yaw_rad: pose.yaw_rad,
            tracking: message.tracking,
        }
    }
}

/// Camera coordinates (x right, y down, z along the optical axis) into the map.
#[derive(Clone, Copy, Debug)]
pub struct CameraPose {
    pub camera_to_map: Mat3,
    /// The camera's optical centre in the map.
    pub position: Vec3,
}

impl CameraPose {
    pub fn new(gravity_reaction: Vec3, pose: &KeyframePose) -> CameraPose {
        let floor = FloorFrame::from_gravity_reaction(gravity_reaction);
        let camera_to_floor = Mat3::from_cols(floor.to_floor(Vec3::X), floor.to_floor(Vec3::Y), floor.to_floor(Vec3::Z));
        CameraPose { camera_to_map: Mat3::from_rotation_z(pose.yaw_rad) * camera_to_floor, position: pose.translation }
    }

    #[inline]
    pub fn to_map(&self, camera_point: Vec3) -> Vec3 {
        self.camera_to_map * camera_point + self.position
    }
}

/// A keyframe's depth, placed in the map.
#[derive(Clone, Copy)]
pub struct PosedDepth<'a> {
    pub width: usize,
    pub height: usize,
    pub intrinsics: CameraIntrinsics,
    pub depth_mm: &'a [u16],
    pub pose: CameraPose,
}

impl<'a> PosedDepth<'a> {
    pub fn new(keyframe: &'a DepthKeyframe, pose: &KeyframePose) -> PosedDepth<'a> {
        PosedDepth {
            width: keyframe.width,
            height: keyframe.height,
            intrinsics: keyframe.intrinsics,
            depth_mm: &keyframe.depth_mm,
            pose: CameraPose::new(keyframe.gravity_reaction, pose),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_level_camera_turned_a_quarter_left_looks_along_minus_x() {
        let pose = KeyframePose {
            frame_id: FrameId(1),
            map_version: MapVersion(1),
            translation: Vec3::new(1.0, 2.0, 0.5),
            yaw_rad: std::f32::consts::FRAC_PI_2,
            tracking: Tracking::Tracking,
        };
        let camera = CameraPose::new(Vec3::new(0.0, -9.81, 0.0), &pose);
        let ahead = camera.to_map(Vec3::new(0.0, 0.0, 2.0));
        assert!((ahead - Vec3::new(-1.0, 2.0, 0.5)).length() < 1e-5, "{ahead}");
        let below = camera.to_map(Vec3::new(0.0, 1.0, 0.0));
        assert!((below - Vec3::new(1.0, 2.0, -0.5)).length() < 1e-5, "{below}");
    }
}
