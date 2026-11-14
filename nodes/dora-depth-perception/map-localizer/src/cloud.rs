//! A keyframe's depth as the thinned floor-frame point cloud that registration works on.

use depth_perception::{sample_point_cloud, DepthImage, FloorFrame, RangeLimits};
use glam::Vec3;
use perception_wire::Keyframe;
use room_map::grid::voxel_downsample;

#[derive(Clone, Copy, Debug)]
pub struct CloudSettings {
    /// Every this-many-th pixel each way is back-projected.
    pub stride_px: usize,
    /// Points are thinned to one per voxel this wide.
    pub voxel_m: f32,
    pub range: RangeLimits,
}

impl Default for CloudSettings {
    fn default() -> Self {
        CloudSettings { stride_px: 4, voxel_m: 0.05, range: RangeLimits::D455 }
    }
}

/// The keyframe's points in its gravity-aligned floor frame (x right, y forward, z up, at the camera).
pub fn registration_cloud(keyframe: &Keyframe, settings: &CloudSettings) -> Vec<Vec3> {
    let depth_m = keyframe.depth_m();
    let depth = DepthImage::new(keyframe.width, keyframe.height, &depth_m);
    let floor = FloorFrame::from_gravity_reaction(keyframe.gravity_reaction);
    let cloud = sample_point_cloud(&depth, &keyframe.intrinsics, &floor, settings.range, settings.stride_px);
    voxel_downsample(&cloud.points, settings.voxel_m)
}
