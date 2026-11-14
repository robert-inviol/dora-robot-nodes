//! What one keyframe's depth says about whether an object is still where the map has it.

use depth_perception::{CameraIntrinsics, FloorFrame, RangeLimits};
use glam::{Mat3, Vec3};
use room_map::Pose;

use crate::settings::AbsenceSettings;
use crate::voxels::{VoxelGrid, VoxelSet};

/// The keyframe's camera and depth, to look at map points through.
pub struct CameraView<'a> {
    map_to_floor: Pose,
    floor_to_camera: Mat3,
    intrinsics: CameraIntrinsics,
    width: usize,
    height: usize,
    depth_m: &'a [f32],
}

impl<'a> CameraView<'a> {
    pub fn new(
        pose: Pose,
        floor: &FloorFrame,
        intrinsics: CameraIntrinsics,
        width: usize,
        height: usize,
        depth_m: &'a [f32],
    ) -> CameraView<'a> {
        let camera_to_floor = Mat3::from_cols(floor.to_floor(Vec3::X), floor.to_floor(Vec3::Y), floor.to_floor(Vec3::Z));
        CameraView { map_to_floor: pose.inverse(), floor_to_camera: camera_to_floor.transpose(), intrinsics, width, height, depth_m }
    }

    /// The pixel a map point falls on and its depth along the optical axis, if it is in front of the camera and
    /// inside the image.
    fn project(&self, map_point: Vec3) -> Option<(usize, f32)> {
        let camera = self.floor_to_camera * self.map_to_floor.apply(map_point);
        if camera.z <= 0.0 {
            return None;
        }
        let column = self.intrinsics.fx * camera.x / camera.z + self.intrinsics.ppx;
        let row = self.intrinsics.fy * camera.y / camera.z + self.intrinsics.ppy;
        let inside = (0.0..self.width as f32).contains(&column) && (0.0..self.height as f32).contains(&row);
        inside.then(|| (row as usize * self.width + column as usize, camera.z))
    }
}

/// How an object's voxels compare with the measured depth where they should be.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VoxelTally {
    /// Projected inside the image, within the trusted depth range.
    pub in_view: usize,
    /// The measured depth is at the voxel: the camera sees the object's surface.
    pub surface: usize,
    /// The measured depth is clearly behind the voxel: the camera sees through where the object was.
    pub seen_past: usize,
    /// Something is in front of the voxel, or the sensor has no reading there: no evidence either way.
    pub hidden: usize,
}

impl VoxelTally {
    pub fn judged(&self) -> usize {
        self.surface + self.seen_past
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Too few voxels were judged: out of view, hidden, or too far.
    NoEvidence,
    /// Enough of the surface is where the map has it.
    Present,
    /// The camera sees clearly past most of where the object should be.
    Missing,
    /// Judged voxels disagree; neither present nor missing.
    Inconclusive,
}

pub fn judge(voxels: &VoxelSet, grid: VoxelGrid, view: &CameraView, settings: &AbsenceSettings) -> (Verdict, VoxelTally) {
    let range = RangeLimits { min_m: RangeLimits::D455.min_m, max_m: settings.max_depth_m };
    let mut tally = VoxelTally::default();
    for cell in voxels.cells() {
        let Some((pixel, expected_m)) = view.project(grid.centre(cell)) else { continue };
        if !range.contains(expected_m) {
            continue;
        }
        tally.in_view += 1;
        let measured_m = view.depth_m[pixel];
        if !RangeLimits::D455.contains(measured_m) || measured_m < expected_m - settings.surface_margin_m {
            tally.hidden += 1;
        } else if measured_m > expected_m + settings.surface_margin_m {
            tally.seen_past += 1;
        } else {
            tally.surface += 1;
        }
    }
    let judged = tally.judged();
    let verdict = if judged < settings.min_judged_voxels {
        Verdict::NoEvidence
    } else if tally.surface as f32 >= settings.surface_fraction * judged as f32 {
        Verdict::Present
    } else if tally.seen_past as f32 >= settings.seen_past_fraction * judged as f32 {
        Verdict::Missing
    } else {
        Verdict::Inconclusive
    };
    (verdict, tally)
}
