//! One keyframe's masks lifted into the map as voxel sets.

use depth_perception::{object_footprint, DepthImage, FloorFrame, ObjectFootprint, ObjectMask, RangeLimits};
use glam::{Vec2, Vec3};
use perception_wire::NO_SEGMENT;
use rayon::prelude::*;
use tracing::debug;

use crate::crop::PixelBox;
use crate::label::Label;
use crate::settings::DetectionSettings;
use crate::voxels::{VoxelGrid, VoxelSet};
use crate::ObservedFrame;

/// One mask of one keyframe, placed in the map.
#[derive(Clone, Debug)]
pub struct Detection {
    /// Index into the labels of the segments message.
    pub segment_index: u16,
    pub label: Label,
    pub score: f32,
    pub voxels: VoxelSet,
    pub mask_box: PixelBox,
    pub mask_pixels: usize,
}

impl Detection {
    /// How good a view of the object this mask gives: a confident mask that covers many pixels.
    pub fn view_quality(&self) -> f32 {
        self.score * self.mask_pixels as f32
    }
}

/// The keyframe's depth in metres and its floor frame, shared by every mask and by the absence check.
pub struct FrameGeometry {
    pub depth_m: Vec<f32>,
    pub floor: FloorFrame,
}

impl FrameGeometry {
    pub fn of(frame: &ObservedFrame) -> FrameGeometry {
        FrameGeometry {
            depth_m: frame.keyframe.depth_m(),
            floor: FloorFrame::from_gravity_reaction(frame.keyframe.gravity_reaction),
        }
    }
}

/// Every mask with enough depth to place, best score first.
pub fn detect(frame: &ObservedFrame, geometry: &FrameGeometry, grid: VoxelGrid, settings: &DetectionSettings) -> Vec<Detection> {
    let mut detections: Vec<Detection> = (0..frame.segments.labels.len())
        .into_par_iter()
        .filter_map(|index| place_mask(frame, geometry, grid, settings, index as u16))
        .collect();
    detections.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.segment_index.cmp(&b.segment_index)));
    detections
}

fn place_mask(
    frame: &ObservedFrame,
    geometry: &FrameGeometry,
    grid: VoxelGrid,
    settings: &DetectionSettings,
    segment_index: u16,
) -> Option<Detection> {
    let (keyframe, segments) = (&frame.keyframe, &frame.segments);
    let label = Label::new(&segments.labels[segment_index as usize]);
    let score = segments.scores[segment_index as usize];
    let frame_id = keyframe.frame_id;
    if score < settings.min_score {
        debug!(frame_id, %label, score, "mask ignored: score below the minimum");
        return None;
    }
    let pixels: Vec<bool> = segments.segment_ids.iter().map(|&id| id != NO_SEGMENT && id as u16 == segment_index).collect();
    let mask_pixels = pixels.iter().filter(|&&p| p).count();
    let mask_box = PixelBox::around(keyframe.width, pixels.iter().enumerate().filter(|(_, &p)| p).map(|(i, _)| i))?;
    let mask = ObjectMask { width: keyframe.width, height: keyframe.height, pixels: &pixels };
    let depth = DepthImage::new(keyframe.width, keyframe.height, &geometry.depth_m);
    let Some(footprint) =
        object_footprint(&mask, &depth, &keyframe.intrinsics, &geometry.floor, RangeLimits::D455, &settings.footprint())
    else {
        debug!(frame_id, %label, mask_pixels, "mask ignored: too few points with depth on the object's surface");
        return None;
    };
    let voxels = surface_voxels(frame, geometry, grid, &pixels, mask_box, &footprint);
    if voxels.len() < settings.min_voxels {
        debug!(frame_id, %label, voxels = voxels.len(), "mask ignored: too few voxels");
        return None;
    }
    Some(Detection { segment_index, label, score, voxels, mask_box, mask_pixels })
}

/// The map cells of the mask's points that lie within the object's footprint rectangle and height band, which
/// leaves out background seen through or around the mask.
fn surface_voxels(
    frame: &ObservedFrame,
    geometry: &FrameGeometry,
    grid: VoxelGrid,
    pixels: &[bool],
    mask_box: PixelBox,
    footprint: &ObjectFootprint,
) -> VoxelSet {
    let keyframe = &frame.keyframe;
    let slack_m = grid.size_m / 2.0;
    let rectangle = Rectangle::new(&footprint.corners_m, slack_m);
    let band = (footprint.top_above_camera_m - footprint.height_m - slack_m)..=(footprint.top_above_camera_m + slack_m);
    let mut voxels = VoxelSet::default();
    for row in mask_box.top..mask_box.bottom {
        for column in mask_box.left..mask_box.right {
            let index = row * keyframe.width + column;
            let depth_m = geometry.depth_m[index];
            if !pixels[index] || !RangeLimits::D455.contains(depth_m) {
                continue;
            }
            let point: Vec3 = geometry.floor.to_floor(keyframe.intrinsics.deproject(column as f32, row as f32, depth_m));
            if band.contains(&point.z) && rectangle.contains(point.truncate()) {
                voxels.insert(grid.cell(frame.pose.apply(point)));
            }
        }
    }
    voxels
}

/// A rectangle given by four corners in order, grown by a slack on every side.
struct Rectangle {
    origin: Vec2,
    axes: [Vec2; 2],
    lengths: [f32; 2],
    slack_m: f32,
}

impl Rectangle {
    fn new(corners: &[Vec2; 4], slack_m: f32) -> Rectangle {
        let sides = [corners[1] - corners[0], corners[3] - corners[0]];
        Rectangle {
            origin: corners[0],
            axes: sides.map(|s| s.normalize_or_zero()),
            lengths: sides.map(|s| s.length()),
            slack_m,
        }
    }

    fn contains(&self, point: Vec2) -> bool {
        let offset = point - self.origin;
        (0..2).all(|k| (-self.slack_m..=self.lengths[k] + self.slack_m).contains(&offset.dot(self.axes[k])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rectangle_holds_points_within_its_slack() {
        let corners = [Vec2::new(0.0, 0.0), Vec2::new(2.0, 0.0), Vec2::new(2.0, 1.0), Vec2::new(0.0, 1.0)];
        let rectangle = Rectangle::new(&corners, 0.1);
        assert!(rectangle.contains(Vec2::new(1.0, 0.5)));
        assert!(rectangle.contains(Vec2::new(2.05, -0.05)));
        assert!(!rectangle.contains(Vec2::new(2.2, 0.5)));
        assert!(!rectangle.contains(Vec2::new(1.0, 1.2)));
    }
}
