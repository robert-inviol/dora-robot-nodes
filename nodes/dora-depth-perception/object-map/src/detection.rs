//! One keyframe's masks lifted into the map as voxel sets. Each mask's pixels with depth are lifted into the
//! keyframe's floor frame and cleaned in 3D (`cleaning`): flying pixels and noise are dropped, and of the clusters
//! left only the largest is kept. The kept points within the eroded mask become the detection's voxels; the mask's
//! pixels less those whose points were dropped become the mask its view is cut with.

use depth_perception::objects::eroded;
use depth_perception::{FloorFrame, ObjectMask, RangeLimits};
use glam::Vec3;
use perception_wire::NO_INSTANCE;
use rayon::prelude::*;
use tracing::{debug, info};

use crate::cleaning::{clean, CleaningReport};
use crate::crop::PixelBox;
use crate::label::Label;
use crate::settings::DetectionSettings;
use crate::voxels::{VoxelGrid, VoxelSet};
use crate::ObservedFrame;

/// One mask of one keyframe, in the map frame.
#[derive(Clone, Debug)]
pub struct Detection {
    /// Index into the labels of the instance masks message.
    pub instance_index: u16,
    pub label: Label,
    pub score: f32,
    pub voxels: VoxelSet,
    /// The pixels of the object's view, row-major indices in ascending order: the mask less the pixels whose points
    /// cleaning dropped.
    pub view_pixels: Vec<usize>,
    pub mask_box: PixelBox,
    pub cleaning: CleaningReport,
}

impl Detection {
    /// How good a view of the object this mask gives: a confident mask that covers many pixels.
    pub fn view_quality(&self) -> f32 {
        self.score * self.view_pixels.len() as f32
    }

    /// Whether a pixel, by its row-major index, shows the object.
    pub fn shows(&self, pixel: usize) -> bool {
        self.view_pixels.binary_search(&pixel).is_ok()
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
            floor: FloorFrame::from_specific_force(frame.keyframe.specific_force),
        }
    }
}

/// Every mask with enough depth to place, best score first.
pub fn detect(frame: &ObservedFrame, geometry: &FrameGeometry, grid: VoxelGrid, settings: &DetectionSettings) -> Vec<Detection> {
    let mut detections: Vec<Detection> = (0..frame.instance_masks.labels.len())
        .into_par_iter()
        .filter_map(|index| place_mask(frame, geometry, grid, settings, index as u16))
        .collect();
    detections.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.instance_index.cmp(&b.instance_index)));
    detections
}

fn place_mask(
    frame: &ObservedFrame,
    geometry: &FrameGeometry,
    grid: VoxelGrid,
    settings: &DetectionSettings,
    instance_index: u16,
) -> Option<Detection> {
    let (keyframe, instance_masks) = (&frame.keyframe, &frame.instance_masks);
    let label = Label::new(&instance_masks.labels[instance_index as usize]);
    let score = instance_masks.scores[instance_index as usize];
    let frame_id = keyframe.frame_id;
    if score < settings.min_score {
        debug!(frame_id, %label, score, "mask ignored: score below the minimum");
        return None;
    }
    let pixels: Vec<bool> = instance_masks.instance_ids.iter().map(|&id| id != NO_INSTANCE && id as u16 == instance_index).collect();
    let lifted: Vec<(usize, Vec3)> = pixels
        .iter()
        .enumerate()
        .filter(|(index, &in_mask)| in_mask && RangeLimits::D455.contains(geometry.depth_m[*index]))
        .map(|(index, _)| {
            let (column, row) = ((index % keyframe.width) as f32, (index / keyframe.width) as f32);
            (index, geometry.floor.to_floor(keyframe.intrinsics.deproject(column, row, geometry.depth_m[index])))
        })
        .collect();
    let points: Vec<Vec3> = lifted.iter().map(|(_, point)| *point).collect();
    let (keep, cleaning) = clean(&points, &settings.cleaning);
    for points in cleaning.large_dropped_clusters(settings.cleaning.large_cluster_share) {
        let share = points as f32 / cleaning.points as f32;
        info!(frame_id, %label, points, share, mask_points = cleaning.points, "a large cluster under the mask was dropped: possibly a second object");
    }
    let mut dropped = vec![false; pixels.len()];
    for ((index, _), kept) in lifted.iter().zip(&keep) {
        dropped[*index] = !kept;
    }
    let view_pixels: Vec<usize> = (0..pixels.len()).filter(|&index| pixels[index] && !dropped[index]).collect();
    let mask_box = PixelBox::around(keyframe.width, view_pixels.iter().copied())?;
    let mask = ObjectMask { width: keyframe.width, height: keyframe.height, pixels: &pixels };
    let mut inside = vec![false; pixels.len()];
    for index in eroded(&mask, settings.erosion_px) {
        inside[index] = true;
    }
    let surface: Vec<Vec3> = lifted.iter().zip(&keep).filter(|((index, _), &kept)| kept && inside[*index]).map(|((_, point), _)| *point).collect();
    if surface.len() < settings.min_points {
        debug!(frame_id, %label, points = surface.len(), ?cleaning, "mask ignored: too few points on the object's surface");
        return None;
    }
    let voxels: VoxelSet = surface.iter().map(|&point| grid.cell(frame.pose.apply(point))).collect();
    if voxels.len() < settings.min_voxels {
        debug!(frame_id, %label, voxels = voxels.len(), "mask ignored: too few voxels");
        return None;
    }
    Some(Detection { instance_index, label, score, voxels, view_pixels, mask_box, cleaning })
}

