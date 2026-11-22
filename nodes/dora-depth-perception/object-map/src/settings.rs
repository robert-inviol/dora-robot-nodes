//! Every threshold that decides an object's identity, in one place, readable from a TOML file.

use std::time::Duration;

use depth_perception::objects::FootprintSettings;
use serde::Deserialize;

use crate::cleaning::CleaningSettings;
use crate::crop::CropBackground;

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ObjectMapSettings {
    /// Edge of the cubes that object points are counted in.
    pub voxel_size_m: f32,
    /// Keyframes that wait for their pose and instance masks; the oldest is dropped beyond this.
    pub pending_frames: usize,
    pub detection: DetectionSettings,
    pub association: AssociationSettings,
    pub confirmation: ConfirmationSettings,
    pub merge: MergeSettings,
    pub absence: AbsenceSettings,
    pub crop: CropSettings,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DetectionSettings {
    /// Masks scored below this are ignored.
    pub min_score: f32,
    /// A mask's edge can lie on the surface the object stands on or leans against, which cleaning cannot tell from
    /// the object where they touch; only points this far inside the mask place the object.
    pub erosion_px: usize,
    /// Fewer points than this on the object's surface, after cleaning and erosion, and the mask is ignored.
    pub min_points: usize,
    /// Fewer voxels than this and the mask is ignored.
    pub min_voxels: usize,
    pub cleaning: CleaningSettings,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AssociationSettings {
    /// A detection is of an object when at least this fraction of its voxels touch the object's voxels and, unless
    /// their labels agree, this fraction of the object's touch its own.
    pub min_overlap: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConfirmationSettings {
    /// Observations, in different keyframes, that make a candidate an object.
    pub observations: u32,
    /// A candidate not confirmed this long after its first observation is dropped.
    pub window_s: f64,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MergeSettings {
    /// Two objects are one when this fraction of the smaller one's voxels touch the larger and, unless their labels
    /// agree, this fraction of the larger one's touch the smaller.
    pub min_overlap: f32,
    /// Captured time between merge passes.
    pub interval_s: f64,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AbsenceSettings {
    /// A measured depth within this of an object voxel sees its surface; farther than this behind it sees past it.
    pub surface_margin_m: f32,
    /// Object voxels farther than this from the camera are not judged.
    pub max_depth_m: f32,
    /// Fewer judged voxels (surface seen or seen past) than this and the keyframe says nothing about the object.
    pub min_judged_voxels: usize,
    /// At least this fraction of judged voxels on the surface and the object is there.
    pub surface_fraction: f32,
    /// At least this fraction of judged voxels seen past and the keyframe counts a miss.
    pub seen_past_fraction: f32,
    /// Misses in a row that make an object disappear.
    pub misses_to_disappear: u32,
    /// An object with an agreeing label confirmed this soon after a disappearance is the same object, moved.
    pub moved_window_s: f64,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CropSettings {
    pub jpeg_quality: u8,
    /// A view replaces the best one when its score times visible area is at least this many times larger.
    pub min_improvement: f32,
    /// The view is the box around the mask grown on every side by this share of the box's width or height.
    pub context_margin: f32,
    /// What the view shows around the mask.
    pub background: CropBackground,
}

impl Default for ObjectMapSettings {
    fn default() -> Self {
        ObjectMapSettings {
            voxel_size_m: 0.05,
            pending_frames: 32,
            detection: DetectionSettings::default(),
            association: AssociationSettings::default(),
            confirmation: ConfirmationSettings::default(),
            merge: MergeSettings::default(),
            absence: AbsenceSettings::default(),
            crop: CropSettings::default(),
        }
    }
}

impl Default for DetectionSettings {
    fn default() -> Self {
        let footprint = FootprintSettings::default();
        DetectionSettings {
            min_score: 0.3,
            erosion_px: footprint.erosion_px,
            min_points: footprint.min_points,
            min_voxels: 4,
            cleaning: CleaningSettings::default(),
        }
    }
}

impl Default for AssociationSettings {
    fn default() -> Self {
        AssociationSettings { min_overlap: 0.3 }
    }
}

impl Default for ConfirmationSettings {
    fn default() -> Self {
        ConfirmationSettings { observations: 3, window_s: 60.0 }
    }
}

impl Default for MergeSettings {
    fn default() -> Self {
        MergeSettings { min_overlap: 0.6, interval_s: 10.0 }
    }
}

impl Default for AbsenceSettings {
    fn default() -> Self {
        AbsenceSettings {
            surface_margin_m: 0.15,
            max_depth_m: 5.0,
            min_judged_voxels: 10,
            surface_fraction: 0.3,
            seen_past_fraction: 0.7,
            misses_to_disappear: 3,
            moved_window_s: 600.0,
        }
    }
}

impl Default for CropSettings {
    fn default() -> Self {
        CropSettings { jpeg_quality: 85, min_improvement: 1.2, context_margin: 0.25, background: CropBackground::Dimmed }
    }
}

impl ConfirmationSettings {
    pub fn window(&self) -> Duration {
        Duration::from_secs_f64(self.window_s)
    }
}

impl MergeSettings {
    pub fn interval(&self) -> Duration {
        Duration::from_secs_f64(self.interval_s)
    }
}

impl AbsenceSettings {
    pub fn moved_window(&self) -> Duration {
        Duration::from_secs_f64(self.moved_window_s)
    }
}
