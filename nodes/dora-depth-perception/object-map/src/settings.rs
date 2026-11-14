//! Every threshold that decides an object's identity, in one place, readable from a TOML file.

use std::time::Duration;

use depth_perception::objects::FootprintSettings;
use serde::Deserialize;

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ObjectMapSettings {
    /// Edge of the cubes that object points are counted in.
    pub voxel_size_m: f32,
    /// Keyframes that wait for their pose and segments; the oldest is dropped beyond this.
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
    /// Mask edges smear foreground and background depth together; this much is shaved off first.
    pub erosion_px: usize,
    /// Fewer mask pixels with depth than this, after erosion, and the mask is ignored.
    pub min_points: usize,
    /// A jump in depth this large separates the object's surface from background seen through the mask.
    pub max_depth_gap_m: f32,
    /// Fewer voxels than this and the mask is ignored.
    pub min_voxels: usize,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AssociationSettings {
    /// A detection is of an object when at least this fraction of its voxels touch the object's voxels.
    pub min_overlap: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConfirmationSettings {
    /// Sightings, in different keyframes, that make a candidate an object.
    pub sightings: u32,
    /// A candidate not confirmed this long after its first sighting is dropped.
    pub window_s: f64,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MergeSettings {
    /// Two objects with agreeing labels are one when this fraction of the smaller one's voxels touch the larger.
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
            max_depth_gap_m: footprint.max_depth_gap_m,
            min_voxels: 4,
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
        ConfirmationSettings { sightings: 3, window_s: 60.0 }
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
        CropSettings { jpeg_quality: 85, min_improvement: 1.2 }
    }
}

impl DetectionSettings {
    pub fn footprint(&self) -> FootprintSettings {
        FootprintSettings {
            erosion_px: self.erosion_px,
            min_points: self.min_points,
            max_depth_gap_m: self.max_depth_gap_m,
            ..FootprintSettings::default()
        }
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
