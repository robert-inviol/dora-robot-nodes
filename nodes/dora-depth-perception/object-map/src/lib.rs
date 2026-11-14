//! Persistent objects built from SAM 3 masks: matching, confirmation, merging and negative evidence.
//!
//! Each keyframe's masks are lifted into the map with the keyframe's pose and voxelised. A detection belongs to
//! the existing object with the same label whose voxels it overlaps most; otherwise it starts a candidate, which
//! becomes an object after enough sightings. Objects that overlap and share a label are merged. Where the camera
//! sees clearly past an object's surface, often enough, the object has gone; if one with its label turns up
//! elsewhere soon after, it has moved.

pub mod crop;
pub mod detection;
pub mod evidence;
pub mod join;
pub mod label;
pub mod map;
pub mod settings;
pub mod synthetic;
pub mod voxels;

use std::time::Duration;

use perception_wire::navigation::{MapVersion, UnixNanos};
use perception_wire::{Keyframe, Segments};
use room_map::Pose;

pub use join::{FrameJoiner, Placement};
pub use map::ObjectMap;
pub use settings::ObjectMapSettings;

/// Time from `earlier` to `later`, zero if `earlier` is later.
pub fn elapsed(earlier: UnixNanos, later: UnixNanos) -> Duration {
    Duration::from_nanos(later.0.saturating_sub(earlier.0).max(0) as u64)
}

/// A keyframe with its segments and its place in the map: everything one round of object mapping needs.
pub struct ObservedFrame {
    pub keyframe: Keyframe,
    pub segments: Segments,
    /// Maps the keyframe's gravity-aligned floor frame into the map.
    pub pose: Pose,
    pub map_version: MapVersion,
}

impl ObservedFrame {
    pub fn captured(&self) -> UnixNanos {
        UnixNanos(self.keyframe.captured_unix_ns)
    }
}
