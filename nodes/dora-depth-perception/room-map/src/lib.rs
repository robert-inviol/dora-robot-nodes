//! One map of a room from a walk of keyframes.
//!
//! Every keyframe's points are in its own gravity-aligned floor frame (x right, y forward, z up, at the
//! camera), so the keyframes differ only by a turn about the vertical and a translation: a 4-DoF pose.
//! Consecutive keyframes are registered against a local model of the last few, the walk's drift is
//! closed against its start, and every keyframe is fused into one voxel map.

pub mod fusion;
pub mod grid;
pub mod placement;
pub mod pose;
pub mod recording;
pub mod registration;
pub mod trajectory;

pub use pose::Pose;
