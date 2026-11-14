//! Registers keyframes against the map and keeps a 4-DoF pose graph.
//!
//! Every keyframe's points are in its own gravity-aligned floor frame, so keyframes differ only by a turn about
//! the vertical and a translation. `localizer` registers each keyframe against a local submap of the map built so
//! far, seeded by the gyro's turn (`gyro`), and keeps the poses in a `pose_graph` that loop closures correct.
//! A keyframe whose turn the gyro cannot give is lost unless `HeadingWithoutGyro::Search` asks for its heading
//! to be searched for.
//! `map_to_odom` turns keyframe poses and odometry into the odometry frame's correction.

pub mod cloud;
pub mod gyro;
pub mod localizer;
pub mod map_to_odom;
pub mod pose_graph;
pub mod synthetic;

pub use localizer::{HeadingHint, HeadingWithoutGyro, KeyframeEstimate, Localizer, LocalizerInput, LocalizerSettings};
