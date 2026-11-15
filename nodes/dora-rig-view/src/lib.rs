//! A live 3D page of the robot's map for people to watch: the dense map's surface with the objects painted on it,
//! keyframe clouds at their map poses from the localizer, the robot from odometry, people with their skeletons, and rooms. The
//! page gets the scene once when it connects, then only what changes.

pub mod cloud;
pub mod clouds;
pub mod rig;
pub mod scene;
pub mod server;
pub mod surface;
pub mod surface_stream;
pub mod view;
