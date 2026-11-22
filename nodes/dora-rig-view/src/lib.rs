//! A live 3D page of the robot's map for people to watch: keyframe clouds at their map poses from the localizer, tinted
//! where the objects are, the objects with their labels and sizes, the robot from odometry, people with their
//! skeletons, and, as diagnostic layers, the dense map's mesh and rooms. The page gets the scene once when it
//! connects, then only what changes.

pub mod cloud;
pub mod clouds;
pub mod rig;
pub mod scene;
pub mod server;
pub mod surface;
pub mod surface_stream;
pub mod view;
