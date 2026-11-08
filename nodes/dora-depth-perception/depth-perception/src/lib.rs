//! Room geometry from one RGB-D frame, in a gravity-aligned floor frame (x right, y forward, z up,
//! metres, origin at the camera).

pub mod camera;
pub mod geometry2d;
pub mod objects;
pub mod planes;
pub mod room;
pub mod synthetic;

pub use camera::{sample_point_cloud, CameraIntrinsics, DepthImage, FloorFrame, PointCloud, RangeLimits};
pub use objects::{object_footprint, ObjectFootprint, ObjectMask};
pub use planes::{find_planes, Plane, PlaneFit, PlaneSettings, SurfaceKind};
pub use room::{infer_room, Room, RoomError, RoomSettings};
