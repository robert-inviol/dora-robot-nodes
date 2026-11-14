//! Live odometry for a depth camera with an IMU.
//!
//! The IMU predicts how the body moved since the last cloud; each cloud is then registered against the
//! surfaces a local map holds, and an error-state Kalman filter blends the two, learning the IMU's
//! biases as it goes. The world frame starts at the body, with z where the accelerometer at rest said up
//! was; the state carries the true up as the filter learns it, and [`State::levelled_pose`] gives the
//! pose in a frame whose z is up. The body frame is the IMU's.
//!
//! Other sensors join through [`filter::PoseInformation`]: anything that can say "this body point lies
//! on that world plane" adds its measurements to the same correction as the cloud's.

pub mod association;
pub mod filter;
pub mod map;
pub mod odometry;
pub mod simulation;
pub mod surfaces;
pub mod thinning;

pub use filter::State;
pub use map::{KdTreeMap, LocalMap, VoxelHashMap};
pub use odometry::{DepthInertialOdometry, ImuSample, OdometrySettings};
pub use surfaces::{NeighbourPlaneSettings, NeighbourPlanes, PlaneVoxelMap, PlaneVoxelSettings, SurfaceMap};
