//! Live odometry for a depth camera with an IMU.
//!
//! The IMU predicts how the body moved since the last cloud; each cloud is then registered against the
//! surfaces a local map holds, and an error-state Kalman filter blends the two, learning the IMU's
//! biases as it goes. The world frame starts at the body, with z where the accelerometer at rest said up
//! was; the state carries the true up as the filter learns it, and [`State::levelled_pose`] gives the
//! pose in a frame whose z is up. The body frame is the IMU's.
//!
//! [`PoseTracker`] runs the whole of it from a camera's raw streams: the depth images
//! ([`depth_image`]), the accelerometer's and the gyro's separate samples ([`imu_pairing`]) and the
//! start at rest.
//!
//! Other sensors join through [`filter::PoseInformation`]: anything that can say "this body point lies
//! on that world plane" adds its measurements to the same correction as the cloud's.

pub mod association;
pub mod depth_image;
pub mod filter;
pub mod imu_pairing;
pub mod map;
pub mod odometry;
pub mod simulation;
pub mod surfaces;
pub mod thinning;
pub mod tracker;

pub use filter::State;
pub use map::{KdTreeMap, LocalMap, VoxelHashMap};
pub use odometry::{DepthInertialOdometry, ImuSample, OdometrySettings};
pub use surfaces::{NeighbourPlaneSettings, NeighbourPlanes, PlaneVoxelMap, PlaneVoxelSettings, SurfaceMap};
pub use tracker::{DepthFrameOutcome, FramePose, PoseTracker, TrackerSettings};
