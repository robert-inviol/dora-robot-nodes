//! The dense map: a TSDF fused from posed keyframes, the free space sliced from it above the floor, rooms cut at
//! doorways with places along the middle of the free space, and keep-out zones around glass and mirrors.

pub mod grid;
pub mod keep_out;
pub mod keyframe;
pub mod levels;
pub mod mapper;
pub mod outline;
pub mod settings;
pub mod synthetic;
pub mod topology;
pub mod tsdf;

pub use grid::{slice_field, squared_distance_transform, Cell, CellGrid, GridGeometry, Slice};
pub use keep_out::{KeepOutZones, MergedIntoNothing, ObjectChange, ObjectUpdate};
pub use keyframe::{CameraPose, DepthKeyframe, FrameId, KeyframePose, PosedDepth, Tracking};
pub use mapper::{free_space, Arrival, DenseMapper, MapOutput, MapperError, TickReport};
pub use settings::{DenseMapSettings, SettingsError};
pub use topology::{Segmentation, Topologist, TopologyPlan};
pub use tsdf::{BlockTsdf, Bounds, DistanceField, Sample, SurfacePoint};
