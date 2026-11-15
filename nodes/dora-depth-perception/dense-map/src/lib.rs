//! The dense map: a TSDF fused from posed keyframes and posed depth frames, the occupancy grid sliced from it above the floor, rooms cut at
//! doorways with places along the middle of the free space, keep-out zones around glass and mirrors, and the
//! surface as triangles, one mesh block per block.

pub mod derived;
pub mod grid;
pub mod keep_out;
pub mod keyframe;
pub mod kept;
pub mod levels;
pub mod mapper;
pub mod outline;
pub mod settings;
pub mod block_mesher;
pub mod synthetic;
pub mod topology;
pub mod tsdf;

pub use grid::{slice_field, squared_distance_transform, Cell, OccupancyCells, GridGeometry, Slice};
pub use keep_out::{KeepOutZones, MergedIntoNothing, ObjectChange, ObjectUpdate};
pub use keyframe::{CameraPose, DepthKeyframe, FrameId, KeyframePose, PosedDepth, Tracking};
pub use derived::{occupancy_grid, DerivedMaps, Sliced};
pub use kept::{Footprint, KeptKeyframes, PoseCell};
pub use mapper::{Arrival, DenseMapper, MapOutput, MapperError, PosedDepthArrival, RebuildReport, TickReport, Unfused};
pub use settings::{DenseMapSettings, KeyframeSettings, SettingsError};
pub use topology::{Segmentation, TopologyBuilder, TopologyPlan};
pub use block_mesher::{mesh_block, BlockMesher};
pub use tsdf::{BlockField, BlockTsdf, Bounds, Tsdf, Sample, SurfacePoint, BLOCK_EDGE_VOXELS};
