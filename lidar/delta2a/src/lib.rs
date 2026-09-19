//! Decoder for the serial stream of a 3irobotics Delta-2A lidar.
//!
//! The lidar only talks: 230400 baud, 8N1, one frame per 22.5 degree sector, sixteen sectors per
//! revolution. Feed the bytes to a [`FrameDecoder`] and the scan sectors it yields to a
//! [`RevolutionAssembler`].

mod decoder;
mod frame;
mod revolution;
mod units;

pub use decoder::FrameDecoder;
pub use frame::{Frame, RangeSample, ScanSector};
pub use revolution::{Revolution, RevolutionAssembler};
pub use units::{Bearing, Range, SpinRate};

pub const BAUD_RATE: u32 = 230_400;
