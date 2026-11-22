//! Decoder for the serial stream of an LDRobot LD06 lidar.
//!
//! The lidar only talks: 230400 baud, 8N1, a 47-byte packet of twelve readings about every 2.6 ms. Feed the bytes
//! to a [`PacketDecoder`] and the packets it yields to a [`RevolutionAssembler`].

mod decoder;
mod packet;
mod revolution;
mod units;

pub use decoder::PacketDecoder;
pub use packet::{Packet, Reading, READINGS_PER_PACKET};
pub use revolution::{Revolution, RevolutionAssembler};
pub use units::{Bearing, Intensity, LidarStamp, Range, SpinRate};

pub const BAUD_RATE: u32 = 230_400;
