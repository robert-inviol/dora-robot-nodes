//! One packet of the lidar's protocol.
//!
//! Layout, all integers little-endian: header `54`, `2C` (packet type and twelve readings), spin rate (2, degrees
//! per second), start bearing (2, hundredths of a degree), twelve readings of distance (2, millimetres) and
//! intensity (1), end bearing (2), timestamp (2, milliseconds), CRC-8 (1, of every byte before it).

use crate::units::{Bearing, Intensity, LidarStamp, Range, SpinRate};

pub(crate) const HEADER: [u8; 2] = [0x54, 0x2C];
pub(crate) const PACKET_BYTES: usize = 47;
pub const READINGS_PER_PACKET: usize = 12;

const SPIN_RATE_FIELD: usize = 2;
const START_BEARING_FIELD: usize = 4;
const FIRST_READING_OFFSET: usize = 6;
const READING_BYTES: usize = 3;
const END_BEARING_FIELD: usize = 42;
const STAMP_FIELD: usize = 44;
const CRC_OFFSET: usize = 46;
const CRC_POLYNOMIAL: u8 = 0x4D;

#[derive(Debug, Clone, PartialEq)]
pub struct Packet {
    pub spin_rate: SpinRate,
    /// The twelve readings in the order measured, clockwise from the packet's start bearing to its end bearing.
    pub readings: Vec<Reading>,
    /// When the lidar measured the packet's readings, within the 2.6 ms the packet spans.
    pub stamp: LidarStamp,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reading {
    pub bearing: Bearing,
    /// `None` when nothing came back from this direction.
    pub range: Option<Range>,
    pub intensity: Intensity,
}

impl Packet {
    /// `None` when the CRC does not match. `bytes` is one whole packet, header included.
    pub(crate) fn parse(bytes: &[u8; PACKET_BYTES]) -> Option<Self> {
        if crc8(&bytes[..CRC_OFFSET]) != bytes[CRC_OFFSET] {
            return None;
        }
        let start = Bearing::from_wire(little_endian(bytes, START_BEARING_FIELD));
        let end = Bearing::from_wire(little_endian(bytes, END_BEARING_FIELD));
        // The readings are evenly spread from the start bearing to the end bearing, both included.
        let step_degrees = start.clockwise_to(end) / (READINGS_PER_PACKET - 1) as f32;
        let readings = (0..READINGS_PER_PACKET)
            .map(|index| {
                let offset = FIRST_READING_OFFSET + READING_BYTES * index;
                Reading {
                    bearing: Bearing::from_degrees(start.degrees() + step_degrees * index as f32),
                    range: Range::from_wire(little_endian(bytes, offset)),
                    intensity: Intensity(bytes[offset + 2]),
                }
            })
            .collect();
        Some(Self {
            spin_rate: SpinRate::from_wire(little_endian(bytes, SPIN_RATE_FIELD)),
            readings,
            stamp: LidarStamp {
                milliseconds: little_endian(bytes, STAMP_FIELD),
            },
        })
    }
}

fn little_endian(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

/// CRC-8 with polynomial 0x4D, no reflection, starting from zero, as the manufacturer's table computes it.
fn crc8(bytes: &[u8]) -> u8 {
    bytes.iter().fold(0, |crc, &byte| {
        (0..8).fold(crc ^ byte, |crc, _| {
            if crc & 0x80 != 0 {
                (crc << 1) ^ CRC_POLYNOMIAL
            } else {
                crc << 1
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_crc_matches_the_first_entries_of_the_manufacturers_table() {
        let table_start: Vec<u8> = (0u8..4).map(|byte| crc8(&[byte])).collect();

        assert_eq!(table_start, [0x00, 0x4D, 0x9A, 0xD7]);
    }
}
