//! One frame of the lidar's protocol, parsed from bytes whose checksum already matched.
//!
//! Layout, all integers big-endian: header `AA`, frame length (2, counted from the header to the
//! byte before the checksum), protocol version (1), frame type `61`, command (1), parameter
//! length (2), parameters, checksum (2, the sum of every byte before it).

use crate::units::{Bearing, Range, SpinRate};

pub(crate) const HEADER: u8 = 0xAA;
pub(crate) const LENGTH_FIELD: std::ops::Range<usize> = 1..3;
pub(crate) const CHECKSUM_BYTES: usize = 2;
/// Header, length, version, type, command and parameter length.
pub(crate) const PREAMBLE_BYTES: usize = 8;

const FRAME_TYPE: u8 = 0x61;
const FRAME_TYPE_OFFSET: usize = 4;
const COMMAND_OFFSET: usize = 5;
const PARAMETER_LENGTH_FIELD: std::ops::Range<usize> = 6..8;
const COMMAND_SCAN: u8 = 0xAD;
const COMMAND_SPEED_FAULT: u8 = 0xAE;

/// Spin rate (1), zero offset (2, a factory debug value) and start bearing (2).
const SCAN_PREFIX_BYTES: usize = 5;
const SCAN_START_BEARING_FIELD: std::ops::Range<usize> = 3..5;
/// Signal strength (1, a factory debug value) and distance (2).
const SAMPLE_BYTES: usize = 3;
const SAMPLE_DISTANCE_FIELD: std::ops::Range<usize> = 1..3;
pub(crate) const SECTOR_DEGREES: f32 = 22.5;

#[derive(Debug, Clone, PartialEq)]
pub enum Frame {
    Scan(ScanSector),
    /// The head is not turning at a speed the lidar can measure with.
    SpeedFault(SpinRate),
}

/// The samples of one 22.5 degree sector, in the order they were measured.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanSector {
    pub spin_rate: SpinRate,
    pub start: Bearing,
    pub samples: Vec<RangeSample>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeSample {
    pub bearing: Bearing,
    /// `None` when nothing came back from this direction.
    pub range: Option<Range>,
}

impl Frame {
    /// `None` for a frame this decoder has no meaning for.
    pub(crate) fn parse(frame: &[u8]) -> Option<Self> {
        if frame[FRAME_TYPE_OFFSET] != FRAME_TYPE {
            return None;
        }
        let parameter_length = usize::from(big_endian(&frame[PARAMETER_LENGTH_FIELD]));
        let parameters = frame.get(PREAMBLE_BYTES..PREAMBLE_BYTES + parameter_length)?;
        match frame[COMMAND_OFFSET] {
            COMMAND_SCAN => ScanSector::parse(parameters).map(Frame::Scan),
            COMMAND_SPEED_FAULT => parameters
                .first()
                .map(|&count| Frame::SpeedFault(SpinRate::from_wire(count))),
            _ => None,
        }
    }
}

impl ScanSector {
    fn parse(parameters: &[u8]) -> Option<Self> {
        let sample_bytes = parameters.get(SCAN_PREFIX_BYTES..)?;
        if sample_bytes.is_empty() || sample_bytes.len() % SAMPLE_BYTES != 0 {
            return None;
        }
        let start = Bearing::from_wire(big_endian(&parameters[SCAN_START_BEARING_FIELD]));
        let sample_count = sample_bytes.len() / SAMPLE_BYTES;
        let samples = sample_bytes
            .chunks_exact(SAMPLE_BYTES)
            .enumerate()
            .map(|(index, sample)| RangeSample {
                // The samples of a sector are evenly spread across it.
                bearing: Bearing::from_degrees(
                    start.degrees() + SECTOR_DEGREES * index as f32 / sample_count as f32,
                ),
                range: Range::from_wire(big_endian(&sample[SAMPLE_DISTANCE_FIELD])),
            })
            .collect();
        Some(Self {
            spin_rate: SpinRate::from_wire(parameters[0]),
            start,
            samples,
        })
    }
}

pub(crate) fn big_endian(pair: &[u8]) -> u16 {
    u16::from_be_bytes([pair[0], pair[1]])
}
