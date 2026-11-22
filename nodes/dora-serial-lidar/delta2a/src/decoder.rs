//! Finds frames in the byte stream, however it is chopped up, and drops what fails its checksum.

use crate::frame::{big_endian, Frame, CHECKSUM_BYTES, HEADER, LENGTH_FIELD, PREAMBLE_BYTES};

/// A sector at the slowest spin and the fastest sampling is under 250 bytes. A longer length field
/// is noise, and waiting for that many bytes to arrive would stall the stream.
const MAX_FRAME_BYTES: usize = 512;

#[derive(Debug, Default)]
pub struct FrameDecoder {
    pending: Vec<u8>,
    discarded_bytes: u64,
}

impl FrameDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes the next bytes off the wire and returns every frame they complete.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<Frame> {
        self.pending.extend_from_slice(bytes);
        let mut frames = Vec::new();
        let mut cursor = 0;
        while let Some(candidate) = Candidate::at(&self.pending[cursor..]) {
            match candidate {
                Candidate::Noise => {
                    cursor += 1;
                    self.discarded_bytes += 1;
                }
                Candidate::Incomplete => break,
                Candidate::Checked(frame_bytes) => {
                    let consumed = frame_bytes.len() + CHECKSUM_BYTES;
                    match Frame::parse(frame_bytes) {
                        Some(frame) => frames.push(frame),
                        None => self.discarded_bytes += consumed as u64,
                    }
                    cursor += consumed;
                }
            }
        }
        self.pending.drain(..cursor);
        frames
    }

    /// Bytes that belonged to no frame this decoder understood: line noise, failed checksums.
    pub fn discarded_bytes(&self) -> u64 {
        self.discarded_bytes
    }
}

enum Candidate<'a> {
    /// The first byte starts no valid frame.
    Noise,
    /// The bytes so far could be a frame, but more have to arrive to tell.
    Incomplete,
    /// A frame whose checksum matched, without the checksum.
    Checked(&'a [u8]),
}

impl<'a> Candidate<'a> {
    fn at(bytes: &'a [u8]) -> Option<Self> {
        let &first = bytes.first()?;
        if first != HEADER {
            return Some(Self::Noise);
        }
        let Some(length_field) = bytes.get(LENGTH_FIELD) else {
            return Some(Self::Incomplete);
        };
        let frame_length = usize::from(big_endian(length_field));
        if !(PREAMBLE_BYTES..=MAX_FRAME_BYTES).contains(&frame_length) {
            return Some(Self::Noise);
        }
        let Some(checksum_field) = bytes.get(frame_length..frame_length + CHECKSUM_BYTES) else {
            return Some(Self::Incomplete);
        };
        let frame_bytes = &bytes[..frame_length];
        let sum = frame_bytes
            .iter()
            .fold(0u16, |sum, &byte| sum.wrapping_add(u16::from(byte)));
        Some(if sum == big_endian(checksum_field) {
            Self::Checked(frame_bytes)
        } else {
            Self::Noise
        })
    }
}
