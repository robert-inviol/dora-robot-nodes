//! Finds packets in the byte stream, however it is chopped up, and drops what fails its CRC.

use crate::packet::{Packet, HEADER, PACKET_BYTES};

#[derive(Debug, Default)]
pub struct PacketDecoder {
    pending: Vec<u8>,
    discarded_bytes: u64,
}

impl PacketDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes the next bytes off the wire and returns every packet they complete.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<Packet> {
        self.pending.extend_from_slice(bytes);
        let mut packets = Vec::new();
        let mut cursor = 0;
        while let Some(candidate) = self.pending.get(cursor..cursor + PACKET_BYTES) {
            if candidate[..HEADER.len()] != HEADER {
                cursor += 1;
                self.discarded_bytes += 1;
                continue;
            }
            let candidate: &[u8; PACKET_BYTES] =
                candidate.try_into().expect("the slice is one packet long");
            match Packet::parse(candidate) {
                Some(packet) => {
                    packets.push(packet);
                    cursor += PACKET_BYTES;
                }
                // The header bytes can turn up inside a packet, so only the first byte is given up.
                None => {
                    cursor += 1;
                    self.discarded_bytes += 1;
                }
            }
        }
        self.pending.drain(..cursor);
        packets
    }

    /// Bytes that belonged to no packet: line noise, failed CRCs.
    pub fn discarded_bytes(&self) -> u64 {
        self.discarded_bytes
    }
}
