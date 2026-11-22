//! The LD06's turns, timed by its own stamps placed on the host's clock.

use ld06::{PacketDecoder, Revolution, RevolutionAssembler};

use crate::clock::LidarClock;
use crate::turn::{Reading, Readings, Turn, TurnDecoder, UnixNanos};

#[derive(Debug, Default)]
pub struct Ld06Turns {
    decoder: PacketDecoder,
    assembler: RevolutionAssembler<UnixNanos>,
    clock: LidarClock,
    reported_restarts: u64,
}

impl Ld06Turns {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TurnDecoder for Ld06Turns {
    fn push(&mut self, bytes: &[u8], arrived: UnixNanos) -> Vec<Turn> {
        let mut turns = Vec::new();
        for packet in self.decoder.push(bytes) {
            let measured_at = self.clock.place(packet.stamp, arrived);
            turns.extend(self.assembler.push(packet, measured_at).map(turn));
        }
        if self.clock.restarts() != self.reported_restarts {
            println!("[lidar] the LD06's stamps broke off, so its clock was placed again");
            self.reported_restarts = self.clock.restarts();
        }
        turns
    }
}

fn turn(revolution: Revolution<UnixNanos>) -> Turn {
    Turn {
        spin_rev_per_s: revolution.spin_rate.revolutions_per_second(),
        first_reading_at: revolution.first_reading_at,
        last_reading_at: revolution.last_reading_at,
        readings: Readings::WithIntensity(
            revolution
                .readings
                .iter()
                .map(|reading| {
                    let bearing_and_range = Reading {
                        bearing_deg: reading.bearing.degrees(),
                        range_m: reading.range.map(|range| range.metres()),
                    };
                    (bearing_and_range, reading.intensity.0)
                })
                .collect(),
        ),
    }
}
