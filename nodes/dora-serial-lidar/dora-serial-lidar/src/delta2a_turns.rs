//! The Delta-2A's turns. It stamps nothing, so its readings are timed by when their bytes reached the host.

use delta2a::{Frame, FrameDecoder, Revolution, RevolutionAssembler};

use crate::turn::{Reading, Readings, Turn, TurnDecoder, UnixNanos};

#[derive(Debug, Default)]
pub struct Delta2aTurns {
    decoder: FrameDecoder,
    assembler: RevolutionAssembler,
    gathering_since: Option<UnixNanos>,
    latest_sector_arrived: Option<UnixNanos>,
    speed_fault: bool,
}

impl Delta2aTurns {
    pub fn new() -> Self {
        Self::default()
    }

    fn note_speed_fault(&mut self, speed_fault: bool) {
        if speed_fault != self.speed_fault {
            let report = if speed_fault {
                "the head is not up to speed"
            } else {
                "the head is up to speed"
            };
            println!("[lidar] {report}");
            self.speed_fault = speed_fault;
        }
    }
}

impl TurnDecoder for Delta2aTurns {
    fn push(&mut self, bytes: &[u8], arrived: UnixNanos) -> Vec<Turn> {
        let mut turns = Vec::new();
        for frame in self.decoder.push(bytes) {
            match frame {
                Frame::Scan(sector) => {
                    self.note_speed_fault(false);
                    // A sector that ends a turn starts the next, so the finished turn's last sector is the one before.
                    if let Some(revolution) = self.assembler.push(sector) {
                        let first_reading_at = self
                            .gathering_since
                            .expect("a finished turn began with an earlier sector");
                        let last_reading_at = self
                            .latest_sector_arrived
                            .expect("a finished turn ended with an earlier sector");
                        turns.push(turn(revolution, first_reading_at, last_reading_at));
                        self.gathering_since = Some(arrived);
                    }
                    self.gathering_since.get_or_insert(arrived);
                    self.latest_sector_arrived = Some(arrived);
                }
                Frame::SpeedFault(_) => self.note_speed_fault(true),
            }
        }
        turns
    }
}

fn turn(revolution: Revolution, first_reading_at: UnixNanos, last_reading_at: UnixNanos) -> Turn {
    Turn {
        spin_rev_per_s: revolution.spin_rate.revolutions_per_second(),
        first_reading_at,
        last_reading_at,
        readings: Readings::RangeOnly(
            revolution
                .samples
                .iter()
                .map(|sample| Reading {
                    bearing_deg: sample.bearing.degrees(),
                    range_m: sample.range.map(|range| range.metres()),
                })
                .collect(),
        ),
    }
}
