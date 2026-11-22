//! Collects readings into whole turns of the head.

use crate::packet::{Packet, Reading};
use crate::units::{Bearing, SpinRate};

/// The readings of one turn of the head, in the order measured, and when its first and last reading were
/// measured, in whatever time the caller tags packets with. The first turn after the assembler starts is partial.
#[derive(Debug, Clone, PartialEq)]
pub struct Revolution<Time> {
    pub spin_rate: SpinRate,
    pub readings: Vec<Reading>,
    pub first_reading_at: Time,
    pub last_reading_at: Time,
}

#[derive(Debug)]
pub struct RevolutionAssembler<Time> {
    gathering: Option<Gathering<Time>>,
}

#[derive(Debug)]
struct Gathering<Time> {
    latest_bearing: Bearing,
    revolution: Revolution<Time>,
}

impl<Time> Default for RevolutionAssembler<Time> {
    fn default() -> Self {
        Self { gathering: None }
    }
}

impl<Time: Copy> RevolutionAssembler<Time> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a packet measured at `measured_at` and returns the revolution it completes. A turn ends where the
    /// bearing passes the zero mark, which can be partway through a packet.
    pub fn push(&mut self, packet: Packet, measured_at: Time) -> Option<Revolution<Time>> {
        let mut finished = None;
        for reading in packet.readings {
            let crosses_zero = matches!(&self.gathering, Some(gathering) if reading.bearing < gathering.latest_bearing);
            if crosses_zero {
                finished = self.gathering.take().map(|gathering| gathering.revolution);
            }
            let gathering = self.gathering.get_or_insert_with(|| Gathering {
                latest_bearing: reading.bearing,
                revolution: Revolution {
                    spin_rate: packet.spin_rate,
                    readings: Vec::new(),
                    first_reading_at: measured_at,
                    last_reading_at: measured_at,
                },
            });
            gathering.latest_bearing = reading.bearing;
            gathering.revolution.spin_rate = packet.spin_rate;
            gathering.revolution.last_reading_at = measured_at;
            gathering.revolution.readings.push(reading);
        }
        finished
    }
}
