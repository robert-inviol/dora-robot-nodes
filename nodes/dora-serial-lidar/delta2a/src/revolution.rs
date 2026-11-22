//! Collects scan sectors into whole turns of the head.

use crate::frame::{RangeSample, ScanSector};
use crate::units::{Bearing, SpinRate};

/// The samples of one turn of the head, in bearing order. Sectors lost on the wire leave a gap.
#[derive(Debug, Clone, PartialEq)]
pub struct Revolution {
    pub spin_rate: SpinRate,
    pub samples: Vec<RangeSample>,
}

#[derive(Debug, Default)]
pub struct RevolutionAssembler {
    gathering: Option<Gathering>,
}

#[derive(Debug)]
struct Gathering {
    latest_start: Bearing,
    spin_rate: SpinRate,
    samples: Vec<RangeSample>,
}

impl RevolutionAssembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the revolution that `sector` ends, which it does by starting the next one.
    pub fn push(&mut self, sector: ScanSector) -> Option<Revolution> {
        let finished = match &self.gathering {
            Some(gathering) if sector.start < gathering.latest_start => self.gathering.take(),
            _ => None,
        };
        let gathering = self.gathering.get_or_insert_with(|| Gathering {
            latest_start: sector.start,
            spin_rate: sector.spin_rate,
            samples: Vec::new(),
        });
        gathering.latest_start = sector.start;
        gathering.spin_rate = sector.spin_rate;
        gathering.samples.extend(sector.samples);
        finished.map(|gathering| Revolution {
            spin_rate: gathering.spin_rate,
            samples: gathering.samples,
        })
    }
}
