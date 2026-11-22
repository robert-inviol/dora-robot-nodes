//! Which lidar is on the port.

use std::str::FromStr;

use crate::delta2a_turns::Delta2aTurns;
use crate::ld06_turns::Ld06Turns;
use crate::turn::TurnDecoder;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LidarModel {
    Ld06,
    Delta2a,
}

impl LidarModel {
    pub fn baud_rate(self) -> u32 {
        match self {
            Self::Ld06 => ld06::BAUD_RATE,
            Self::Delta2a => delta2a::BAUD_RATE,
        }
    }

    /// A decoder with nothing pending, for a freshly opened port.
    pub fn turn_decoder(self) -> Box<dyn TurnDecoder> {
        match self {
            Self::Ld06 => Box::new(Ld06Turns::new()),
            Self::Delta2a => Box::new(Delta2aTurns::new()),
        }
    }
}

impl FromStr for LidarModel {
    type Err = eyre::Report;

    fn from_str(name: &str) -> eyre::Result<Self> {
        match name {
            "ld06" => Ok(Self::Ld06),
            "delta2a" => Ok(Self::Delta2a),
            other => Err(eyre::eyre!(
                "the lidar model {other:?} is not one of ld06, delta2a"
            )),
        }
    }
}
