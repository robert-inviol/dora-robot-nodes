//! One turn of the head, whichever lidar measured it: what the `scan` output carries.

use std::time::{SystemTime, UNIX_EPOCH};

/// An instant on the host's clock, in nanoseconds since the unix epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct UnixNanos(pub i64);

impl UnixNanos {
    pub fn now() -> Self {
        let since_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("the host clock is after 1970");
        Self(i64::try_from(since_epoch.as_nanos()).expect("the host clock is before 2262"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reading {
    /// Clockwise from the lidar's zero mark, seen from above.
    pub bearing_deg: f32,
    /// `None` when nothing came back.
    pub range_m: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Readings {
    RangeOnly(Vec<Reading>),
    /// Each reading with its return strength on the lidar's own 0 to 255 scale.
    WithIntensity(Vec<(Reading, u8)>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub spin_rev_per_s: f32,
    pub first_reading_at: UnixNanos,
    pub last_reading_at: UnixNanos,
    /// In the order measured.
    pub readings: Readings,
}

/// Turns a lidar's byte stream into turns of its head.
pub trait TurnDecoder {
    /// Takes the bytes read off the port at `arrived` and returns every turn they complete.
    fn push(&mut self, bytes: &[u8], arrived: UnixNanos) -> Vec<Turn>;
}
