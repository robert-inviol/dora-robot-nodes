//! Places the LD06's own millisecond stamps on the host's clock.
//!
//! A packet reaches the host some time after the lidar stamped it: the serial line, the USB adapter and the node's
//! polling all add delay, never subtract it. So the smallest gap between arrival and stamp over the last few seconds
//! is the closest to the true offset between the two clocks, and every stamp is placed with that offset. The window
//! is kept short so the drift between the two clocks does not build up inside it.

use std::collections::VecDeque;

use ld06::LidarStamp;

use crate::turn::UnixNanos;

/// The datasheet's stamp counts up to 30000 ms and starts again from zero.
const STAMP_WRAP_MS: i64 = 30_000;
/// Packets come every 2.6 ms, so a longer step between stamps, or between arrivals, means the stream broke.
const MAX_STEP_MS: i64 = 1_000;
const OFFSET_WINDOW_MS: i64 = 10_000;
const NANOS_PER_MILLI: i64 = 1_000_000;

#[derive(Debug, Default)]
pub struct LidarClock {
    latest: Option<Latest>,
    /// (lidar time in ms, arrival minus lidar time in ns), with offsets rising from front to back, so the front is
    /// the smallest offset of the window.
    offsets: VecDeque<(i64, i64)>,
    restarts: u64,
}

#[derive(Debug, Clone, Copy)]
struct Latest {
    stamp: LidarStamp,
    lidar_ms: i64,
    arrived: UnixNanos,
}

impl LidarClock {
    /// When the packet stamped `stamp`, which arrived at `arrived`, was measured, on the host's clock.
    pub fn place(&mut self, stamp: LidarStamp, arrived: UnixNanos) -> UnixNanos {
        let lidar_ms = match self.latest {
            Some(latest) => match continued_ms(latest, stamp, arrived) {
                Some(step_ms) => latest.lidar_ms + step_ms,
                None => {
                    self.offsets.clear();
                    self.restarts += 1;
                    0
                }
            },
            None => 0,
        };
        self.latest = Some(Latest {
            stamp,
            lidar_ms,
            arrived,
        });

        let offset_ns = arrived.0 - lidar_ms * NANOS_PER_MILLI;
        while self
            .offsets
            .back()
            .is_some_and(|&(_, held)| held >= offset_ns)
        {
            self.offsets.pop_back();
        }
        self.offsets.push_back((lidar_ms, offset_ns));
        while self
            .offsets
            .front()
            .is_some_and(|&(held_ms, _)| held_ms < lidar_ms - OFFSET_WINDOW_MS)
        {
            self.offsets.pop_front();
        }
        let smallest_offset_ns = self
            .offsets
            .front()
            .expect("the latest offset was just added")
            .1;
        UnixNanos(lidar_ms * NANOS_PER_MILLI + smallest_offset_ns)
    }

    /// How often the stream broke and the clock started again from the next packet.
    pub fn restarts(&self) -> u64 {
        self.restarts
    }
}

/// The milliseconds from the latest stamp to `stamp`, or `None` when the stream broke between them.
fn continued_ms(latest: Latest, stamp: LidarStamp, arrived: UnixNanos) -> Option<i64> {
    let raw_step = i64::from(stamp.milliseconds) - i64::from(latest.stamp.milliseconds);
    let step_ms = if raw_step < 0 {
        raw_step + STAMP_WRAP_MS
    } else {
        raw_step
    };
    let arrival_gap_ms = (arrived.0 - latest.arrived.0) / NANOS_PER_MILLI;
    ((0..=MAX_STEP_MS).contains(&step_ms) && arrival_gap_ms <= MAX_STEP_MS).then_some(step_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    const START: i64 = 1_762_560_000_000_000_000;

    fn stamp(milliseconds: u16) -> LidarStamp {
        LidarStamp { milliseconds }
    }

    fn at_ms(milliseconds: i64) -> UnixNanos {
        UnixNanos(START + milliseconds * NANOS_PER_MILLI)
    }

    #[test]
    fn a_stamp_is_placed_at_the_earliest_arrival_of_the_window() {
        let mut clock = LidarClock::default();
        clock.place(stamp(100), at_ms(9));
        clock.place(stamp(103), at_ms(5));

        assert_eq!(clock.place(stamp(106), at_ms(20)), at_ms(8));
    }

    #[test]
    fn a_late_arrival_does_not_move_the_placing() {
        let mut clock = LidarClock::default();
        clock.place(stamp(100), at_ms(4));

        assert_eq!(clock.place(stamp(103), at_ms(40)), at_ms(7));
    }

    #[test]
    fn the_stamp_carries_on_across_its_wrap() {
        let mut clock = LidarClock::default();
        clock.place(stamp(29_998), at_ms(0));

        assert_eq!(clock.place(stamp(1), at_ms(3)), at_ms(3));
        assert_eq!(clock.restarts(), 0);
    }

    #[test]
    fn an_offset_older_than_the_window_is_forgotten() {
        let mut clock = LidarClock::default();
        clock.place(stamp(0), at_ms(0));
        clock.place(stamp(900), at_ms(905));

        // Stepping on in sub-second hops until the first offset has left the window.
        let mut milliseconds = 900;
        while milliseconds < OFFSET_WINDOW_MS as u16 + 900 {
            milliseconds += 900;
            clock.place(stamp(milliseconds), at_ms(i64::from(milliseconds) + 5));
        }

        assert_eq!(
            clock.place(stamp(milliseconds + 3), at_ms(i64::from(milliseconds) + 50)),
            at_ms(i64::from(milliseconds) + 8)
        );
    }

    #[test]
    fn a_long_silence_restarts_the_clock() {
        let mut clock = LidarClock::default();
        clock.place(stamp(100), at_ms(0));

        let placed = clock.place(stamp(103), at_ms(5_000));

        assert_eq!(placed, at_ms(5_000));
        assert_eq!(clock.restarts(), 1);
    }

    #[test]
    fn a_stamp_that_jumps_restarts_the_clock() {
        let mut clock = LidarClock::default();
        clock.place(stamp(100), at_ms(0));

        clock.place(stamp(5_000), at_ms(3));

        assert_eq!(clock.restarts(), 1);
    }
}
