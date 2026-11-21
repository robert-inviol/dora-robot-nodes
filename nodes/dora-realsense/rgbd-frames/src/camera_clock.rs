//! The camera's hardware clock as one 64-bit timeline for as long as the node runs.
//!
//! The D455 stamps every frame of every stream with one free-running 32-bit microsecond counter, which counts from
//! the camera's firmware start, not from a stream start, and wraps every 2^32 us (71 min 35 s). librealsense passes
//! that counter on as it is, in milliseconds: for colour and depth the UVC metadata's 32-bit timestamp, for the
//! accelerometer and gyro the low 32 bits of the HID report's timestamp, which librealsense trims to 32 bits so that
//! motion and video share the counter. All four streams therefore read the same counter, with the same zero and the
//! same wrap, and one `CameraClock` unwraps them all onto one timeline.
//!
//! The rules:
//! - The origin, the reading a clock starts at or else its first reading, is placed in epoch 1 (the counter's second
//!   pass), so a late reading from the pass before it still has a time; camera time is the counter range plus the
//!   counter from then on, and never starts near zero.
//! - Every later reading is placed in the epoch that puts it nearest the latest time placed so far, of any stream. A
//!   reading more than half the counter range below the latest's counter value is the counter having wrapped; a
//!   reading more than half the range above it is a late one from the epoch before. Streams arrive on different
//!   threads and cross the wrap at slightly different wall times, and each is placed against the same latest time,
//!   so no stream's epoch drifts from another's. Every stream reads the counter many times a second, far more often
//!   than once per half range (35 min 47 s), so a real step is never near half the range.
//! - Each stream's times must rise: a reading placed at or before the previous one of its own stream is an error
//!   (`CameraTimeWentBack`), never a wrap. A reading may be older than another stream's latest.

use std::fmt;

/// The range of the camera's counter, in microseconds: it wraps from 2^32 - 1 to 0.
pub const COUNTER_RANGE_US: u64 = 1 << 32;
const HALF_COUNTER_RANGE_US: u64 = COUNTER_RANGE_US / 2;
/// The epoch the first reading is placed in.
const FIRST_EPOCH: u64 = 1;

/// A stream of the camera, each read against the same counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CameraStream {
    Accel,
    Gyro,
    Depth,
    Color,
}

impl CameraStream {
    const COUNT: usize = 4;

    fn index(self) -> usize {
        match self {
            CameraStream::Accel => 0,
            CameraStream::Gyro => 1,
            CameraStream::Depth => 2,
            CameraStream::Color => 3,
        }
    }
}

impl fmt::Display for CameraStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            CameraStream::Accel => "accel",
            CameraStream::Gyro => "gyro",
            CameraStream::Depth => "depth",
            CameraStream::Color => "colour",
        })
    }
}

/// One reading of the camera's 32-bit microsecond counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CounterReading {
    micros: u32,
}

impl CounterReading {
    pub fn from_micros(micros: u32) -> CounterReading {
        CounterReading { micros }
    }

    /// The reading librealsense reports as a frame's timestamp: the counter times 0.001, in milliseconds. Anything
    /// else (negative, fractional microseconds, beyond the counter's range) is not a reading of the counter.
    pub fn from_librealsense_ms(ms: f64) -> Result<CounterReading, NotACounterReading> {
        let micros = (ms * 1e3).round();
        let whole_micros = (ms * 1e3 - micros).abs() < 1e-2;
        if !ms.is_finite() || !whole_micros || micros < 0.0 || micros >= COUNTER_RANGE_US as f64 {
            return Err(NotACounterReading { ms });
        }
        Ok(CounterReading { micros: micros as u32 })
    }

    pub fn micros(self) -> u32 {
        self.micros
    }
}

/// A frame timestamp that is not a reading of the camera's 32-bit microsecond counter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NotACounterReading {
    pub ms: f64,
}

impl fmt::Display for NotACounterReading {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "the timestamp {} ms is not a reading of the camera's 32-bit microsecond counter", self.ms)
    }
}

impl std::error::Error for NotACounterReading {}

/// A time on the camera's clock, unwrapped: whole microseconds on one 64-bit timeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CameraTime {
    micros: u64,
}

impl CameraTime {
    pub fn micros(self) -> u64 {
        self.micros
    }

    /// The time in nanoseconds, as every `camera_ns` the node sends.
    pub fn nanos(self) -> i64 {
        (self.micros * 1_000) as i64
    }

    fn epoch(self) -> u64 {
        self.micros / COUNTER_RANGE_US
    }
}

impl fmt::Display for CameraTime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}.{:06} s", self.micros / 1_000_000, self.micros % 1_000_000)
    }
}

/// A stream's reading placed at or before the previous reading of the same stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CameraTimeWentBack {
    pub stream: CameraStream,
    pub previous: CameraTime,
    pub reading: CameraTime,
}

impl fmt::Display for CameraTimeWentBack {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "the {} stream's camera time went from {} to {}: it is not newer than the stream's previous reading",
            self.stream, self.previous, self.reading
        )
    }
}

impl std::error::Error for CameraTimeWentBack {}

/// The camera's counter unwrapped across every stream (see the module's rules): the latest time placed, which
/// holds the latest counter value and the wraps so far, and each stream's previous time.
#[derive(Clone, Debug, Default)]
pub struct CameraClock {
    latest: Option<CameraTime>,
    previous_by_stream: [Option<CameraTime>; CameraStream::COUNT],
}

impl CameraClock {
    /// A clock whose origin is the first reading it unwraps.
    pub fn new() -> CameraClock {
        CameraClock::default()
    }

    /// A clock whose origin is `first`, a reading of no stream's sequence: it places later readings and holds no
    /// stream's previous time.
    pub fn starting_at(first: CounterReading) -> CameraClock {
        let latest = CameraTime { micros: FIRST_EPOCH * COUNTER_RANGE_US + u64::from(first.micros) };
        CameraClock { latest: Some(latest), ..CameraClock::default() }
    }

    /// The unwrapped time of `reading` from `stream`. On an error nothing changes.
    pub fn unwrap_reading(&mut self, stream: CameraStream, reading: CounterReading) -> Result<CameraTime, CameraTimeWentBack> {
        let counter = u64::from(reading.micros);
        let time = match self.latest {
            None => CameraTime { micros: FIRST_EPOCH * COUNTER_RANGE_US + counter },
            Some(latest) => {
                let in_latest_epoch = latest.epoch() * COUNTER_RANGE_US + counter;
                let micros = if in_latest_epoch + HALF_COUNTER_RANGE_US < latest.micros {
                    in_latest_epoch + COUNTER_RANGE_US
                } else if in_latest_epoch > latest.micros + HALF_COUNTER_RANGE_US {
                    in_latest_epoch - COUNTER_RANGE_US
                } else {
                    in_latest_epoch
                };
                CameraTime { micros }
            }
        };
        let previous = &mut self.previous_by_stream[stream.index()];
        if let Some(previous) = *previous {
            if time <= previous {
                return Err(CameraTimeWentBack { stream, previous, reading: time });
            }
        }
        *previous = Some(time);
        self.latest = Some(self.latest.map_or(time, |latest| latest.max(time)));
        Ok(time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RANGE: u64 = COUNTER_RANGE_US;

    fn reading(counter_us: u64) -> CounterReading {
        CounterReading::from_micros((counter_us % RANGE) as u32)
    }

    fn place(clock: &mut CameraClock, stream: CameraStream, counter_us: u64) -> u64 {
        clock.unwrap_reading(stream, reading(counter_us)).expect("a newer reading").micros()
    }

    #[test]
    fn librealsense_milliseconds_are_read_back_as_the_counter() {
        for micros in [0u32, 1, 996_725_664, 4_294_965_386, u32::MAX] {
            let ms = f64::from(micros) * 0.001;
            assert_eq!(CounterReading::from_librealsense_ms(ms), Ok(CounterReading::from_micros(micros)));
        }
    }

    #[test]
    fn a_timestamp_that_is_not_a_counter_reading_is_refused() {
        for ms in [-0.001, 4_294_967.296, 1.7e12, 12.3456, f64::NAN, f64::INFINITY] {
            assert!(CounterReading::from_librealsense_ms(ms).is_err(), "{ms} ms was taken as a counter reading");
        }
    }

    #[test]
    fn the_first_reading_is_placed_in_the_counters_second_pass() {
        let mut clock = CameraClock::new();
        let first = clock.unwrap_reading(CameraStream::Depth, reading(996_725_664)).unwrap();
        assert_eq!(first.micros(), RANGE + 996_725_664);
        assert_eq!(first.nanos(), (RANGE + 996_725_664) as i64 * 1_000);
        assert_eq!(first.to_string(), "5291.692960 s");
    }

    #[test]
    fn a_clock_started_at_a_reading_places_that_reading_again_for_any_stream() {
        let mut clock = CameraClock::starting_at(reading(RANGE - 2_000));
        assert_eq!(place(&mut clock, CameraStream::Depth, RANGE - 2_000), 2 * RANGE - 2_000);
        assert_eq!(place(&mut clock, CameraStream::Accel, RANGE - 9_000), 2 * RANGE - 9_000);
        assert_eq!(place(&mut clock, CameraStream::Gyro, RANGE + 1_000), 2 * RANGE + 1_000);
    }

    #[test]
    fn one_stream_crosses_the_wrap_without_a_step() {
        let mut clock = CameraClock::new();
        let start = RANGE - 20_000;
        let times: Vec<u64> = (0..10).map(|k| place(&mut clock, CameraStream::Accel, start + k * 5_000)).collect();
        assert!(times.windows(2).all(|pair| pair[1] - pair[0] == 5_000), "{times:?}");
        assert_eq!(times[0], RANGE + start);
        assert_eq!(times[9], 2 * RANGE + 25_000);
    }

    #[test]
    fn interleaved_streams_crossing_the_wrap_at_different_times_share_one_timeline() {
        // Each stream's counter value and where it lands; the gyro and colour cross first, the depth last.
        let order = [
            (CameraStream::Accel, RANGE - 3_000),
            (CameraStream::Gyro, RANGE + 500),
            (CameraStream::Depth, RANGE - 2_500),
            (CameraStream::Color, RANGE + 1_000),
            (CameraStream::Accel, RANGE + 2_000),
            (CameraStream::Depth, RANGE - 100),
            (CameraStream::Gyro, RANGE + 3_000),
            (CameraStream::Depth, RANGE + 33_000),
        ];
        let mut clock = CameraClock::new();
        for (stream, counter_us) in order {
            assert_eq!(place(&mut clock, stream, counter_us), RANGE + counter_us, "{stream} at {counter_us} us");
        }
    }

    #[test]
    fn a_late_reading_from_the_epoch_before_belongs_to_it() {
        let mut clock = CameraClock::new();
        assert_eq!(place(&mut clock, CameraStream::Gyro, RANGE - 1_000), 2 * RANGE - 1_000);
        assert_eq!(place(&mut clock, CameraStream::Gyro, RANGE + 1_500), 2 * RANGE + 1_500);
        assert_eq!(place(&mut clock, CameraStream::Depth, RANGE - 10), 2 * RANGE - 10);
        assert_eq!(place(&mut clock, CameraStream::Depth, RANGE + 33_000), 2 * RANGE + 33_000);
    }

    #[test]
    fn a_late_reading_from_before_the_first_wrap_seen_has_a_time() {
        let mut clock = CameraClock::new();
        assert_eq!(place(&mut clock, CameraStream::Gyro, 5), RANGE + 5);
        assert_eq!(place(&mut clock, CameraStream::Depth, RANGE - 5), RANGE - 5);
        assert!(clock.unwrap_reading(CameraStream::Depth, reading(RANGE - 6)).is_err());
    }

    #[test]
    fn a_small_backward_step_of_one_stream_is_refused_and_changes_nothing() {
        let mut clock = CameraClock::new();
        place(&mut clock, CameraStream::Accel, 1_000_000);
        let went_back = clock.unwrap_reading(CameraStream::Accel, reading(995_000)).expect_err("a step back");
        assert_eq!(went_back.stream, CameraStream::Accel);
        assert_eq!(went_back.previous.micros(), RANGE + 1_000_000);
        assert_eq!(went_back.reading.micros(), RANGE + 995_000);
        assert!(clock.unwrap_reading(CameraStream::Accel, reading(1_000_000)).is_err(), "a repeated reading");
        assert_eq!(place(&mut clock, CameraStream::Accel, 1_005_000), RANGE + 1_005_000);
        // Another stream may read slightly older than the latest.
        assert_eq!(place(&mut clock, CameraStream::Gyro, 1_002_500), RANGE + 1_002_500);
    }

    #[test]
    fn a_step_back_across_the_wrap_is_refused_not_taken_for_a_wrap() {
        let mut clock = CameraClock::new();
        place(&mut clock, CameraStream::Accel, RANGE - 5_000);
        place(&mut clock, CameraStream::Accel, RANGE + 5);
        assert!(clock.unwrap_reading(CameraStream::Accel, reading(RANGE - 10)).is_err());
    }

    #[test]
    fn several_wraps_add_up() {
        let mut clock = CameraClock::new();
        let step = RANGE / 8;
        let start = 996_725_664;
        for k in 0..40 {
            let counter_us = start + k * step;
            assert_eq!(place(&mut clock, CameraStream::Depth, counter_us), RANGE + counter_us);
        }
    }

    /// One stream of the simulated camera: its period, and its delivery latency, which varies by less than the period
    /// so the stream arrives in order.
    struct SimulatedStream {
        stream: CameraStream,
        period_us: u64,
        latency_us: u64,
        latency_jitter_us: u64,
    }

    impl SimulatedStream {
        fn capture_us(&self, start_us: u64, index: u64) -> u64 {
            start_us + index * self.period_us
        }

        fn arrival_us(&self, start_us: u64, index: u64) -> u64 {
            self.capture_us(start_us, index) + self.latency_us + (index * 7_919) % self.latency_jitter_us
        }
    }

    #[test]
    fn the_four_streams_run_past_two_and_three_counter_ranges_on_one_timeline() {
        let streams = [
            SimulatedStream { stream: CameraStream::Accel, period_us: 5_000, latency_us: 1_000, latency_jitter_us: 3_000 },
            SimulatedStream { stream: CameraStream::Gyro, period_us: 2_500, latency_us: 500, latency_jitter_us: 2_000 },
            SimulatedStream { stream: CameraStream::Depth, period_us: 33_333, latency_us: 20_000, latency_jitter_us: 15_000 },
            SimulatedStream { stream: CameraStream::Color, period_us: 33_333, latency_us: 25_000, latency_jitter_us: 15_000 },
        ];
        // The counter at stream start as on the rig, run until it has passed 2^33 us.
        let start_us = 996_725_664;
        let end_us = 2 * RANGE + 60_000_000;
        let mut next = [0u64; 4];
        let mut clock = CameraClock::new();
        let (mut placed, mut latest) = (0u64, 0u64);
        while let Some(due) = (0..streams.len())
            .filter(|&s| streams[s].capture_us(start_us, next[s]) < end_us)
            .min_by_key(|&s| streams[s].arrival_us(start_us, next[s]))
        {
            let stream = &streams[due];
            let counter_us = stream.capture_us(start_us, next[due]);
            let time = place(&mut clock, stream.stream, counter_us);
            assert_eq!(time, RANGE + counter_us, "{} sample {}", stream.stream, next[due]);
            latest = latest.max(time);
            next[due] += 1;
            placed += 1;
        }
        assert!(latest > 3 * RANGE, "the run reached {latest} us");
        assert!(placed > 4_000_000, "{placed} readings");
    }
}
