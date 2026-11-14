//! Pairs a camera's two IMU streams into the single stream the odometry takes.
//!
//! The accelerometer and the gyro each stamp their own samples on the camera's clock, at their own
//! rates, and arrive in batches. The odometry wants both readings at one time, so there is one paired
//! sample per gyro sample, with the accelerometer interpolated to the gyro's time.

use std::collections::VecDeque;
use std::time::Duration;

use glam::DVec3;

use crate::odometry::ImuSample;

/// One accelerometer sample, on the camera's clock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccelReading {
    pub time: Duration,
    pub m_s2: DVec3,
}

/// One gyro sample, on the camera's clock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GyroReading {
    pub time: Duration,
    pub rad_s: DVec3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImuStream {
    Accelerometer,
    Gyro,
}

impl std::fmt::Display for ImuStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ImuStream::Accelerometer => "accelerometer",
            ImuStream::Gyro => "gyro",
        })
    }
}

/// A sample that is not newer than the one before it in its stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImuOutOfOrder {
    pub stream: ImuStream,
    pub at: Duration,
    pub previous: Duration,
}

impl std::fmt::Display for ImuOutOfOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} sample at {:?} is not newer than the one before it at {:?}", self.stream, self.at, self.previous)
    }
}

impl std::error::Error for ImuOutOfOrder {}

/// Pairs the two streams. A gyro sample newer than the latest accelerometer sample is held until an
/// accelerometer sample at or after its time arrives. Gyro samples older than the first accelerometer
/// sample ever given have nothing to pair with and are left out.
#[derive(Clone, Debug, Default)]
pub struct ImuPairer {
    /// Gyro samples not yet paired, oldest first.
    held_gyro: VecDeque<GyroReading>,
    /// The accelerometer samples from the last one at or before the oldest held gyro sample onwards.
    accel: VecDeque<AccelReading>,
    latest_gyro: Option<Duration>,
}

impl ImuPairer {
    /// When the newest sample of a stream was taken.
    pub fn latest(&self, stream: ImuStream) -> Option<Duration> {
        match stream {
            ImuStream::Accelerometer => self.accel.back().map(|reading| reading.time),
            ImuStream::Gyro => self.latest_gyro,
        }
    }

    /// The span of camera time over the samples of `stream` that are held for the other stream to
    /// reach them. It grows for as long as the other stream gives nothing.
    pub fn held_span(&self, stream: ImuStream) -> Duration {
        let span = |oldest: Option<Duration>, newest: Option<Duration>| oldest.zip(newest).map_or(Duration::ZERO, |(oldest, newest)| newest - oldest);
        match stream {
            ImuStream::Accelerometer => span(self.accel.front().map(|reading| reading.time), self.accel.back().map(|reading| reading.time)),
            ImuStream::Gyro => span(self.held_gyro.front().map(|reading| reading.time), self.held_gyro.back().map(|reading| reading.time)),
        }
    }

    /// Takes every sample since the previous batch, each stream oldest first, and appends to `paired`
    /// the samples that can now be paired, in time order. A batch with a sample out of order is
    /// refused whole.
    pub fn add_batch(&mut self, accel: &[AccelReading], gyro: &[GyroReading], paired: &mut Vec<ImuSample>) -> Result<(), ImuOutOfOrder> {
        in_order(ImuStream::Accelerometer, self.latest(ImuStream::Accelerometer), accel.iter().map(|reading| reading.time))?;
        in_order(ImuStream::Gyro, self.latest_gyro, gyro.iter().map(|reading| reading.time))?;
        self.accel.extend(accel);
        self.held_gyro.extend(gyro);
        self.latest_gyro = gyro.last().map(|reading| reading.time).or(self.latest_gyro);
        while let Some(&gyro) = self.held_gyro.front() {
            while self.accel.len() >= 2 && self.accel[1].time <= gyro.time {
                self.accel.pop_front();
            }
            let Some(&before) = self.accel.front() else { break };
            if gyro.time > before.time {
                let Some(&after) = self.accel.get(1) else { break };
                let share = (gyro.time - before.time).as_secs_f64() / (after.time - before.time).as_secs_f64();
                paired.push(ImuSample { time: gyro.time, gyro_rad_s: gyro.rad_s, accel_m_s2: before.m_s2.lerp(after.m_s2, share) });
            } else if gyro.time == before.time {
                paired.push(ImuSample { time: gyro.time, gyro_rad_s: gyro.rad_s, accel_m_s2: before.m_s2 });
            }
            self.held_gyro.pop_front();
        }
        Ok(())
    }

    /// Appends to `paired` the held gyro samples taken at or before `through`, each with the latest
    /// accelerometer reading as it stands. For a consumer that must act at `through` and cannot wait
    /// for the accelerometer to catch up; it says how stale a reading it accepts by checking
    /// [`Self::latest`] first. Without any accelerometer sample nothing is paired.
    pub fn pair_held_with_latest_accel(&mut self, through: Duration, paired: &mut Vec<ImuSample>) {
        let Some(&latest) = self.accel.back() else { return };
        while let Some(gyro) = self.held_gyro.front().copied().filter(|gyro| gyro.time <= through) {
            paired.push(ImuSample { time: gyro.time, gyro_rad_s: gyro.rad_s, accel_m_s2: latest.m_s2 });
            self.held_gyro.pop_front();
        }
    }
}

fn in_order(stream: ImuStream, mut previous: Option<Duration>, times: impl Iterator<Item = Duration>) -> Result<(), ImuOutOfOrder> {
    for at in times {
        if let Some(previous) = previous.filter(|&previous| at <= previous) {
            return Err(ImuOutOfOrder { stream, at, previous });
        }
        previous = Some(at);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(millis: f64) -> Duration {
        Duration::from_micros((millis * 1e3).round() as u64)
    }

    /// An accelerometer whose x reads its own time in milliseconds, so an interpolated value is the
    /// time it was interpolated to.
    fn accel(millis: f64) -> AccelReading {
        AccelReading { time: ms(millis), m_s2: DVec3::new(millis, 0.0, 9.81) }
    }

    fn gyro(millis: f64) -> GyroReading {
        GyroReading { time: ms(millis), rad_s: DVec3::new(0.0, millis, 0.0) }
    }

    fn times_ms(paired: &[ImuSample]) -> Vec<f64> {
        paired.iter().map(|sample| (sample.time.as_secs_f64() * 1e4).round() / 10.0).collect()
    }

    #[test]
    fn each_gyro_sample_is_paired_with_the_accelerometer_interpolated_to_its_time() {
        let mut paired = Vec::new();
        ImuPairer::default().add_batch(&[accel(0.0), accel(5.0), accel(10.0)], &[gyro(2.5), gyro(5.0), gyro(7.5), gyro(9.0)], &mut paired).expect("in order");
        assert_eq!(times_ms(&paired), vec![2.5, 5.0, 7.5, 9.0]);
        for sample in &paired {
            let millis = sample.time.as_secs_f64() * 1e3;
            assert!((sample.accel_m_s2 - DVec3::new(millis, 0.0, 9.81)).length() < 1e-9 && (sample.gyro_rad_s.y - millis).abs() < 1e-9, "{sample:?}");
        }
    }

    #[test]
    fn gyro_samples_newer_than_the_latest_accelerometer_sample_wait_for_the_next_batch() {
        let mut pairer = ImuPairer::default();
        let mut paired = Vec::new();
        pairer.add_batch(&[accel(0.0), accel(5.0)], &[gyro(2.5), gyro(5.0), gyro(7.5), gyro(10.0)], &mut paired).expect("in order");
        assert_eq!(times_ms(&paired), vec![2.5, 5.0]);
        pairer.add_batch(&[accel(10.0)], &[gyro(12.5)], &mut paired).expect("in order");
        assert_eq!(times_ms(&paired), vec![2.5, 5.0, 7.5, 10.0]);
        assert!((paired[2].accel_m_s2.x - 7.5).abs() < 1e-9, "{:?}", paired[2]);
    }

    #[test]
    fn gyro_samples_older_than_the_first_accelerometer_sample_are_left_out() {
        let mut pairer = ImuPairer::default();
        let mut paired = Vec::new();
        pairer.add_batch(&[], &[gyro(1.0), gyro(3.5)], &mut paired).expect("in order");
        assert!(paired.is_empty());
        pairer.add_batch(&[accel(3.0), accel(8.0)], &[gyro(6.0)], &mut paired).expect("in order");
        assert_eq!(times_ms(&paired), vec![3.5, 6.0]);
    }

    #[test]
    fn a_sample_older_than_the_one_before_it_is_refused_and_the_batch_with_it() {
        let mut pairer = ImuPairer::default();
        let mut paired = Vec::new();
        pairer.add_batch(&[accel(0.0), accel(5.0)], &[gyro(2.5)], &mut paired).expect("in order");
        let within_a_batch = pairer.add_batch(&[accel(10.0)], &[gyro(5.0), gyro(4.0)], &mut paired);
        assert_eq!(within_a_batch, Err(ImuOutOfOrder { stream: ImuStream::Gyro, at: ms(4.0), previous: ms(5.0) }));
        let across_batches = pairer.add_batch(&[accel(5.0)], &[], &mut paired);
        assert_eq!(across_batches, Err(ImuOutOfOrder { stream: ImuStream::Accelerometer, at: ms(5.0), previous: ms(5.0) }));
        // Neither refused batch left anything behind.
        assert_eq!(times_ms(&paired), vec![2.5]);
        pairer.add_batch(&[accel(10.0)], &[gyro(5.0), gyro(7.5)], &mut paired).expect("in order");
        assert_eq!(times_ms(&paired), vec![2.5, 5.0, 7.5]);
    }

    #[test]
    fn held_gyro_samples_up_to_a_time_can_be_paired_with_the_latest_accelerometer_reading() {
        let mut pairer = ImuPairer::default();
        let mut paired = Vec::new();
        pairer.add_batch(&[accel(0.0), accel(5.0)], &[gyro(5.0), gyro(7.5), gyro(10.0), gyro(12.5)], &mut paired).expect("in order");
        pairer.pair_held_with_latest_accel(ms(10.0), &mut paired);
        assert_eq!(times_ms(&paired), vec![5.0, 7.5, 10.0]);
        assert!(paired[1..].iter().all(|sample| sample.accel_m_s2.x == 5.0), "{paired:?}");
        // The sample still held is interpolated as usual once the accelerometer passes it.
        pairer.add_batch(&[accel(10.0), accel(15.0)], &[], &mut paired).expect("in order");
        assert_eq!(times_ms(&paired), vec![5.0, 7.5, 10.0, 12.5]);
        assert!((paired[3].accel_m_s2.x - 12.5).abs() < 1e-9, "{:?}", paired[3]);
    }

    #[test]
    fn the_span_held_of_a_stream_grows_while_the_other_stream_is_silent_and_falls_back_when_it_speaks() {
        let mut pairer = ImuPairer::default();
        let mut paired = Vec::new();
        assert_eq!((pairer.held_span(ImuStream::Accelerometer), pairer.held_span(ImuStream::Gyro)), (Duration::ZERO, Duration::ZERO));
        pairer.add_batch(&[accel(0.0)], &[gyro(2.5), gyro(5.0), gyro(7.5)], &mut paired).expect("in order");
        pairer.add_batch(&[], &[gyro(10.0), gyro(12.5)], &mut paired).expect("in order");
        assert_eq!((pairer.held_span(ImuStream::Accelerometer), pairer.held_span(ImuStream::Gyro)), (Duration::ZERO, ms(10.0)));
        pairer.add_batch(&[accel(5.0), accel(10.0), accel(15.0), accel(20.0), accel(25.0)], &[], &mut paired).expect("in order");
        assert_eq!(times_ms(&paired), vec![2.5, 5.0, 7.5, 10.0, 12.5]);
        assert_eq!((pairer.held_span(ImuStream::Accelerometer), pairer.held_span(ImuStream::Gyro)), (ms(15.0), Duration::ZERO));
        pairer.add_batch(&[], &[gyro(24.0)], &mut paired).expect("in order");
        assert_eq!((pairer.held_span(ImuStream::Accelerometer), pairer.held_span(ImuStream::Gyro)), (ms(5.0), Duration::ZERO));
    }

    #[test]
    fn the_latest_time_of_each_stream_is_known() {
        let mut pairer = ImuPairer::default();
        assert_eq!((pairer.latest(ImuStream::Accelerometer), pairer.latest(ImuStream::Gyro)), (None, None));
        pairer.add_batch(&[accel(0.0), accel(5.0)], &[gyro(2.5)], &mut Vec::new()).expect("in order");
        assert_eq!((pairer.latest(ImuStream::Accelerometer), pairer.latest(ImuStream::Gyro)), (Some(ms(5.0)), Some(ms(2.5))));
    }
}
