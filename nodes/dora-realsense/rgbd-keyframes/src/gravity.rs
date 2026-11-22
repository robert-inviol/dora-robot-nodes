//! Which way is up in the IMU's axes while the camera moves.
//!
//! A hand-held or driven camera's accelerometer reads gravity's reaction plus every push it is given, so its direction
//! alone swings with each step. [`GravityFilter`] is a complementary filter: up is turned by every gyro sample, which
//! follows the camera's rotation without lag, and pulled towards the accelerometer's direction with a time constant,
//! which removes the gyro's drift while averaging the pushes away. It starts once the IMU has been at rest.

use std::collections::VecDeque;
use std::fmt;
use std::time::Duration;

use glam::{Quat, Vec3};

/// The size of the specific force the filter reports: it estimates only the direction, and at rest the accelerometer
/// reads gravity's reaction.
pub const STANDARD_GRAVITY_M_S2: f32 = 9.80665;

/// One of the IMU's two sample streams.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImuStream {
    Accelerometer,
    Gyro,
}

impl fmt::Display for ImuStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ImuStream::Accelerometer => "accelerometer",
            ImuStream::Gyro => "gyro",
        })
    }
}

/// A sample the filter refuses. Times are camera times.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GravityFilterError {
    /// The sample is not newer than the one before it in its stream.
    OutOfOrder { stream: ImuStream, at: Duration, previous: Duration },
    /// The gyro was silent for longer than [`GravityFilter::MAX_GYRO_GAP`], so the turn in between is unknown.
    GyroGap { previous: Duration, at: Duration },
    /// The samples of `stream` held for the other stream to catch up span longer than
    /// [`GravityFilter::MAX_HELD_SPAN`].
    Backlog { stream: ImuStream, span: Duration },
}

impl fmt::Display for GravityFilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GravityFilterError::OutOfOrder { stream, at, previous } => {
                write!(f, "{stream} sample at {at:?} is not newer than the one before it at {previous:?}")
            }
            GravityFilterError::GyroGap { previous, at } => write!(
                f,
                "no gyro sample between {previous:?} and {at:?}, longer than {:?}, so the turn in between is unknown",
                GravityFilter::MAX_GYRO_GAP
            ),
            GravityFilterError::Backlog { stream, span } => write!(
                f,
                "{span:?} of {stream} samples are waiting for the other stream, more than {:?}",
                GravityFilter::MAX_HELD_SPAN
            ),
        }
    }
}

impl std::error::Error for GravityFilterError {}

#[derive(Clone, Copy, Debug)]
struct Sample {
    at: Duration,
    reading: Vec3,
}

#[derive(Clone, Debug)]
enum Phase {
    /// The latest samples of each stream, oldest first, reaching back at least [`GravityFilter::REST_SPAN`] once
    /// the stream has run that long.
    AwaitingRest { accel_m_s2: VecDeque<Sample>, gyro_rad_s: VecDeque<Sample> },
    /// `up` is a unit vector in the IMU's axes, as of the latest sample applied of each stream.
    Tracking { up: Vec3, last_accel: Duration, last_gyro: Option<Duration> },
}

/// Up in the IMU's axes, from the accelerometer and gyro samples of one IMU, both in those axes and on one clock.
///
/// Samples are given stream by stream, each stream in time order, and held until [`GravityFilter::advance_to`]
/// applies them to the estimate in time order across both streams: only up to the newest sample of the stream that
/// is behind, so a sample still to come from it is never older than one applied. Each gyro sample turns up by its
/// rate over the time since the previous gyro sample; each accelerometer sample pulls up towards its own direction
/// by the time since the previous accelerometer sample over the time constant.
///
/// The estimate starts at the mean accelerometer direction of the first [`GravityFilter::REST_SPAN`] in which the
/// IMU was at rest, by the odometry's static initialization rules: neither stream spreads, the mean gyro rate is no
/// more than a bias, and the mean acceleration is the size of gravity. The gyro's bias is not estimated.
#[derive(Clone, Debug)]
pub struct GravityFilter {
    time_constant: Duration,
    held_accel: VecDeque<Sample>,
    held_gyro: VecDeque<Sample>,
    latest_accel: Option<Duration>,
    latest_gyro: Option<Duration>,
    phase: Phase,
}

impl GravityFilter {
    /// How long the IMU has to be at rest for the estimate to start.
    pub const REST_SPAN: Duration = Duration::from_secs(1);
    /// At rest, the largest standard deviation of any accelerometer axis, as in the odometry's static initialization.
    pub const MAX_REST_ACCEL_SPREAD_M_S2: f32 = 0.3;
    /// At rest, the largest standard deviation of any gyro axis, as in the odometry's static initialization.
    pub const MAX_REST_GYRO_SPREAD_RAD_S: f32 = 0.02;
    /// At rest the mean gyro rate is its bias, well under this for a calibrated IMU; a steady turn shows as more.
    pub const MAX_REST_MEAN_GYRO_RAD_S: f32 = 0.05;
    /// At rest the accelerometer's mean reading is gravity, off only by its bias and scale error.
    pub const MAX_REST_GRAVITY_ERROR_M_S2: f32 = 0.5;
    /// The D455's gyro sends a sample every 2.5 ms; silent for this long, the turn in between is unknown.
    pub const MAX_GYRO_GAP: Duration = Duration::from_millis(100);
    /// One stream's samples wait at most this long for the other stream; longer means the other stream stopped.
    pub const MAX_HELD_SPAN: Duration = Duration::from_secs(1);

    /// A filter that pulls up towards the accelerometer's direction with `time_constant`: a disagreement between
    /// the two shrinks by a factor e in that time, and a push shorter than it barely moves up.
    pub fn new(time_constant: Duration) -> GravityFilter {
        GravityFilter {
            time_constant,
            held_accel: VecDeque::new(),
            held_gyro: VecDeque::new(),
            latest_accel: None,
            latest_gyro: None,
            phase: Phase::AwaitingRest { accel_m_s2: VecDeque::new(), gyro_rad_s: VecDeque::new() },
        }
    }

    /// Holds one accelerometer sample, the specific force at camera time `at`, for [`Self::advance_to`].
    pub fn add_accel(&mut self, at: Duration, specific_force_m_s2: Vec3) -> Result<(), GravityFilterError> {
        if let Some(previous) = self.latest_accel.filter(|&previous| at <= previous) {
            return Err(GravityFilterError::OutOfOrder { stream: ImuStream::Accelerometer, at, previous });
        }
        self.latest_accel = Some(at);
        self.held_accel.push_back(Sample { at, reading: specific_force_m_s2 });
        within_held_span(ImuStream::Accelerometer, &self.held_accel)
    }

    /// Holds one gyro sample, the angular velocity at camera time `at`, for [`Self::advance_to`].
    pub fn add_gyro(&mut self, at: Duration, angular_velocity_rad_s: Vec3) -> Result<(), GravityFilterError> {
        if let Some(previous) = self.latest_gyro {
            if at <= previous {
                return Err(GravityFilterError::OutOfOrder { stream: ImuStream::Gyro, at, previous });
            }
            if at - previous > Self::MAX_GYRO_GAP {
                return Err(GravityFilterError::GyroGap { previous, at });
            }
        }
        self.latest_gyro = Some(at);
        self.held_gyro.push_back(Sample { at, reading: angular_velocity_rad_s });
        within_held_span(ImuStream::Gyro, &self.held_gyro)
    }

    /// Applies, in time order, every held sample taken at or before `at` and at or before the newest sample of each
    /// stream. Samples after that stay held for a later call.
    pub fn advance_to(&mut self, at: Duration) {
        let (Some(latest_accel), Some(latest_gyro)) = (self.latest_accel, self.latest_gyro) else { return };
        let through = at.min(latest_accel).min(latest_gyro);
        loop {
            let accel = self.held_accel.front().copied().filter(|sample| sample.at <= through);
            let gyro = self.held_gyro.front().copied().filter(|sample| sample.at <= through);
            match (accel, gyro) {
                (Some(accel), Some(gyro)) if gyro.at < accel.at => {
                    self.held_gyro.pop_front();
                    self.apply_gyro(gyro);
                }
                (Some(accel), _) => {
                    self.held_accel.pop_front();
                    self.apply_accel(accel);
                }
                (None, Some(gyro)) => {
                    self.held_gyro.pop_front();
                    self.apply_gyro(gyro);
                }
                (None, None) => break,
            }
        }
    }

    /// The specific force the IMU would read at rest in its present orientation: up in the IMU's axes, scaled to
    /// [`STANDARD_GRAVITY_M_S2`]. None until the IMU has been at rest for [`Self::REST_SPAN`].
    pub fn specific_force(&self) -> Option<Vec3> {
        match self.phase {
            Phase::AwaitingRest { .. } => None,
            Phase::Tracking { up, .. } => Some(up * STANDARD_GRAVITY_M_S2),
        }
    }

    fn apply_accel(&mut self, sample: Sample) {
        match &mut self.phase {
            Phase::AwaitingRest { accel_m_s2, gyro_rad_s } => {
                accel_m_s2.push_back(sample);
                keep_rest_span(accel_m_s2, sample.at);
                keep_rest_span(gyro_rad_s, sample.at);
                if let Some(up) = rest_up(accel_m_s2, gyro_rad_s, sample.at) {
                    let last_gyro = gyro_rad_s.back().map(|gyro| gyro.at);
                    self.phase = Phase::Tracking { up, last_accel: sample.at, last_gyro };
                }
            }
            Phase::Tracking { up, last_accel, .. } => {
                let weight = ((sample.at - *last_accel).as_secs_f32() / self.time_constant.as_secs_f32()).min(1.0);
                *up = up.lerp(sample.reading.normalize(), weight).normalize();
                *last_accel = sample.at;
            }
        }
    }

    fn apply_gyro(&mut self, sample: Sample) {
        match &mut self.phase {
            Phase::AwaitingRest { gyro_rad_s, .. } => gyro_rad_s.push_back(sample),
            Phase::Tracking { up, last_gyro, .. } => {
                if let Some(previous) = *last_gyro {
                    let held_s = (sample.at - previous).as_secs_f32();
                    // A direction fixed in the world, seen from a body turning at `rate`, turns at minus that rate.
                    *up = (Quat::from_scaled_axis(-sample.reading * held_s) * *up).normalize();
                }
                *last_gyro = Some(sample.at);
            }
        }
    }
}

fn within_held_span(stream: ImuStream, held: &VecDeque<Sample>) -> Result<(), GravityFilterError> {
    let (Some(oldest), Some(newest)) = (held.front(), held.back()) else { return Ok(()) };
    let span = newest.at - oldest.at;
    if span > GravityFilter::MAX_HELD_SPAN {
        return Err(GravityFilterError::Backlog { stream, span });
    }
    Ok(())
}

/// Drops the samples before the latest one at or before `now - REST_SPAN`, so the rest reaches back exactly the span.
fn keep_rest_span(samples: &mut VecDeque<Sample>, now: Duration) {
    let Some(start) = now.checked_sub(GravityFilter::REST_SPAN) else { return };
    while samples.get(1).is_some_and(|next| next.at <= start) {
        samples.pop_front();
    }
}

/// Up as the mean accelerometer direction, when both streams reach back [`GravityFilter::REST_SPAN`] from `now`
/// and say the IMU was at rest over it.
fn rest_up(accel_m_s2: &VecDeque<Sample>, gyro_rad_s: &VecDeque<Sample>, now: Duration) -> Option<Vec3> {
    let start = now.checked_sub(GravityFilter::REST_SPAN)?;
    let reaches_back = |samples: &VecDeque<Sample>| samples.front().is_some_and(|oldest| oldest.at <= start);
    if !reaches_back(accel_m_s2) || !reaches_back(gyro_rad_s) {
        return None;
    }
    let (accel, accel_spread) = mean_and_spread(accel_m_s2);
    let (gyro, gyro_spread) = mean_and_spread(gyro_rad_s);
    let at_rest = accel_spread <= GravityFilter::MAX_REST_ACCEL_SPREAD_M_S2
        && gyro_spread <= GravityFilter::MAX_REST_GYRO_SPREAD_RAD_S
        && gyro.length() <= GravityFilter::MAX_REST_MEAN_GYRO_RAD_S
        && (accel.length() - STANDARD_GRAVITY_M_S2).abs() <= GravityFilter::MAX_REST_GRAVITY_ERROR_M_S2;
    at_rest.then(|| accel.normalize())
}

/// The mean reading and the largest standard deviation of any axis.
fn mean_and_spread(samples: &VecDeque<Sample>) -> (Vec3, f32) {
    let count = samples.len() as f32;
    let mean = samples.iter().map(|sample| sample.reading).sum::<Vec3>() / count;
    let variance = samples.iter().map(|sample| (sample.reading - mean) * (sample.reading - mean)).sum::<Vec3>() / count;
    (mean, variance.max_element().sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ACCEL_PERIOD: Duration = Duration::from_millis(5);
    const GYRO_PERIOD: Duration = Duration::from_micros(2500);
    const FRAME_PERIOD: Duration = Duration::from_micros(33_333);
    const TIME_CONSTANT: Duration = Duration::from_secs(3);
    /// Up in camera axes with the camera level: image y points down.
    const LEVEL_UP: Vec3 = Vec3::NEG_Y;

    /// Drives a filter the way the camera node does: the samples of each frame period stream by stream, then an
    /// advance to the frame's time. `specific_force` and `angular_velocity` give each stream's reading at a time.
    struct Imu {
        filter: GravityFilter,
        now: Duration,
        next_accel: Duration,
        next_gyro: Duration,
    }

    impl Imu {
        fn new() -> Imu {
            let start = Duration::from_secs(100);
            Imu { filter: GravityFilter::new(TIME_CONSTANT), now: start, next_accel: start, next_gyro: start }
        }

        fn run(&mut self, span: Duration, specific_force: impl Fn(Duration) -> Vec3, angular_velocity: impl Fn(Duration) -> Vec3) {
            let end = self.now + span;
            while self.now < end {
                self.now += FRAME_PERIOD;
                while self.next_accel <= self.now {
                    self.filter.add_accel(self.next_accel, specific_force(self.next_accel)).expect("accelerometer sample");
                    self.next_accel += ACCEL_PERIOD;
                }
                while self.next_gyro <= self.now {
                    self.filter.add_gyro(self.next_gyro, angular_velocity(self.next_gyro)).expect("gyro sample");
                    self.next_gyro += GYRO_PERIOD;
                }
                self.filter.advance_to(self.now);
            }
        }

        fn rest(&mut self, span: Duration, up: Vec3) {
            self.run(span, |_| up.normalize() * 9.79, |_| Vec3::ZERO);
        }

        fn up(&self) -> Vec3 {
            self.filter.specific_force().expect("an estimate").normalize()
        }
    }

    fn degrees_between(a: Vec3, b: Vec3) -> f32 {
        a.angle_between(b).to_degrees()
    }

    fn secs(value: f32) -> Duration {
        Duration::from_secs_f32(value)
    }

    #[test]
    fn at_rest_the_estimate_is_the_accelerometer_direction_scaled_to_standard_gravity() {
        let tilted = Vec3::new(0.15, -1.0, 0.3).normalize();
        let mut imu = Imu::new();
        imu.run(secs(2.0), |t| tilted * 9.79 + Vec3::new(0.05, -0.04, 0.03) * (t.as_secs_f32() * 200.0).sin(), |_| Vec3::ZERO);
        let specific_force = imu.filter.specific_force().expect("an estimate after a second at rest");
        assert!(degrees_between(specific_force, tilted) < 0.05, "{specific_force}");
        assert!((specific_force.length() - STANDARD_GRAVITY_M_S2).abs() < 1e-4, "{specific_force}");
    }

    #[test]
    fn there_is_no_estimate_before_the_imu_has_been_at_rest_for_the_rest_span() {
        let mut imu = Imu::new();
        imu.rest(secs(0.9), LEVEL_UP);
        assert_eq!(imu.filter.specific_force(), None);
        imu.rest(secs(0.2), LEVEL_UP);
        assert!(imu.filter.specific_force().is_some());
    }

    #[test]
    fn a_shaking_or_turning_imu_gives_no_estimate() {
        let mut shaken = Imu::new();
        shaken.run(secs(3.0), |t| LEVEL_UP * 9.79 + Vec3::X * 2.0 * (t.as_secs_f32() * 20.0).sin(), |_| Vec3::ZERO);
        assert_eq!(shaken.filter.specific_force(), None);
        let mut turning = Imu::new();
        turning.run(secs(3.0), |_| LEVEL_UP * 9.79, |_| Vec3::Y * 0.2);
        assert_eq!(turning.filter.specific_force(), None);
    }

    #[test]
    fn a_linear_acceleration_burst_shorter_than_the_time_constant_barely_moves_the_direction() {
        let mut imu = Imu::new();
        imu.rest(secs(1.5), LEVEL_UP);
        let push = LEVEL_UP * 9.79 + Vec3::Z * 3.0;
        imu.run(secs(0.5), |_| push, |_| Vec3::ZERO);
        let (moved, accelerometer_lean) = (degrees_between(imu.up(), LEVEL_UP), degrees_between(push, LEVEL_UP));
        assert!(moved < 0.2 * accelerometer_lean, "moved {moved} degrees of the accelerometer's {accelerometer_lean}");
        imu.rest(secs(0.5), LEVEL_UP);
        assert!(degrees_between(imu.up(), LEVEL_UP) < moved, "pulled back once the push ends");
    }

    #[test]
    fn a_pure_rotation_is_followed_by_the_gyro() {
        let mut imu = Imu::new();
        imu.rest(secs(1.5), LEVEL_UP);
        // A roll about the optical axis at one radian a second for 1.5 s; the accelerometer reads up as the body sees it.
        let (rate, started) = (Vec3::Z, imu.now);
        let up_at = move |t: Duration| Quat::from_scaled_axis(-rate * t.saturating_sub(started).as_secs_f32()) * LEVEL_UP;
        imu.run(secs(1.5), move |t| up_at(t) * 9.79, move |t| if t > started { rate } else { Vec3::ZERO });
        let truth = up_at(imu.now);
        assert!(degrees_between(LEVEL_UP, truth) > 80.0);
        assert!(degrees_between(imu.up(), truth) < 0.5, "{} against {truth}", imu.up());
    }

    #[test]
    fn the_gyro_turns_up_even_when_the_accelerometer_disagrees() {
        let mut imu = Imu::new();
        imu.rest(secs(1.5), LEVEL_UP);
        // The gyro alone says the body rolled 10 degrees; the accelerometer still reads level.
        let started = imu.now;
        let roll_for = secs(10f32.to_radians());
        imu.run(roll_for, |_| LEVEL_UP * 9.79, move |t| if t > started && t <= started + roll_for { Vec3::Z } else { Vec3::ZERO });
        let gyro_only = Quat::from_rotation_z(-10f32.to_radians()) * LEVEL_UP;
        assert!(degrees_between(imu.up(), gyro_only) < 0.6, "{}", imu.up());
    }

    #[test]
    fn a_wrong_direction_converges_on_the_accelerometer_with_the_time_constant() {
        let mut imu = Imu::new();
        imu.rest(secs(1.5), LEVEL_UP);
        // The accelerometer now reads 30 degrees away and the gyro says nothing turned: the estimate is 30 degrees off.
        let reading = Quat::from_rotation_x(30f32.to_radians()) * LEVEL_UP;
        let off_at = |imu: &Imu| degrees_between(imu.up(), reading);
        let start_deg = off_at(&imu);
        let started = imu.now;
        for _ in 0..2 {
            imu.run(TIME_CONSTANT, |_| reading * 9.79, |_| Vec3::ZERO);
            let elapsed = imu.now - started;
            // Pulling a unit vector towards another by a share of their difference shrinks tan(angle / 2) exponentially.
            let expected = 2.0 * ((start_deg / 2.0).to_radians().tan() * (-elapsed.as_secs_f32() / TIME_CONSTANT.as_secs_f32()).exp()).atan().to_degrees();
            assert!((off_at(&imu) - expected).abs() < 0.3, "after {elapsed:?}: {} degrees off, expected {expected}", off_at(&imu));
        }
    }

    #[test]
    fn samples_given_stream_by_stream_are_applied_in_time_order() {
        let rate = Vec3::new(0.3, 0.0, 0.8);
        let up_at = |t: Duration| Quat::from_scaled_axis(-rate * t.as_secs_f32()) * LEVEL_UP;
        // Both filters start at rest over the same samples, from 1 s to 2 s; then the body turns at `rate`.
        let moving_from = secs(2.0);
        let accel_times: Vec<Duration> = (1..200).map(|k| moving_from + ACCEL_PERIOD * k).collect();
        let gyro_times: Vec<Duration> = (1..400).map(|k| moving_from + GYRO_PERIOD * k).collect();
        let mut whole = GravityFilter::new(TIME_CONSTANT);
        let mut framed = GravityFilter::new(TIME_CONSTANT);
        for filter in [&mut whole, &mut framed] {
            for k in 0..=200 {
                filter.add_accel(secs(1.0) + ACCEL_PERIOD * k, LEVEL_UP * 9.79).unwrap();
            }
            for k in 0..=400 {
                filter.add_gyro(secs(1.0) + GYRO_PERIOD * k, Vec3::ZERO).unwrap();
            }
            filter.advance_to(moving_from);
            assert!(filter.specific_force().is_some());
        }
        let specific_force = |t: Duration| up_at(t - moving_from) * 9.79;
        // One filter is given each stream whole and advanced once, the other frame by frame.
        accel_times.iter().for_each(|&t| whole.add_accel(t, specific_force(t)).unwrap());
        gyro_times.iter().for_each(|&t| whole.add_gyro(t, rate).unwrap());
        whole.advance_to(secs(4.0));
        for frame in 0..31 {
            let (from, to) = (moving_from + FRAME_PERIOD * frame, moving_from + FRAME_PERIOD * (frame + 1));
            accel_times.iter().filter(|&&t| t > from && t <= to).for_each(|&t| framed.add_accel(t, specific_force(t)).unwrap());
            gyro_times.iter().filter(|&&t| t > from && t <= to).for_each(|&t| framed.add_gyro(t, rate).unwrap());
            framed.advance_to(to);
        }
        framed.advance_to(secs(4.0));
        assert_ne!(whole.specific_force(), Some(LEVEL_UP * STANDARD_GRAVITY_M_S2), "the body turned");
        assert_eq!(whole.specific_force(), framed.specific_force());
    }

    #[test]
    fn a_sample_not_newer_than_the_one_before_it_is_refused() {
        let mut filter = GravityFilter::new(TIME_CONSTANT);
        filter.add_gyro(secs(1.0), Vec3::ZERO).unwrap();
        assert_eq!(
            filter.add_gyro(secs(1.0), Vec3::ZERO),
            Err(GravityFilterError::OutOfOrder { stream: ImuStream::Gyro, at: secs(1.0), previous: secs(1.0) })
        );
        filter.add_accel(secs(1.0), LEVEL_UP).unwrap();
        assert!(matches!(filter.add_accel(secs(0.5), LEVEL_UP), Err(GravityFilterError::OutOfOrder { stream: ImuStream::Accelerometer, .. })));
    }

    #[test]
    fn a_gyro_silent_for_longer_than_the_gap_limit_is_refused() {
        let mut filter = GravityFilter::new(TIME_CONSTANT);
        filter.add_gyro(secs(1.0), Vec3::ZERO).unwrap();
        filter.add_gyro(secs(1.0) + GravityFilter::MAX_GYRO_GAP, Vec3::ZERO).unwrap();
        let late = secs(1.0) + GravityFilter::MAX_GYRO_GAP * 3;
        assert_eq!(
            filter.add_gyro(late, Vec3::ZERO),
            Err(GravityFilterError::GyroGap { previous: secs(1.0) + GravityFilter::MAX_GYRO_GAP, at: late })
        );
    }

    #[test]
    fn one_stream_waiting_too_long_for_the_other_is_refused() {
        let mut filter = GravityFilter::new(TIME_CONSTANT);
        let mut refusal = None;
        for k in 0..400 {
            if let Err(error) = filter.add_accel(secs(1.0) + ACCEL_PERIOD * k, LEVEL_UP * 9.79) {
                refusal = Some(error);
                break;
            }
            filter.advance_to(secs(1.0) + ACCEL_PERIOD * k);
        }
        assert!(matches!(refusal, Some(GravityFilterError::Backlog { stream: ImuStream::Accelerometer, span }) if span > GravityFilter::MAX_HELD_SPAN), "{refusal:?}");
    }
}
