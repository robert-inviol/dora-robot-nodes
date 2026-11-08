//! The camera's turn about the vertical between keyframes, from the IMU.
//!
//! The gyro reports angular velocity in the camera's own axes; its component along the up direction (the
//! keyframe's gravity reaction) is the rate of turning about the vertical, anticlockwise seen from above,
//! which is the sense of a pose's yaw. The gyro reads a small, slowly drifting rate even when still: that
//! bias is measured on every span where the camera is still, and interpolated between them.

use std::fmt;

use glam::Vec3;
use perception_wire::Imu;

/// Every accel and gyro sample of a walk, each stream in camera-clock order.
#[derive(Clone, Debug)]
pub struct ImuStream {
    gyro_ns: Vec<i64>,
    gyro_rad_s: Vec<Vec3>,
    accel_ns: Vec<i64>,
    accel_m_s2: Vec<Vec3>,
}

#[derive(Clone, Debug)]
pub struct StillSettings {
    /// The walk is judged still or moving in windows this long.
    pub window_ns: i64,
    /// A still window has no gyro sample turning faster than this; a hand-held camera never stays under it.
    pub max_rate_rad_s: f32,
    /// A still window's mean accel magnitude is within this of standard gravity, allowing for the
    /// accelerometer's own scale error.
    pub max_gravity_error_m_s2: f32,
    /// ...and its accel magnitude varies by no more than this (standard deviation).
    pub max_accel_spread_m_s2: f32,
    /// The bias needs at least this much still time in all to average down the gyro's noise.
    pub min_still_ns: i64,
}

impl Default for StillSettings {
    fn default() -> Self {
        StillSettings {
            window_ns: 500_000_000,
            max_rate_rad_s: 0.05,
            max_gravity_error_m_s2: 0.5,
            max_accel_spread_m_s2: 0.05,
            min_still_ns: 1_000_000_000,
        }
    }
}

const STANDARD_GRAVITY_M_S2: f32 = 9.80665;

/// Adjacent still windows, merged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StillSpan {
    pub start_ns: i64,
    pub end_ns: i64,
    /// The mean gyro reading over the span: the bias at that time.
    pub gyro_bias_rad_s: Vec3,
}

/// The gyro bias over the walk: each still span's reading at its midpoint, linearly interpolated between
/// spans and held constant before the first and after the last.
#[derive(Clone, Debug)]
pub struct GyroBias {
    times_ns: Vec<i64>,
    biases_rad_s: Vec<Vec3>,
}

impl GyroBias {
    pub fn constant(bias_rad_s: Vec3) -> GyroBias {
        GyroBias { times_ns: vec![0], biases_rad_s: vec![bias_rad_s] }
    }

    pub fn from_spans(spans: &[StillSpan], settings: &StillSettings) -> Result<GyroBias, HeadingError> {
        let still_ns: i64 = spans.iter().map(|s| s.end_ns - s.start_ns).sum();
        if still_ns < settings.min_still_ns {
            return Err(HeadingError::TooLittleStillTime { still_ns, needed_ns: settings.min_still_ns });
        }
        Ok(GyroBias {
            times_ns: spans.iter().map(|s| (s.start_ns + s.end_ns) / 2).collect(),
            biases_rad_s: spans.iter().map(|s| s.gyro_bias_rad_s).collect(),
        })
    }

    pub fn at(&self, time_ns: i64) -> Vec3 {
        let after = self.times_ns.partition_point(|&t| t < time_ns);
        if after == 0 {
            return self.biases_rad_s[0];
        }
        if after == self.times_ns.len() {
            return self.biases_rad_s[after - 1];
        }
        let (t0, t1) = (self.times_ns[after - 1], self.times_ns[after]);
        self.biases_rad_s[after - 1].lerp(self.biases_rad_s[after], (time_ns - t0) as f32 / (t1 - t0) as f32)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum HeadingError {
    NoGyroSamples,
    TooLittleStillTime { still_ns: i64, needed_ns: i64 },
    NotCovered { from_ns: i64, to_ns: i64, first_ns: i64, last_ns: i64 },
    KeyframesOutOfOrder { index: usize },
}

impl fmt::Display for HeadingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeadingError::NoGyroSamples => write!(f, "the imu recording has no gyro samples"),
            HeadingError::TooLittleStillTime { still_ns, needed_ns } => write!(
                f,
                "the camera is still for only {:.1} s, {:.1} s are needed to measure the gyro's bias; start a walk by holding the camera still",
                *still_ns as f64 * 1e-9,
                *needed_ns as f64 * 1e-9
            ),
            HeadingError::NotCovered { from_ns, to_ns, first_ns, last_ns } => write!(
                f,
                "gyro samples run from {first_ns} to {last_ns} ns, not over the keyframe span {from_ns} to {to_ns} ns"
            ),
            HeadingError::KeyframesOutOfOrder { index } => write!(f, "keyframe {index} is not later than the one before it"),
        }
    }
}

impl std::error::Error for HeadingError {}

fn sorted_unique(samples: impl Iterator<Item = (i64, Vec3)>) -> (Vec<i64>, Vec<Vec3>) {
    let mut samples: Vec<(i64, Vec3)> = samples.collect();
    samples.sort_by_key(|(t, _)| *t);
    samples.dedup_by_key(|(t, _)| *t);
    samples.into_iter().unzip()
}

impl ImuStream {
    /// Joins the samples of every imu message; a sample repeated across messages is kept once.
    pub fn from_imu(messages: &[Imu]) -> Result<ImuStream, HeadingError> {
        let (gyro_ns, gyro_rad_s) = sorted_unique(messages.iter().flat_map(|m| m.gyro_camera_ns.iter().copied().zip(m.gyro_rad_s.iter().copied())));
        let (accel_ns, accel_m_s2) = sorted_unique(messages.iter().flat_map(|m| m.accel_camera_ns.iter().copied().zip(m.accel_m_s2.iter().copied())));
        ImuStream::new(gyro_ns, gyro_rad_s, accel_ns, accel_m_s2)
    }

    pub fn new(gyro_ns: Vec<i64>, gyro_rad_s: Vec<Vec3>, accel_ns: Vec<i64>, accel_m_s2: Vec<Vec3>) -> Result<ImuStream, HeadingError> {
        assert_eq!(gyro_ns.len(), gyro_rad_s.len(), "a gyro rate for every time");
        assert_eq!(accel_ns.len(), accel_m_s2.len(), "an acceleration for every time");
        assert!(gyro_ns.windows(2).all(|w| w[0] < w[1]) && accel_ns.windows(2).all(|w| w[0] < w[1]), "imu times must increase");
        if gyro_ns.is_empty() {
            return Err(HeadingError::NoGyroSamples);
        }
        Ok(ImuStream { gyro_ns, gyro_rad_s, accel_ns, accel_m_s2 })
    }

    /// The spans where the camera is still: every window with a steady gyro and an accelerometer reading only
    /// gravity, steadily; adjacent still windows are merged.
    pub fn still_spans(&self, settings: &StillSettings) -> Vec<StillSpan> {
        let mut spans: Vec<(i64, i64, Vec3, usize)> = Vec::new();
        let mut start = 0;
        while start < self.gyro_ns.len() {
            let window_end = self.gyro_ns[start] + settings.window_ns;
            let end = start + self.gyro_ns[start..].partition_point(|&t| t < window_end);
            let (from_ns, to_ns) = (self.gyro_ns[start], self.gyro_ns[end - 1]);
            let gyro = &self.gyro_rad_s[start..end];
            let accel_from = self.accel_ns.partition_point(|&t| t < from_ns);
            let accel_to = self.accel_ns.partition_point(|&t| t <= to_ns);
            let magnitudes: Vec<f32> = self.accel_m_s2[accel_from..accel_to].iter().map(|a| a.length()).collect();
            let still = to_ns - from_ns >= settings.window_ns * 9 / 10
                && gyro.iter().all(|r| r.length() < settings.max_rate_rad_s)
                && magnitudes.len() >= 2
                && {
                    let mean = magnitudes.iter().sum::<f32>() / magnitudes.len() as f32;
                    let spread = (magnitudes.iter().map(|m| (m - mean).powi(2)).sum::<f32>() / magnitudes.len() as f32).sqrt();
                    (mean - STANDARD_GRAVITY_M_S2).abs() <= settings.max_gravity_error_m_s2 && spread <= settings.max_accel_spread_m_s2
                };
            if still {
                let sum: Vec3 = gyro.iter().copied().sum();
                match spans.last_mut() {
                    Some(last) if last.1 >= self.gyro_ns[start.saturating_sub(1)] => {
                        last.1 = to_ns;
                        last.2 += sum;
                        last.3 += gyro.len();
                    }
                    _ => spans.push((from_ns, to_ns, sum, gyro.len())),
                }
            }
            start = end;
        }
        spans.into_iter().map(|(start_ns, end_ns, sum, count)| StillSpan { start_ns, end_ns, gyro_bias_rad_s: sum / count as f32 }).collect()
    }

    fn rate_at(&self, k: usize, up: Vec3, bias: &GyroBias) -> f32 {
        (self.gyro_rad_s[k] - bias.at(self.gyro_ns[k])).dot(up)
    }

    /// The turn about `up` from `from_ns` to `to_ns`: the bias-corrected rate integrated by the trapezoid
    /// rule, with the rate interpolated at both ends.
    pub fn turn_about(&self, up: Vec3, bias: &GyroBias, from_ns: i64, to_ns: i64) -> Result<f32, HeadingError> {
        let times = &self.gyro_ns;
        let (first, last) = (times[0], times[times.len() - 1]);
        if from_ns < first || to_ns > last {
            return Err(HeadingError::NotCovered { from_ns, to_ns, first_ns: first, last_ns: last });
        }
        let rate_at_time = |t: i64| -> f32 {
            let after = times.partition_point(|&s| s < t).min(times.len() - 1);
            if times[after] == t || after == 0 {
                return self.rate_at(after, up, bias);
            }
            let (t0, t1) = (times[after - 1], times[after]);
            let share = (t - t0) as f32 / (t1 - t0) as f32;
            self.rate_at(after - 1, up, bias) * (1.0 - share) + self.rate_at(after, up, bias) * share
        };
        let inside_start = times.partition_point(|&s| s <= from_ns);
        let inside_end = times.partition_point(|&s| s < to_ns);
        let mut points: Vec<(i64, f32)> = Vec::with_capacity(inside_end.saturating_sub(inside_start) + 2);
        points.push((from_ns, rate_at_time(from_ns)));
        points.extend((inside_start..inside_end).map(|k| (times[k], self.rate_at(k, up, bias))));
        points.push((to_ns, rate_at_time(to_ns)));
        Ok(points.windows(2).map(|w| 0.5 * (w[0].1 + w[1].1) * (w[1].0 - w[0].0) as f32 * 1e-9).sum())
    }
}

/// The turn about the vertical from each keyframe to the next, from the keyframes' camera times and their
/// up directions in camera axes (their gravity reactions, any length).
pub fn keyframe_turns(stream: &ImuStream, bias: &GyroBias, times_ns: &[i64], gravity_reactions: &[Vec3]) -> Result<Vec<f32>, HeadingError> {
    assert_eq!(times_ns.len(), gravity_reactions.len(), "an up direction for every keyframe");
    (1..times_ns.len())
        .map(|k| {
            if times_ns[k] <= times_ns[k - 1] {
                return Err(HeadingError::KeyframesOutOfOrder { index: k });
            }
            let up = (gravity_reactions[k - 1].normalize() + gravity_reactions[k].normalize()).normalize();
            stream.turn_about(up, bias, times_ns[k - 1], times_ns[k])
        })
        .collect()
}

/// For each keyframe from the second on, whether it and the one before lie inside one still span: the camera
/// did not move between them.
pub fn still_between(spans: &[StillSpan], times_ns: &[i64]) -> Vec<bool> {
    (1..times_ns.len())
        .map(|k| spans.iter().any(|s| s.start_ns <= times_ns[k - 1] && times_ns[k] <= s.end_ns))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const UP: Vec3 = Vec3::new(0.0, -1.0, 0.0);
    const GYRO_STEP_NS: i64 = 2_500_000;
    const ACCEL_STEP_NS: i64 = 5_000_000;

    /// Gyro at 400 Hz and accel at 200 Hz; `moving` decides whether the accelerometer also feels the hand.
    fn stream(seconds: f32, rate: impl Fn(f32) -> Vec3, moving: impl Fn(f32) -> bool) -> ImuStream {
        let end = (seconds * 1e9) as i64;
        let gyro_ns: Vec<i64> = (0..=end / GYRO_STEP_NS).map(|k| k * GYRO_STEP_NS).collect();
        let accel_ns: Vec<i64> = (0..=end / ACCEL_STEP_NS).map(|k| k * ACCEL_STEP_NS).collect();
        let gyro = gyro_ns.iter().map(|&t| rate(t as f32 * 1e-9)).collect();
        let accel = accel_ns
            .iter()
            .map(|&t| {
                let s = t as f32 * 1e-9;
                -UP * 9.7 + if moving(s) { Vec3::new((s * 23.0).sin(), (s * 17.0).cos(), 0.0) * 0.8 } else { Vec3::ZERO }
            })
            .collect();
        ImuStream::new(gyro_ns, gyro, accel_ns, accel).unwrap()
    }

    #[test]
    fn a_steady_turn_integrates_to_rate_times_time() {
        let imu = stream(4.0, |_| UP * 0.5, |_| true);
        let turn = imu.turn_about(UP, &GyroBias::constant(Vec3::ZERO), 1_000_000_000, 3_001_250_000).unwrap();
        assert!((turn - 0.5 * 2.00125).abs() < 1e-4, "{turn}");
    }

    #[test]
    fn still_spans_are_found_anywhere_and_give_the_bias_there() {
        let bias = |s: f32| Vec3::new(0.002 + 0.001 * s, -0.002, 0.001);
        let turning = |s: f32| (2.0..4.0).contains(&s) || (6.0..8.0).contains(&s);
        let imu = stream(10.0, |s| bias(s) + if turning(s) { UP * 0.8 } else { Vec3::ZERO }, turning);
        let spans = imu.still_spans(&StillSettings::default());
        let bounds: Vec<(f32, f32)> = spans.iter().map(|s| (s.start_ns as f32 * 1e-9, s.end_ns as f32 * 1e-9)).collect();
        assert_eq!(spans.len(), 3, "{bounds:?}");
        for span in &spans {
            let middle = (span.start_ns + span.end_ns) as f32 * 0.5e-9;
            assert!((span.gyro_bias_rad_s - bias(middle)).length() < 1e-4, "{span:?}");
        }
        let model = GyroBias::from_spans(&spans, &StillSettings::default()).unwrap();
        assert!((model.at(5_000_000_000) - bias(5.0)).length() < 2e-4, "{}", model.at(5_000_000_000));
    }

    #[test]
    fn a_shaking_accelerometer_is_not_still_even_with_a_quiet_gyro() {
        let imu = stream(3.0, |_| Vec3::ZERO, |_| true);
        assert!(imu.still_spans(&StillSettings::default()).is_empty());
    }

    #[test]
    fn a_walk_that_never_holds_still_cannot_measure_its_bias() {
        let imu = stream(3.0, |s| UP * (0.3 + 0.2 * s.sin()), |_| true);
        let spans = imu.still_spans(&StillSettings::default());
        assert!(matches!(GyroBias::from_spans(&spans, &StillSettings::default()), Err(HeadingError::TooLittleStillTime { .. })));
    }

    #[test]
    fn keyframe_turns_follow_the_gyro_and_refuse_spans_it_does_not_cover() {
        let imu = stream(2.0, |s| UP * if s < 1.0 { 0.2 } else { -0.4 }, |_| true);
        let zero = GyroBias::constant(Vec3::ZERO);
        let times = [250_000_000, 750_000_000, 1_500_000_000];
        let turns = keyframe_turns(&imu, &zero, &times, &[UP * 9.8; 3]).unwrap();
        assert!((turns[0] - 0.1).abs() < 1e-3 && (turns[1] + 0.15).abs() < 2e-3, "{turns:?}");
        let error = keyframe_turns(&imu, &zero, &[250_000_000, 2_500_000_000], &[UP; 2]).unwrap_err();
        assert!(matches!(error, HeadingError::NotCovered { .. }), "{error}");
    }

    #[test]
    fn keyframes_inside_one_still_span_are_still_between() {
        let spans = [StillSpan { start_ns: 0, end_ns: 1_000, gyro_bias_rad_s: Vec3::ZERO }];
        assert_eq!(still_between(&spans, &[100, 900, 1_500]), vec![true, false]);
    }
}
