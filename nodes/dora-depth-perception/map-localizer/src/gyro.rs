//! The turn about the vertical between keyframes, from a rolling buffer of every IMU sample.
//!
//! The gyro's bias is measured on the spans of the buffer where the camera is still (room-map's heading model).
//! Until the buffer has held `still.min_still_ns` of still time the bias is unknown and no turn is given; once
//! measured, the latest measurement is kept while the buffer holds too little still time to measure it again.

use std::collections::{HashMap, VecDeque};

use glam::Vec3;
use perception_wire::localization::{CameraNanos, FrameId};
use perception_wire::Imu;
use room_map::heading::{GyroBias, HeadingError, ImuStream, StillSettings};

use crate::localizer::{HeadingHint, HeadingWithoutGyro};

#[derive(Clone, Debug)]
pub struct GyroSettings {
    pub still: StillSettings,
    /// Samples older than this before the newest are dropped; it bounds the gap between two keyframes whose turn
    /// can be measured.
    pub buffer_ns: i64,
    /// Camera times of this many recent keyframes are kept for matching keyframes to their imu messages.
    pub remembered_keyframes: usize,
}

impl Default for GyroSettings {
    fn default() -> Self {
        GyroSettings { still: StillSettings::default(), buffer_ns: 60_000_000_000, remembered_keyframes: 64 }
    }
}

/// A keyframe's camera time and its up direction in camera axes (its gravity reaction, any length).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyframeTiming {
    pub camera_time: CameraNanos,
    pub gravity_reaction: Vec3,
}

/// A keyframe's heading hint, and why the gyro could not give its turn when it could not.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyframeHeading {
    pub hint: HeadingHint,
    pub gyro_problem: Option<HeadingError>,
}

pub struct GyroHeading {
    settings: GyroSettings,
    gyro: VecDeque<(i64, Vec3)>,
    accel: VecDeque<(i64, Vec3)>,
    keyframe_times: HashMap<FrameId, CameraNanos>,
    keyframe_order: VecDeque<FrameId>,
    /// `None` until a still span has been measured.
    bias: Option<GyroBias>,
}

/// Appends the samples later than the last one held, in time order; earlier or repeated samples are already held.
fn append(buffer: &mut VecDeque<(i64, Vec3)>, times: &[i64], values: &[Vec3]) {
    let last = buffer.back().map_or(i64::MIN, |s| s.0);
    let mut fresh: Vec<(i64, Vec3)> = times.iter().copied().zip(values.iter().copied()).filter(|(t, _)| *t > last).collect();
    fresh.sort_by_key(|s| s.0);
    fresh.dedup_by_key(|s| s.0);
    buffer.extend(fresh);
}

fn drop_before(buffer: &mut VecDeque<(i64, Vec3)>, oldest_ns: i64) {
    while buffer.front().is_some_and(|s| s.0 < oldest_ns) {
        buffer.pop_front();
    }
}

impl GyroHeading {
    pub fn new(settings: GyroSettings) -> GyroHeading {
        GyroHeading {
            settings,
            gyro: VecDeque::new(),
            accel: VecDeque::new(),
            keyframe_times: HashMap::new(),
            keyframe_order: VecDeque::new(),
            bias: None,
        }
    }

    pub fn add(&mut self, imu: &Imu) {
        append(&mut self.gyro, &imu.gyro_camera_ns, &imu.gyro_rad_s);
        append(&mut self.accel, &imu.accel_camera_ns, &imu.accel_m_s2);
        let newest = self.gyro.back().map_or(imu.keyframe_camera_ns, |s| s.0.max(imu.keyframe_camera_ns));
        drop_before(&mut self.gyro, newest - self.settings.buffer_ns);
        drop_before(&mut self.accel, newest - self.settings.buffer_ns);
        let frame = FrameId(imu.frame_id);
        if self.keyframe_times.insert(frame, CameraNanos(imu.keyframe_camera_ns)).is_none() {
            self.keyframe_order.push_back(frame);
        }
        while self.keyframe_order.len() > self.settings.remembered_keyframes {
            let oldest = self.keyframe_order.pop_front().expect("longer than the limit");
            self.keyframe_times.remove(&oldest);
        }
    }

    /// The camera time of a keyframe whose imu message has arrived.
    pub fn camera_time(&self, frame: FrameId) -> Option<CameraNanos> {
        self.keyframe_times.get(&frame).copied()
    }

    fn stream(&self) -> Result<ImuStream, HeadingError> {
        let (gyro_ns, gyro_rad_s) = self.gyro.iter().copied().unzip();
        let (accel_ns, accel_m_s2) = self.accel.iter().copied().unzip();
        ImuStream::new(gyro_ns, gyro_rad_s, accel_ns, accel_m_s2)
    }

    /// Re-measures the bias on the buffer's still spans. Fails with `HeadingError::TooLittleStillTime` until a
    /// measurement has succeeded once.
    fn measure_bias(&mut self, stream: &ImuStream) -> Result<&GyroBias, HeadingError> {
        let spans = stream.still_spans(&self.settings.still);
        match (GyroBias::from_spans(&spans, &self.settings.still), self.bias.is_some()) {
            (Ok(bias), _) => self.bias = Some(bias),
            (Err(_), true) => {}
            (Err(not_yet), false) => return Err(not_yet),
        }
        Ok(self.bias.as_ref().expect("measured above or before"))
    }

    /// Whether the gyro can give turns: its bias has been measured.
    pub fn ready(&mut self) -> Result<(), HeadingError> {
        let stream = self.stream()?;
        self.measure_bias(&stream).map(|_| ())
    }

    /// The turn about the vertical from one keyframe to the next, anticlockwise seen from above. The bias is
    /// re-measured first if the buffer holds enough still time.
    pub fn turn_between(&mut self, from: KeyframeTiming, to: KeyframeTiming) -> Result<f32, HeadingError> {
        if to.camera_time <= from.camera_time {
            return Err(HeadingError::KeyframesOutOfOrder { index: 1 });
        }
        let stream = self.stream()?;
        let bias = self.measure_bias(&stream)?;
        let up = (from.gravity_reaction.normalize() + to.gravity_reaction.normalize()).normalize();
        stream.turn_about(up, bias, from.camera_time.0, to.camera_time.0)
    }

    /// The heading hint for a keyframe: the gyro's turn from the previous keyframe given to the localizer, or for
    /// the first, `First` once the gyro is ready. When the gyro cannot give it, `without_gyro` decides.
    pub fn keyframe_heading(&mut self, previous: Option<KeyframeTiming>, keyframe: KeyframeTiming, without_gyro: HeadingWithoutGyro) -> KeyframeHeading {
        let from_gyro = match previous {
            Some(before) => self.turn_between(before, keyframe).map(|turn_rad| HeadingHint::GyroTurn { turn_rad }),
            None => self.ready().map(|()| HeadingHint::First),
        };
        match from_gyro {
            Ok(hint) => KeyframeHeading { hint, gyro_problem: None },
            Err(problem) => KeyframeHeading { hint: without_gyro.hint(), gyro_problem: Some(problem) },
        }
    }

    /// The bias the next turn would be corrected with, at a camera time; `None` until it has been measured.
    pub fn bias_at(&self, time: CameraNanos) -> Option<Vec3> {
        self.bias.as_ref().map(|bias| bias.at(time.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UP: Vec3 = Vec3::new(0.0, -1.0, 0.0);
    const STEP_NS: i64 = 2_500_000;

    /// Imu messages, one per `keyframe_ns` gap, for a camera still for two seconds then turning at `rate`.
    fn messages(rate: f32, bias: Vec3, seconds: i64, keyframe_ns: i64) -> Vec<Imu> {
        let end = seconds * 1_000_000_000;
        (1..=end / keyframe_ns)
            .map(|k| {
                let (from, to) = ((k - 1) * keyframe_ns, k * keyframe_ns);
                let times: Vec<i64> = (from / STEP_NS..to / STEP_NS).map(|s| s * STEP_NS).collect();
                let turning = |t: i64| t >= 2_000_000_000;
                Imu {
                    frame_id: k as u64,
                    keyframe_camera_ns: to,
                    accel_camera_ns: times.clone(),
                    accel_m_s2: times.iter().map(|&t| -UP * 9.8 + if turning(t) { Vec3::X * (t as f32 * 1e-8).sin() } else { Vec3::ZERO }).collect(),
                    gyro_camera_ns: times.clone(),
                    gyro_rad_s: times.iter().map(|&t| bias + if turning(t) { UP * rate } else { Vec3::ZERO }).collect(),
                }
            })
            .collect()
    }

    #[test]
    fn the_turn_between_keyframes_follows_the_gyro_less_its_bias() {
        let bias = Vec3::new(0.01, -0.02, 0.005);
        let mut heading = GyroHeading::new(GyroSettings::default());
        for imu in messages(0.5, bias, 4, 500_000_000) {
            heading.add(&imu);
        }
        let timing = |frame: u64| KeyframeTiming { camera_time: heading.camera_time(FrameId(frame)).unwrap(), gravity_reaction: UP };
        let (from, to) = (timing(5), timing(7));
        let turn = heading.turn_between(from, to).unwrap();
        assert!((turn - 0.5).abs() < 0.01, "{turn}");
        assert!((heading.bias_at(CameraNanos(1_000_000_000)).unwrap() - bias).length() < 1e-4);
    }

    /// Imu messages for a camera turning at `rate` from the start: never still.
    fn turning_from_the_start(rate: f32, seconds: i64, keyframe_ns: i64) -> Vec<Imu> {
        let mut all = messages(rate, Vec3::ZERO, seconds + 2, keyframe_ns);
        let still_messages = (2_000_000_000 / keyframe_ns) as usize;
        all.drain(..still_messages);
        all
    }

    #[test]
    fn no_turn_is_given_until_a_still_span_has_measured_the_bias() {
        let mut heading = GyroHeading::new(GyroSettings::default());
        for imu in turning_from_the_start(0.5, 4, 500_000_000) {
            heading.add(&imu);
        }
        let timing = |frame: u64| KeyframeTiming { camera_time: heading.camera_time(FrameId(frame)).unwrap(), gravity_reaction: UP };
        let (from, to) = (timing(5), timing(7));
        let error = heading.turn_between(from, to).unwrap_err();
        assert!(matches!(error, HeadingError::TooLittleStillTime { .. }), "{error}");
        assert_eq!(heading.bias_at(CameraNanos(0)), None);
        for without_gyro in [HeadingWithoutGyro::Refuse, HeadingWithoutGyro::Search] {
            let first = heading.keyframe_heading(None, from, without_gyro);
            assert_eq!(first.hint, without_gyro.hint());
            assert!(matches!(first.gyro_problem, Some(HeadingError::TooLittleStillTime { .. })), "{first:?}");
        }
    }

    #[test]
    fn the_minimum_still_time_decides_when_the_bias_is_measured() {
        // Half a second still, then turning.
        let mut all = messages(0.5, Vec3::ZERO, 4, 500_000_000);
        all.drain(..3);
        let fed = |min_still_ns: i64| {
            let still = StillSettings { min_still_ns, ..StillSettings::default() };
            let mut heading = GyroHeading::new(GyroSettings { still, ..GyroSettings::default() });
            for imu in &all {
                heading.add(imu);
            }
            heading
        };
        assert!(matches!(fed(1_000_000_000).ready(), Err(HeadingError::TooLittleStillTime { .. })));
        assert_eq!(fed(400_000_000).ready(), Ok(()));
    }

    #[test]
    fn once_measured_the_bias_is_kept_after_its_still_span_leaves_the_buffer() {
        let settings = GyroSettings { buffer_ns: 1_500_000_000, ..GyroSettings::default() };
        let mut heading = GyroHeading::new(settings);
        let all = messages(0.5, Vec3::new(0.0, 0.02, 0.0), 6, 500_000_000);
        for imu in &all[..4] {
            heading.add(imu);
        }
        assert_eq!(heading.ready(), Ok(()));
        for imu in &all[4..] {
            heading.add(imu);
        }
        let timing = |frame: u64| KeyframeTiming { camera_time: heading.camera_time(FrameId(frame)).unwrap(), gravity_reaction: UP };
        let (from, to) = (timing(10), timing(11));
        let turn = heading.turn_between(from, to).unwrap();
        assert!((turn - 0.25).abs() < 0.002, "{turn}");
    }

    #[test]
    fn samples_repeated_across_messages_are_kept_once() {
        let mut heading = GyroHeading::new(GyroSettings::default());
        let all = messages(0.0, Vec3::ZERO, 2, 500_000_000);
        heading.add(&all[0]);
        heading.add(&all[0]);
        heading.add(&all[1]);
        assert_eq!(heading.gyro.len(), all[0].gyro_camera_ns.len() + all[1].gyro_camera_ns.len());
    }

    #[test]
    fn a_turn_the_buffer_no_longer_covers_is_refused() {
        let still = StillSettings { min_still_ns: 500_000_000, ..StillSettings::default() };
        let settings = GyroSettings { buffer_ns: 1_000_000_000, still, ..GyroSettings::default() };
        let mut heading = GyroHeading::new(settings);
        let all = messages(0.1, Vec3::ZERO, 4, 500_000_000);
        for imu in &all[..4] {
            heading.add(imu);
        }
        assert_eq!(heading.ready(), Ok(()), "the bias is measured while the still start is in the buffer");
        for imu in &all[4..] {
            heading.add(imu);
        }
        let at = |ns: i64| KeyframeTiming { camera_time: CameraNanos(ns), gravity_reaction: UP };
        let error = heading.turn_between(at(500_000_000), at(3_500_000_000)).unwrap_err();
        assert!(matches!(error, HeadingError::NotCovered { .. }), "{error}");
    }
}
