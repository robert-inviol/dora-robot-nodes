//! Which camera frames are worth sending on as keyframes, and which way is up.

use std::time::Duration;

use glam::Vec3;

#[derive(Clone, Debug)]
pub struct KeyframeSettings {
    /// A keyframe goes out at least this often, even when the camera is still.
    pub max_interval: Duration,
    /// Never more often than this, however fast the camera turns.
    pub min_interval: Duration,
    /// Turning this far since the last keyframe calls for a new one.
    pub turn_threshold_rad: f32,
}

impl Default for KeyframeSettings {
    fn default() -> Self {
        KeyframeSettings {
            max_interval: Duration::from_secs(1),
            min_interval: Duration::from_millis(200),
            turn_threshold_rad: 10f32.to_radians(),
        }
    }
}

/// Times are durations since any fixed start, so the selector can be driven by camera timestamps.
#[derive(Clone, Debug)]
pub struct KeyframeSelector {
    settings: KeyframeSettings,
    last_keyframe: Option<Duration>,
    turned_rad: f32,
}

impl KeyframeSelector {
    pub fn new(settings: KeyframeSettings) -> KeyframeSelector {
        KeyframeSelector { settings, last_keyframe: None, turned_rad: 0.0 }
    }

    /// Adds one gyro reading held for `held`.
    pub fn add_rotation(&mut self, angular_velocity_rad_s: Vec3, held: Duration) {
        self.turned_rad += angular_velocity_rad_s.length() * held.as_secs_f32();
    }

    /// Whether the frame at `now` should be a keyframe; if so, it is counted as the latest one.
    pub fn take(&mut self, now: Duration) -> bool {
        let due = match self.last_keyframe {
            None => true,
            Some(last) => {
                let since = now.saturating_sub(last);
                since >= self.settings.max_interval
                    || (since >= self.settings.min_interval && self.turned_rad >= self.settings.turn_threshold_rad)
            }
        };
        if due {
            self.last_keyframe = Some(now);
            self.turned_rad = 0.0;
        }
        due
    }
}

/// The accelerometer smoothed over a time constant, so the specific force at rest, the reaction to gravity, stands out from shakes.
#[derive(Clone, Debug)]
pub struct GravityEstimate {
    time_constant: Duration,
    specific_force: Option<Vec3>,
}

impl GravityEstimate {
    pub fn new(time_constant: Duration) -> GravityEstimate {
        GravityEstimate { time_constant, specific_force: None }
    }

    pub fn add(&mut self, acceleration_m_s2: Vec3, since_last: Duration) {
        let weight = (since_last.as_secs_f32() / self.time_constant.as_secs_f32()).min(1.0);
        self.specific_force = Some(match self.specific_force {
            None => acceleration_m_s2,
            Some(previous) => previous.lerp(acceleration_m_s2, weight),
        });
    }

    /// The specific force (at rest, the reaction to gravity) in the accelerometer's frame, once any reading has arrived.
    pub fn specific_force(&self) -> Option<Vec3> {
        self.specific_force
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(value: u64) -> Duration {
        Duration::from_millis(value)
    }

    #[test]
    fn the_first_frame_is_a_keyframe() {
        assert!(KeyframeSelector::new(KeyframeSettings::default()).take(ms(5)));
    }

    #[test]
    fn a_still_camera_sends_one_keyframe_per_max_interval() {
        let mut selector = KeyframeSelector::new(KeyframeSettings::default());
        let taken: Vec<u64> = (0..=3000).step_by(33).filter(|&t| selector.take(ms(t))).collect();
        assert_eq!(taken, vec![0, 1023, 2046]);
    }

    #[test]
    fn turning_past_the_threshold_brings_the_next_keyframe_forward() {
        let mut selector = KeyframeSelector::new(KeyframeSettings::default());
        assert!(selector.take(ms(0)));
        selector.add_rotation(Vec3::new(0.0, 1.0, 0.0), ms(200));
        assert!(!selector.take(ms(150)), "the minimum interval has not passed");
        assert!(selector.take(ms(250)), "turned about 11 degrees after the minimum interval");
    }

    #[test]
    fn a_small_turn_waits_for_the_max_interval() {
        let mut selector = KeyframeSelector::new(KeyframeSettings::default());
        assert!(selector.take(ms(0)));
        selector.add_rotation(Vec3::new(0.0, 0.0, 0.1), ms(500));
        assert!(!selector.take(ms(600)));
        assert!(selector.take(ms(1000)));
    }

    #[test]
    fn gravity_settles_on_the_steady_reading_through_a_shake() {
        let mut gravity = GravityEstimate::new(ms(500));
        gravity.add(Vec3::new(0.0, -9.8, 0.0), ms(5));
        for k in 0..400 {
            let shake = if k % 2 == 0 { 2.0 } else { -2.0 };
            gravity.add(Vec3::new(shake, -9.8, 0.0), ms(5));
        }
        let specific_force = gravity.specific_force().unwrap();
        assert!(specific_force.x.abs() < 0.1 && (specific_force.y + 9.8).abs() < 0.01, "{specific_force}");
    }
}
