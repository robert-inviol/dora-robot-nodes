//! Which camera frames are worth sending on as keyframes, and which way is up.

pub mod gravity;

use std::time::Duration;

use glam::Vec3;

pub use gravity::{GravityFilter, GravityFilterError, ImuStream, STANDARD_GRAVITY_M_S2};

#[derive(Clone, Debug)]
pub struct KeyframeSettings {
    /// A keyframe goes out at least this often, even when the camera is still.
    pub max_interval: Duration,
    /// Never more often than this, however fast the camera turns.
    pub min_interval: Duration,
    /// Turning this far since the last keyframe calls for a new one.
    pub turn_threshold_rad: f32,
    /// Moving this far from where the last keyframe was taken calls for a new one.
    pub travel_threshold_m: f32,
}

impl Default for KeyframeSettings {
    fn default() -> Self {
        KeyframeSettings {
            max_interval: Duration::from_secs(1),
            min_interval: Duration::from_millis(200),
            turn_threshold_rad: 10f32.to_radians(),
            travel_threshold_m: 0.10,
        }
    }
}

/// Times are durations since any fixed start, so the selector can be driven by camera timestamps.
#[derive(Clone, Debug)]
pub struct KeyframeSelector {
    settings: KeyframeSettings,
    last_keyframe: Option<Duration>,
    turned_rad: f32,
    travelled_m: f32,
}

impl KeyframeSelector {
    pub fn new(settings: KeyframeSettings) -> KeyframeSelector {
        KeyframeSelector { settings, last_keyframe: None, turned_rad: 0.0, travelled_m: 0.0 }
    }

    /// Sets how far the camera now is from where the last keyframe was taken.
    pub fn set_travel(&mut self, travelled_m: f32) {
        self.travelled_m = travelled_m;
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
                    || (since >= self.settings.min_interval
                        && (self.turned_rad >= self.settings.turn_threshold_rad || self.travelled_m >= self.settings.travel_threshold_m))
            }
        };
        if due {
            self.last_keyframe = Some(now);
            self.turned_rad = 0.0;
            self.travelled_m = 0.0;
        }
        due
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
    fn travelling_past_the_threshold_brings_the_next_keyframe_forward() {
        let mut selector = KeyframeSelector::new(KeyframeSettings::default());
        assert!(selector.take(ms(0)));
        selector.set_travel(0.12);
        assert!(!selector.take(ms(150)), "the minimum interval has not passed");
        assert!(selector.take(ms(250)));
        assert!(!selector.take(ms(500)), "a keyframe starts the travel again from where it was taken");
    }

    #[test]
    fn a_short_move_waits_for_the_max_interval() {
        let mut selector = KeyframeSelector::new(KeyframeSettings::default());
        assert!(selector.take(ms(0)));
        selector.set_travel(0.08);
        assert!(!selector.take(ms(600)));
        assert!(selector.take(ms(1000)));
    }

    #[test]
    fn a_small_turn_waits_for_the_max_interval() {
        let mut selector = KeyframeSelector::new(KeyframeSettings::default());
        assert!(selector.take(ms(0)));
        selector.add_rotation(Vec3::new(0.0, 0.0, 0.1), ms(500));
        assert!(!selector.take(ms(600)));
        assert!(selector.take(ms(1000)));
    }
}
