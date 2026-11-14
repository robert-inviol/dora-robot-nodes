//! The One Euro filter (Casiez, Roussel and Vogel, CHI 2012): smooths a jittery signal hard while it is still and
//! lightly while it moves, so it adds little lag. The rate of change of the raw samples, low-pass filtered, is the
//! velocity.

use glam::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct OneEuroSettings {
    /// The cutoff while still, in hertz: lower smooths more.
    pub min_cutoff_hz: f32,
    /// How fast the cutoff rises with speed: higher lags less.
    pub beta: f32,
    /// The cutoff for the derivative, in hertz.
    pub derivative_cutoff_hz: f32,
}

impl Default for OneEuroSettings {
    fn default() -> OneEuroSettings {
        OneEuroSettings { min_cutoff_hz: 1.0, beta: 0.7, derivative_cutoff_hz: 1.0 }
    }
}

#[derive(Clone, Debug)]
pub struct OneEuro {
    settings: OneEuroSettings,
    /// The filtered value, its filtered rate of change, and the last raw sample.
    state: Option<(Vec3, Vec3, Vec3)>,
}

fn smoothing(cutoff_hz: f32, dt_s: f32) -> f32 {
    let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
    1.0 / (1.0 + tau / dt_s)
}

impl OneEuro {
    pub fn new(settings: OneEuroSettings) -> OneEuro {
        OneEuro { settings, state: None }
    }

    /// Adds a sample `dt_s` after the previous one; returns the filtered value and its rate of change.
    pub fn add(&mut self, sample: Vec3, dt_s: f32) -> (Vec3, Vec3) {
        let (value, rate) = match self.state {
            Some((previous, previous_rate, previous_sample)) if dt_s > 0.0 => {
                let raw_rate = (sample - previous_sample) / dt_s;
                let rate = previous_rate.lerp(raw_rate, smoothing(self.settings.derivative_cutoff_hz, dt_s));
                let cutoff = self.settings.min_cutoff_hz + self.settings.beta * rate.length();
                (previous.lerp(sample, smoothing(cutoff, dt_s)), rate)
            }
            Some((value, rate, _)) => (value, rate),
            None => (sample, Vec3::ZERO),
        };
        self.state = Some((value, rate, sample));
        (value, rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_sample_passes_through_still() {
        let mut filter = OneEuro::new(OneEuroSettings::default());
        assert_eq!(filter.add(Vec3::new(1.0, 2.0, 3.0), 0.033), (Vec3::new(1.0, 2.0, 3.0), Vec3::ZERO));
    }

    #[test]
    fn jitter_around_a_still_point_is_smoothed_towards_it() {
        let mut filter = OneEuro::new(OneEuroSettings::default());
        filter.add(Vec3::ZERO, 0.033);
        let mut worst = 0.0f32;
        for i in 0..60 {
            let jitter = if i % 2 == 0 { 0.05 } else { -0.05 };
            worst = worst.max(filter.add(Vec3::splat(jitter), 0.033).0.abs().max_element());
        }
        assert!(worst < 0.02, "{worst}");
    }

    #[test]
    fn a_steady_walk_is_followed_with_its_speed() {
        let mut filter = OneEuro::new(OneEuroSettings::default());
        let mut last = (Vec3::ZERO, Vec3::ZERO);
        for i in 0..90 {
            last = filter.add(Vec3::new(1.2 * i as f32 * 0.033, 0.0, 0.0), 0.033);
        }
        let (position, velocity) = last;
        assert!((velocity.x - 1.2).abs() < 0.05, "{velocity}");
        assert!((position.x - 1.2 * 89.0 * 0.033).abs() < 0.15, "{position}");
    }
}
