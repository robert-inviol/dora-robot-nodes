//! Each fitted joint smoothed over time by its own One Euro filter, in the frame tracks are kept in, so a still
//! person's joints hold still while a moving person's joints follow with little lag. In the odom frame a person
//! standing still is still whatever the robot does, and the depth noise that makes joints jitter, mostly along the
//! camera's ray, averages out.

use std::time::Duration;

use glam::Vec3;

use crate::one_euro::{OneEuro, OneEuroSettings};

#[derive(Clone, Copy, Debug)]
pub struct JointSmoothingSettings {
    pub filter: OneEuroSettings,
    /// A joint not fitted for longer than this starts its filter afresh where it is next fitted, rather than glide
    /// there from where it was.
    pub restart_after: Duration,
}

impl JointSmoothingSettings {
    /// Still, a 0.3 Hz cutoff: a time constant of half a second. Each metre a second of speed adds 4 Hz, so at
    /// walking pace the cutoff is near 5 Hz, a time constant of about 30 ms. The speed itself is smoothed at
    /// 0.3 Hz, since a still joint's raw speed is mostly depth noise.
    pub const DEFAULT: JointSmoothingSettings = JointSmoothingSettings {
        filter: OneEuroSettings { min_cutoff_hz: 0.3, beta: 4.0, derivative_cutoff_hz: 0.3 },
        restart_after: Duration::from_millis(300),
    };
}

/// One joint's filter and when it last took a sample.
struct JointFilter {
    filter: OneEuro,
    sampled_ns: i64,
}

/// The joint filters of one person, in keypoint order.
pub struct JointSmoothing {
    settings: JointSmoothingSettings,
    joints: Vec<Option<JointFilter>>,
}

impl JointSmoothing {
    pub fn new(settings: JointSmoothingSettings) -> JointSmoothing {
        JointSmoothing { settings, joints: Vec::new() }
    }

    /// The fitted joints of one frame at `now_ns`, each smoothed; a joint not fitted this frame stays None.
    pub fn smooth(&mut self, now_ns: i64, fitted: &[Option<Vec3>]) -> Vec<Option<Vec3>> {
        if self.joints.len() < fitted.len() {
            self.joints.resize_with(fitted.len(), || None);
        }
        let restart_after_ns = self.settings.restart_after.as_nanos() as i64;
        fitted
            .iter()
            .zip(&mut self.joints)
            .map(|(joint, held)| {
                let joint = (*joint)?;
                let dt_s = match held {
                    Some(held) if now_ns - held.sampled_ns <= restart_after_ns => (now_ns - held.sampled_ns) as f32 * 1e-9,
                    _ => {
                        *held = Some(JointFilter { filter: OneEuro::new(self.settings.filter), sampled_ns: now_ns });
                        0.0
                    }
                };
                let held = held.as_mut().expect("set above");
                held.sampled_ns = now_ns;
                Some(held.filter.add(joint, dt_s).0)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME_NS: i64 = 33_333_333;

    #[test]
    fn a_joint_seen_first_passes_through_and_a_missing_joint_stays_missing() {
        let mut smoothing = JointSmoothing::new(JointSmoothingSettings::DEFAULT);
        let out = smoothing.smooth(0, &[Some(Vec3::ONE), None]);
        assert_eq!(out, vec![Some(Vec3::ONE), None]);
    }

    #[test]
    fn a_joint_back_after_a_long_gap_starts_where_it_is_seen() {
        let mut smoothing = JointSmoothing::new(JointSmoothingSettings::DEFAULT);
        smoothing.smooth(0, &[Some(Vec3::ZERO)]);
        smoothing.smooth(FRAME_NS, &[None]);
        let back = smoothing.smooth(20 * FRAME_NS, &[Some(Vec3::X)]);
        assert_eq!(back, vec![Some(Vec3::X)], "a gap past restart_after does not glide from the old place");
        let soon = smoothing.smooth(21 * FRAME_NS, &[Some(Vec3::new(1.1, 0.0, 0.0))]);
        assert!(soon[0].unwrap().x < 1.1, "within restart_after the filter smooths: {soon:?}");
    }
}
