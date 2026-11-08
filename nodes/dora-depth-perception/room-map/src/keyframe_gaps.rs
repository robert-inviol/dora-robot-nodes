//! What the camera node's keyframe picker reveals about the turn between keyframes, for recordings with no
//! IMU series.
//!
//! The picker (`rgbd_keyframes::KeyframePicker`) adds up the gyro's rotation rate over all axes and takes a
//! keyframe as soon as that reaches its turn threshold, once its minimum interval has passed, and otherwise
//! at its maximum interval. So a keyframe that came early followed a turn of at least the threshold, and one
//! that came at the maximum interval followed a turn under it. Gravity keeps roll and pitch nearly fixed, so
//! the turn is mostly about the vertical, but the sum over all axes and any back-and-forth make it larger
//! than the net change in heading.

use std::time::Duration;

use rgbd_keyframes::KeyframeSettings;

/// What a keyframe's gap from the one before says about the turn between them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GapTurn {
    /// The gap fits neither rule, as when a keyframe in between was dropped.
    Unknown,
    /// The keyframe came early: the camera turned at least this much in all.
    AtLeast { turn_rad: f32 },
    /// The keyframe came at the maximum interval: the camera turned less than this.
    AtMost { turn_rad: f32 },
}

#[derive(Clone, Debug)]
pub struct GapSettings {
    pub picker: KeyframeSettings,
    /// Keyframe times are stamped after alignment, not when the picker chose the frame, so a gap may be off by
    /// a couple of video frames.
    pub timing_tolerance: Duration,
    /// A heading change this far over an `AtMost` bound contradicts it, allowing for registration noise.
    pub over_margin_rad: f32,
    /// A heading change this far under an `AtLeast` bound contradicts it; generous, because the picker's
    /// turn sums every axis and any back-and-forth, which the net heading change does not.
    pub under_margin_rad: f32,
}

impl Default for GapSettings {
    fn default() -> Self {
        GapSettings {
            picker: KeyframeSettings::default(),
            timing_tolerance: Duration::from_millis(70),
            over_margin_rad: 2f32.to_radians(),
            under_margin_rad: 5f32.to_radians(),
        }
    }
}

/// The turn bound for each keyframe from the second on, from capture times in nanoseconds.
pub fn gap_turns(times_ns: &[i64], settings: &GapSettings) -> Vec<GapTurn> {
    let picker = &settings.picker;
    let tolerance = settings.timing_tolerance;
    times_ns
        .windows(2)
        .map(|w| {
            let gap = Duration::from_nanos(w[1].saturating_sub(w[0]).max(0) as u64);
            if gap + tolerance < picker.min_interval || gap > picker.max_interval + tolerance {
                GapTurn::Unknown
            } else if gap + tolerance >= picker.max_interval {
                GapTurn::AtMost { turn_rad: picker.turn_threshold_rad }
            } else {
                GapTurn::AtLeast { turn_rad: picker.turn_threshold_rad }
            }
        })
        .collect()
}

/// Whether a heading change between two keyframes contradicts what their gap says.
pub fn contradicts(bound: GapTurn, heading_change_rad: f32, settings: &GapSettings) -> bool {
    match bound {
        GapTurn::Unknown => false,
        GapTurn::AtMost { turn_rad } => heading_change_rad.abs() > turn_rad + settings.over_margin_rad,
        GapTurn::AtLeast { turn_rad } => heading_change_rad.abs() < turn_rad - settings.under_margin_rad,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: i64 = 1_000_000;

    #[test]
    fn gaps_are_read_against_the_picker_settings() {
        let settings = GapSettings::default();
        let threshold = settings.picker.turn_threshold_rad;
        let times = [0, 1000 * MS, 1400 * MS, 1550 * MS, 3600 * MS, 4560 * MS];
        assert_eq!(
            gap_turns(&times, &settings),
            vec![
                GapTurn::AtMost { turn_rad: threshold },
                GapTurn::AtLeast { turn_rad: threshold },
                GapTurn::AtLeast { turn_rad: threshold },
                GapTurn::Unknown,
                GapTurn::AtMost { turn_rad: threshold },
            ]
        );
    }

    #[test]
    fn a_turn_contradicts_a_bound_only_beyond_its_margin() {
        let settings = GapSettings::default();
        let threshold = settings.picker.turn_threshold_rad;
        let at_most = GapTurn::AtMost { turn_rad: threshold };
        let at_least = GapTurn::AtLeast { turn_rad: threshold };
        assert!(!contradicts(at_most, threshold + 1f32.to_radians(), &settings));
        assert!(contradicts(at_most, -(threshold + 3f32.to_radians()), &settings));
        assert!(!contradicts(at_least, threshold - 4f32.to_radians(), &settings));
        assert!(contradicts(at_least, 1f32.to_radians(), &settings));
        assert!(!contradicts(GapTurn::Unknown, 3.0, &settings));
    }
}
