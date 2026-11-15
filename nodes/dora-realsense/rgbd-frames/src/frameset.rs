//! Which colour and depth framesets the camera sends. A frameset goes out only once the accelerometer and gyro have
//! both delivered a sample, so every imu message has samples. Every frameset that goes out has its colour and depth
//! frames captured at most `COLOR_DEPTH_GAP_LIMIT` apart; one that would go out further apart is a camera fault and
//! stops the node. Framesets before the motion streams deliver are not sent and so not held to the limit: the first
//! frameset after the camera starts pairs frames captured far apart.

use std::fmt;
use std::time::Duration;

/// The furthest apart the colour and depth frames of a sent frameset may have been captured.
pub const COLOR_DEPTH_GAP_LIMIT: Duration = Duration::from_millis(5);

/// What the camera does with a frameset of a new colour and a new depth frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FramesetDisposition {
    /// The accelerometer or gyro has not yet delivered a sample: the frameset is not sent.
    AwaitingMotion,
    /// The frameset goes out.
    Send,
}

/// A frameset that would go out with its colour and depth frames captured more than `COLOR_DEPTH_GAP_LIMIT` apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorDepthGapExceeded {
    pub gap: Duration,
}

impl fmt::Display for ColorDepthGapExceeded {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "colour and depth were captured {:.3} ms apart, more than the {:.1} ms a sent frameset may be",
            self.gap.as_secs_f64() * 1e3,
            COLOR_DEPTH_GAP_LIMIT.as_secs_f64() * 1e3
        )
    }
}

impl std::error::Error for ColorDepthGapExceeded {}

/// The disposition of a frameset whose colour and depth frames were captured at these camera times, given whether
/// the accelerometer and gyro have both delivered.
pub fn frameset_disposition(
    motion_delivering: bool,
    color_camera_ns: i64,
    depth_camera_ns: i64,
) -> Result<FramesetDisposition, ColorDepthGapExceeded> {
    if !motion_delivering {
        return Ok(FramesetDisposition::AwaitingMotion);
    }
    let gap = Duration::from_nanos(color_camera_ns.abs_diff(depth_camera_ns));
    if gap > COLOR_DEPTH_GAP_LIMIT {
        return Err(ColorDepthGapExceeded { gap });
    }
    Ok(FramesetDisposition::Send)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: i64 = 1_000_000;

    #[test]
    fn a_frameset_before_motion_is_not_sent_whatever_its_gap() {
        assert_eq!(frameset_disposition(false, 0, 300 * MS), Ok(FramesetDisposition::AwaitingMotion));
        assert_eq!(frameset_disposition(false, 0, 0), Ok(FramesetDisposition::AwaitingMotion));
    }

    #[test]
    fn a_frameset_within_the_limit_is_sent_once_motion_delivers() {
        assert_eq!(frameset_disposition(true, 1_000 * MS, 1_000 * MS + 8_000), Ok(FramesetDisposition::Send));
        assert_eq!(frameset_disposition(true, 1_000 * MS + 5 * MS, 1_000 * MS), Ok(FramesetDisposition::Send));
    }

    #[test]
    fn a_frameset_beyond_the_limit_once_motion_delivers_is_an_error() {
        let gap = frameset_disposition(true, 0, 5 * MS + 1).expect_err("beyond the limit");
        assert_eq!(gap.gap, Duration::from_nanos(5_000_001));
        assert!(frameset_disposition(true, 300 * MS, 0).is_err());
    }
}
