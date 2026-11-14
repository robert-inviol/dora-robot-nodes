//! Turns a raw depth image into the sensor-frame cloud the odometry takes.
//!
//! The sensor frame is the depth stream's optical frame: the origin at the pinhole, x to the right of
//! the image, y down it and z forward along the optical axis. It is the frame a camera's depth-to-IMU
//! transform starts from, so [`crate::thinning::SensorMount::from_depth_to_imu`] takes these points to
//! the body.

use std::num::NonZeroUsize;

use glam::Vec3;

const METRES_PER_MILLIMETRE: f32 = 1e-3;

/// A pinhole model of the depth stream at its streamed size, without lens distortion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PinholeIntrinsics {
    pub width: usize,
    pub height: usize,
    /// Focal lengths in pixels.
    pub fx: f32,
    pub fy: f32,
    /// The pixel the optical axis passes through.
    pub ppx: f32,
    pub ppy: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DepthImageError {
    /// An image with no pixels, or a focal length that is not a positive number.
    UnusableIntrinsics(PinholeIntrinsics),
    /// The readings do not fill the image the intrinsics describe.
    WrongSize { readings: usize, width: usize, height: usize },
}

impl std::fmt::Display for DepthImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DepthImageError::UnusableIntrinsics(intrinsics) => write!(f, "depth intrinsics cannot project an image: {intrinsics:?}"),
            DepthImageError::WrongSize { readings, width, height } => write!(f, "depth image has {readings} readings for {width}x{height}"),
        }
    }
}

impl std::error::Error for DepthImageError {}

/// Projects depth images of one camera to sensor-frame points, reading every `stride`-th pixel of
/// every `stride`-th row.
#[derive(Clone, Debug)]
pub struct DepthProjector {
    intrinsics: PinholeIntrinsics,
    stride: usize,
    /// For each column read, the point's x per millimetre of depth.
    x_per_mm: Vec<f32>,
    /// For each row read, its index and the point's y per millimetre of depth.
    rows: Vec<(usize, f32)>,
}

impl DepthProjector {
    pub fn new(intrinsics: PinholeIntrinsics, stride: NonZeroUsize) -> Result<DepthProjector, DepthImageError> {
        let usable = intrinsics.width > 0 && intrinsics.height > 0 && intrinsics.fx.is_finite() && intrinsics.fy.is_finite() && intrinsics.fx > 0.0 && intrinsics.fy > 0.0 && intrinsics.ppx.is_finite() && intrinsics.ppy.is_finite();
        if !usable {
            return Err(DepthImageError::UnusableIntrinsics(intrinsics));
        }
        let stride = stride.get();
        // The pixels read sit in the middle of each stride-sized block.
        let read = |size: usize| (stride / 2..size).step_by(stride);
        Ok(DepthProjector {
            intrinsics,
            stride,
            x_per_mm: read(intrinsics.width).map(|column| (column as f32 - intrinsics.ppx) / intrinsics.fx * METRES_PER_MILLIMETRE).collect(),
            rows: read(intrinsics.height).map(|row| (row, (row as f32 - intrinsics.ppy) / intrinsics.fy * METRES_PER_MILLIMETRE)).collect(),
        })
    }

    pub fn intrinsics(&self) -> &PinholeIntrinsics {
        &self.intrinsics
    }

    /// The most points one image can give.
    pub fn pixels_read(&self) -> usize {
        self.x_per_mm.len() * self.rows.len()
    }

    /// Whether an image of this many readings is one of this camera's.
    pub fn fits(&self, readings: usize) -> Result<(), DepthImageError> {
        let PinholeIntrinsics { width, height, .. } = self.intrinsics;
        if readings != width * height {
            return Err(DepthImageError::WrongSize { readings, width, height });
        }
        Ok(())
    }

    /// Fills `points` with the sensor-frame points of a row-major image of depths along the optical
    /// axis in millimetres, where zero means no reading.
    pub fn project(&self, depth_mm: &[u16], points: &mut Vec<Vec3>) -> Result<(), DepthImageError> {
        self.fits(depth_mm.len())?;
        let width = self.intrinsics.width;
        points.clear();
        points.reserve(self.pixels_read());
        for &(row, y_per_mm) in &self.rows {
            let line = &depth_mm[row * width..(row + 1) * width];
            for (&mm, &x_per_mm) in line.iter().skip(self.stride / 2).step_by(self.stride).zip(&self.x_per_mm) {
                if mm != 0 {
                    let mm = mm as f32;
                    points.push(Vec3::new(mm * x_per_mm, mm * y_per_mm, mm * METRES_PER_MILLIMETRE));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EVERY_PIXEL: NonZeroUsize = NonZeroUsize::MIN;

    fn close(points: &[Vec3], expected: &[Vec3]) -> bool {
        points.len() == expected.len() && points.iter().zip(expected).all(|(point, expected)| (*point - *expected).length() < 1e-5)
    }

    fn small() -> PinholeIntrinsics {
        PinholeIntrinsics { width: 4, height: 3, fx: 2.0, fy: 4.0, ppx: 1.0, ppy: 2.0 }
    }

    #[test]
    fn a_reading_becomes_a_point_x_right_y_down_z_forward_at_its_depth() {
        let mut depth_mm = vec![0u16; 12];
        // Column 3 is two pixels right of the optical axis; row 0 is two pixels above it.
        depth_mm[3] = 2000;
        let mut points = Vec::new();
        DepthProjector::new(small(), EVERY_PIXEL).expect("usable intrinsics").project(&depth_mm, &mut points).expect("the image fits");
        assert!(close(&points, &[Vec3::new(2.0, -1.0, 2.0)]), "{points:?}");
    }

    #[test]
    fn a_reading_on_the_optical_axis_lies_straight_ahead() {
        let mut depth_mm = vec![0u16; 12];
        depth_mm[2 * 4 + 1] = 1500;
        let mut points = Vec::new();
        DepthProjector::new(small(), EVERY_PIXEL).expect("usable intrinsics").project(&depth_mm, &mut points).expect("the image fits");
        assert!(close(&points, &[Vec3::new(0.0, 0.0, 1.5)]), "{points:?}");
    }

    #[test]
    fn pixels_without_a_reading_give_no_point() {
        let mut points = vec![Vec3::ONE];
        DepthProjector::new(small(), EVERY_PIXEL).expect("usable intrinsics").project(&[0u16; 12], &mut points).expect("the image fits");
        assert!(points.is_empty());
    }

    #[test]
    fn a_stride_reads_the_middle_pixel_of_each_block() {
        let intrinsics = PinholeIntrinsics { width: 8, height: 4, fx: 1.0, fy: 1.0, ppx: 0.0, ppy: 0.0 };
        let depth_mm: Vec<u16> = (0..32).map(|index| 1000 + index).collect();
        let projector = DepthProjector::new(intrinsics, NonZeroUsize::new(4).expect("not zero")).expect("usable intrinsics");
        let mut points = Vec::new();
        projector.project(&depth_mm, &mut points).expect("the image fits");
        // Row 2, columns 2 and 6.
        assert!(close(&points, &[Vec3::new(2.0 * 1.018, 2.0 * 1.018, 1.018), Vec3::new(6.0 * 1.022, 2.0 * 1.022, 1.022)]), "{points:?}");
        assert_eq!(projector.pixels_read(), 2);
    }

    #[test]
    fn an_image_of_another_size_is_refused() {
        let error = DepthProjector::new(small(), EVERY_PIXEL).expect("usable intrinsics").project(&[0u16; 11], &mut Vec::new()).expect_err("one reading short");
        assert_eq!(error, DepthImageError::WrongSize { readings: 11, width: 4, height: 3 });
    }

    #[test]
    fn intrinsics_without_a_positive_focal_length_or_any_pixels_are_refused() {
        for unusable in [PinholeIntrinsics { fx: 0.0, ..small() }, PinholeIntrinsics { fy: f32::NAN, ..small() }, PinholeIntrinsics { width: 0, ..small() }] {
            assert!(matches!(DepthProjector::new(unusable, EVERY_PIXEL), Err(DepthImageError::UnusableIntrinsics(_))), "{unusable:?}");
        }
    }
}
