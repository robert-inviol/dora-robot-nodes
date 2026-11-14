//! Reading depth for a pixel of the colour image from a depth frame that is not aligned to it.

use glam::Vec3;
use perception_wire::camera::CameraCalibration;

/// One depth frame, row major, millimetres, 0 where the sensor has no reading.
pub struct DepthImage<'a> {
    pub width: usize,
    pub height: usize,
    pub depth_mm: &'a [u16],
}

/// Half the side of the square of depth pixels whose median stands for one point.
const WINDOW_RADIUS: isize = 2;
/// Where the search along a colour pixel's ray starts, before any depth is known.
const FIRST_GUESS_M: f32 = 2.0;
/// Each step moves the guess to where the depth camera saw the ray; the cameras' small baseline settles it quickly.
const REFINEMENTS: usize = 3;

impl DepthImage<'_> {
    /// The median of the readings in the window around a depth pixel, in metres, or None when it has none.
    fn median_m(&self, x: f32, y: f32) -> Option<f32> {
        let (cx, cy) = (x.round() as isize, y.round() as isize);
        let mut readings: Vec<u16> = Vec::with_capacity(25);
        for row in cy - WINDOW_RADIUS..=cy + WINDOW_RADIUS {
            for column in cx - WINDOW_RADIUS..=cx + WINDOW_RADIUS {
                if row < 0 || column < 0 || row as usize >= self.height || column as usize >= self.width {
                    continue;
                }
                let reading = self.depth_mm[row as usize * self.width + column as usize];
                if reading != 0 {
                    readings.push(reading);
                }
            }
        }
        if readings.is_empty() {
            return None;
        }
        let middle = readings.len() / 2;
        Some(*readings.select_nth_unstable(middle).1 as f32 * 1e-3)
    }

    /// The point a colour pixel sees, in colour camera coordinates, or None where depth has no reading.
    pub fn point_at_color_pixel(&self, calibration: &CameraCalibration, x_px: f32, y_px: f32) -> Option<Vec3> {
        let color_to_depth = calibration.depth_to_color.inverse();
        let mut z_m = FIRST_GUESS_M;
        for _ in 0..REFINEMENTS {
            let in_depth = color_to_depth.apply(calibration.color.deproject(x_px, y_px, z_m));
            let (depth_x, depth_y) = calibration.depth.project(in_depth)?;
            let seen_m = self.median_m(depth_x, depth_y)?;
            z_m = calibration.depth_to_color.apply(calibration.depth.deproject(depth_x, depth_y, seen_m)).z;
            if z_m <= 0.0 {
                return None;
            }
        }
        Some(calibration.color.deproject(x_px, y_px, z_m))
    }
}

#[cfg(test)]
mod tests {
    use perception_wire::camera::{RigidTransform, StreamIntrinsics};

    use super::*;

    const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    fn calibration(baseline_m: f32) -> CameraCalibration {
        let intrinsics = StreamIntrinsics { width: 64, height: 48, fx: 50.0, fy: 50.0, ppx: 32.0, ppy: 24.0 };
        let transform = RigidTransform { rotation: IDENTITY, translation_m: [baseline_m, 0.0, 0.0] };
        CameraCalibration { color: intrinsics, depth: intrinsics, depth_to_color: transform, depth_to_imu: transform }
    }

    #[test]
    fn a_flat_wall_reads_back_at_its_distance_through_the_baseline() {
        let depth_mm = vec![3000u16; 64 * 48];
        let image = DepthImage { width: 64, height: 48, depth_mm: &depth_mm };
        let point = image.point_at_color_pixel(&calibration(-0.06), 40.0, 30.0).unwrap();
        assert!((point.z - 3.0).abs() < 1e-4, "{point}");
        assert!((point.x - 8.0 / 50.0 * 3.0).abs() < 1e-4, "{point}");
    }

    #[test]
    fn a_window_with_some_readings_takes_their_median_and_one_with_none_gives_nothing() {
        let mut depth_mm = vec![0u16; 64 * 48];
        for (i, value) in [1000, 1100, 5000].into_iter().enumerate() {
            depth_mm[24 * 64 + 31 + i] = value;
        }
        let image = DepthImage { width: 64, height: 48, depth_mm: &depth_mm };
        assert_eq!(image.median_m(32.0, 24.0), Some(1.1));
        assert_eq!(image.median_m(5.0, 5.0), None);
        assert_eq!(image.point_at_color_pixel(&calibration(0.0), 5.0, 5.0), None);
    }

    #[test]
    fn the_search_finds_a_near_object_in_front_of_a_far_wall() {
        // A 0.06 m baseline shifts a point at 1 m by 3 depth pixels; the person stands at 1 m before a wall at 4 m.
        let mut depth_mm = vec![4000u16; 64 * 48];
        for row in 10..40 {
            for column in 20..34 {
                depth_mm[row * 64 + column] = 1000;
            }
        }
        let image = DepthImage { width: 64, height: 48, depth_mm: &depth_mm };
        let point = image.point_at_color_pixel(&calibration(-0.06), 24.0, 24.0).unwrap();
        assert!((point.z - 1.0).abs() < 1e-4, "{point}");
    }
}
