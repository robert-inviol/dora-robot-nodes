//! Raw depth redrawn onto the colour image: each depth pixel is carried into the colour camera's coordinates with the
//! calibration and painted over the colour pixels it covers, the nearest reading winning where two overlap, as
//! librealsense's align does. Lens distortion is not modelled.

use perception_wire::camera::CameraCalibration;

/// Depth in millimetres, row major, aligned to the colour image; 0 where no depth pixel lands.
pub fn align_depth_to_color(depth_mm: &[u16], calibration: &CameraCalibration) -> Vec<u16> {
    let (depth, color) = (&calibration.depth, &calibration.color);
    let (color_width, color_height) = (color.width as usize, color.height as usize);
    let mut aligned = vec![0u16; color_width * color_height];
    let corner_pixel = |x_px: f32, y_px: f32, depth_m: f32| -> Option<(f32, f32)> {
        color.project(calibration.depth_to_color.apply(depth.deproject(x_px, y_px, depth_m)))
    };
    for row in 0..depth.height as usize {
        for column in 0..depth.width as usize {
            let mm = depth_mm[row * depth.width as usize + column];
            if mm == 0 {
                continue;
            }
            let depth_m = f32::from(mm) * 1e-3;
            let (x, y) = (column as f32, row as f32);
            let (Some(top_left), Some(bottom_right)) = (corner_pixel(x - 0.5, y - 0.5, depth_m), corner_pixel(x + 0.5, y + 0.5, depth_m)) else {
                continue;
            };
            let first_column = (top_left.0 + 0.5).floor().max(0.0) as usize;
            let first_row = (top_left.1 + 0.5).floor().max(0.0) as usize;
            let last_column = ((bottom_right.0 + 0.5).floor() as isize).min(color_width as isize - 1);
            let last_row = ((bottom_right.1 + 0.5).floor() as isize).min(color_height as isize - 1);
            for painted_row in first_row as isize..=last_row {
                for painted_column in first_column as isize..=last_column {
                    let cell = &mut aligned[painted_row as usize * color_width + painted_column as usize];
                    if *cell == 0 || mm < *cell {
                        *cell = mm;
                    }
                }
            }
        }
    }
    aligned
}

#[cfg(test)]
mod tests {
    use perception_wire::camera::{RigidTransform, StreamIntrinsics};

    use super::*;

    const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    /// Depth 8x6 and colour 12x9 with the same field of view, colour `baseline_m` to the right of depth.
    fn calibration(baseline_m: f32) -> CameraCalibration {
        CameraCalibration {
            depth: StreamIntrinsics { width: 8, height: 6, fx: 8.0, fy: 8.0, ppx: 3.5, ppy: 2.5 },
            color: StreamIntrinsics { width: 12, height: 9, fx: 12.0, fy: 12.0, ppx: 5.5, ppy: 4.0 },
            depth_to_color: RigidTransform { rotation: IDENTITY, translation_m: [-baseline_m, 0.0, 0.0] },
            depth_to_imu: RigidTransform { rotation: IDENTITY, translation_m: [0.0; 3] },
        }
    }

    #[test]
    fn a_flat_wall_seen_by_both_cameras_from_one_place_fills_the_colour_image() {
        let aligned = align_depth_to_color(&[2000; 48], &calibration(0.0));
        assert!(aligned.iter().all(|&mm| mm == 2000), "{aligned:?}");
    }

    #[test]
    fn a_reading_lands_shifted_by_the_baseline() {
        // One reading in the middle at 1 m; colour 0.25 m to the right sees it 3 px further left.
        let mut depth_mm = vec![0u16; 48];
        depth_mm[2 * 8 + 4] = 1000;
        let aligned = align_depth_to_color(&depth_mm, &calibration(0.25));
        let painted: Vec<(usize, usize)> = (0..aligned.len()).filter(|&i| aligned[i] != 0).map(|i| (i % 12, i / 12)).collect();
        // Without the baseline the depth pixel (4, 2) covers colour columns 6..7, rows 3..4.
        assert_eq!(painted, vec![(3, 3), (4, 3), (3, 4), (4, 4)]);
    }

    #[test]
    fn where_two_readings_overlap_the_nearer_one_wins() {
        let mut depth_mm = vec![3000u16; 48];
        depth_mm[2 * 8 + 4] = 1000;
        let aligned = align_depth_to_color(&depth_mm, &calibration(0.0));
        assert_eq!(aligned[3 * 12 + 6], 1000);
        assert_eq!(aligned[0], 3000);
    }
}
