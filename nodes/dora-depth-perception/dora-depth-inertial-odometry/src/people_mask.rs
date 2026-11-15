//! People cut out of the depth odometry tracks: a person moves while the room stays, so matching their surface would
//! drag the pose after them, and adding it to the map would leave them there. The pose model finds people in the
//! colour image, later than the depth frame is tracked, so a depth frame is cut with the newest boxes there are. Each
//! box is grown by a margin and by as far as a person walking could have gone in the time between the box's frame and
//! the depth frame, then carried into the depth image, which is not aligned to colour, by bounding where it can fall
//! for anything between the nearest and the furthest depth. Boxes older than `max_age_ns` cut nothing.

use perception_wire::camera::CameraCalibration;
use perception_wire::localization::CameraNanos;
use perception_wire::people::{PersonPoses, PixelBox};

/// Nearer than this the camera reads no depth; further than this the box is not cut.
const NEAREST_M: f32 = 0.4;
const FURTHEST_M: f32 = 6.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskSettings {
    /// Each side of a box grows by this fraction of the box's width or height, for a box from the depth frame's own
    /// time.
    pub margin: f32,
    /// How fast people are taken to walk.
    pub walking_speed_m_s: f32,
    /// How wide a person is taken to be: a box grows on each side by the share of its size that the walking speed
    /// covers, over this width, in the box's age.
    pub person_width_m: f32,
    /// Boxes stand for depth frames taken up to this long before or after them, and for none further off.
    pub max_age_ns: i64,
}

impl MaskSettings {
    pub const DEFAULT: MaskSettings = MaskSettings { margin: 0.15, walking_speed_m_s: 1.5, person_width_m: 0.5, max_age_ns: 300_000_000 };

    /// The fraction of a box's size each side grows by for a depth frame `age_ns` from the box's frame.
    pub fn growth(&self, age_ns: i64) -> f32 {
        self.margin + self.walking_speed_m_s * (age_ns.unsigned_abs() as f32 * 1e-9) / self.person_width_m
    }
}

/// A rectangle of depth pixels: columns `left..right`, rows `top..bottom`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthRect {
    pub left: usize,
    pub top: usize,
    pub right: usize,
    pub bottom: usize,
}

#[derive(Debug, thiserror::Error, PartialEq)]
#[error("person poses come from a {width}x{height} image; the calibration's colour stream is {color_width}x{color_height}")]
pub struct WrongImageSize {
    pub width: u16,
    pub height: u16,
    pub color_width: u16,
    pub color_height: u16,
}

/// The depth pixels where a person in a colour frame can be, whatever their distance, with each side of the box grown
/// by `growth` of its size.
fn depth_rect(calibration: &CameraCalibration, bbox: &PixelBox, growth: f32) -> Option<DepthRect> {
    let (grow_x, grow_y) = (bbox.w_px * growth, bbox.h_px * growth);
    let (x0, y0) = (bbox.x_px - grow_x, bbox.y_px - grow_y);
    let (x1, y1) = (bbox.x_px + bbox.w_px + grow_x, bbox.y_px + bbox.h_px + grow_y);
    let color_to_depth = calibration.depth_to_color.inverse();
    let mut low = (f32::INFINITY, f32::INFINITY);
    let mut high = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (x, y) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
        for depth_m in [NEAREST_M, FURTHEST_M] {
            let Some((u, v)) = calibration.depth.project(color_to_depth.apply(calibration.color.deproject(x, y, depth_m))) else { continue };
            low = (low.0.min(u), low.1.min(v));
            high = (high.0.max(u), high.1.max(v));
        }
    }
    let (width, height) = (f32::from(calibration.depth.width), f32::from(calibration.depth.height));
    let clamp = |value: f32, limit: f32| value.clamp(0.0, limit) as usize;
    let rect = DepthRect {
        left: clamp(low.0.floor(), width),
        top: clamp(low.1.floor(), height),
        right: clamp(high.0.ceil(), width),
        bottom: clamp(high.1.ceil(), height),
    };
    (rect.left < rect.right && rect.top < rect.bottom).then_some(rect)
}

/// The newest people the pose model found.
#[derive(Clone, Debug, Default)]
pub struct PeopleMask {
    seen_at: Option<CameraNanos>,
    boxes: Vec<PixelBox>,
    rects: Vec<DepthRect>,
}

impl PeopleMask {
    pub fn update(&mut self, poses: &PersonPoses, calibration: &CameraCalibration) -> Result<(), WrongImageSize> {
        let color = calibration.color;
        if (poses.width, poses.height) != (color.width, color.height) {
            return Err(WrongImageSize { width: poses.width, height: poses.height, color_width: color.width, color_height: color.height });
        }
        let at = CameraNanos(poses.captured_camera_ns);
        if self.seen_at.is_some_and(|seen| at.0 < seen.0) {
            return Ok(());
        }
        self.seen_at = Some(at);
        self.boxes.clear();
        self.boxes.extend(poses.people.iter().map(|person| person.bbox));
        Ok(())
    }

    /// The depth rectangles to cut out of a depth frame taken at `time`: none when the newest boxes are too far
    /// from it in time.
    pub fn rects_for(&mut self, time: CameraNanos, calibration: &CameraCalibration, settings: &MaskSettings) -> &[DepthRect] {
        self.rects.clear();
        let Some(seen) = self.seen_at else { return &self.rects };
        let age_ns = time.0 - seen.0;
        if age_ns.abs() <= settings.max_age_ns {
            let growth = settings.growth(age_ns);
            self.rects.extend(self.boxes.iter().filter_map(|bbox| depth_rect(calibration, bbox, growth)));
        }
        &self.rects
    }
}

/// The depth with every pixel inside the rectangles read as no reading, written into `cut`; the frame itself when
/// there are none.
pub fn cut_out<'a>(depth_mm: &'a [u16], width: usize, rects: &[DepthRect], cut: &'a mut Vec<u16>) -> &'a [u16] {
    if rects.is_empty() {
        return depth_mm;
    }
    cut.clear();
    cut.extend_from_slice(depth_mm);
    for rect in rects {
        for row in rect.top..rect.bottom {
            cut[row * width + rect.left..row * width + rect.right].fill(0);
        }
    }
    cut
}

/// How many pixels of the image the rectangles cover together, each pixel counted once. `spans` is scratch space.
pub fn covered_pixels(rects: &[DepthRect], spans: &mut Vec<(usize, usize)>) -> usize {
    let (Some(top), Some(bottom)) = (rects.iter().map(|rect| rect.top).min(), rects.iter().map(|rect| rect.bottom).max()) else { return 0 };
    let mut covered = 0;
    for row in top..bottom {
        // The columns of the rectangles crossing this row, merged where they overlap.
        spans.clear();
        spans.extend(rects.iter().filter(|rect| (rect.top..rect.bottom).contains(&row)).map(|rect| (rect.left, rect.right)));
        spans.sort_unstable();
        let mut reached = 0;
        for &(left, right) in spans.iter() {
            covered += right.saturating_sub(left.max(reached));
            reached = reached.max(right);
        }
    }
    covered
}

#[cfg(test)]
mod tests {
    use perception_wire::camera::{RigidTransform, StreamIntrinsics};
    use perception_wire::people::PersonPose;

    use super::*;

    const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    /// Colour and depth streams alike, the depth camera `baseline_m` along x from the colour camera.
    fn calibration(baseline_m: f32) -> CameraCalibration {
        let intrinsics = StreamIntrinsics { width: 64, height: 48, fx: 50.0, fy: 50.0, ppx: 32.0, ppy: 24.0 };
        let depth_to_color = RigidTransform { rotation: IDENTITY, translation_m: [baseline_m, 0.0, 0.0] };
        CameraCalibration { color: intrinsics, depth: intrinsics, depth_to_color, depth_to_imu: depth_to_color }
    }

    fn poses(captured_camera_ns: i64, boxes: &[PixelBox]) -> PersonPoses {
        let people = boxes.iter().map(|&bbox| PersonPose { track_id: None, score: 0.9, bbox, keypoints: vec![] }).collect();
        PersonPoses { frame_id: 1, captured_camera_ns, width: 64, height: 48, people }
    }

    const PERSON: PixelBox = PixelBox { x_px: 20.0, y_px: 10.0, w_px: 10.0, h_px: 20.0 };

    #[test]
    fn with_no_baseline_a_box_is_the_same_pixels_grown_by_the_margin() {
        let rect = depth_rect(&calibration(0.0), &PERSON, 0.1).unwrap();
        let near = |got: usize, want: usize| got.abs_diff(want) <= 1;
        assert!(near(rect.left, 19) && near(rect.top, 8) && near(rect.right, 31) && near(rect.bottom, 32), "{rect:?}");
    }

    #[test]
    fn a_baseline_widens_the_box_to_cover_every_distance() {
        let rect = depth_rect(&calibration(-0.05), &PERSON, 0.0).unwrap();
        // 5 cm moves a pixel 50 * 0.05 / 0.4 = 6.25 px at 0.4 m, and 0.4 px at 6 m.
        assert!(rect.right - rect.left > 15, "{rect:?}");
        assert!(rect.top.abs_diff(10) <= 1 && rect.bottom.abs_diff(30) <= 1, "{rect:?}");
    }

    #[test]
    fn the_pixels_inside_a_person_read_as_no_depth() {
        let calibration = calibration(0.0);
        let mut mask = PeopleMask::default();
        mask.update(&poses(1_000, &[PERSON]), &calibration).unwrap();
        let depth = vec![2000u16; 64 * 48];
        let rects = mask.rects_for(CameraNanos(1_000), &calibration, &MaskSettings { margin: 0.0, ..MaskSettings::DEFAULT });
        let mut buffer = Vec::new();
        let cut = cut_out(&depth, 64, rects, &mut buffer);
        let cut_pixels = cut.iter().filter(|d| **d == 0).count();
        assert!((10 * 20..=11 * 21).contains(&cut_pixels), "{cut_pixels}");
        assert_eq!(covered_pixels(rects, &mut Vec::new()), cut_pixels);
        assert_eq!(cut[15 * 64 + 25], 0);
        assert_eq!(cut[15 * 64 + 35], 2000);
    }

    #[test]
    fn boxes_too_old_for_a_frame_cut_nothing_and_a_frame_without_people_is_not_copied() {
        let (mut mask, calibration) = (PeopleMask::default(), calibration(0.0));
        let settings = MaskSettings::DEFAULT;
        mask.update(&poses(1_000, &[PERSON]), &calibration).unwrap();
        assert_eq!(mask.rects_for(CameraNanos(1_000 + settings.max_age_ns), &calibration, &settings).len(), 1);
        assert_eq!(mask.rects_for(CameraNanos(1_000 - settings.max_age_ns), &calibration, &settings).len(), 1);
        assert!(mask.rects_for(CameraNanos(1_001 + settings.max_age_ns), &calibration, &settings).is_empty());
        let depth = vec![2000u16; 64 * 48];
        assert!(std::ptr::eq(cut_out(&depth, 64, &[], &mut Vec::new()), depth.as_slice()));
        mask.update(&poses(2_000, &[]), &calibration).unwrap();
        assert!(mask.rects_for(CameraNanos(2_000), &calibration, &settings).is_empty(), "nobody in the newest frame");
    }

    #[test]
    fn an_older_box_is_grown_by_as_far_as_a_person_walks_in_its_age() {
        let (mut mask, calibration) = (PeopleMask::default(), calibration(0.0));
        let settings = MaskSettings { margin: 0.0, ..MaskSettings::DEFAULT };
        mask.update(&poses(0, &[PERSON]), &calibration).unwrap();
        let width_at = |mask: &mut PeopleMask, age_ns: i64| {
            let rect = mask.rects_for(CameraNanos(age_ns), &calibration, &settings)[0];
            rect.right - rect.left
        };
        assert!(width_at(&mut mask, 0).abs_diff(10) <= 1);
        // A tenth of a second at 1.5 m/s is 0.15 m, three tenths of a half metre wide person: 3 px each side of 10.
        assert!(width_at(&mut mask, 100_000_000).abs_diff(16) <= 1, "{}", width_at(&mut mask, 100_000_000));
    }

    #[test]
    fn overlapping_rectangles_cover_each_pixel_once() {
        let a = DepthRect { left: 0, top: 0, right: 10, bottom: 10 };
        let b = DepthRect { left: 5, top: 5, right: 15, bottom: 15 };
        let inside = DepthRect { left: 2, top: 2, right: 4, bottom: 4 };
        assert_eq!(covered_pixels(&[a, b, inside], &mut Vec::new()), 100 + 100 - 25);
        assert_eq!(covered_pixels(&[], &mut Vec::new()), 0);
    }

    #[test]
    fn poses_from_an_image_of_another_size_are_refused() {
        let mut wrong = poses(1, &[PERSON]);
        wrong.width = 640;
        let refused = PeopleMask::default().update(&wrong, &calibration(0.0));
        assert_eq!(refused, Err(WrongImageSize { width: 640, height: 48, color_width: 64, color_height: 48 }));
    }
}
