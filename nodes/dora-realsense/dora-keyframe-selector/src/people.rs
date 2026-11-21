//! People cut out of a keyframe's depth: a person moves while the room stays, and the dense map keeps keyframes as
//! its durable record and fuses them again whenever a pose graph update moves them, so a person left in a keyframe's depth
//! would come back into the mesh.
//!
//! The pose model's boxes are in colour image pixels, and a keyframe's depth is aligned to the colour image, so a box
//! falls on exactly the aligned depth pixels it covers in colour, at any distance. Each box is grown by the
//! odometry's rule (its `people_mask::MaskSettings`): a margin, and as far as a person walking could have gone between
//! the box's colour frame and the keyframe's depth frame. The person poses used are those nearest the keyframe's
//! depth in camera time, among the recent ones; a keyframe waits until person poses at or after its time have
//! arrived, so none nearer can still come.

use std::collections::VecDeque;

use dora_depth_inertial_odometry::people_mask::{covered_pixels, DepthRect, MaskSettings, WrongImageSize};
use perception_wire::camera::{CameraCalibration, StreamIntrinsics};
use perception_wire::localization::CameraNanos;
use perception_wire::people::{PersonPoses, PixelBox};

/// Person poses kept to choose from: about half a second of the pose model's frames.
const POSES_KEPT: usize = 16;

/// The people the pose model found in one colour frame.
struct SeenPeople {
    captured: CameraNanos,
    boxes: Vec<PixelBox>,
}

/// What was cut out of one keyframe's depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeopleCut {
    /// The person poses nearest the keyframe, `age_ns` from its depth frame, found `people` people, whose grown boxes
    /// cover `pixels` aligned depth pixels, all set to 0.
    Cut { people: usize, pixels: usize, age_ns: i64 },
    /// No person poses within `MaskSettings::max_age_ns` of the keyframe: the nearest were `nearest_age_ns` away.
    /// Nothing is cut.
    NoPosesNear { nearest_age_ns: i64 },
}

/// The recent person poses, oldest first.
pub struct RecentPeople {
    settings: MaskSettings,
    seen: VecDeque<SeenPeople>,
    rects: Vec<DepthRect>,
}

impl RecentPeople {
    pub fn new(settings: MaskSettings) -> RecentPeople {
        RecentPeople { settings, seen: VecDeque::new(), rects: Vec::new() }
    }

    /// Person poses from the pose model. Poses older than the newest kept are let go, as the odometry does.
    pub fn add(&mut self, poses: &PersonPoses, calibration: &CameraCalibration) -> Result<(), WrongImageSize> {
        let color = calibration.color;
        if (poses.width, poses.height) != (color.width, color.height) {
            return Err(WrongImageSize { width: poses.width, height: poses.height, color_width: color.width, color_height: color.height });
        }
        let captured = CameraNanos(poses.captured_camera_ns);
        if self.newest().is_some_and(|newest| captured <= newest) {
            return Ok(());
        }
        if self.seen.len() == POSES_KEPT {
            self.seen.pop_front();
        }
        self.seen.push_back(SeenPeople { captured, boxes: poses.people.iter().map(|person| person.bbox).collect() });
        Ok(())
    }

    /// The capture time of the newest person poses, None before the first.
    pub fn newest(&self) -> Option<CameraNanos> {
        self.seen.back().map(|seen| seen.captured)
    }

    /// Whether person poses at or after `time` have arrived, so that none nearer to it can still come.
    pub fn have_reached(&self, time: CameraNanos) -> bool {
        self.newest().is_some_and(|newest| newest >= time)
    }

    /// Sets to 0 every pixel of `aligned_depth_mm` (depth aligned to the `color` image, row major) inside the grown
    /// boxes of the person poses nearest `time`.
    ///
    /// # Panics
    /// When no person poses have arrived; a keyframe is only selected after the first.
    pub fn cut_out(&mut self, time: CameraNanos, color: &StreamIntrinsics, aligned_depth_mm: &mut [u16]) -> PeopleCut {
        let nearest = self.seen.iter().min_by_key(|seen| (time.0 - seen.captured.0).unsigned_abs()).expect("person poses arrived before any keyframe");
        let age_ns = time.0 - nearest.captured.0;
        if age_ns.abs() > self.settings.max_age_ns {
            return PeopleCut::NoPosesNear { nearest_age_ns: age_ns };
        }
        let growth = self.settings.growth(age_ns);
        self.rects.clear();
        self.rects.extend(nearest.boxes.iter().filter_map(|bbox| color_rect(color, bbox, growth)));
        let width = color.width as usize;
        for rect in &self.rects {
            for row in rect.top..rect.bottom {
                aligned_depth_mm[row * width + rect.left..row * width + rect.right].fill(0);
            }
        }
        PeopleCut::Cut { people: nearest.boxes.len(), pixels: covered_pixels(&self.rects, &mut Vec::new()), age_ns }
    }
}

/// The colour image pixels a box covers, each side grown by `growth` of its size, clipped to the image.
fn color_rect(color: &StreamIntrinsics, bbox: &PixelBox, growth: f32) -> Option<DepthRect> {
    let (grow_x, grow_y) = (bbox.w_px * growth, bbox.h_px * growth);
    let clamp = |value: f32, limit: u16| value.clamp(0.0, f32::from(limit)) as usize;
    let rect = DepthRect {
        left: clamp((bbox.x_px - grow_x).floor(), color.width),
        top: clamp((bbox.y_px - grow_y).floor(), color.height),
        right: clamp((bbox.x_px + bbox.w_px + grow_x).ceil(), color.width),
        bottom: clamp((bbox.y_px + bbox.h_px + grow_y).ceil(), color.height),
    };
    (rect.left < rect.right && rect.top < rect.bottom).then_some(rect)
}

#[cfg(test)]
mod tests {
    use perception_wire::camera::RigidTransform;
    use perception_wire::people::PersonPose;

    use super::*;

    const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    const COLOR: StreamIntrinsics = StreamIntrinsics { width: 20, height: 10, fx: 20.0, fy: 20.0, ppx: 9.5, ppy: 4.5 };
    const PERSON: PixelBox = PixelBox { x_px: 4.0, y_px: 2.0, w_px: 4.0, h_px: 6.0 };
    const EXACT: MaskSettings = MaskSettings { margin: 0.0, ..MaskSettings::DEFAULT };

    fn calibration() -> CameraCalibration {
        let transform = RigidTransform { rotation: IDENTITY, translation_m: [0.05, 0.0, 0.0] };
        let depth = StreamIntrinsics { width: 8, height: 6, fx: 8.0, fy: 8.0, ppx: 3.5, ppy: 2.5 };
        CameraCalibration { color: COLOR, depth, depth_to_color: transform, depth_to_imu: transform }
    }

    fn poses(captured_camera_ns: i64, boxes: &[PixelBox]) -> PersonPoses {
        let people = boxes.iter().map(|&bbox| PersonPose { track_id: None, score: 0.9, bbox, keypoints: vec![] }).collect();
        PersonPoses { frame_id: 1, captured_camera_ns, width: COLOR.width, height: COLOR.height, people }
    }

    fn zeroed(depth_mm: &[u16]) -> Vec<(usize, usize)> {
        (0..depth_mm.len()).filter(|&i| depth_mm[i] == 0).map(|i| (i % COLOR.width as usize, i / COLOR.width as usize)).collect()
    }

    #[test]
    fn a_box_of_the_keyframes_own_time_cuts_exactly_its_colour_pixels_whatever_the_baseline() {
        let mut people = RecentPeople::new(EXACT);
        people.add(&poses(1_000, &[PERSON]), &calibration()).unwrap();
        let mut depth = vec![1500u16; 200];
        let cut = people.cut_out(CameraNanos(1_000), &COLOR, &mut depth);
        assert_eq!(cut, PeopleCut::Cut { people: 1, pixels: 24, age_ns: 0 });
        let expected: Vec<(usize, usize)> = (2..8).flat_map(|y| (4..8).map(move |x| (x, y))).collect();
        assert_eq!(zeroed(&depth), expected);
    }

    #[test]
    fn the_nearest_poses_in_time_are_used_and_grown_by_their_age() {
        let mut people = RecentPeople::new(EXACT);
        let elsewhere = PixelBox { x_px: 14.0, ..PERSON };
        people.add(&poses(0, &[elsewhere]), &calibration()).unwrap();
        people.add(&poses(33_000_000, &[PERSON]), &calibration()).unwrap();
        people.add(&poses(66_000_000, &[elsewhere]), &calibration()).unwrap();
        let mut depth = vec![1500u16; 200];
        // 10 ms at 1.5 m/s over a half metre wide person grows each side by 3% of the box: under a pixel, rounded out.
        let cut = people.cut_out(CameraNanos(43_000_000), &COLOR, &mut depth);
        assert_eq!(cut, PeopleCut::Cut { people: 1, pixels: 6 * 8, age_ns: 10_000_000 });
        assert!(zeroed(&depth).iter().all(|&(x, y)| (3..9).contains(&x) && (1..9).contains(&y)));
    }

    #[test]
    fn a_box_reaching_past_the_image_is_clipped_and_nobody_cuts_nothing() {
        let mut people = RecentPeople::new(MaskSettings::DEFAULT);
        let at_the_edge = PixelBox { x_px: 17.0, y_px: -2.0, w_px: 6.0, h_px: 4.0 };
        people.add(&poses(0, &[at_the_edge]), &calibration()).unwrap();
        let mut depth = vec![1500u16; 200];
        let PeopleCut::Cut { pixels, .. } = people.cut_out(CameraNanos(0), &COLOR, &mut depth) else { panic!("poses at its time") };
        assert_eq!(pixels, zeroed(&depth).len());
        assert!(zeroed(&depth).iter().all(|&(x, y)| x >= 16 && y <= 2));
        people.add(&poses(1, &[]), &calibration()).unwrap();
        let mut depth = vec![1500u16; 200];
        assert_eq!(people.cut_out(CameraNanos(1), &COLOR, &mut depth), PeopleCut::Cut { people: 0, pixels: 0, age_ns: 0 });
        assert!(zeroed(&depth).is_empty());
    }

    #[test]
    fn poses_too_far_from_the_keyframe_cut_nothing_and_say_so() {
        let mut people = RecentPeople::new(MaskSettings::DEFAULT);
        people.add(&poses(0, &[PERSON]), &calibration()).unwrap();
        let late = MaskSettings::DEFAULT.max_age_ns + 1;
        let mut depth = vec![1500u16; 200];
        assert_eq!(people.cut_out(CameraNanos(late), &COLOR, &mut depth), PeopleCut::NoPosesNear { nearest_age_ns: late });
        assert!(zeroed(&depth).is_empty());
    }

    #[test]
    fn older_poses_than_the_newest_are_let_go_and_another_image_size_is_refused() {
        let mut people = RecentPeople::new(EXACT);
        assert!(!people.have_reached(CameraNanos(0)));
        people.add(&poses(2_000, &[]), &calibration()).unwrap();
        people.add(&poses(1_000, &[PERSON]), &calibration()).unwrap();
        assert_eq!(people.newest(), Some(CameraNanos(2_000)));
        assert!(people.have_reached(CameraNanos(2_000)) && !people.have_reached(CameraNanos(2_001)));
        let mut depth = vec![1500u16; 200];
        assert_eq!(people.cut_out(CameraNanos(1_000), &COLOR, &mut depth), PeopleCut::Cut { people: 0, pixels: 0, age_ns: -1_000 });
        let wrong = PersonPoses { width: 640, ..poses(3_000, &[]) };
        assert_eq!(people.add(&wrong, &calibration()), Err(WrongImageSize { width: 640, height: 10, color_width: 20, color_height: 10 }));
    }
}
