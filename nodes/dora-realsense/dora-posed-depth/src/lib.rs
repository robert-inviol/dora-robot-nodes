//! Posed depth frames: every few depth frames, downsampled for mapping and joined to the odometry pose with exactly
//! the same camera time, the pose carried from the body (the IMU) to the depth camera by the camera's extrinsics.
//!
//! The downsample halves each side by keeping, in each 2x2 block of pixels, the nearest valid reading. Every pixel
//! out is a reading the sensor made, never an average, so no depth is made up across a depth edge; the nearest
//! reading keeps thin and near obstacles that plain decimation would drop three times in four, at the cost of
//! widening the near side of an edge by at most one full-resolution pixel; and a block with any reading has one,
//! so holes shrink. A block with no reading stays 0.
//!
//! Each valid reading is then rounded to the nearest multiple of a depth step. The lowest bits of D455 depth are
//! sensor noise, which does not compress; rounding them away shrinks the compressed message, and a step of a few
//! millimetres is far below what a 5 cm voxel map resolves. Rounding never turns a reading into 0 or past `u16::MAX`.
//!
//! People are cut out of each chosen frame before it is downsampled, by the odometry's rule (its `people_mask`): the
//! newest boxes the pose model found, grown by a margin and by how far a person walks in their age, read as no
//! reading. People are drawn from their skeletons and are not part of the map; a pixel without a reading neither
//! adds surface to the map nor carves it free.
//!
//! A chosen depth frame waits for its pose; a frame whose pose never comes (the odometry skipped it) is dropped, never
//! sent with another frame's pose. Each frame carries the odom frame of the pose it was joined to, so when odometry
//! starts again the frames that follow are in the new odom frame. Odometry whose camera time or odom frame start
//! goes back, and a camera calibration that changes, stop the node.

use std::collections::VecDeque;
use std::num::NonZeroU16;

use glam::{Mat3, Quat};
use perception_wire::camera::{CameraCalibration, StreamIntrinsics};
use perception_wire::localization::{CameraNanos, FrameId};
use perception_wire::odometry::OdomPose;
use perception_wire::people::PersonPoses;
use dora_depth_inertial_odometry::people_mask::{cut_out, MaskSettings, PeopleMask, WrongImageSize};
use perception_wire::posed_depth::PosedDepthFrame;

/// What to send.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosedDepthSettings {
    /// Every this-many-th depth frame is sent, by frame id: 3 of the camera's 30 frames a second is 10 a second.
    pub frame_stride: u64,
    /// Chosen frames waiting for their pose beyond this many drop the oldest.
    pub max_waiting: usize,
    /// Odometry poses kept for depth frames that arrive after their pose.
    pub poses_kept: usize,
    /// Every valid depth reading sent is rounded to the nearest multiple of this; 1 sends readings as the camera made
    /// them.
    pub depth_step: DepthStep,
    /// How the people the pose model found are cut out of the depth: the odometry's own settings.
    pub people: MaskSettings,
}

impl PosedDepthSettings {
    pub const DEFAULT: PosedDepthSettings =
        PosedDepthSettings { frame_stride: 3, max_waiting: 8, poses_kept: 8, depth_step: DepthStep::DEFAULT, people: MaskSettings::DEFAULT };
}

/// The step, in millimetres, that depth readings are rounded to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthStep(NonZeroU16);

impl DepthStep {
    pub const DEFAULT: DepthStep = DepthStep::millimetres(2);

    /// A step of `mm` millimetres; zero is no step and does not compile in a constant.
    pub const fn millimetres(mm: u16) -> DepthStep {
        match NonZeroU16::new(mm) {
            Some(step) => DepthStep(step),
            None => panic!("a depth step is at least 1 mm"),
        }
    }

    /// A reading rounded to the nearest multiple of the step, halves up. 0, no reading, stays 0; a valid reading stays
    /// valid, at least one step, and is never rounded past the largest multiple that fits a `u16`.
    pub fn round(self, depth_mm: u16) -> u16 {
        if depth_mm == 0 {
            return 0;
        }
        let step = u32::from(self.0.get());
        let rounded = (u32::from(depth_mm) + step / 2) / step * step;
        let largest = u32::from(u16::MAX) / step * step;
        rounded.clamp(step, largest) as u16
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum PosedDepthError {
    #[error("depth is {width}x{height}; halving it needs an even width and height")]
    OddSize { width: usize, height: usize },
    #[error("depth frame {frame_id} has {values} values for {width}x{height}")]
    WrongLength { frame_id: u64, values: usize, width: usize, height: usize },
    #[error("depth frame {frame_id} is {width}x{height}, but the calibration's depth stream is {calibrated_width}x{calibrated_height}")]
    NotTheCalibratedSize { frame_id: u64, width: usize, height: usize, calibrated_width: u16, calibrated_height: u16 },
    #[error("the camera calibration changed while running")]
    CalibrationChanged,
    #[error("an odom pose at camera time {received} ns is not later than the last, at {last} ns")]
    OdometryWentBack { last: i64, received: i64 },
    #[error("odometry started at camera time {received} ns, before the odom frame in use, started at {current} ns")]
    OdomStartWentBack { current: i64, received: i64 },
    #[error("the odom pose at camera time {camera_ns} ns is for depth frame {pose_frame}, not depth frame {depth_frame}")]
    PoseOfAnotherFrame { camera_ns: i64, pose_frame: u64, depth_frame: u64 },
    #[error(transparent)]
    PersonPoses(#[from] WrongImageSize),
}

/// Counts of what happened to depth frames and poses, for the node's progress line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    pub sent: u32,
    pub not_chosen: u32,
    pub before_calibration: u32,
    /// Chosen frames whose pose never came.
    pub without_pose: u32,
    /// Chosen frames pushed out of the wait for their pose.
    pub dropped_waiting: u32,
    pub odometry_restarts: u32,
    /// Chosen frames with people cut out of them.
    pub with_people: u32,
}

/// The pinhole model of an image halved each way: a downsampled pixel's centre is the centre of its 2x2 block.
pub fn halved_intrinsics(k: &StreamIntrinsics) -> StreamIntrinsics {
    StreamIntrinsics { width: k.width / 2, height: k.height / 2, fx: k.fx / 2.0, fy: k.fy / 2.0, ppx: (k.ppx - 0.5) / 2.0, ppy: (k.ppy - 0.5) / 2.0 }
}

/// Depth halved each way, row major, each pixel the nearest valid reading of its 2x2 block, 0 where it has none.
pub fn downsample_nearest(width: usize, height: usize, depth_mm: &[u16]) -> Result<Vec<u16>, PosedDepthError> {
    if width % 2 != 0 || height % 2 != 0 {
        return Err(PosedDepthError::OddSize { width, height });
    }
    let (half_width, half_height) = (width / 2, height / 2);
    let mut out = Vec::with_capacity(half_width * half_height);
    for row in 0..half_height {
        let (upper, lower) = (&depth_mm[2 * row * width..][..width], &depth_mm[(2 * row + 1) * width..][..width]);
        for column in 0..half_width {
            let block = [upper[2 * column], upper[2 * column + 1], lower[2 * column], lower[2 * column + 1]];
            out.push(block.into_iter().filter(|&mm| mm != 0).min().unwrap_or(0));
        }
    }
    Ok(out)
}

/// A chosen depth frame, downsampled, waiting for its pose.
struct HeldDepth {
    frame_id: u64,
    camera_ns: i64,
    depth_mm: Vec<u16>,
}

pub struct PosedDepthJoiner {
    settings: PosedDepthSettings,
    calibration: Option<CameraCalibration>,
    /// Oldest first.
    waiting: VecDeque<HeldDepth>,
    /// The latest odom poses, oldest first, all of one odom frame.
    poses: VecDeque<OdomPose>,
    people: PeopleMask,
    /// A chosen frame with people cut out of it, before it is downsampled.
    cut_depth_mm: Vec<u16>,
    tally: Tally,
}

impl PosedDepthJoiner {
    pub fn new(settings: PosedDepthSettings) -> PosedDepthJoiner {
        PosedDepthJoiner {
            settings,
            calibration: None,
            waiting: VecDeque::new(),
            poses: VecDeque::new(),
            people: PeopleMask::default(),
            cut_depth_mm: Vec::new(),
            tally: Tally::default(),
        }
    }

    /// The people the pose model found in a colour frame; before the calibration there is no depth to cut them out
    /// of, and they are let go, as the odometry does.
    pub fn on_person_poses(&mut self, poses: &PersonPoses) -> Result<(), PosedDepthError> {
        let Some(calibration) = &self.calibration else { return Ok(()) };
        Ok(self.people.update(poses, calibration)?)
    }

    /// The counts since the last call.
    pub fn take_tally(&mut self) -> Tally {
        std::mem::take(&mut self.tally)
    }

    pub fn on_calibration(&mut self, calibration: CameraCalibration) -> Result<(), PosedDepthError> {
        match self.calibration {
            Some(known) if known != calibration => Err(PosedDepthError::CalibrationChanged),
            _ => {
                self.calibration = Some(calibration);
                Ok(())
            }
        }
    }

    /// A depth frame from the camera; returns it posed when it is chosen and its pose has already arrived.
    pub fn on_depth(&mut self, frame_id: u64, camera_ns: i64, width: usize, height: usize, depth_mm: &[u16]) -> Result<Vec<PosedDepthFrame>, PosedDepthError> {
        if depth_mm.len() != width * height {
            return Err(PosedDepthError::WrongLength { frame_id, values: depth_mm.len(), width, height });
        }
        if frame_id % self.settings.frame_stride != 0 {
            self.tally.not_chosen += 1;
            return Ok(Vec::new());
        }
        let Some(calibration) = self.calibration else {
            self.tally.before_calibration += 1;
            return Ok(Vec::new());
        };
        let depth = calibration.depth;
        if (width, height) != (depth.width as usize, depth.height as usize) {
            return Err(PosedDepthError::NotTheCalibratedSize { frame_id, width, height, calibrated_width: depth.width, calibrated_height: depth.height });
        }
        let people = self.people.rects_for(CameraNanos(camera_ns), &calibration, &self.settings.people);
        self.tally.with_people += u32::from(!people.is_empty());
        let depth_mm = cut_out(depth_mm, width, people, &mut self.cut_depth_mm);
        let mut downsampled = downsample_nearest(width, height, depth_mm)?;
        downsampled.iter_mut().for_each(|mm| *mm = self.settings.depth_step.round(*mm));
        let held = HeldDepth { frame_id, camera_ns, depth_mm: downsampled };
        if let Some(pose) = self.poses.iter().find(|pose| pose.camera_ns == camera_ns).copied() {
            return Ok(vec![self.posed(held, &pose, &calibration)?]);
        }
        if self.poses.back().is_some_and(|newest| newest.camera_ns > camera_ns) {
            self.tally.without_pose += 1;
            return Ok(Vec::new());
        }
        self.waiting.push_back(held);
        if self.waiting.len() > self.settings.max_waiting {
            self.waiting.pop_front();
            self.tally.dropped_waiting += 1;
        }
        Ok(Vec::new())
    }

    /// An odometry pose; returns the waiting frame it poses, if any.
    pub fn on_odom_pose(&mut self, pose: OdomPose) -> Result<Vec<PosedDepthFrame>, PosedDepthError> {
        if let Some(last) = self.poses.back() {
            if pose.camera_ns <= last.camera_ns {
                return Err(PosedDepthError::OdometryWentBack { last: last.camera_ns, received: pose.camera_ns });
            }
            if pose.odom_started_camera_ns < last.odom_started_camera_ns {
                return Err(PosedDepthError::OdomStartWentBack { current: last.odom_started_camera_ns, received: pose.odom_started_camera_ns });
            }
            if pose.odom_started_camera_ns > last.odom_started_camera_ns {
                self.poses.clear();
                self.tally.odometry_restarts += 1;
            }
        }
        self.poses.push_back(pose);
        while self.poses.len() > self.settings.poses_kept {
            self.poses.pop_front();
        }
        let mut posed = Vec::new();
        while let Some(held) = self.waiting.front() {
            if held.camera_ns > pose.camera_ns {
                break;
            }
            let held = self.waiting.pop_front().expect("a waiting frame");
            if held.camera_ns < pose.camera_ns {
                self.tally.without_pose += 1;
                continue;
            }
            let calibration = self.calibration.expect("a frame waits only once the calibration is known");
            posed.push(self.posed(held, &pose, &calibration)?);
        }
        Ok(posed)
    }

    fn posed(&mut self, held: HeldDepth, pose: &OdomPose, calibration: &CameraCalibration) -> Result<PosedDepthFrame, PosedDepthError> {
        if pose.depth_frame_id != held.frame_id {
            return Err(PosedDepthError::PoseOfAnotherFrame { camera_ns: pose.camera_ns, pose_frame: pose.depth_frame_id, depth_frame: held.frame_id });
        }
        let extrinsics = calibration.depth_to_imu;
        let camera_to_body = Mat3::from_cols_array(&extrinsics.rotation);
        self.tally.sent += 1;
        Ok(PosedDepthFrame {
            frame_id: FrameId(held.frame_id),
            camera_ns: CameraNanos(held.camera_ns),
            odom_started: CameraNanos(pose.odom_started_camera_ns),
            intrinsics: halved_intrinsics(&calibration.depth),
            camera_position_m: pose.position_m + pose.orientation * glam::Vec3::from_array(extrinsics.translation_m),
            camera_orientation: (pose.orientation * Quat::from_mat3(&camera_to_body)).normalize(),
            depth_mm: held.depth_mm.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;
    use perception_wire::camera::RigidTransform;

    use super::*;

    const FRAME_NS: i64 = 33_333_333;

    /// A 4x2 depth stream; the depth camera 2 cm right of the IMU, axes shared.
    fn calibration() -> CameraCalibration {
        let intrinsics = StreamIntrinsics { width: 4, height: 2, fx: 4.0, fy: 4.0, ppx: 1.5, ppy: 0.5 };
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        CameraCalibration {
            color: intrinsics,
            depth: intrinsics,
            depth_to_color: RigidTransform { rotation: identity, translation_m: [0.0; 3] },
            depth_to_imu: RigidTransform { rotation: identity, translation_m: [0.02, 0.0, 0.0] },
        }
    }

    fn pose(frame_id: u64, odom_started_camera_ns: i64) -> OdomPose {
        OdomPose {
            camera_ns: frame_id as i64 * FRAME_NS,
            odom_started_camera_ns,
            depth_frame_id: frame_id,
            position_m: Vec3::new(1.0, 2.0, 0.5),
            // A quarter turn about odom z: body x is odom y.
            orientation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            velocity_m_s: Vec3::ZERO,
            position_sigma_m: Vec3::splat(0.01),
            tilt_sigma_rad: 0.01,
            heading_sigma_rad: 0.01,
            corrected: true,
            matched_points: 100,
            degenerate: None,
        }
    }

    fn joiner() -> PosedDepthJoiner {
        let mut joiner = PosedDepthJoiner::new(PosedDepthSettings::DEFAULT);
        joiner.on_calibration(calibration()).unwrap();
        joiner
    }

    fn depth(joiner: &mut PosedDepthJoiner, frame_id: u64) -> Vec<PosedDepthFrame> {
        joiner.on_depth(frame_id, frame_id as i64 * FRAME_NS, 4, 2, &[1000, 2000, 0, 0, 3000, 0, 0, 1500]).unwrap()
    }

    #[test]
    fn the_downsample_keeps_the_nearest_reading_across_a_depth_edge_and_never_averages() {
        let depth = [
            1000, 3000, 0, 0, //
            3000, 3000, 0, 2500, //
            0, 0, 4000, 4010, //
            0, 0, 4020, 4030,
        ];
        assert_eq!(downsample_nearest(4, 4, &depth).unwrap(), vec![1000, 2500, 0, 4000]);
        assert_eq!(downsample_nearest(3, 2, &[0; 6]), Err(PosedDepthError::OddSize { width: 3, height: 2 }));
    }

    #[test]
    fn a_depth_step_rounds_to_its_nearest_multiple_and_keeps_no_reading_and_every_valid_one() {
        let step = DepthStep::millimetres(2);
        assert_eq!([0, 1, 2, 3, 1001, 1002, 65534, 65535].map(|mm| step.round(mm)), [0, 2, 2, 4, 1002, 1002, 65534, 65534]);
        let coarse = DepthStep::millimetres(4);
        assert_eq!([0, 1, 2, 5, 6, 65535].map(|mm| coarse.round(mm)), [0, 4, 4, 4, 8, 65532], "never 0 and never past u16::MAX");
        assert_eq!([1, 1001, 65535].map(|mm| DepthStep::millimetres(1).round(mm)), [1, 1001, 65535], "a 1 mm step keeps every reading");
    }

    #[test]
    fn a_posed_depth_frame_carries_its_readings_rounded_to_the_step() {
        let settings = PosedDepthSettings { depth_step: DepthStep::millimetres(4), ..PosedDepthSettings::DEFAULT };
        let mut joiner = PosedDepthJoiner::new(settings);
        joiner.on_calibration(calibration()).unwrap();
        joiner.on_odom_pose(pose(3, 0)).unwrap();
        let sent = joiner.on_depth(3, 3 * FRAME_NS, 4, 2, &[1001, 2000, 0, 0, 3000, 0, 0, 1503]).unwrap();
        assert_eq!(&sent[0].depth_mm[..], &[1000, 1504]);
    }

    #[test]
    fn halved_intrinsics_put_a_pixel_where_its_block_was() {
        let full = StreamIntrinsics { width: 848, height: 480, fx: 424.0, fy: 424.0, ppx: 423.5, ppy: 239.5 };
        let half = halved_intrinsics(&full);
        assert_eq!((half.width, half.height, half.fx), (424, 240, 212.0));
        // The point full-resolution pixels (10, 20) and (11, 21) share a block with lands on its centre.
        let (x, y) = (10.5, 20.5);
        let point = full.deproject(x, y, 2.0);
        let (column, row) = half.project(point).unwrap();
        assert!((column - 5.0).abs() < 1e-4 && (row - 10.0).abs() < 1e-4, "{column} {row}");
    }

    #[test]
    fn every_third_frame_is_sent_once_its_exact_pose_arrives() {
        let mut joiner = joiner();
        let mut sent = Vec::new();
        for frame_id in 0..9 {
            sent.extend(depth(&mut joiner, frame_id));
            sent.extend(joiner.on_odom_pose(pose(frame_id, 0)).unwrap());
        }
        assert_eq!(sent.iter().map(|f| f.frame_id.0).collect::<Vec<_>>(), vec![0, 3, 6]);
        let frame = &sent[1];
        assert_eq!(frame.camera_ns, CameraNanos(3 * FRAME_NS));
        assert_eq!((frame.intrinsics.width, frame.intrinsics.height), (2, 1));
        assert_eq!(&frame.depth_mm[..], &[1000, 1500]);
        // 2 cm along body x, which the quarter turn makes odom y.
        assert!(frame.camera_position_m.abs_diff_eq(Vec3::new(1.0, 2.02, 0.5), 1e-6), "{}", frame.camera_position_m);
        assert!(frame.camera_orientation.abs_diff_eq(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2), 1e-6));
        assert_eq!(joiner.take_tally(), Tally { sent: 3, not_chosen: 6, ..Tally::default() });
    }

    #[test]
    fn a_frame_whose_pose_was_skipped_is_never_sent_with_another_pose() {
        let mut joiner = joiner();
        assert!(depth(&mut joiner, 3).is_empty());
        assert!(joiner.on_odom_pose(pose(2, 0)).unwrap().is_empty(), "an earlier pose is not its pose");
        assert!(joiner.on_odom_pose(pose(4, 0)).unwrap().is_empty(), "frame 3's pose never came");
        assert!(depth(&mut joiner, 6).is_empty());
        assert_eq!(joiner.on_odom_pose(pose(6, 0)).unwrap().len(), 1);
        assert!(depth(&mut joiner, 0).is_empty(), "older than the poses: its pose has gone by");
        let tally = joiner.take_tally();
        assert_eq!((tally.sent, tally.without_pose), (1, 2));
    }

    #[test]
    fn a_pose_that_arrives_before_its_depth_frame_still_poses_it() {
        let mut joiner = joiner();
        joiner.on_odom_pose(pose(3, 0)).unwrap();
        assert_eq!(depth(&mut joiner, 3).len(), 1);
    }

    #[test]
    fn frames_after_an_odometry_restart_are_in_the_new_odom_frame() {
        let mut joiner = joiner();
        depth(&mut joiner, 3);
        assert_eq!(joiner.on_odom_pose(pose(3, 0)).unwrap()[0].odom_started, CameraNanos(0));
        depth(&mut joiner, 6);
        let restarted = 5 * FRAME_NS;
        assert_eq!(joiner.on_odom_pose(pose(6, restarted)).unwrap()[0].odom_started, CameraNanos(restarted));
        assert_eq!(joiner.take_tally().odometry_restarts, 1);
        assert_eq!(
            joiner.on_odom_pose(pose(7, 0)).unwrap_err(),
            PosedDepthError::OdomStartWentBack { current: restarted, received: 0 }
        );
    }

    #[test]
    fn odometry_going_back_a_pose_of_another_frame_and_a_changed_calibration_stop_the_node() {
        let mut joiner = joiner();
        joiner.on_odom_pose(pose(4, 0)).unwrap();
        assert!(matches!(joiner.on_odom_pose(pose(4, 0)), Err(PosedDepthError::OdometryWentBack { .. })));
        depth(&mut joiner, 6);
        let mismatched = OdomPose { depth_frame_id: 7, ..pose(6, 0) };
        assert!(matches!(joiner.on_odom_pose(mismatched), Err(PosedDepthError::PoseOfAnotherFrame { .. })));
        let moved = CameraCalibration { depth_to_imu: RigidTransform { rotation: calibration().depth_to_imu.rotation, translation_m: [0.0; 3] }, ..calibration() };
        assert_eq!(joiner.on_calibration(moved), Err(PosedDepthError::CalibrationChanged));
    }

    #[test]
    fn frames_before_the_calibration_are_not_sent() {
        let mut joiner = PosedDepthJoiner::new(PosedDepthSettings::DEFAULT);
        joiner.on_odom_pose(pose(3, 0)).unwrap();
        assert!(depth(&mut joiner, 3).is_empty());
        assert_eq!(joiner.take_tally().before_calibration, 1);
    }

    fn person_in_the_left_half(captured_camera_ns: i64) -> PersonPoses {
        use perception_wire::people::{Keypoint, PersonPose, PixelBox};
        let bbox = PixelBox { x_px: 0.0, y_px: 0.0, w_px: 1.0, h_px: 2.0 };
        let person = PersonPose { track_id: Some(1), score: 0.9, bbox, keypoints: vec![Keypoint { x_px: 0.5, y_px: 1.0, score: 0.9 }; 17] };
        PersonPoses { frame_id: 3, captured_camera_ns, width: 4, height: 2, people: vec![person] }
    }

    #[test]
    fn people_are_cut_out_as_no_reading_and_boxes_too_old_cut_nothing() {
        let mut joiner = joiner();
        joiner.on_person_poses(&person_in_the_left_half(3 * FRAME_NS)).unwrap();
        depth(&mut joiner, 3);
        let sent = joiner.on_odom_pose(pose(3, 0)).unwrap();
        // The box, grown by its margin, covers the two left columns: the left 2x2 block has no reading left.
        assert_eq!(&sent[0].depth_mm[..], &[0, 1500]);
        assert_eq!(joiner.take_tally().with_people, 1);
        // Frame 30 is a second after the box: too old to stand for it.
        depth(&mut joiner, 30);
        let sent = joiner.on_odom_pose(pose(30, 0)).unwrap();
        assert_eq!(&sent[0].depth_mm[..], &[1000, 1500]);
    }

    #[test]
    fn people_before_the_calibration_are_let_go_and_from_another_image_size_stop_the_node() {
        let mut joiner = PosedDepthJoiner::new(PosedDepthSettings::DEFAULT);
        assert_eq!(joiner.on_person_poses(&person_in_the_left_half(0)), Ok(()));
        joiner.on_calibration(calibration()).unwrap();
        let wrong = PersonPoses { width: 8, ..person_in_the_left_half(0) };
        assert!(matches!(joiner.on_person_poses(&wrong), Err(PosedDepthError::PersonPoses(_))));
    }
}
