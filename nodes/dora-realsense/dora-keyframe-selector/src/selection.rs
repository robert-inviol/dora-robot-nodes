//! Selecting keyframes from a stream of camera frames: which depth frames become keyframes, what is held while their
//! colour is fetched, and which IMU samples each keyframe carries.
//!
//! A keyframe's imu message carries every sample since the previous keyframe's, up to and including the first gyro
//! sample at or after the keyframe's time, so the gyro covers the keyframe's whole span. That sample usually arrives
//! with the next frame's imu message, so a keyframe goes out once its colour and that sample have both arrived, and
//! person poses at or after its time (see `people`): people are cut out of its aligned depth.

use std::collections::BTreeMap;
use std::time::Duration;

use depth_perception::CameraIntrinsics;
use glam::Vec3;
use dora_depth_inertial_odometry::people_mask::{MaskSettings, WrongImageSize};
use perception_wire::camera::CameraCalibration;
use perception_wire::localization::CameraNanos;
use perception_wire::odometry::OdomPose;
use perception_wire::people::PersonPoses;
use perception_wire::{Imu, Keyframe};
use rgbd_keyframes::{KeyframeSelector, KeyframeSettings};

use crate::align::align_depth_to_color;
use crate::people::{PeopleCut, RecentPeople};

/// Selected keyframes held beyond this many are an error: the pose model has fallen behind or stopped, and no person
/// poses at their time have come to cut people out of them.
pub const MOST_HELD: usize = 8;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SelectionError {
    #[error("{held} selected keyframes are waiting, the oldest (frame {oldest_frame_id}, camera time {oldest_camera_ns} ns) for person poses at or after its time; the newest person poses are from camera time {newest_poses_camera_ns} ns: the pose model has fallen behind or stopped")]
    PeopleBehind { held: usize, oldest_frame_id: u64, oldest_camera_ns: i64, newest_poses_camera_ns: i64 },
    #[error("{held} selected keyframes are waiting, the oldest (frame {oldest_frame_id}) for its colour or for the gyro to pass its time")]
    TooManyHeld { held: usize, oldest_frame_id: u64 },
    #[error(transparent)]
    PersonPoses(#[from] WrongImageSize),
}

/// A selected depth frame waiting for its colour and for the gyro to pass its time.
struct Held {
    camera_ns: i64,
    captured_unix_ns: i64,
    specific_force: Vec3,
    depth_mm: Vec<u16>,
    color_jpeg: Option<Vec<u8>>,
}

/// One selected frame, ready to send: its imu message goes out before the keyframe.
pub struct SelectedKeyframe {
    pub imu: Imu,
    pub keyframe: Keyframe,
    /// What was cut out of the keyframe's depth.
    pub people: PeopleCut,
}

/// IMU samples not yet sent with a keyframe.
#[derive(Default)]
struct Samples {
    accel: Vec<(i64, Vec3)>,
    gyro: Vec<(i64, Vec3)>,
}

/// Where the camera is in one odom frame, against where it was when the last keyframe was taken.
struct Travel {
    odom_started_camera_ns: i64,
    at_last_keyframe_m: Vec3,
    latest_m: Vec3,
}

pub struct KeyframeAssembler {
    selector: KeyframeSelector,
    travel: Option<Travel>,
    first_camera_ns: Option<i64>,
    last_gyro_ns: Option<i64>,
    samples: Samples,
    held: BTreeMap<u64, Held>,
    people: RecentPeople,
}

/// Takes the samples at or before `until_ns` out of `samples`, in time order.
fn take_until(samples: &mut Vec<(i64, Vec3)>, until_ns: i64) -> (Vec<i64>, Vec<Vec3>) {
    let later = samples.split_off(samples.partition_point(|(t, _)| *t <= until_ns));
    std::mem::replace(samples, later).into_iter().unzip()
}

impl KeyframeAssembler {
    pub fn new(settings: KeyframeSettings) -> KeyframeAssembler {
        KeyframeAssembler {
            selector: KeyframeSelector::new(settings),
            travel: None,
            first_camera_ns: None,
            last_gyro_ns: None,
            samples: Samples::default(),
            held: BTreeMap::new(),
            people: RecentPeople::new(MaskSettings::DEFAULT),
        }
    }

    /// Adds the people the pose model found in one colour frame.
    pub fn add_person_poses(&mut self, poses: &PersonPoses, calibration: &CameraCalibration) -> Result<(), SelectionError> {
        Ok(self.people.add(poses, calibration)?)
    }

    /// Adds one frame's imu message from the camera.
    pub fn add_imu(&mut self, imu: &Imu) {
        for (&t, &sample) in imu.gyro_camera_ns.iter().zip(&imu.gyro_rad_s) {
            let held = self.last_gyro_ns.map_or(Duration::ZERO, |last| Duration::from_nanos((t - last).max(0) as u64));
            self.selector.add_rotation(sample, held);
            self.last_gyro_ns = Some(t);
            self.samples.gyro.push((t, sample));
        }
        self.samples.accel.extend(imu.accel_camera_ns.iter().copied().zip(imu.accel_m_s2.iter().copied()));
    }

    /// Adds the odometry's latest pose, so that travel calls for a keyframe. A pose in a new odom frame starts the
    /// travel again from there: positions in different odom frames do not compare.
    pub fn add_odom_pose(&mut self, pose: &OdomPose) {
        match self.travel.as_mut() {
            Some(travel) if travel.odom_started_camera_ns == pose.odom_started_camera_ns => travel.latest_m = pose.position_m,
            _ => {
                self.travel = Some(Travel {
                    odom_started_camera_ns: pose.odom_started_camera_ns,
                    at_last_keyframe_m: pose.position_m,
                    latest_m: pose.position_m,
                })
            }
        }
        let travel = self.travel.as_ref().expect("just set");
        self.selector.set_travel(travel.latest_m.distance(travel.at_last_keyframe_m));
    }

    /// Adds one depth frame. Returns true when it is selected, and its colour should be fetched. Frames before the
    /// camera has a gravity estimate, or before the pose model's first person poses, are never selected.
    pub fn add_depth(&mut self, frame_id: u64, camera_ns: i64, captured_unix_ns: i64, specific_force: Option<Vec3>, depth_mm: &[u16]) -> bool {
        let first = *self.first_camera_ns.get_or_insert(camera_ns);
        let Some(specific_force) = specific_force else { return false };
        if self.people.newest().is_none() {
            return false;
        }
        if !self.selector.take(Duration::from_nanos((camera_ns - first).max(0) as u64)) {
            return false;
        }
        if let Some(travel) = self.travel.as_mut() {
            travel.at_last_keyframe_m = travel.latest_m;
        }
        self.held.insert(frame_id, Held { camera_ns, captured_unix_ns, specific_force, depth_mm: depth_mm.to_vec(), color_jpeg: None });
        true
    }

    /// Whether `frame_id` is selected and waiting for its colour.
    pub fn is_waiting_for(&self, frame_id: u64) -> bool {
        self.held.get(&frame_id).is_some_and(|held| held.color_jpeg.is_none())
    }

    /// Gives a selected frame its colour image, encoded as `color_jpeg`. Returns false for a frame not waiting for one.
    pub fn add_color(&mut self, frame_id: u64, color_jpeg: Vec<u8>) -> bool {
        match self.held.get_mut(&frame_id) {
            Some(held) if held.color_jpeg.is_none() => {
                held.color_jpeg = Some(color_jpeg);
                true
            }
            _ => false,
        }
    }

    /// The selected frames now complete, oldest first: each has its colour, the gyro has reached its time, and person
    /// poses at or after its time have arrived. More than `MOST_HELD` frames still waiting is an error.
    pub fn take_ready(&mut self, calibration: &CameraCalibration) -> Result<Vec<SelectedKeyframe>, SelectionError> {
        let mut ready = Vec::new();
        while let Some(entry) = self.held.first_entry() {
            let held = entry.get();
            let Some(&(covered_to_ns, _)) = self.samples.gyro.iter().find(|(t, _)| *t >= held.camera_ns) else { break };
            if held.color_jpeg.is_none() || !self.people.have_reached(CameraNanos(held.camera_ns)) {
                break;
            }
            let (frame_id, held) = entry.remove_entry();
            let (accel_camera_ns, accel_m_s2) = take_until(&mut self.samples.accel, covered_to_ns);
            let (gyro_camera_ns, gyro_rad_s) = take_until(&mut self.samples.gyro, covered_to_ns);
            let imu = Imu { frame_id, keyframe_camera_ns: held.camera_ns, accel_camera_ns, accel_m_s2, gyro_camera_ns, gyro_rad_s };
            let color = calibration.color;
            let mut depth_mm = align_depth_to_color(&held.depth_mm, calibration);
            let people = self.people.cut_out(CameraNanos(held.camera_ns), &color, &mut depth_mm);
            let keyframe = Keyframe {
                frame_id,
                captured_unix_ns: held.captured_unix_ns,
                width: color.width as usize,
                height: color.height as usize,
                intrinsics: CameraIntrinsics { fx: color.fx, fy: color.fy, ppx: color.ppx, ppy: color.ppy },
                specific_force: held.specific_force,
                depth_mm: depth_mm.into(),
                color_jpeg: held.color_jpeg.expect("checked above"),
            };
            ready.push(SelectedKeyframe { imu, keyframe, people });
        }
        if let Some((&oldest_frame_id, oldest)) = self.held.first_key_value().filter(|_| self.held.len() > MOST_HELD) {
            let held = self.held.len();
            return Err(match self.people.newest() {
                Some(newest) if newest.0 < oldest.camera_ns => {
                    SelectionError::PeopleBehind { held, oldest_frame_id, oldest_camera_ns: oldest.camera_ns, newest_poses_camera_ns: newest.0 }
                }
                _ => SelectionError::TooManyHeld { held, oldest_frame_id },
            });
        }
        Ok(ready)
    }
}

#[cfg(test)]
mod tests {
    use perception_wire::camera::{RigidTransform, StreamIntrinsics};
    use perception_wire::people::{PersonPose, PixelBox};

    use super::*;

    const FRAME_NS: i64 = 33_333_333;

    fn calibration() -> CameraCalibration {
        let identity = RigidTransform { rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], translation_m: [0.0; 3] };
        let intrinsics = StreamIntrinsics { width: 4, height: 3, fx: 4.0, fy: 4.0, ppx: 1.5, ppy: 1.0 };
        CameraCalibration { color: intrinsics, depth: intrinsics, depth_to_color: identity, depth_to_imu: identity }
    }

    /// The pose model's people in one colour frame of `calibration()`.
    fn poses(captured_camera_ns: i64, boxes: &[PixelBox]) -> PersonPoses {
        let people = boxes.iter().map(|&bbox| PersonPose { track_id: None, score: 0.9, bbox, keypoints: vec![] }).collect();
        PersonPoses { frame_id: 1, captured_camera_ns, width: 4, height: 3, people }
    }

    /// An assembler that has had person poses, with nobody in them, far later than any frame of a test.
    fn assembler() -> KeyframeAssembler {
        let mut assembler = KeyframeAssembler::new(KeyframeSettings::default());
        assembler.add_person_poses(&poses(1_000 * FRAME_NS, &[]), &calibration()).unwrap();
        assembler
    }

    /// The camera's imu message for one frame: one accel and two gyro samples during it, turning at `turn_rad_s`.
    fn imu(frame: u64, turn_rad_s: f32) -> Imu {
        let end = frame as i64 * FRAME_NS;
        Imu {
            frame_id: frame,
            keyframe_camera_ns: end,
            accel_camera_ns: vec![end - FRAME_NS / 2],
            accel_m_s2: vec![Vec3::new(0.0, -9.8, 0.0)],
            gyro_camera_ns: vec![end - FRAME_NS / 2, end],
            gyro_rad_s: vec![Vec3::new(0.0, turn_rad_s, 0.0); 2],
        }
    }

    /// Feeds frames `range` and returns the ones selected.
    fn run(assembler: &mut KeyframeAssembler, frames: std::ops::Range<u64>, turn_rad_s: f32) -> Vec<u64> {
        let depth = [1000u16; 12];
        frames
            .filter(|&frame| {
                assembler.add_imu(&imu(frame, turn_rad_s));
                assembler.add_depth(frame, frame as i64 * FRAME_NS, 0, Some(Vec3::new(0.0, -9.8, 0.0)), &depth)
            })
            .collect()
    }

    /// The odometry's pose at `position_m` in the odom frame started at `odom_started_camera_ns`.
    fn odom_pose(odom_started_camera_ns: i64, position_m: Vec3) -> OdomPose {
        OdomPose {
            camera_ns: 0,
            odom_started_camera_ns,
            depth_frame_id: 0,
            position_m,
            orientation: glam::Quat::IDENTITY,
            velocity_m_s: Vec3::ZERO,
            position_sigma_m: Vec3::ZERO,
            tilt_sigma_rad: 0.0,
            heading_sigma_rad: 0.0,
            corrected: true,
            matched_points: 100,
            degenerate: None,
        }
    }

    /// Feeds frames `range` with the camera moving `step_m` along x each frame, and returns the ones selected.
    fn run_moving(assembler: &mut KeyframeAssembler, frames: std::ops::Range<u64>, odom_started_camera_ns: i64, step_m: f32) -> Vec<u64> {
        let depth = [1000u16; 12];
        frames
            .filter(|&frame| {
                assembler.add_imu(&imu(frame, 0.0));
                assembler.add_odom_pose(&odom_pose(odom_started_camera_ns, Vec3::new(frame as f32 * step_m, 0.0, 0.0)));
                assembler.add_depth(frame, frame as i64 * FRAME_NS, 0, Some(Vec3::new(0.0, -9.8, 0.0)), &depth)
            })
            .collect()
    }

    #[test]
    fn a_moving_camera_gives_a_keyframe_every_ten_centimetres() {
        // 1.1 cm a frame passes 10 cm on the tenth frame after a keyframe.
        assert_eq!(run_moving(&mut assembler(), 0..35, 1, 0.011), vec![0, 10, 20, 30]);
    }

    #[test]
    fn a_new_odom_frame_starts_the_travel_again() {
        let mut assembler = assembler();
        assert_eq!(run_moving(&mut assembler, 0..8, 1, 0.01), vec![0]);
        // The new frame's positions start 5 m away, which is not travel.
        assembler.add_odom_pose(&odom_pose(2, Vec3::new(5.0, 0.0, 0.0)));
        let depth = [1000u16; 12];
        assert!(!assembler.add_depth(8, 8 * FRAME_NS, 0, Some(Vec3::new(0.0, -9.8, 0.0)), &depth));
    }

    #[test]
    fn a_still_camera_gives_a_keyframe_a_second() {
        // 30 frames at 30 fps fall a hair short of a second, so the next keyframe is the 31st frame on.
        let mut assembler = assembler();
        assert_eq!(run(&mut assembler, 0..95, 0.0), vec![0, 31, 62, 93]);
    }

    #[test]
    fn a_turning_camera_gives_them_sooner() {
        // 1 rad/s turns 10 degrees in about 5 frames, but keyframes are never closer than 200 ms: 7 frames.
        let mut assembler = assembler();
        assert_eq!(run(&mut assembler, 0..20, 1.0), vec![0, 7, 14]);
    }

    #[test]
    fn frames_before_gravity_settles_are_not_selected() {
        let mut assembler = assembler();
        assert!(!assembler.add_depth(0, 0, 0, None, &[1000; 12]));
        assert!(!assembler.is_waiting_for(0));
    }

    #[test]
    fn each_keyframe_carries_every_sample_since_the_last_one_and_none_after_it() {
        let mut assembler = assembler();
        assert_eq!(run(&mut assembler, 0..32, 0.0), vec![0, 31]);
        assert!(assembler.add_color(0, vec![1]) && assembler.add_color(31, vec![2]));
        let mut selected = assembler.take_ready(&calibration()).unwrap().into_iter();
        let first = selected.next().unwrap();
        assert_eq!((first.imu.frame_id, first.imu.keyframe_camera_ns), (0, 0));
        assert_eq!(first.imu.gyro_camera_ns, vec![-FRAME_NS / 2, 0]);
        let second = selected.next().unwrap();
        assert_eq!(second.imu.gyro_camera_ns.len(), 62);
        assert_eq!(second.imu.accel_camera_ns.len(), 31);
        assert_eq!(*second.imu.gyro_camera_ns.last().unwrap(), 31 * FRAME_NS);
        assert_eq!(second.keyframe.frame_id, 31);
        assert_eq!(&second.keyframe.depth_mm[..], &[1000u16; 12]);
    }

    #[test]
    fn colour_for_a_frame_not_picked_completes_nothing() {
        let mut assembler = assembler();
        run(&mut assembler, 0..3, 0.0);
        assert!(!assembler.add_color(1, vec![]));
    }

    #[test]
    fn a_keyframe_waits_for_the_gyro_to_pass_its_time_and_carries_that_sample() {
        // The camera's gyro runs a little behind its depth frames: each frame's imu message ends 2 ms before it.
        let late = |frame: u64| {
            let end = frame as i64 * FRAME_NS;
            Imu { frame_id: frame, keyframe_camera_ns: end, gyro_camera_ns: vec![end - 2_000_000], gyro_rad_s: vec![Vec3::ZERO], ..Imu::default() }
        };
        let mut assembler = assembler();
        assembler.add_imu(&late(0));
        assert!(assembler.add_depth(0, 0, 0, Some(Vec3::new(0.0, -9.8, 0.0)), &[1000; 12]));
        assert!(assembler.add_color(0, vec![1]));
        assert!(assembler.take_ready(&calibration()).unwrap().is_empty(), "the gyro has not reached the keyframe yet");
        assembler.add_imu(&late(1));
        let selected = assembler.take_ready(&calibration()).unwrap();
        assert_eq!(selected[0].imu.gyro_camera_ns, vec![-2_000_000, FRAME_NS - 2_000_000]);
    }

    #[test]
    fn frames_before_the_first_person_poses_are_not_selected() {
        let mut assembler = KeyframeAssembler::new(KeyframeSettings::default());
        assert!(run(&mut assembler, 0..40, 0.0).is_empty());
        assembler.add_person_poses(&poses(0, &[]), &calibration()).unwrap();
        assert_eq!(run(&mut assembler, 40..41, 0.0), vec![40]);
    }

    #[test]
    fn a_keyframe_waits_for_person_poses_at_its_time_and_has_their_people_cut_out_of_its_depth() {
        let mut assembler = KeyframeAssembler::new(KeyframeSettings::default());
        assembler.add_person_poses(&poses(-FRAME_NS, &[]), &calibration()).unwrap();
        assert_eq!(run(&mut assembler, 0..2, 0.0), vec![0]);
        assert!(assembler.add_color(0, vec![1]));
        assert!(assembler.take_ready(&calibration()).unwrap().is_empty(), "no person poses at the keyframe's time yet");
        let left_column = PixelBox { x_px: 0.0, y_px: 0.0, w_px: 1.0, h_px: 3.0 };
        assembler.add_person_poses(&poses(0, &[left_column]), &calibration()).unwrap();
        let selected = assembler.take_ready(&calibration()).unwrap();
        // The margin grows the one pixel wide box by 0.15 px a side, which rounds out to the next column.
        assert_eq!(selected[0].people, PeopleCut::Cut { people: 1, pixels: 6, age_ns: 0 });
        assert_eq!(&selected[0].keyframe.depth_mm[..], &[0, 0, 1000, 1000, 0, 0, 1000, 1000, 0, 0, 1000, 1000]);
        assert_eq!(selected[0].keyframe.color_jpeg, vec![1]);
    }

    #[test]
    fn keyframes_piling_up_behind_the_pose_model_stop_the_node() {
        let mut assembler = KeyframeAssembler::new(KeyframeSettings::default());
        assembler.add_person_poses(&poses(-FRAME_NS, &[]), &calibration()).unwrap();
        let selected = run(&mut assembler, 0..250, 0.0);
        assert_eq!(selected.len(), MOST_HELD + 1);
        for &frame in &selected {
            assembler.add_color(frame, vec![1]);
        }
        let error = assembler.take_ready(&calibration()).err().unwrap();
        assert!(matches!(error, SelectionError::PeopleBehind { oldest_frame_id: 0, newest_poses_camera_ns, .. } if newest_poses_camera_ns == -FRAME_NS), "{error}");
    }
}
