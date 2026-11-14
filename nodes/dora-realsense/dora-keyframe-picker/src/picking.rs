//! Picking keyframes from a stream of camera frames: which depth frames become keyframes, what is held while their
//! colour is fetched, and which IMU samples each keyframe carries.

use std::collections::BTreeMap;
use std::time::Duration;

use depth_perception::CameraIntrinsics;
use glam::Vec3;
use perception_wire::camera::CameraCalibration;
use perception_wire::{Imu, Keyframe};
use rgbd_keyframes::{KeyframePicker, KeyframeSettings};

use crate::align::align_depth_to_color;

/// A picked depth frame waiting for its colour.
struct Held {
    camera_ns: i64,
    captured_unix_ns: i64,
    gravity_reaction: Vec3,
    depth_mm: Vec<u16>,
}

/// One picked frame, ready to send: its imu message goes out before the keyframe.
pub struct Picked {
    pub imu: Imu,
    pub keyframe: Keyframe,
}

/// IMU samples not yet sent with a keyframe.
#[derive(Default)]
struct Samples {
    accel: Vec<(i64, Vec3)>,
    gyro: Vec<(i64, Vec3)>,
}

pub struct KeyframeAssembler {
    picker: KeyframePicker,
    first_camera_ns: Option<i64>,
    last_gyro_ns: Option<i64>,
    samples: Samples,
    held: BTreeMap<u64, Held>,
}

/// Takes the samples at or before `until_ns` out of `samples`, in time order.
fn take_until(samples: &mut Vec<(i64, Vec3)>, until_ns: i64) -> (Vec<i64>, Vec<Vec3>) {
    let later = samples.split_off(samples.partition_point(|(t, _)| *t <= until_ns));
    std::mem::replace(samples, later).into_iter().unzip()
}

impl KeyframeAssembler {
    pub fn new(settings: KeyframeSettings) -> KeyframeAssembler {
        KeyframeAssembler { picker: KeyframePicker::new(settings), first_camera_ns: None, last_gyro_ns: None, samples: Samples::default(), held: BTreeMap::new() }
    }

    /// Adds one frame's imu message from the camera.
    pub fn add_imu(&mut self, imu: &Imu) {
        for (&t, &sample) in imu.gyro_camera_ns.iter().zip(&imu.gyro_rad_s) {
            let held = self.last_gyro_ns.map_or(Duration::ZERO, |last| Duration::from_nanos((t - last).max(0) as u64));
            self.picker.add_rotation(sample, held);
            self.last_gyro_ns = Some(t);
            self.samples.gyro.push((t, sample));
        }
        self.samples.accel.extend(imu.accel_camera_ns.iter().copied().zip(imu.accel_m_s2.iter().copied()));
    }

    /// Adds one depth frame. Returns true when it is picked, and its colour should be fetched. Frames before the
    /// camera has a gravity estimate are never picked.
    pub fn add_depth(&mut self, frame_id: u64, camera_ns: i64, captured_unix_ns: i64, gravity_reaction: Option<Vec3>, depth_mm: &[u16]) -> bool {
        let first = *self.first_camera_ns.get_or_insert(camera_ns);
        let Some(gravity_reaction) = gravity_reaction else { return false };
        if !self.picker.take(Duration::from_nanos((camera_ns - first).max(0) as u64)) {
            return false;
        }
        self.held.insert(frame_id, Held { camera_ns, captured_unix_ns, gravity_reaction, depth_mm: depth_mm.to_vec() });
        true
    }

    /// Whether `frame_id` is picked and waiting for its colour.
    pub fn is_waiting_for(&self, frame_id: u64) -> bool {
        self.held.contains_key(&frame_id)
    }

    /// Completes a picked frame with its colour image, encoded as `color_jpeg`.
    pub fn complete(&mut self, frame_id: u64, color_jpeg: Vec<u8>, calibration: &CameraCalibration) -> Option<Picked> {
        let held = self.held.remove(&frame_id)?;
        let (accel_camera_ns, accel_m_s2) = take_until(&mut self.samples.accel, held.camera_ns);
        let (gyro_camera_ns, gyro_rad_s) = take_until(&mut self.samples.gyro, held.camera_ns);
        let imu = Imu { frame_id, keyframe_camera_ns: held.camera_ns, accel_camera_ns, accel_m_s2, gyro_camera_ns, gyro_rad_s };
        let color = calibration.color;
        let keyframe = Keyframe {
            frame_id,
            captured_unix_ns: held.captured_unix_ns,
            width: color.width as usize,
            height: color.height as usize,
            intrinsics: CameraIntrinsics { fx: color.fx, fy: color.fy, ppx: color.ppx, ppy: color.ppy },
            gravity_reaction: held.gravity_reaction,
            depth_mm: align_depth_to_color(&held.depth_mm, calibration).into(),
            color_jpeg,
        };
        Some(Picked { imu, keyframe })
    }
}

#[cfg(test)]
mod tests {
    use perception_wire::camera::{RigidTransform, StreamIntrinsics};

    use super::*;

    const FRAME_NS: i64 = 33_333_333;

    fn calibration() -> CameraCalibration {
        let identity = RigidTransform { rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], translation_m: [0.0; 3] };
        let intrinsics = StreamIntrinsics { width: 4, height: 3, fx: 4.0, fy: 4.0, ppx: 1.5, ppy: 1.0 };
        CameraCalibration { color: intrinsics, depth: intrinsics, depth_to_color: identity, depth_to_imu: identity }
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

    /// Feeds frames `range` and returns the ones picked.
    fn run(assembler: &mut KeyframeAssembler, frames: std::ops::Range<u64>, turn_rad_s: f32) -> Vec<u64> {
        let depth = [1000u16; 12];
        frames
            .filter(|&frame| {
                assembler.add_imu(&imu(frame, turn_rad_s));
                assembler.add_depth(frame, frame as i64 * FRAME_NS, 0, Some(Vec3::new(0.0, -9.8, 0.0)), &depth)
            })
            .collect()
    }

    #[test]
    fn a_still_camera_gives_a_keyframe_a_second() {
        // 30 frames at 30 fps fall a hair short of a second, so the next keyframe is the 31st frame on.
        let mut assembler = KeyframeAssembler::new(KeyframeSettings::default());
        assert_eq!(run(&mut assembler, 0..95, 0.0), vec![0, 31, 62, 93]);
    }

    #[test]
    fn a_turning_camera_gives_them_sooner() {
        // 1 rad/s turns 10 degrees in about 5 frames, but keyframes are never closer than 200 ms: 7 frames.
        let mut assembler = KeyframeAssembler::new(KeyframeSettings::default());
        assert_eq!(run(&mut assembler, 0..20, 1.0), vec![0, 7, 14]);
    }

    #[test]
    fn frames_before_gravity_settles_are_not_picked() {
        let mut assembler = KeyframeAssembler::new(KeyframeSettings::default());
        assert!(!assembler.add_depth(0, 0, 0, None, &[1000; 12]));
        assert!(!assembler.is_waiting_for(0));
    }

    #[test]
    fn each_keyframe_carries_every_sample_since_the_last_one_and_none_after_it() {
        let mut assembler = KeyframeAssembler::new(KeyframeSettings::default());
        assert_eq!(run(&mut assembler, 0..32, 0.0), vec![0, 31]);
        let first = assembler.complete(0, vec![1], &calibration()).unwrap();
        assert_eq!((first.imu.frame_id, first.imu.keyframe_camera_ns), (0, 0));
        assert_eq!(first.imu.gyro_camera_ns, vec![-FRAME_NS / 2, 0]);
        let second = assembler.complete(31, vec![2], &calibration()).unwrap();
        assert_eq!(second.imu.gyro_camera_ns.len(), 62);
        assert_eq!(second.imu.accel_camera_ns.len(), 31);
        assert_eq!(*second.imu.gyro_camera_ns.last().unwrap(), 31 * FRAME_NS);
        assert_eq!(second.keyframe.frame_id, 31);
        assert_eq!(&second.keyframe.depth_mm[..], &[1000u16; 12]);
    }

    #[test]
    fn colour_for_a_frame_not_picked_completes_nothing() {
        let mut assembler = KeyframeAssembler::new(KeyframeSettings::default());
        run(&mut assembler, 0..3, 0.0);
        assert!(assembler.complete(1, vec![], &calibration()).is_none());
    }
}
