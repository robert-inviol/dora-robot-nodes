//! The transform from the robot's odometry frame to the map, from keyframes whose pose is known in both.
//!
//! For each registered keyframe, the odometry pose at the keyframe's camera time is interpolated from the
//! odometry history, and the correction is the keyframe's map pose composed with that odometry pose's
//! inverse. The estimate averages the corrections of the latest keyframes, each recomputed from the keyframe's
//! current map pose, so an optimisation of the map moves it too; their spread gives the standard deviations.
//! Odometry poses describe the same body as keyframe poses: the camera's gravity-aligned floor frame.
//! `CameraMount` turns the odometry's IMU body pose into that floor frame.

use std::collections::VecDeque;

use glam::{Quat, Vec3};
use perception_wire::localization::{CameraNanos, FrameId, MapToOdom, Pose4};
use perception_wire::navigation::MapVersion;
use room_map::pose::wrap_angle;
use room_map::Pose;

/// The shortest level part of the camera's forward direction that still gives its floor frame a heading.
const MIN_LEVEL_FORWARD: f32 = 0.1;

/// Where the colour camera sits on the IMU body, whose axes are the camera's (x right, y down, z forward).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraMount {
    /// The camera's optical centre in IMU coordinates.
    pub camera_in_imu_m: Vec3,
}

impl CameraMount {
    /// The camera's floor frame in the odometry frame (z up), from the IMU body's pose there: at the camera, facing
    /// its forward direction made level. `None` when the camera faces too near straight up or down to have a heading.
    pub fn floor_pose(&self, imu_position: Vec3, imu_orientation: Quat) -> Option<Pose> {
        let level_forward = (imu_orientation * Vec3::Z).truncate();
        if level_forward.length() < MIN_LEVEL_FORWARD {
            return None;
        }
        Some(Pose { yaw_rad: (-level_forward.x).atan2(level_forward.y), translation: imu_position + imu_orientation * self.camera_in_imu_m })
    }
}

/// Where odometry puts the camera's floor frame at a camera time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OdomPose {
    pub time: CameraNanos,
    pub pose: Pose,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum OdomError {
    #[error("odometry at camera time {received} ns is not later than the last, at {last} ns")]
    OutOfOrder { last: i64, received: i64 },
    #[error("odometry covers camera times {first} to {last} ns, not {wanted} ns")]
    NotCovered { wanted: i64, first: i64, last: i64 },
    #[error("there is no odometry yet")]
    Empty,
    #[error("odometry at camera time {time} ns has the camera facing too near straight up or down to have a heading")]
    NoHeading { time: i64 },
}

/// Recent odometry poses in camera-time order.
#[derive(Clone, Debug)]
pub struct OdomHistory {
    poses: VecDeque<OdomPose>,
    retention_ns: i64,
}

impl OdomHistory {
    pub fn new(retention_ns: i64) -> OdomHistory {
        OdomHistory { poses: VecDeque::new(), retention_ns }
    }

    pub fn push(&mut self, odom: OdomPose) -> Result<(), OdomError> {
        if let Some(last) = self.poses.back() {
            if odom.time <= last.time {
                return Err(OdomError::OutOfOrder { last: last.time.0, received: odom.time.0 });
            }
        }
        self.poses.push_back(odom);
        while self.poses.front().is_some_and(|p| p.time.0 < odom.time.0 - self.retention_ns) {
            self.poses.pop_front();
        }
        Ok(())
    }

    /// The odometry pose at a camera time, interpolated between the poses either side.
    pub fn at(&self, time: CameraNanos) -> Result<Pose, OdomError> {
        let (first, last) = match (self.poses.front(), self.poses.back()) {
            (Some(first), Some(last)) => (first, last),
            _ => return Err(OdomError::Empty),
        };
        if time < first.time || time > last.time {
            return Err(OdomError::NotCovered { wanted: time.0, first: first.time.0, last: last.time.0 });
        }
        let after = self.poses.partition_point(|p| p.time < time);
        let later = self.poses[after];
        if later.time == time {
            return Ok(later.pose);
        }
        let earlier = self.poses[after - 1];
        let share = (time.0 - earlier.time.0) as f32 / (later.time.0 - earlier.time.0) as f32;
        Ok(Pose {
            yaw_rad: wrap_angle(earlier.pose.yaw_rad + share * wrap_angle(later.pose.yaw_rad - earlier.pose.yaw_rad)),
            translation: earlier.pose.translation.lerp(later.pose.translation, share),
        })
    }
}

#[derive(Clone, Debug)]
pub struct MapToOdomSettings {
    /// The correction is averaged over this many of the latest registered keyframes.
    pub window_keyframes: usize,
    /// Standard deviations never reported below these, the registration's own accuracy.
    pub min_sigma_xy_m: f32,
    pub min_sigma_z_m: f32,
    pub min_sigma_yaw_rad: f32,
}

impl Default for MapToOdomSettings {
    fn default() -> Self {
        MapToOdomSettings { window_keyframes: 5, min_sigma_xy_m: 0.02, min_sigma_z_m: 0.01, min_sigma_yaw_rad: 0.5f32.to_radians() }
    }
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    frame: FrameId,
    time: CameraNanos,
    odom: Pose,
}

pub struct MapToOdomEstimator {
    settings: MapToOdomSettings,
    samples: VecDeque<Sample>,
}

impl MapToOdomEstimator {
    pub fn new(settings: MapToOdomSettings) -> MapToOdomEstimator {
        MapToOdomEstimator { settings, samples: VecDeque::new() }
    }

    /// Records the odometry pose at a registered keyframe's camera time.
    pub fn add_keyframe(&mut self, frame: FrameId, time: CameraNanos, odometry: &OdomHistory) -> Result<(), OdomError> {
        let odom = odometry.at(time)?;
        self.samples.push_back(Sample { frame, time, odom });
        while self.samples.len() > self.settings.window_keyframes {
            self.samples.pop_front();
        }
        Ok(())
    }

    /// The correction from the recorded keyframes that are still in the map, with `map_pose` giving each one's
    /// current map pose; `None` before any.
    pub fn estimate(&self, map_version: MapVersion, map_pose: impl Fn(FrameId) -> Option<Pose>) -> Option<MapToOdom> {
        let corrections: Vec<(CameraNanos, Pose)> =
            self.samples.iter().filter_map(|s| map_pose(s.frame).map(|map| (s.time, map.compose(&s.odom.inverse())))).collect();
        let valid_from = corrections.iter().map(|c| c.0).max()?;
        let count = corrections.len() as f32;
        let mean_translation: Vec3 = corrections.iter().map(|c| c.1.translation).sum::<Vec3>() / count;
        let (sin, cos) = corrections.iter().fold((0.0f32, 0.0f32), |(s, c), (_, p)| (s + p.yaw_rad.sin(), c + p.yaw_rad.cos()));
        let mean_yaw = sin.atan2(cos);
        let spread = |deviation: &dyn Fn(&Pose) -> f32| (corrections.iter().map(|c| deviation(&c.1).powi(2)).sum::<f32>() / count).sqrt();
        let sigma_xy = spread(&|p| (p.translation - mean_translation).truncate().length() / std::f32::consts::SQRT_2);
        let sigma_z = spread(&|p| p.translation.z - mean_translation.z);
        let sigma_yaw = spread(&|p| wrap_angle(p.yaw_rad - mean_yaw));
        Some(MapToOdom {
            map_version,
            valid_from,
            correction: Pose4 { x_m: mean_translation.x, y_m: mean_translation.y, z_m: mean_translation.z, yaw_rad: mean_yaw },
            sigma_xy_m: sigma_xy.max(self.settings.min_sigma_xy_m),
            sigma_z_m: sigma_z.max(self.settings.min_sigma_z_m),
            sigma_yaw_rad: sigma_yaw.max(self.settings.min_sigma_yaw_rad),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn pose(x: f32, y: f32, yaw: f32) -> Pose {
        Pose { yaw_rad: yaw, translation: Vec3::new(x, y, 0.0) }
    }

    #[test]
    fn the_floor_pose_faces_the_cameras_level_forward_and_sits_at_the_camera() {
        let mount = CameraMount { camera_in_imu_m: Vec3::new(0.03, 0.0, 0.0) };
        // The camera's optical axes (x right, y down, z forward) turned to face the odometry frame's +x, level.
        let facing_x = Quat::from_mat3(&glam::Mat3::from_cols(Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 0.0, -1.0), Vec3::X));
        let floor = mount.floor_pose(Vec3::new(1.0, 2.0, 0.5), facing_x).unwrap();
        assert!(wrap_angle(floor.yaw_rad + std::f32::consts::FRAC_PI_2).abs() < 1e-5, "floor y (forward) is odometry +x: {}", floor.yaw_rad);
        assert!((floor.translation - Vec3::new(1.0, 1.97, 0.5)).length() < 1e-5, "{:?}", floor.translation);
        let tilted_down = facing_x * Quat::from_rotation_x(-0.5);
        assert!(wrap_angle(mount.floor_pose(Vec3::ZERO, tilted_down).unwrap().yaw_rad - floor.yaw_rad).abs() < 1e-5, "tilt leaves the heading");
        let straight_down = facing_x * Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
        assert_eq!(mount.floor_pose(Vec3::ZERO, straight_down), None);
    }

    #[test]
    fn odometry_is_interpolated_between_its_poses_across_the_turn_wrap() {
        let mut history = OdomHistory::new(10_000_000_000);
        history.push(OdomPose { time: CameraNanos(1_000), pose: pose(0.0, 0.0, 3.0) }).unwrap();
        history.push(OdomPose { time: CameraNanos(2_000), pose: pose(1.0, 2.0, -3.0) }).unwrap();
        let middle = history.at(CameraNanos(1_500)).unwrap();
        assert!((middle.translation - Vec3::new(0.5, 1.0, 0.0)).length() < 1e-6);
        assert!(wrap_angle(middle.yaw_rad - std::f32::consts::PI).abs() < 1e-5, "{}", middle.yaw_rad);
        assert_eq!(history.at(CameraNanos(2_500)), Err(OdomError::NotCovered { wanted: 2_500, first: 1_000, last: 2_000 }));
        assert!(matches!(history.push(OdomPose { time: CameraNanos(2_000), pose: Pose::IDENTITY }), Err(OdomError::OutOfOrder { .. })));
    }

    #[test]
    fn the_correction_maps_odometry_poses_onto_map_poses() {
        let truth = pose(1.0, -0.5, 0.4);
        let mut history = OdomHistory::new(10_000_000_000);
        let mut map_poses = HashMap::new();
        let mut estimator = MapToOdomEstimator::new(MapToOdomSettings::default());
        for k in 0..8 {
            let time = CameraNanos(k * 100);
            let odom = pose(0.2 * k as f32, 0.1 * k as f32, 0.05 * k as f32);
            history.push(OdomPose { time, pose: odom }).unwrap();
            map_poses.insert(FrameId(k as u64), truth.compose(&odom));
            estimator.add_keyframe(FrameId(k as u64), time, &history).unwrap();
        }
        let estimate = estimator.estimate(MapVersion(3), |frame| map_poses.get(&frame).copied()).unwrap();
        let c = estimate.correction;
        assert!((c.x_m - 1.0).abs() < 1e-5 && (c.y_m + 0.5).abs() < 1e-5 && (c.yaw_rad - 0.4).abs() < 1e-5, "{c:?}");
        assert_eq!((estimate.valid_from, estimate.map_version), (CameraNanos(700), MapVersion(3)));
        assert_eq!(estimate.sigma_xy_m, MapToOdomSettings::default().min_sigma_xy_m, "agreeing keyframes give the floor sigma");
    }

    #[test]
    fn keyframes_that_disagree_widen_the_sigma_and_follow_their_map_poses() {
        let mut history = OdomHistory::new(10_000_000_000);
        let mut estimator = MapToOdomEstimator::new(MapToOdomSettings::default());
        for k in 0..4 {
            history.push(OdomPose { time: CameraNanos(k * 100), pose: Pose::IDENTITY }).unwrap();
            estimator.add_keyframe(FrameId(k as u64), CameraNanos(k * 100), &history).unwrap();
        }
        let estimate = estimator.estimate(MapVersion(1), |frame| Some(pose(if frame.0 % 2 == 0 { 0.2 } else { -0.2 }, 0.0, 0.0))).unwrap();
        assert!(estimate.correction.x_m.abs() < 1e-6);
        assert!((estimate.sigma_xy_m - 0.2 / std::f32::consts::SQRT_2).abs() < 1e-5, "{}", estimate.sigma_xy_m);
        assert!(estimator.estimate(MapVersion(1), |_| None).is_none(), "no keyframe left in the map, no estimate");
    }
}
