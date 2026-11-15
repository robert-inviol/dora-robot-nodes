//! Where the colour camera was in the odom frame at a given camera time, from the odometry's poses of the IMU body.

use std::collections::VecDeque;

use glam::{Affine3A, Mat3, Vec3};
use perception_wire::camera::{CameraCalibration, RigidTransform};
use perception_wire::localization::CameraNanos;
use perception_wire::odometry::OdomPose;

/// What the odometry can say about one camera time.
#[derive(Clone, Copy, Debug)]
pub enum CameraPoseLookup {
    /// The transform from the colour camera's frame into the odom frame `odom_started` names: the odom frame of the
    /// odometry poses the camera time was interpolated between.
    Interpolated { camera_to_odom: Affine3A, odom_started: CameraNanos },
    /// The odometry has not reached this time yet.
    NotYet,
    /// The odometry will never cover this time: it is before the current odom frame's oldest pose kept here.
    Never,
}

fn affine(transform: &RigidTransform) -> Affine3A {
    let r = transform.rotation;
    let rotation = Mat3::from_cols(Vec3::new(r[0], r[1], r[2]), Vec3::new(r[3], r[4], r[5]), Vec3::new(r[6], r[7], r[8]));
    Affine3A::from_mat3_translation(rotation, Vec3::from_array(transform.translation_m))
}

/// The odometry's recent poses, all in one odom frame.
pub struct OdomHistory {
    poses: VecDeque<OdomPose>,
    keep_ns: i64,
}

impl OdomHistory {
    pub fn new(keep_ns: i64) -> OdomHistory {
        OdomHistory { poses: VecDeque::new(), keep_ns }
    }

    /// Adds the next pose. Returns true when it starts a new odom frame, so every pose before it is dropped.
    pub fn push(&mut self, pose: OdomPose) -> bool {
        let restarted = self.poses.back().is_some_and(|last| last.odom_started_camera_ns != pose.odom_started_camera_ns);
        if restarted {
            self.poses.clear();
        }
        while self.poses.front().is_some_and(|oldest| pose.camera_ns - oldest.camera_ns > self.keep_ns) {
            self.poses.pop_front();
        }
        self.poses.push_back(pose);
        restarted
    }

    /// Where the colour camera was at `camera_ns`, between the two poses either side of it.
    pub fn camera_at(&self, camera_ns: i64, calibration: &CameraCalibration) -> CameraPoseLookup {
        let (Some(oldest), Some(newest)) = (self.poses.front(), self.poses.back()) else { return CameraPoseLookup::NotYet };
        if camera_ns < oldest.camera_ns {
            return CameraPoseLookup::Never;
        }
        if camera_ns > newest.camera_ns {
            return CameraPoseLookup::NotYet;
        }
        let after = self.poses.partition_point(|p| p.camera_ns < camera_ns);
        let later = &self.poses[after];
        let earlier = &self.poses[after.saturating_sub(1)];
        let span_ns = later.camera_ns - earlier.camera_ns;
        let t = if span_ns == 0 { 0.0 } else { (camera_ns - earlier.camera_ns) as f32 / span_ns as f32 };
        let body_to_odom = Affine3A::from_rotation_translation(
            earlier.orientation.slerp(later.orientation, t),
            earlier.position_m.lerp(later.position_m, t),
        );
        let color_to_body = affine(&calibration.depth_to_imu) * affine(&calibration.depth_to_color.inverse());
        CameraPoseLookup::Interpolated { camera_to_odom: body_to_odom * color_to_body, odom_started: CameraNanos(later.odom_started_camera_ns) }
    }
}

#[cfg(test)]
mod tests {
    use glam::Quat;
    use perception_wire::camera::StreamIntrinsics;

    use super::*;

    fn pose(camera_ns: i64, odom_started_camera_ns: i64, position_m: Vec3, orientation: Quat) -> OdomPose {
        OdomPose {
            camera_ns,
            odom_started_camera_ns,
            depth_frame_id: 0,
            position_m,
            orientation,
            velocity_m_s: Vec3::ZERO,
            position_sigma_m: Vec3::splat(0.01),
            tilt_sigma_rad: 0.01,
            heading_sigma_rad: 0.01,
            corrected: true,
            matched_points: 100,
            degenerate: None,
        }
    }

    /// Colour 5 cm right of depth; the IMU 1 cm right of depth.
    fn calibration() -> CameraCalibration {
        let intrinsics = StreamIntrinsics { width: 64, height: 48, fx: 50.0, fy: 50.0, ppx: 32.0, ppy: 24.0 };
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        CameraCalibration {
            color: intrinsics,
            depth: intrinsics,
            depth_to_color: RigidTransform { rotation: identity, translation_m: [-0.05, 0.0, 0.0] },
            depth_to_imu: RigidTransform { rotation: identity, translation_m: [-0.01, 0.0, 0.0] },
        }
    }

    fn interpolated(lookup: CameraPoseLookup) -> Affine3A {
        match lookup {
            CameraPoseLookup::Interpolated { camera_to_odom, .. } => camera_to_odom,
            other => panic!("not interpolated: {other:?}"),
        }
    }

    #[test]
    fn a_point_before_the_colour_camera_lands_where_the_body_pose_puts_it() {
        let mut history = OdomHistory::new(2_000_000_000);
        // The body 1 m up odom z, turned a quarter about odom z: body x points along odom y.
        let turned = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        history.push(pose(100, 0, Vec3::new(0.0, 0.0, 1.0), turned));
        let camera_to_odom = interpolated(history.camera_at(100, &calibration()));
        // The colour camera's origin is 4 cm right of the IMU, along body x, so 4 cm along odom y.
        let origin = camera_to_odom.transform_point3(Vec3::ZERO);
        assert!(origin.abs_diff_eq(Vec3::new(0.0, 0.04, 1.0), 1e-6), "{origin}");
        // 2 m forward of the camera is 2 m along body z, which the turn leaves as odom z.
        let ahead = camera_to_odom.transform_point3(Vec3::new(0.0, 0.0, 2.0));
        assert!(ahead.abs_diff_eq(Vec3::new(0.0, 0.04, 3.0), 1e-6), "{ahead}");
    }

    #[test]
    fn a_time_between_two_poses_is_interpolated_between_them() {
        let mut history = OdomHistory::new(2_000_000_000);
        history.push(pose(100, 0, Vec3::ZERO, Quat::IDENTITY));
        history.push(pose(200, 0, Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY));
        let origin = interpolated(history.camera_at(150, &calibration())).transform_point3(Vec3::ZERO);
        assert!(origin.abs_diff_eq(Vec3::new(0.54, 0.0, 0.0), 1e-6), "{origin}");
    }

    #[test]
    fn times_outside_the_poses_are_not_yet_or_never() {
        let mut history = OdomHistory::new(2_000_000_000);
        assert!(matches!(history.camera_at(100, &calibration()), CameraPoseLookup::NotYet));
        history.push(pose(100, 0, Vec3::ZERO, Quat::IDENTITY));
        assert!(matches!(history.camera_at(150, &calibration()), CameraPoseLookup::NotYet));
        assert!(matches!(history.camera_at(50, &calibration()), CameraPoseLookup::Never));
    }

    #[test]
    fn a_restarted_odometry_drops_the_old_frame() {
        let mut history = OdomHistory::new(2_000_000_000);
        assert!(!history.push(pose(100, 0, Vec3::ZERO, Quat::IDENTITY)));
        assert!(history.push(pose(300, 250, Vec3::ZERO, Quat::IDENTITY)));
        assert!(matches!(history.camera_at(200, &calibration()), CameraPoseLookup::Never));
        match history.camera_at(300, &calibration()) {
            CameraPoseLookup::Interpolated { odom_started, .. } => assert_eq!(odom_started, CameraNanos(250)),
            other => panic!("{other:?}"),
        }
    }
}
