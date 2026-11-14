//! The node's handling of its inputs, apart from dora: calibration, imu messages and depth frames in, odom poses
//! out.
//!
//! Nothing is tracked until the first calibration has arrived: it gives the depth intrinsics and where the depth
//! sensor sits on the IMU. Imu messages and depth frames before it are dropped. A later calibration must equal the
//! first in the parts odometry uses; one that differs is an error.
//!
//! Every imu message is fed to the tracker (it needs every sample). A depth frame is tracked once the imu message
//! leading up to it has arrived; a frame that arrives first is held for it, and a newer frame replaces a held one.
//! The tracker gives no pose until the body has rested, so until then frames get none. From one pose to the next
//! `camera_ns` strictly increases.

use std::time::Duration;

use depth_inertial_odometry::depth_image::{DepthImageError, PinholeIntrinsics};
use depth_inertial_odometry::imu_pairing::{AccelReading, GyroReading, ImuStream};
use depth_inertial_odometry::odometry::StillStartError;
use depth_inertial_odometry::thinning::{NotARotation, SensorMount};
use depth_inertial_odometry::tracker::{DepthFrameId, DepthFrameOutcome, TrackerError, TrackerPhase};
use depth_inertial_odometry::{FramePose, PlaneVoxelMap, PlaneVoxelSettings, PoseTracker, TrackerSettings};
use glam::Vec3;
use perception_wire::camera::{CameraCalibration, RigidTransform, StreamIntrinsics};
use perception_wire::localization::CameraNanos;
use perception_wire::odometry::{OdomPose, WeakestDirection};
use perception_wire::Imu;

/// The parts of a camera calibration that odometry is built from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthCalibration {
    pub intrinsics: StreamIntrinsics,
    /// Takes depth-stream points into the IMU's axes, which are the odometry's body frame.
    pub depth_to_imu: RigidTransform,
}

impl DepthCalibration {
    pub fn of(calibration: &CameraCalibration) -> DepthCalibration {
        DepthCalibration { intrinsics: calibration.depth, depth_to_imu: calibration.depth_to_imu }
    }

    pub fn pinhole(&self) -> PinholeIntrinsics {
        let StreamIntrinsics { width, height, fx, fy, ppx, ppy } = self.intrinsics;
        PinholeIntrinsics { width: usize::from(width), height: usize::from(height), fx, fy, ppx, ppy }
    }

    /// Where the depth sensor sits on the body.
    pub fn sensor_mount(&self) -> Result<SensorMount, NotARotation> {
        // Both sides store the rotation column major.
        SensorMount::from_depth_to_imu(self.depth_to_imu.rotation, self.depth_to_imu.translation_m)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NodeSettings {
    pub tracker: TrackerSettings,
    pub map: PlaneVoxelSettings,
}

impl NodeSettings {
    /// Settings for an Intel RealSense D455 indoors streaming 848x480 depth.
    pub fn d455() -> NodeSettings {
        NodeSettings { tracker: TrackerSettings::d455(), map: PlaneVoxelSettings::indoors() }
    }
}

/// One depth image as the camera node sends it: row major, millimetres along the optical axis, zero where there
/// is no reading.
#[derive(Clone, Copy, Debug)]
pub struct DepthFrame<'a> {
    pub frame_id: DepthFrameId,
    /// When the frame was taken.
    pub time: CameraNanos,
    pub width: usize,
    pub height: usize,
    pub depth_mm: &'a [u16],
}

/// A pose with a value the odom pose message cannot carry.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum UnrepresentablePose {
    #[error("a pose at camera time {0:?} is beyond what camera_ns can hold")]
    Time(Duration),
    #[error("{0} matched points are more than matched_points can hold")]
    MatchedPoints(usize),
}

/// The odom pose message for a tracked frame. The tracker's level frame is the message's odom frame.
pub fn odom_pose(pose: &FramePose) -> Result<OdomPose, UnrepresentablePose> {
    Ok(OdomPose {
        camera_ns: i64::try_from(pose.time.as_nanos()).map_err(|_| UnrepresentablePose::Time(pose.time))?,
        depth_frame_id: pose.frame_id.0,
        position_m: pose.position_m.as_vec3(),
        orientation: pose.orientation.as_quat(),
        velocity_m_s: pose.velocity_m_s.as_vec3(),
        position_sigma_m: pose.uncertainty.position_sigma_m.as_vec3(),
        tilt_sigma_rad: pose.uncertainty.tilt_sigma_rad as f32,
        heading_sigma_rad: pose.uncertainty.heading_sigma_rad as f32,
        corrected: pose.corrected(),
        matched_points: u32::try_from(pose.matched_points()).map_err(|_| UnrepresentablePose::MatchedPoints(pose.matched_points()))?,
        weakest: pose.weakest().map(|weakest| WeakestDirection { direction: weakest.direction.as_vec3(), sigma_m: weakest.sigma_m as f32 }),
    })
}

/// The calibration odometry started with and a later one that differs from it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationChange {
    pub started: DepthCalibration,
    pub now: DepthCalibration,
}

#[derive(Debug, thiserror::Error)]
pub enum OdometryNodeError {
    #[error("the calibration cannot mount the depth sensor: {0}")]
    Mount(#[from] NotARotation),
    #[error("the calibration's depth stream cannot be tracked: {0}")]
    DepthStream(DepthImageError),
    #[error("the calibration changed while odometry runs: it started with {:?} and is now {:?}", .0.started, .0.now)]
    CalibrationChanged(Box<CalibrationChange>),
    #[error("a camera time of {} ns is before the camera clock's zero", .0 .0)]
    NegativeCameraTime(CameraNanos),
    #[error("an imu message has {times} {stream} times for {readings} readings")]
    UnevenImu { stream: ImuStream, times: usize, readings: usize },
    #[error("the imu message for frame {} is not newer than the one before it, for frame {}", .frame_id.0, .previous.0)]
    ImuMessageOutOfOrder { frame_id: DepthFrameId, previous: DepthFrameId },
    #[error("depth frame {} is {width}x{height}, and the calibration's depth stream is {calibrated_width}x{calibrated_height}", .frame_id.0)]
    DepthSize { frame_id: DepthFrameId, width: usize, height: usize, calibrated_width: usize, calibrated_height: usize },
    #[error(transparent)]
    Tracker(TrackerError),
    #[error(transparent)]
    Pose(#[from] UnrepresentablePose),
}

/// What a calibration message did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CalibrationOutcome {
    /// The first calibration: the tracker now exists and waits for the body to rest.
    TrackerBuilt(DepthCalibration),
    /// The calibration odometry already runs on.
    Unchanged,
}

/// How far the start of odometry has come.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StartProgress {
    /// The body has rested this long of the time it must.
    Resting { rested: Duration },
    /// The body moved during the time it had to rest; that time has begun again.
    NotAtRest(StillStartError),
    /// Odometry started with this imu message.
    Started,
    Tracking,
}

/// What became of a depth frame the tracker was given.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FrameOutcome {
    /// The frame was taken before odometry started; there is no pose for it.
    BeforeStart,
    Posed(OdomPose),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImuOutcome {
    /// Dropped: no calibration has arrived.
    BeforeCalibration,
    Fed {
        start: StartProgress,
        /// The frame that was held for this imu message.
        held_frame: Option<FrameOutcome>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DepthOutcome {
    /// Dropped: no calibration has arrived.
    BeforeCalibration,
    /// The imu message leading up to the frame has not arrived, so the frame is held for it. A frame that was
    /// already held is dropped in its favour.
    HeldForImu { replaced: Option<DepthFrameId> },
    Tracked(FrameOutcome),
}

struct HeldFrame {
    frame_id: DepthFrameId,
    time: Duration,
    depth_mm: Vec<u16>,
}

struct Tracking {
    calibration: DepthCalibration,
    tracker: PoseTracker<PlaneVoxelMap>,
    started: bool,
    /// The frame the newest imu message leads up to.
    imu_through: Option<DepthFrameId>,
    /// A frame newer than `imu_through`.
    held: Option<HeldFrame>,
    accel: Vec<AccelReading>,
    gyro: Vec<GyroReading>,
}

impl Tracking {
    fn new(calibration: DepthCalibration, settings: &NodeSettings) -> Result<Tracking, OdometryNodeError> {
        let tracker = PoseTracker::new(calibration.pinhole(), calibration.sensor_mount()?, PlaneVoxelMap::new(settings.map), settings.tracker)
            .map_err(OdometryNodeError::DepthStream)?;
        Ok(Tracking { calibration, tracker, started: false, imu_through: None, held: None, accel: Vec::new(), gyro: Vec::new() })
    }

    fn add_imu(&mut self, imu: &Imu) -> Result<StartProgress, OdometryNodeError> {
        let frame_id = DepthFrameId(imu.frame_id);
        if let Some(previous) = self.imu_through.filter(|&previous| frame_id <= previous) {
            return Err(OdometryNodeError::ImuMessageOutOfOrder { frame_id, previous });
        }
        self.accel.clear();
        for (time, reading) in timed(ImuStream::Accelerometer, &imu.accel_camera_ns, &imu.accel_m_s2)? {
            self.accel.push(AccelReading { time: camera_time(time)?, m_s2: reading.as_dvec3() });
        }
        self.gyro.clear();
        for (time, reading) in timed(ImuStream::Gyro, &imu.gyro_camera_ns, &imu.gyro_rad_s)? {
            self.gyro.push(GyroReading { time: camera_time(time)?, rad_s: reading.as_dvec3() });
        }
        let start = match self.tracker.add_imu(&self.accel, &self.gyro) {
            Ok(TrackerPhase::Resting { rested }) => StartProgress::Resting { rested },
            Ok(TrackerPhase::Tracking) if self.started => StartProgress::Tracking,
            Ok(TrackerPhase::Tracking) => StartProgress::Started,
            Err(TrackerError::NotAtRest(moving)) => StartProgress::NotAtRest(moving),
            Err(other) => return Err(OdometryNodeError::Tracker(other)),
        };
        self.started = matches!(start, StartProgress::Started | StartProgress::Tracking);
        self.imu_through = Some(frame_id);
        Ok(start)
    }

    fn add_frame(&mut self, frame_id: DepthFrameId, time: Duration, depth_mm: &[u16]) -> Result<FrameOutcome, OdometryNodeError> {
        match self.tracker.add_depth_frame(time, frame_id, depth_mm).map_err(OdometryNodeError::Tracker)? {
            DepthFrameOutcome::AwaitingStart => Ok(FrameOutcome::BeforeStart),
            DepthFrameOutcome::Posed(pose) => Ok(FrameOutcome::Posed(odom_pose(&pose)?)),
        }
    }
}

fn timed<'a>(stream: ImuStream, times: &'a [i64], readings: &'a [Vec3]) -> Result<impl Iterator<Item = (CameraNanos, Vec3)> + 'a, OdometryNodeError> {
    if times.len() != readings.len() {
        return Err(OdometryNodeError::UnevenImu { stream, times: times.len(), readings: readings.len() });
    }
    Ok(times.iter().map(|&time| CameraNanos(time)).zip(readings.iter().copied()))
}

fn camera_time(time: CameraNanos) -> Result<Duration, OdometryNodeError> {
    u64::try_from(time.0).map(Duration::from_nanos).map_err(|_| OdometryNodeError::NegativeCameraTime(time))
}

pub struct OdometryNode {
    settings: NodeSettings,
    /// Absent until the first calibration.
    tracking: Option<Box<Tracking>>,
}

impl OdometryNode {
    pub fn new(settings: NodeSettings) -> OdometryNode {
        OdometryNode { settings, tracking: None }
    }

    pub fn on_calibration(&mut self, calibration: &CameraCalibration) -> Result<CalibrationOutcome, OdometryNodeError> {
        let now = DepthCalibration::of(calibration);
        match &self.tracking {
            None => {
                self.tracking = Some(Box::new(Tracking::new(now, &self.settings)?));
                Ok(CalibrationOutcome::TrackerBuilt(now))
            }
            Some(tracking) if tracking.calibration == now => Ok(CalibrationOutcome::Unchanged),
            Some(tracking) => Err(OdometryNodeError::CalibrationChanged(Box::new(CalibrationChange { started: tracking.calibration, now }))),
        }
    }

    pub fn on_imu(&mut self, imu: &Imu) -> Result<ImuOutcome, OdometryNodeError> {
        let Some(tracking) = self.tracking.as_deref_mut() else { return Ok(ImuOutcome::BeforeCalibration) };
        let start = tracking.add_imu(imu)?;
        let imu_through = tracking.imu_through;
        let held_frame = match tracking.held.take_if(|held| imu_through.is_some_and(|through| held.frame_id <= through)) {
            Some(held) => Some(tracking.add_frame(held.frame_id, held.time, &held.depth_mm)?),
            None => None,
        };
        Ok(ImuOutcome::Fed { start, held_frame })
    }

    pub fn on_depth(&mut self, frame: DepthFrame) -> Result<DepthOutcome, OdometryNodeError> {
        let Some(tracking) = self.tracking.as_deref_mut() else { return Ok(DepthOutcome::BeforeCalibration) };
        let DepthFrame { frame_id, width, height, depth_mm, .. } = frame;
        let calibrated = tracking.calibration.pinhole();
        if (width, height) != (calibrated.width, calibrated.height) {
            return Err(OdometryNodeError::DepthSize { frame_id, width, height, calibrated_width: calibrated.width, calibrated_height: calibrated.height });
        }
        let time = camera_time(frame.time)?;
        if tracking.imu_through.is_some_and(|through| frame_id <= through) {
            return Ok(DepthOutcome::Tracked(tracking.add_frame(frame_id, time, depth_mm)?));
        }
        let replaced = tracking.held.replace(HeldFrame { frame_id, time, depth_mm: depth_mm.to_vec() });
        Ok(DepthOutcome::HeldForImu { replaced: replaced.map(|held| held.frame_id) })
    }
}

#[cfg(test)]
mod tests {
    use depth_inertial_odometry::filter::{CorrectionError, LevelledUncertainty};
    use depth_inertial_odometry::tracker::{self, CloudUse};
    use glam::{DQuat, DVec3, Quat};

    use super::*;

    /// A pose whose every value a single-precision float holds exactly.
    fn frame_pose(cloud: CloudUse) -> FramePose {
        FramePose {
            time: Duration::new(1_762_560, 133_000_001),
            frame_id: DepthFrameId(43),
            position_m: DVec3::new(1.25, -0.5, 0.25),
            orientation: DQuat::from_xyzw(0.5, -0.5, 0.5, 0.5),
            velocity_m_s: DVec3::new(0.375, 0.0, -0.0625),
            uncertainty: LevelledUncertainty { position_sigma_m: DVec3::new(0.015625, 0.03125, 0.0078125), tilt_sigma_rad: 0.00390625, heading_sigma_rad: 0.0078125 },
            cloud,
        }
    }

    #[test]
    fn a_corrected_pose_becomes_an_odom_pose_with_every_field_carried_over() {
        let weakest = tracker::WeakestDirection { direction: DVec3::new(0.0, -1.0, 0.0), sigma_m: 0.125 };
        let message = odom_pose(&frame_pose(CloudUse::Corrected { matched_points: 1234, weakest })).unwrap();
        let expected = OdomPose {
            camera_ns: 1_762_560_133_000_001,
            depth_frame_id: 43,
            position_m: Vec3::new(1.25, -0.5, 0.25),
            orientation: Quat::from_xyzw(0.5, -0.5, 0.5, 0.5),
            velocity_m_s: Vec3::new(0.375, 0.0, -0.0625),
            position_sigma_m: Vec3::new(0.015625, 0.03125, 0.0078125),
            tilt_sigma_rad: 0.00390625,
            heading_sigma_rad: 0.0078125,
            corrected: true,
            matched_points: 1234,
            weakest: Some(WeakestDirection { direction: Vec3::NEG_Y, sigma_m: 0.125 }),
        };
        assert_eq!(message, expected);
    }

    #[test]
    fn a_pose_the_cloud_did_not_correct_has_no_weakest_direction_and_no_matched_points() {
        for cloud in [CloudUse::StartedMap, CloudUse::NotCorrected(CorrectionError::TooFewRows { rows: 3, needed: 50 })] {
            let message = odom_pose(&frame_pose(cloud)).unwrap();
            assert_eq!((message.corrected, message.matched_points, message.weakest), (false, 0, None), "{cloud:?}");
            assert_eq!((message.camera_ns, message.depth_frame_id, message.position_m), (1_762_560_133_000_001, 43, Vec3::new(1.25, -0.5, 0.25)));
        }
    }

    #[test]
    fn a_pose_time_beyond_camera_ns_is_refused() {
        let late = FramePose { time: Duration::from_secs(u64::MAX / 1_000_000), ..frame_pose(CloudUse::StartedMap) };
        assert_eq!(odom_pose(&late), Err(UnrepresentablePose::Time(late.time)));
    }

    #[test]
    fn the_mount_takes_a_depth_point_where_the_calibrations_depth_to_imu_transform_takes_it() {
        // A turn about no axis of symmetry, so reading the matrix row major would give another mount.
        let rotation = glam::Mat3::from_quat(Quat::from_euler(glam::EulerRot::ZYX, 0.7, -0.4, 1.1));
        let depth_to_imu = RigidTransform { rotation: rotation.to_cols_array(), translation_m: [0.03, -0.01, 0.02] };
        assert!((rotation - rotation.transpose()).to_cols_array().iter().any(|entry| entry.abs() > 0.3));
        let calibration = DepthCalibration { intrinsics: StreamIntrinsics { width: 848, height: 480, fx: 420.0, fy: 420.0, ppx: 424.0, ppy: 240.0 }, depth_to_imu };
        let mount = calibration.sensor_mount().unwrap();
        for point in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(0.4, -1.2, 2.5)] {
            let mounted = mount.rotation * point.as_dvec3() + mount.translation_m;
            assert!((mounted - depth_to_imu.apply(point).as_dvec3()).length() < 1e-6, "{point}: {mounted}");
        }
    }
}
