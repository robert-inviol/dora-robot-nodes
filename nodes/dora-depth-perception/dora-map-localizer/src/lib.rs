//! The node's handling of its inputs, apart from dora: keyframes, imu messages and odometry in, keyframe poses and
//! the map to odom correction out.
//!
//! Every imu message is kept (the gyro needs every sample). A keyframe is localized once the imu message
//! leading up to it has arrived, since that carries its camera time; a keyframe that arrives first waits for
//! it, and a newer keyframe replaces a waiting one. With `imu = "absent"` there is no imu stream: a keyframe is
//! localized as it arrives, timed by its capture time, and every heading is searched for. A keyframe whose turn the gyro cannot give is handled as
//! `heading_without_gyro` says, and reported as a `GyroGap` with the count of such keyframes so far. An imu message
//! whose samples are uneven or out of order is refused with an `ImuMessageError`.
//!
//! After an optimization, every map keyframe already sent that moved is sent again at its new pose, together with
//! the keyframes not selected that follow it (`KeyframeSelection::NotSelected`). A map keyframe that restarted
//! tracking after loss is sent restarted, and its moves are sent like any other map keyframe's.
//!
//! With an `[odometry]` section, every registered keyframe (tracked or relocalized) is joined to the odometry pose at
//! its camera time once odometry has reached that time, and the correction is sent after every keyframe and every
//! odometry pose that joined one. A keyframe that waits longer than the retention, or that falls before the
//! odometry kept, is left out and reported in `unjoined`.
//!
//! Each odometry pose names its odom frame by the camera time odometry started at. A pose with a later start is from
//! odometry that started again in a new odom frame: the poses and correction of the old frame are dropped, a waiting
//! keyframe from before the new start is reported in `unjoined`, no correction is sent until a keyframe joins the new
//! frame, and the restart is reported in `odometry_restarted`.
//! Odometry camera times keep increasing across a restart.

pub mod config;

use std::collections::{HashMap, VecDeque};

use map_localizer::cloud::registration_cloud;
use map_localizer::gyro::{GyroHeading, ImuMessageError, KeyframeTiming};
use map_localizer::map_to_odom::{CameraExtrinsics, ExtrinsicsError, MapToOdomEstimator, OdomError, OdomHistory, OdomPose};
use map_localizer::{HeadingWithoutGyro, KeyframeEstimate, KeyframeSelection, Localizer, LocalizerInput};
use perception_wire::localization::{CameraNanos, FrameId, PoseGraphUpdate, MapToOdom, UpdatedKeyframePose, Pose4Dof, Tracking};
use perception_wire::camera::CameraCalibration;
use perception_wire::odometry;
use perception_wire::navigation::UnixNanos;
use perception_wire::{Imu, Keyframe};
use room_map::heading::HeadingError;

use crate::config::{ImuStream, NodeSettings, OdometrySettings};

/// A keyframe whose turn the gyro could not give.
#[derive(Clone, Debug, PartialEq)]
pub struct GyroGap {
    pub frame_id: FrameId,
    pub problem: HeadingError,
    /// Refuse: the keyframe was reported lost. Search: its heading was searched for.
    pub handled_by: HeadingWithoutGyro,
    /// Keyframes whose turn the gyro could not give since the node started, this one included.
    pub count: u64,
}

/// A registered keyframe left out of the map to odom correction.
#[derive(Clone, Debug, PartialEq)]
pub struct UnjoinedKeyframe {
    pub frame_id: FrameId,
    pub problem: OdomError,
}

#[derive(Debug, thiserror::Error)]
pub enum OdometryInputError {
    #[error("an odom_pose message arrived but the settings have no [odometry] section")]
    NotConfigured,
    #[error("the camera moved against its IMU: the colour camera was at {was:?} m in IMU coordinates and is now at {now:?} m, turned {turned_deg} degrees")]
    CameraMoved { was: [f32; 3], now: [f32; 3], turned_deg: f32 },
    #[error(transparent)]
    Extrinsics(#[from] ExtrinsicsError),
    #[error("odometry started at camera time {received} ns, before the odom frame in use, started at {current} ns")]
    StartWentBack { current: i64, received: i64 },
    #[error(transparent)]
    Odometry(#[from] OdomError),
}

/// Odometry that started again in a new odom frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OdometryRestart {
    pub previous_start: CameraNanos,
    pub start: CameraNanos,
}

/// What handling one input did.
#[derive(Debug, Default)]
pub struct Handled {
    pub estimate: Option<KeyframeEstimate>,
    /// A keyframe that was waiting for its imu message and was replaced by a newer one.
    pub dropped: Option<FrameId>,
    pub gyro_gap: Option<GyroGap>,
    pub map_to_odom: Option<MapToOdom>,
    pub unjoined: Vec<UnjoinedKeyframe>,
    /// An odometry pose left out because the camera faced too near straight up or down to have a heading.
    pub headingless_odometry: Option<CameraNanos>,
    pub odometry_restarted: Option<OdometryRestart>,
    /// An odometry pose left out because no calibration has said where the colour camera sits on the IMU yet.
    pub odometry_before_calibration: Option<CameraNanos>,
    /// The keyframes already sent that an optimization moved, at their new poses.
    pub moved: Option<PoseGraphUpdate>,
}

/// Odometry, and the registered keyframes waiting for it to reach their camera time.
struct OdometryLink {
    settings: OdometrySettings,
    /// Where the colour camera sits on the IMU, from the camera's calibration message.
    extrinsics: Option<CameraExtrinsics>,
    /// The camera time the odometry in use started at, and the camera time of its latest pose.
    odom_frame: Option<(CameraNanos, CameraNanos)>,
    history: OdomHistory,
    estimator: MapToOdomEstimator,
    /// Oldest first.
    waiting: VecDeque<(FrameId, CameraNanos)>,
}

impl OdometryLink {
    fn new(settings: OdometrySettings) -> OdometryLink {
        OdometryLink {
            extrinsics: None,
            odom_frame: None,
            history: OdomHistory::new(settings.retention_ns),
            estimator: MapToOdomEstimator::new(settings.estimator.clone()),
            settings,
            waiting: VecDeque::new(),
        }
    }

    /// Takes a pose's place in camera time and odom frame; a later start drops the old frame's poses and correction.
    fn follow(&mut self, start: CameraNanos, time: CameraNanos) -> Result<Option<OdometryRestart>, OdometryInputError> {
        let Some((current, last)) = self.odom_frame else {
            self.odom_frame = Some((start, time));
            return Ok(None);
        };
        if time <= last {
            return Err(OdomError::OutOfOrder { last: last.0, received: time.0 }.into());
        }
        if start < current {
            return Err(OdometryInputError::StartWentBack { current: current.0, received: start.0 });
        }
        self.odom_frame = Some((start, time));
        if start == current {
            return Ok(None);
        }
        self.history = OdomHistory::new(self.settings.retention_ns);
        self.estimator = MapToOdomEstimator::new(self.settings.estimator.clone());
        Ok(Some(OdometryRestart { previous_start: current, start }))
    }

    /// Joins the waiting keyframes odometry has reached; true when any joined.
    fn join(&mut self, unjoined: &mut Vec<UnjoinedKeyframe>) -> bool {
        let mut joined = false;
        while let Some(&(frame_id, time)) = self.waiting.front() {
            match self.estimator.add_keyframe(frame_id, time, &self.history) {
                Ok(()) => joined = true,
                Err(OdomError::Empty) => break,
                Err(OdomError::NotCovered { wanted, last, .. }) if wanted > last => break,
                Err(problem) => unjoined.push(UnjoinedKeyframe { frame_id, problem }),
            }
            self.waiting.pop_front();
        }
        joined
    }

    /// Gives up on keyframes that have waited longer than the retention before `now`.
    fn expire(&mut self, now: CameraNanos, unjoined: &mut Vec<UnjoinedKeyframe>) {
        while let Some(&(frame_id, time)) = self.waiting.front() {
            if now.0 - time.0 <= self.settings.retention_ns {
                break;
            }
            let problem = self.history.at(time).expect_err("odometry that covered the keyframe would have joined it");
            unjoined.push(UnjoinedKeyframe { frame_id, problem });
            self.waiting.pop_front();
        }
    }
}

/// A map keyframe counts as moved once optimization shifts it this far from the pose last sent for it.
const MOVED_M: f32 = 0.01;
const MOVED_RAD: f32 = 0.004;
/// A calibration whose camera axes turn further than this against the IMU from the first one is a moved camera.
const MOUNT_TURN_TOLERANCE_RAD: f32 = 1e-3;
/// ...and so is one whose camera centre moves further than this.
const MOUNT_SHIFT_TOLERANCE_M: f32 = 1e-4;

pub struct LocalizerNode {
    settings: NodeSettings,
    /// The pose last sent for every map keyframe (pose graph node): one entry per keyframe selected, so it grows
    /// with the space explored, not with time.
    sent: HashMap<FrameId, room_map::Pose>,
    gyro_gaps: u64,
    localizer: Localizer,
    gyro: GyroHeading,
    waiting: Option<Keyframe>,
    previous: Option<KeyframeTiming>,
    odometry: Option<OdometryLink>,
}

impl LocalizerNode {
    pub fn new(settings: NodeSettings) -> LocalizerNode {
        LocalizerNode {
            localizer: Localizer::new(settings.localizer.clone()),
            gyro: GyroHeading::new(settings.gyro.clone()),
            odometry: settings.odometry.clone().map(OdometryLink::new),
            settings,
            sent: HashMap::new(),
            gyro_gaps: 0,
            waiting: None,
            previous: None,
        }
    }

    pub fn localizer(&self) -> &Localizer {
        &self.localizer
    }

    pub fn on_imu(&mut self, imu: &Imu) -> Result<Handled, ImuMessageError> {
        self.gyro.add(imu)?;
        Ok(match self.waiting.take() {
            Some(keyframe) if keyframe.frame_id == imu.frame_id => {
                let camera_time = self.gyro.camera_time(FrameId(keyframe.frame_id)).expect("the imu message just added holds its time");
                self.localize_at(keyframe, camera_time)
            }
            other => {
                self.waiting = other;
                Handled::default()
            }
        })
    }

    /// Takes the colour camera's extrinsics on the IMU from the camera's calibration. Without odometry they are not
    /// needed and are ignored; a calibration whose rotations are not rotations, or that moves or turns the camera
    /// against the IMU, is an error.
    pub fn on_calibration(&mut self, calibration: &CameraCalibration) -> Result<(), OdometryInputError> {
        let Some(link) = self.odometry.as_mut() else { return Ok(()) };
        let now = CameraExtrinsics::from_calibration(calibration)?;
        match link.extrinsics {
            Some(was) if was.camera_in_imu_m.distance(now.camera_in_imu_m) > MOUNT_SHIFT_TOLERANCE_M || was.turn_from(&now) > MOUNT_TURN_TOLERANCE_RAD => {
                Err(OdometryInputError::CameraMoved {
                    was: was.camera_in_imu_m.to_array(),
                    now: now.camera_in_imu_m.to_array(),
                    turned_deg: was.turn_from(&now).to_degrees(),
                })
            }
            _ => {
                link.extrinsics = Some(now);
                Ok(())
            }
        }
    }

    pub fn on_odom_pose(&mut self, pose: &odometry::OdomPose) -> Result<Handled, OdometryInputError> {
        let link = self.odometry.as_mut().ok_or(OdometryInputError::NotConfigured)?;
        let time = CameraNanos(pose.camera_ns);
        let Some(extrinsics) = link.extrinsics else {
            return Ok(Handled { odometry_before_calibration: Some(time), ..Handled::default() });
        };
        let odometry_restarted = link.follow(CameraNanos(pose.odom_started_camera_ns), time)?;
        let Some(floor) = extrinsics.floor_pose(pose.position_m, pose.orientation) else {
            return Ok(Handled { headingless_odometry: Some(time), odometry_restarted, ..Handled::default() });
        };
        link.history.push(OdomPose { time, pose: floor })?;
        let mut handled = Handled { odometry_restarted, ..Handled::default() };
        if link.join(&mut handled.unjoined) {
            handled.map_to_odom = self.map_to_odom();
        }
        Ok(handled)
    }

    fn map_to_odom(&self) -> Option<MapToOdom> {
        let link = self.odometry.as_ref()?;
        // Before the first odom pose there is no odom frame to correct.
        let (odom_started, _) = link.odom_frame?;
        link.estimator.estimate(self.localizer.map_version(), odom_started, |frame| self.localizer.keyframe_pose(frame))
    }

    pub fn imu_stream(&self) -> ImuStream {
        self.settings.imu
    }

    pub fn on_keyframe(&mut self, keyframe: Keyframe) -> Handled {
        if self.settings.imu == ImuStream::Absent {
            let camera_time = CameraNanos(keyframe.captured_unix_ns);
            return self.localize_at(keyframe, camera_time);
        }
        if let Some(camera_time) = self.gyro.camera_time(FrameId(keyframe.frame_id)) {
            return self.localize_at(keyframe, camera_time);
        }
        let dropped = self.waiting.replace(keyframe).map(|k| FrameId(k.frame_id));
        Handled { dropped, ..Handled::default() }
    }

    fn localize_at(&mut self, keyframe: Keyframe, camera_time: CameraNanos) -> Handled {
        let frame_id = FrameId(keyframe.frame_id);
        let timing = KeyframeTiming { camera_time, specific_force: keyframe.specific_force };
        let without_gyro = self.settings.heading_without_gyro;
        let heading = self.gyro.keyframe_heading(self.previous, timing, without_gyro);
        self.previous = Some(timing);
        self.gyro_gaps += heading.gyro_problem.is_some() as u64;
        let gyro_gap = heading.gyro_problem.map(|problem| GyroGap { frame_id, problem, handled_by: without_gyro, count: self.gyro_gaps });
        let cloud = registration_cloud(&keyframe, &self.settings.cloud);
        let estimate =
            self.localizer.localize(LocalizerInput { frame_id, captured: UnixNanos(keyframe.captured_unix_ns), cloud, yaw_prior: heading.yaw_prior });
        let mut unjoined = Vec::new();
        if let Some(link) = self.odometry.as_mut() {
            match estimate.tracking {
                Tracking::Tracking | Tracking::Relocalized => link.waiting.push_back((frame_id, camera_time)),
                // Only a registered keyframe ties the odometry to the map: a restarted keyframe's map pose is dead
                // reckoning, like a lost one's.
                Tracking::Restarted | Tracking::Lost => {}
            }
            link.join(&mut unjoined);
            link.expire(camera_time, &mut unjoined);
        }
        let map_to_odom = self.map_to_odom();
        let moved = estimate.optimization.as_ref().and_then(|_| self.pose_graph_update());
        if estimate.selection == KeyframeSelection::Selected {
            self.sent.insert(frame_id, estimate.pose);
        }
        Handled { estimate: Some(estimate), gyro_gap, map_to_odom, unjoined, moved, ..Handled::default() }
    }

    /// The map keyframes already sent whose map pose has moved since it was last sent, each with the keyframes not
    /// selected that follow it; each map keyframe is counted as sent at its new pose.
    fn pose_graph_update(&mut self) -> Option<PoseGraphUpdate> {
        let mut keyframes = Vec::new();
        let as_moved = |frame_id: FrameId, pose: room_map::Pose| {
            let t = pose.translation;
            UpdatedKeyframePose { frame_id, pose: Pose4Dof { x_m: t.x, y_m: t.y, z_m: t.z, yaw_rad: pose.yaw_rad } }
        };
        for (frame_id, pose) in self.localizer.keyframe_poses() {
            let Some(sent) = self.sent.get_mut(&frame_id) else { continue };
            let shifted = sent.translation.distance(pose.translation) > MOVED_M
                || room_map::pose::wrap_angle(sent.yaw_rad - pose.yaw_rad).abs() > MOVED_RAD;
            if shifted {
                *sent = pose;
                keyframes.push(as_moved(frame_id, pose));
                keyframes.extend(self.localizer.followers(frame_id).map(|(follower, pose)| as_moved(follower, pose)));
            }
        }
        (!keyframes.is_empty()).then(|| PoseGraphUpdate { map_version: self.localizer.map_version(), keyframes })
    }
}
