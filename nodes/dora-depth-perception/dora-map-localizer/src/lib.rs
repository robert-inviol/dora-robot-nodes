//! The node's handling of its inputs, apart from dora: keyframes, imu messages and odometry in, keyframe poses and
//! the map to odom correction out.
//!
//! Every imu message is kept (the gyro needs every sample). A keyframe is localised once the imu message
//! leading up to it has arrived, since that carries its camera time; a keyframe that arrives first waits for
//! it, and a newer keyframe replaces a waiting one. With `imu = "absent"` there is no imu stream: a keyframe is
//! localised as it arrives, timed by its capture time, and every heading is searched for. A keyframe whose turn the gyro cannot give is handled as
//! `heading_without_gyro` says, and reported as a `GyroGap`.
//!
//! With an `[odometry]` section, every registered keyframe (tracked or relocalised) is joined to the odometry pose at
//! its camera time once odometry has reached that time, and the correction is sent after every keyframe and every
//! odometry pose that joined one. A keyframe that waits longer than the retention, or that falls before the
//! odometry kept, is left out and reported in `unjoined`.

pub mod config;

use std::collections::VecDeque;

use map_localizer::cloud::registration_cloud;
use map_localizer::gyro::{GyroHeading, KeyframeTiming};
use map_localizer::map_to_odom::{MapToOdomEstimator, OdomError, OdomHistory, OdomPose};
use map_localizer::{HeadingWithoutGyro, KeyframeEstimate, Localizer, LocalizerInput};
use perception_wire::localization::{CameraNanos, FrameId, MapToOdom, Tracking};
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
    #[error(transparent)]
    Odometry(#[from] OdomError),
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
}

/// Odometry, and the registered keyframes waiting for it to reach their camera time.
struct OdometryLink {
    settings: OdometrySettings,
    history: OdomHistory,
    estimator: MapToOdomEstimator,
    /// Oldest first.
    waiting: VecDeque<(FrameId, CameraNanos)>,
}

impl OdometryLink {
    fn new(settings: OdometrySettings) -> OdometryLink {
        OdometryLink {
            history: OdomHistory::new(settings.retention_ns),
            estimator: MapToOdomEstimator::new(settings.estimator.clone()),
            settings,
            waiting: VecDeque::new(),
        }
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

pub struct LocalizerNode {
    settings: NodeSettings,
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
            waiting: None,
            previous: None,
        }
    }

    pub fn localizer(&self) -> &Localizer {
        &self.localizer
    }

    pub fn on_imu(&mut self, imu: &Imu) -> Handled {
        self.gyro.add(imu);
        match self.waiting.take() {
            Some(keyframe) if keyframe.frame_id == imu.frame_id => {
                let camera_time = self.gyro.camera_time(FrameId(keyframe.frame_id)).expect("the imu message just added holds its time");
                self.localize_at(keyframe, camera_time)
            }
            other => {
                self.waiting = other;
                Handled::default()
            }
        }
    }

    pub fn on_odom_pose(&mut self, pose: &odometry::OdomPose) -> Result<Handled, OdometryInputError> {
        let link = self.odometry.as_mut().ok_or(OdometryInputError::NotConfigured)?;
        let time = CameraNanos(pose.camera_ns);
        let Some(floor) = link.settings.mount.floor_pose(pose.position_m, pose.orientation) else {
            return Ok(Handled { headingless_odometry: Some(time), ..Handled::default() });
        };
        link.history.push(OdomPose { time, pose: floor })?;
        let mut handled = Handled::default();
        if link.join(&mut handled.unjoined) {
            handled.map_to_odom = self.map_to_odom();
        }
        Ok(handled)
    }

    fn map_to_odom(&self) -> Option<MapToOdom> {
        let link = self.odometry.as_ref()?;
        link.estimator.estimate(self.localizer.map_version(), |frame| self.localizer.keyframe_pose(frame))
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
        let timing = KeyframeTiming { camera_time, gravity_reaction: keyframe.gravity_reaction };
        let without_gyro = self.settings.heading_without_gyro;
        let heading = self.gyro.keyframe_heading(self.previous, timing, without_gyro);
        self.previous = Some(timing);
        let gyro_gap = heading.gyro_problem.map(|problem| GyroGap { frame_id, problem, handled_by: without_gyro });
        let cloud = registration_cloud(&keyframe, &self.settings.cloud);
        let estimate =
            self.localizer.localize(LocalizerInput { frame_id, captured: UnixNanos(keyframe.captured_unix_ns), cloud, heading: heading.hint });
        let mut unjoined = Vec::new();
        if let Some(link) = self.odometry.as_mut() {
            if estimate.tracking != Tracking::Lost {
                link.waiting.push_back((frame_id, camera_time));
            }
            link.join(&mut unjoined);
            link.expire(camera_time, &mut unjoined);
        }
        let map_to_odom = self.map_to_odom();
        Handled { estimate: Some(estimate), gyro_gap, map_to_odom, unjoined, ..Handled::default() }
    }
}
