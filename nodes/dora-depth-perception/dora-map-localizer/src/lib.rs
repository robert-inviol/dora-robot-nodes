//! The node's handling of its inputs, apart from dora: keyframes and imu messages in, keyframe poses out.
//!
//! Every imu message is kept (the gyro needs every sample). A keyframe is localised once the imu message
//! leading up to it has arrived, since that carries its camera time; a keyframe that arrives first waits for
//! it, and a newer keyframe replaces a waiting one. With `imu = "absent"` there is no imu stream: a keyframe is
//! localised as it arrives, timed by its capture time, and every heading is searched for. A keyframe whose turn the gyro cannot give is handled as
//! `heading_without_gyro` says, and reported as a `GyroGap`.

pub mod config;

use map_localizer::cloud::registration_cloud;
use map_localizer::gyro::{GyroHeading, KeyframeTiming};
use map_localizer::{HeadingWithoutGyro, KeyframeEstimate, Localizer, LocalizerInput};
use perception_wire::localization::{CameraNanos, FrameId};
use perception_wire::navigation::UnixNanos;
use perception_wire::{Imu, Keyframe};
use room_map::heading::HeadingError;

use crate::config::{ImuStream, NodeSettings};

/// A keyframe whose turn the gyro could not give.
#[derive(Clone, Debug, PartialEq)]
pub struct GyroGap {
    pub frame_id: FrameId,
    pub problem: HeadingError,
    /// Refuse: the keyframe was reported lost. Search: its heading was searched for.
    pub handled_by: HeadingWithoutGyro,
}

/// What handling one input did.
#[derive(Debug, Default)]
pub struct Handled {
    pub estimate: Option<KeyframeEstimate>,
    /// A keyframe that was waiting for its imu message and was replaced by a newer one.
    pub dropped: Option<FrameId>,
    pub gyro_gap: Option<GyroGap>,
}

pub struct LocalizerNode {
    settings: NodeSettings,
    localizer: Localizer,
    gyro: GyroHeading,
    waiting: Option<Keyframe>,
    previous: Option<KeyframeTiming>,
}

impl LocalizerNode {
    pub fn new(settings: NodeSettings) -> LocalizerNode {
        LocalizerNode {
            localizer: Localizer::new(settings.localizer.clone()),
            gyro: GyroHeading::new(settings.gyro.clone()),
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
        Handled { estimate: Some(estimate), dropped: None, gyro_gap }
    }
}
