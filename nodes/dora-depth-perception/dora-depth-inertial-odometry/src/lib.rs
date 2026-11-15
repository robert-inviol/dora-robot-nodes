//! The node's handling of its inputs, apart from dora: calibration, imu messages and depth frames in, odom poses
//! out.
//!
//! Nothing is tracked until the first calibration has arrived: it gives the depth intrinsics and where the depth
//! sensor sits on the IMU. Imu messages and depth frames before it are dropped. A later calibration must equal the
//! first in the parts odometry uses; one that differs is an error.
//!
//! Every imu message is fed to the tracker (it needs every sample). A depth frame is tracked once the imu message
//! leading up to it has arrived; a frame that arrives first is held for it, and a newer frame replaces a held one.
//! The tracker gives no pose until static initialization is done, so until then frames get none. From one pose to the next
//! `camera_ns` strictly increases.
//!
//! People the pose model found are cut out of each depth frame before it is tracked (see [`people_mask`]).
//!
//! While depth frames do not correct the pose, the imu alone carries it (dead reckoning) and the filter's position
//! uncertainty grows. Once it is beyond `lost_beyond_position_sigma_m` on a frame that did not correct it, the pose can
//! no longer be found again against the map: odometry is lost. It sends no pose for that frame, and starts again, in
//! a new odom frame, once the body rests.

pub mod people_mask;
pub mod recording;

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use depth_inertial_odometry::depth_image::{DepthImageError, PinholeIntrinsics};
use depth_inertial_odometry::imu_pairing::{AccelReading, GyroReading, ImuStream};
use depth_inertial_odometry::odometry::{CloudCounts, StaticInitializationError};
use depth_inertial_odometry::downsampling::{NotARotation, SensorExtrinsics};
use depth_inertial_odometry::tracker::{CloudUse, DepthFrameId, DepthFrameOutcome, TrackerError, TrackerPhase};
use depth_inertial_odometry::{FramePose, PlaneVoxelMap, PlaneVoxelSettings, PoseTracker, TrackerSettings};
use dora_node_api::arrow::array::{Array, AsArray};
use dora_node_api::arrow::datatypes::UInt16Type;
use dora_node_api::{MetadataParameters, Parameter};
use glam::Vec3;
use perception_wire::camera::{CameraCalibration, RigidTransform, StreamIntrinsics};
use perception_wire::localization::CameraNanos;
use perception_wire::odometry::{OdomPose, DegenerateDirection};
use perception_wire::people::PersonPoses;
use perception_wire::Imu;

use crate::people_mask::{covered_pixels, cut_out, DepthRect, MaskSettings, PeopleMask, WrongImageSize};

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
    pub fn sensor_extrinsics(&self) -> Result<SensorExtrinsics, NotARotation> {
        // Both sides store the rotation column major.
        SensorExtrinsics::from_depth_to_imu(self.depth_to_imu.rotation, self.depth_to_imu.translation_m)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NodeSettings {
    pub tracker: TrackerSettings,
    pub map: PlaneVoxelSettings,
    pub people: MaskSettings,
    /// Odometry is lost on a frame whose cloud did not correct the pose when the pose's largest one-sigma position
    /// uncertainty against the map (`FramePose::largest_position_sigma_m`) is beyond this many metres.
    pub lost_beyond_position_sigma_m: f64,
}

impl NodeSettings {
    /// Settings for an Intel RealSense D455 indoors streaming 848x480 depth.
    pub fn d455() -> NodeSettings {
        NodeSettings {
            tracker: TrackerSettings::d455(),
            map: PlaneVoxelSettings::indoors(),
            people: MaskSettings::DEFAULT,
            // A sixth of a 25 cm voxel: at three sigmas the pose is still within half a voxel of the truth, so most
            // points still land in the voxel whose plane they lie on and the next clear view corrects it.
            lost_beyond_position_sigma_m: PlaneVoxelSettings::indoors().voxel_m as f64 / 6.0,
        }
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

impl<'a> DepthFrame<'a> {
    /// A depth frame as the camera node sends it: `u16` millimetres, with its `frame_id`, `camera_ns`, `width` and
    /// `height` in the metadata.
    pub fn from_input(metadata: &MetadataParameters, data: &'a dyn Array) -> eyre::Result<DepthFrame<'a>> {
        fn integer(metadata: &MetadataParameters, name: &str) -> eyre::Result<i64> {
            match metadata.get(name) {
                Some(Parameter::Integer(value)) => Ok(*value),
                other => eyre::bail!("depth metadata {name} is {other:?}, not an integer"),
            }
        }
        fn unsigned<T: TryFrom<i64>>(metadata: &MetadataParameters, name: &str) -> eyre::Result<T> {
            let value = integer(metadata, name)?;
            T::try_from(value).map_err(|_| eyre::eyre!("depth metadata {name} is {value}, which is out of range"))
        }
        let depth_mm = data.as_primitive_opt::<UInt16Type>().ok_or_else(|| eyre::eyre!("depth is {:?}, not u16", data.data_type()))?.values();
        Ok(DepthFrame {
            frame_id: DepthFrameId(unsigned(metadata, "frame_id")?),
            time: CameraNanos(integer(metadata, "camera_ns")?),
            width: unsigned(metadata, "width")?,
            height: unsigned(metadata, "height")?,
            depth_mm,
        })
    }
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
        camera_ns: camera_ns(pose.time)?,
        odom_started_camera_ns: camera_ns(pose.odometry_started)?,
        depth_frame_id: pose.frame_id.0,
        position_m: pose.position_m.as_vec3(),
        orientation: pose.orientation.as_quat(),
        velocity_m_s: pose.velocity_m_s.as_vec3(),
        position_sigma_m: pose.uncertainty.position_sigma_m.as_vec3(),
        tilt_sigma_rad: pose.uncertainty.tilt_sigma_rad as f32,
        heading_sigma_rad: pose.uncertainty.heading_sigma_rad as f32,
        corrected: pose.corrected(),
        matched_points: u32::try_from(pose.matched_points()).map_err(|_| UnrepresentablePose::MatchedPoints(pose.matched_points()))?,
        degenerate: pose.degenerate().map(|degenerate| DegenerateDirection { direction: degenerate.direction.as_vec3(), sigma_m: degenerate.sigma_m as f32 }),
    })
}

fn camera_ns(time: Duration) -> Result<i64, UnrepresentablePose> {
    i64::try_from(time.as_nanos()).map_err(|_| UnrepresentablePose::Time(time))
}

/// The calibration odometry started with and a later one that differs from it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationChange {
    pub started: DepthCalibration,
    pub now: DepthCalibration,
}

#[derive(Debug, thiserror::Error)]
pub enum OdometryNodeError {
    #[error("the calibration gives no depth sensor extrinsics: {0}")]
    Extrinsics(#[from] NotARotation),
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
    #[error(transparent)]
    PersonPoses(#[from] WrongImageSize),
}

/// What a calibration message did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CalibrationOutcome {
    /// The first calibration: the tracker now exists and waits for the body to be stationary (static initialization).
    TrackerBuilt(DepthCalibration),
    /// The calibration odometry already runs on.
    Unchanged,
}

/// How far the start of odometry has come.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StartProgress {
    /// Static initialization: the body has been stationary this long of the time it must.
    StaticInitialization { stationary_so_far: Duration },
    /// The body moved during static initialization; its window has begun again.
    NotStationary(StaticInitializationError),
    /// Odometry started with this imu message.
    Started,
    Tracking,
}

/// What became of a depth frame the tracker was given.
#[derive(Clone, Debug, PartialEq)]
pub enum FrameOutcome {
    /// The frame was taken before odometry started; there is no pose for it.
    BeforeStart,
    Posed(OdomPose),
    /// No depth frame corrected the pose for `uncorrected_for`, up to this frame, and its position uncertainty grew
    /// beyond the bound: odometry starts again once the body rests.
    Lost { uncorrected_for: Duration, largest_position_sigma_m: f64, last_second: Box<FramesSummary> },
}

/// What became of one tracked depth frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameRecord {
    /// When the frame was taken.
    pub time: Duration,
    /// Share of the depth image cut out as people, from 0 to 1.
    pub cut_fraction: f32,
    pub counts: CloudCounts,
    pub corrected: bool,
    /// The cloud could not correct the pose (the first cloud of a run starts the map and is neither).
    pub refused: bool,
    pub largest_position_sigma_m: f64,
    /// Wall-clock time the frame took to track: cutting out people, projecting, the odometry.
    pub tracking_time: Duration,
}

/// The smallest, the median and the largest of some values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spread<T> {
    pub min: T,
    pub median: T,
    pub max: T,
}

impl<T: Copy + PartialOrd> Spread<T> {
    /// None when there are no values.
    fn of(values: &mut [T]) -> Option<Spread<T>> {
        values.sort_unstable_by(|a, b| a.partial_cmp(b).expect("the values are ordered"));
        Some(Spread { min: *values.first()?, median: values[values.len() / 2], max: *values.last()? })
    }
}

impl<T: std::fmt::Display> std::fmt::Display for Spread<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}/{}", self.min, self.median, self.max)
    }
}

/// How the points of the frames whose cloud did not correct the pose fared: smallest, median and largest of each count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefusedClouds {
    pub frames: usize,
    pub given: Spread<usize>,
    pub downsampled: Spread<usize>,
    pub found_surface: Spread<usize>,
    pub within_gate: Spread<usize>,
    pub needed: usize,
}

/// A span of tracked frames, for the log.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FramesSummary {
    pub frames: usize,
    pub corrected: usize,
    /// Frames with people cut out of them.
    pub with_people: usize,
    /// Share of each frame cut out as people, over the frames with people.
    pub cut_fraction: Option<Spread<f32>>,
    /// Absent when every frame was corrected.
    pub refused: Option<RefusedClouds>,
    pub largest_position_sigma_m: Option<Spread<f64>>,
    pub tracking_ms: Option<Spread<f64>>,
}

impl FramesSummary {
    pub fn of(records: &[FrameRecord]) -> FramesSummary {
        let spread = |pick: &dyn Fn(&FrameRecord) -> Option<f64>| Spread::of(&mut records.iter().filter_map(pick).collect::<Vec<f64>>());
        let refused: Vec<&FrameRecord> = records.iter().filter(|record| record.refused).collect();
        let counts = |pick: fn(&CloudCounts) -> usize| Spread::of(&mut refused.iter().map(|record| pick(&record.counts)).collect::<Vec<usize>>());
        let refused = match (counts(|c| c.given), counts(|c| c.downsampled), counts(|c| c.found_surface), counts(|c| c.within_gate)) {
            (Some(given), Some(downsampled), Some(found_surface), Some(within_gate)) => {
                Some(RefusedClouds { frames: refused.len(), given, downsampled, found_surface, within_gate, needed: refused[0].counts.needed })
            }
            _ => None,
        };
        let round = |value: f64, places: i32| (value * 10f64.powi(places)).round() / 10f64.powi(places);
        FramesSummary {
            frames: records.len(),
            corrected: records.iter().filter(|record| record.corrected).count(),
            with_people: records.iter().filter(|record| record.cut_fraction > 0.0).count(),
            cut_fraction: spread(&|record| (record.cut_fraction > 0.0).then_some(round(f64::from(record.cut_fraction), 2))).map(|s| Spread { min: s.min as f32, median: s.median as f32, max: s.max as f32 }),
            refused,
            largest_position_sigma_m: spread(&|record| Some(round(record.largest_position_sigma_m, 4))),
            tracking_ms: spread(&|record| Some(round(record.tracking_time.as_secs_f64() * 1e3, 2))),
        }
    }
}

impl std::fmt::Display for FramesSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} frames, {} corrected, {} with people", self.frames, self.corrected, self.with_people)?;
        if let Some(cut) = self.cut_fraction {
            write!(f, " (share cut out min/median/max {cut})")?;
        }
        if let Some(refused) = self.refused {
            let RefusedClouds { frames, given, downsampled, found_surface, within_gate, needed } = refused;
            write!(f, "; {frames} refused, points min/median/max: given {given}, downsampled {downsampled}, found a surface {found_surface}, within the gate {within_gate}, {needed} needed")?;
        }
        if let Some(sigma) = self.largest_position_sigma_m {
            write!(f, "; largest position sigma {sigma} m")?;
        }
        if let Some(time) = self.tracking_ms {
            write!(f, "; tracking {time} ms")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ImuOutcome {
    /// Dropped: no calibration has arrived.
    BeforeCalibration,
    Fed {
        start: StartProgress,
        /// The frame that was held for this imu message.
        held_frame: Option<FrameOutcome>,
    },
}

#[derive(Clone, Debug, PartialEq)]
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
    /// The frame being tracked with people cut out of it.
    cut_depth_mm: Vec<u16>,
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
        let tracker = PoseTracker::new(calibration.pinhole(), calibration.sensor_extrinsics()?, PlaneVoxelMap::new(settings.map), settings.tracker)
            .map_err(OdometryNodeError::DepthStream)?;
        Ok(Tracking {
            calibration,
            tracker,
            cut_depth_mm: Vec::new(),
            started: false,
            imu_through: None,
            held: None,
            accel: Vec::new(),
            gyro: Vec::new(),
        })
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
            Ok(TrackerPhase::StaticInitialization { stationary_so_far }) => StartProgress::StaticInitialization { stationary_so_far },
            Ok(TrackerPhase::Tracking) if self.started => StartProgress::Tracking,
            Ok(TrackerPhase::Tracking) => StartProgress::Started,
            Err(TrackerError::NotStationary(moving)) => StartProgress::NotStationary(moving),
            Err(other) => return Err(OdometryNodeError::Tracker(other)),
        };
        self.started = matches!(start, StartProgress::Started | StartProgress::Tracking);
        self.imu_through = Some(frame_id);
        Ok(start)
    }

    /// Tracks a frame with the people in `cut` cut out of it; None before odometry has started.
    fn add_frame(&mut self, frame_id: DepthFrameId, time: Duration, depth_mm: &[u16], cut: &[DepthRect]) -> Result<Option<FramePose>, OdometryNodeError> {
        let depth_mm = cut_out(depth_mm, self.calibration.pinhole().width, cut, &mut self.cut_depth_mm);
        match self.tracker.add_depth_frame(time, frame_id, depth_mm).map_err(OdometryNodeError::Tracker)? {
            DepthFrameOutcome::AwaitingStart => Ok(None),
            DepthFrameOutcome::Posed(pose) => Ok(Some(pose)),
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
    /// The first calibration, for carrying people's boxes from the colour image into the depth image.
    camera: Option<CameraCalibration>,
    people: PeopleMask,
    /// Frames tracked since the progress was last taken.
    since_progress: Vec<FrameRecord>,
    /// Frames tracked in the last [`OdometryNode::RECENT`] of camera time.
    recent: VecDeque<FrameRecord>,
    spans: Vec<(usize, usize)>,
}

impl OdometryNode {
    /// The span of frames a loss is reported with.
    pub const RECENT: Duration = Duration::from_secs(1);

    pub fn new(settings: NodeSettings) -> OdometryNode {
        OdometryNode { settings, tracking: None, camera: None, people: PeopleMask::default(), since_progress: Vec::new(), recent: VecDeque::new(), spans: Vec::new() }
    }

    /// The odometry's map, in its world frame, once odometry has started.
    pub fn surface_map(&self) -> Option<&PlaneVoxelMap> {
        Some(self.tracking.as_deref()?.tracker.odometry()?.map())
    }

    /// The frames tracked since the last call.
    pub fn take_progress(&mut self) -> FramesSummary {
        let summary = FramesSummary::of(&self.since_progress);
        self.since_progress.clear();
        summary
    }

    fn record(&mut self, record: FrameRecord) {
        self.since_progress.push(record);
        while self.recent.front().is_some_and(|oldest| oldest.time + Self::RECENT <= record.time) {
            self.recent.pop_front();
        }
        self.recent.push_back(record);
    }

    /// Keeps the people of the newest pose frame, to cut out of the depth frames that follow; dropped before the
    /// first calibration.
    pub fn on_person_poses(&mut self, poses: &PersonPoses) -> Result<(), OdometryNodeError> {
        let Some(camera) = &self.camera else { return Ok(()) };
        Ok(self.people.update(poses, camera)?)
    }

    /// Tracks a frame, and starts odometry again from rest when it is lost.
    fn track(&mut self, frame_id: DepthFrameId, time: Duration, depth_mm: &[u16]) -> Result<FrameOutcome, OdometryNodeError> {
        let settings = self.settings;
        let began = Instant::now();
        let camera = self.camera.as_ref().expect("the camera calibration comes with the tracker");
        let cut = self.people.rects_for(CameraNanos(camera_ns(time)?), camera, &settings.people);
        let tracking = self.tracking.as_deref_mut().expect("frames are tracked only once calibrated");
        let image = tracking.calibration.pinhole();
        let cut_fraction = covered_pixels(cut, &mut self.spans) as f32 / (image.width * image.height) as f32;
        let Some(pose) = tracking.add_frame(frame_id, time, depth_mm, cut)? else { return Ok(FrameOutcome::BeforeStart) };
        let tracking_time = began.elapsed();
        let calibration = tracking.calibration;
        self.record(FrameRecord { time, cut_fraction, counts: pose.counts, corrected: pose.corrected(), refused: matches!(pose.cloud, CloudUse::NotCorrected(_)), largest_position_sigma_m: pose.largest_position_sigma_m, tracking_time });
        if !pose.corrected() && pose.largest_position_sigma_m > settings.lost_beyond_position_sigma_m {
            let last_second = FramesSummary::of(self.recent.make_contiguous());
            self.recent.clear();
            self.tracking = Some(Box::new(Tracking::new(calibration, &settings)?));
            return Ok(FrameOutcome::Lost { uncorrected_for: pose.dead_reckoning_for, largest_position_sigma_m: pose.largest_position_sigma_m, last_second: Box::new(last_second) });
        }
        Ok(FrameOutcome::Posed(odom_pose(&pose)?))
    }

    pub fn on_calibration(&mut self, calibration: &CameraCalibration) -> Result<CalibrationOutcome, OdometryNodeError> {
        let now = DepthCalibration::of(calibration);
        match &self.tracking {
            None => {
                self.tracking = Some(Box::new(Tracking::new(now, &self.settings)?));
                self.camera = Some(*calibration);
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
            Some(held) => Some(self.track(held.frame_id, held.time, &held.depth_mm)?),
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
            return Ok(DepthOutcome::Tracked(self.track(frame_id, time, depth_mm)?));
        }
        let replaced = tracking.held.replace(HeldFrame { frame_id, time, depth_mm: depth_mm.to_vec() });
        Ok(DepthOutcome::HeldForImu { replaced: replaced.map(|held| held.frame_id) })
    }
}

#[cfg(test)]
mod tests {
    use depth_inertial_odometry::filter::{MeasurementUpdateError, LevelledUncertainty};
    use depth_inertial_odometry::tracker;
    use glam::{DQuat, DVec3, Quat};

    use super::*;

    /// A pose whose every value a single-precision float holds exactly.
    fn frame_pose(cloud: CloudUse) -> FramePose {
        FramePose {
            time: Duration::new(1_762_560, 133_000_001),
            odometry_started: Duration::new(1_762_550, 7),
            frame_id: DepthFrameId(43),
            position_m: DVec3::new(1.25, -0.5, 0.25),
            orientation: DQuat::from_xyzw(0.5, -0.5, 0.5, 0.5),
            velocity_m_s: DVec3::new(0.375, 0.0, -0.0625),
            uncertainty: LevelledUncertainty { position_sigma_m: DVec3::new(0.015625, 0.03125, 0.0078125), tilt_sigma_rad: 0.00390625, heading_sigma_rad: 0.0078125 },
            largest_position_sigma_m: 0.03125,
            cloud,
            counts: CloudCounts::default(),
            dead_reckoning_for: Duration::ZERO,
        }
    }

    #[test]
    fn a_corrected_pose_becomes_an_odom_pose_with_every_field_carried_over() {
        let degenerate = tracker::DegenerateDirection { direction: DVec3::new(0.0, -1.0, 0.0), sigma_m: 0.125 };
        let message = odom_pose(&frame_pose(CloudUse::Corrected { matched_points: 1234, degenerate })).unwrap();
        let expected = OdomPose {
            camera_ns: 1_762_560_133_000_001,
            odom_started_camera_ns: 1_762_550_000_000_007,
            depth_frame_id: 43,
            position_m: Vec3::new(1.25, -0.5, 0.25),
            orientation: Quat::from_xyzw(0.5, -0.5, 0.5, 0.5),
            velocity_m_s: Vec3::new(0.375, 0.0, -0.0625),
            position_sigma_m: Vec3::new(0.015625, 0.03125, 0.0078125),
            tilt_sigma_rad: 0.00390625,
            heading_sigma_rad: 0.0078125,
            corrected: true,
            matched_points: 1234,
            degenerate: Some(DegenerateDirection { direction: Vec3::NEG_Y, sigma_m: 0.125 }),
        };
        assert_eq!(message, expected);
    }

    #[test]
    fn a_pose_the_cloud_did_not_correct_has_no_degenerate_direction_and_no_matched_points() {
        for cloud in [CloudUse::StartedMap, CloudUse::NotCorrected(MeasurementUpdateError::TooFewRows { rows: 3, needed: 50 })] {
            let message = odom_pose(&frame_pose(cloud)).unwrap();
            assert_eq!((message.corrected, message.matched_points, message.degenerate), (false, 0, None), "{cloud:?}");
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
        // A turn about no axis of symmetry, so reading the matrix row major would give other extrinsics.
        let rotation = glam::Mat3::from_quat(Quat::from_euler(glam::EulerRot::ZYX, 0.7, -0.4, 1.1));
        let depth_to_imu = RigidTransform { rotation: rotation.to_cols_array(), translation_m: [0.03, -0.01, 0.02] };
        assert!((rotation - rotation.transpose()).to_cols_array().iter().any(|entry| entry.abs() > 0.3));
        let calibration = DepthCalibration { intrinsics: StreamIntrinsics { width: 848, height: 480, fx: 420.0, fy: 420.0, ppx: 424.0, ppy: 240.0 }, depth_to_imu };
        let extrinsics = calibration.sensor_extrinsics().unwrap();
        for point in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(0.4, -1.2, 2.5)] {
            let in_body = extrinsics.rotation * point.as_dvec3() + extrinsics.translation_m;
            assert!((in_body - depth_to_imu.apply(point).as_dvec3()).length() < 1e-6, "{point}: {in_body}");
        }
    }
}
