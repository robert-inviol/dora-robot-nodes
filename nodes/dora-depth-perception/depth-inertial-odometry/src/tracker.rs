//! From a camera's raw streams to a pose per depth frame: the start at rest, the pairing of the IMU's
//! two streams, the projection of each depth image and the odometry itself, behind one type.
//!
//! Feed [`PoseTracker::add_imu`] every batch of IMU samples and [`PoseTracker::add_depth_frame`] every
//! depth image, the IMU batch that leads up to a frame before the frame. The tracker first waits for
//! the body to rest for a configured time, then answers every depth frame with a [`FramePose`].

use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::time::Duration;

use glam::{DQuat, DVec3, Vec3};

use crate::depth_image::{DepthImageError, DepthProjector, PinholeIntrinsics};
use crate::filter::{CorrectionError, LevelledUncertainty};
use crate::imu_pairing::{AccelReading, GyroReading, ImuOutOfOrder, ImuPairer, ImuStream};
use crate::odometry::{CloudOutcome, DepthInertialOdometry, ImuSample, OdometryError, OdometrySettings, StillStart, StillStartError};
use crate::surfaces::SurfaceMap;
use crate::thinning::SensorMount;

/// The camera's own number for a depth frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DepthFrameId(pub u64);

#[derive(Clone, Copy, Debug)]
pub struct TrackerSettings {
    /// Its `sensor_mount` is replaced by the mount the tracker is given.
    pub odometry: OdometrySettings,
    /// How long the body must rest before odometry starts.
    pub rest_for: Duration,
    /// Every this many pixels of every this many rows of a depth image become cloud points.
    pub depth_stride: NonZeroUsize,
    /// How far a depth frame's time may be ahead of the newest sample of either IMU stream. Over that
    /// gap the newest reading is held.
    pub max_imu_lag: Duration,
    /// The longest span of camera time over the samples held for what they wait on: one IMU stream's
    /// samples for the other stream, or paired samples for a depth frame. A longer span is refused.
    pub max_backlog: Duration,
}

impl TrackerSettings {
    /// Settings for an Intel RealSense D455 indoors streaming 848x480 depth.
    pub fn d455() -> TrackerSettings {
        TrackerSettings {
            odometry: OdometrySettings::d455(),
            rest_for: Duration::from_secs(2),
            depth_stride: NonZeroUsize::new(4).expect("four is not zero"),
            max_imu_lag: Duration::from_millis(10),
            max_backlog: Duration::from_secs(1),
        }
    }
}

/// What the tracker is doing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrackerPhase {
    /// Waiting for the body to have rested long enough; it has rested this long so far.
    Resting { rested: Duration },
    Tracking,
}

/// The direction of travel a cloud pinned down least, in the level frame, and the one-sigma
/// uncertainty the cloud alone leaves along it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeakestDirection {
    pub direction: DVec3,
    pub sigma_m: f64,
}

/// What a depth frame's cloud did to the estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CloudUse {
    /// The first cloud: it became the map and corrected nothing.
    StartedMap,
    Corrected {
        /// Points that found a surface in the map.
        matched_points: usize,
        weakest: WeakestDirection,
    },
    /// The cloud could not correct the estimate, which is the IMU's prediction; the cloud stayed out
    /// of the map.
    NotCorrected(CorrectionError),
}

/// The estimate at one depth frame, in the level frame: fixed where the body rested at the start,
/// with z up. From one pose to the next `time` strictly increases.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FramePose {
    /// When the depth frame was taken, on the camera's clock.
    pub time: Duration,
    pub frame_id: DepthFrameId,
    /// Where the body (the IMU) is.
    pub position_m: DVec3,
    /// Turns a body-frame vector into the level frame.
    pub orientation: DQuat,
    pub velocity_m_s: DVec3,
    /// How well the pose is known against the odometry's own map; the map's drift is not in it.
    pub uncertainty: LevelledUncertainty,
    pub cloud: CloudUse,
}

impl FramePose {
    pub fn corrected(&self) -> bool {
        matches!(self.cloud, CloudUse::Corrected { .. })
    }

    /// Points of the cloud that corrected the estimate; none when it did not.
    pub fn matched_points(&self) -> usize {
        match self.cloud {
            CloudUse::Corrected { matched_points, .. } => matched_points,
            CloudUse::StartedMap | CloudUse::NotCorrected(_) => 0,
        }
    }

    pub fn weakest(&self) -> Option<WeakestDirection> {
        match self.cloud {
            CloudUse::Corrected { weakest, .. } => Some(weakest),
            CloudUse::StartedMap | CloudUse::NotCorrected(_) => None,
        }
    }
}

/// What the tracker made of a depth frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DepthFrameOutcome {
    /// The frame was taken before odometry started; there is no pose for it.
    AwaitingStart,
    Posed(FramePose),
}

/// What held IMU samples wait on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Awaited {
    /// The stream whose samples the other stream's are paired with.
    Imu(ImuStream),
    DepthFrame,
}

impl std::fmt::Display for Awaited {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Awaited::Imu(stream) => write!(f, "{stream} sample"),
            Awaited::DepthFrame => f.write_str("depth frame"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrackerError {
    Imu(ImuOutOfOrder),
    /// The body did not rest through the window just collected. Collection has begun again with the
    /// samples after that window, so feeding on tries the next one.
    NotAtRest(StillStartError),
    DepthImage(DepthImageError),
    /// A depth frame that is not newer than the last frame that got a pose.
    FrameOutOfOrder { at: Duration, previous: Duration },
    /// A depth frame further ahead of a stream's newest sample than the settings allow; `latest` is
    /// that sample's time, if the stream has given any. The frame is left out.
    ImuBehind { frame_at: Duration, stream: ImuStream, latest: Option<Duration> },
    /// Samples spanning more camera time than the settings allow are held for something that has
    /// not come. They stay held, so every later batch is refused the same way.
    Backlog { waiting_for: Awaited, span: Duration },
    Odometry(OdometryError),
}

impl std::fmt::Display for TrackerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackerError::Imu(cause) => write!(f, "{cause}"),
            TrackerError::NotAtRest(cause) => write!(f, "odometry did not start: {cause}"),
            TrackerError::DepthImage(cause) => write!(f, "{cause}"),
            TrackerError::FrameOutOfOrder { at, previous } => write!(f, "depth frame at {at:?} is not newer than the last posed frame at {previous:?}"),
            TrackerError::ImuBehind { frame_at, stream, latest: Some(latest) } => write!(f, "depth frame at {frame_at:?} is too far ahead of the newest {stream} sample at {latest:?}"),
            TrackerError::ImuBehind { frame_at, stream, latest: None } => write!(f, "depth frame at {frame_at:?} arrived before any {stream} sample"),
            TrackerError::Backlog { waiting_for, span } => write!(f, "IMU samples spanning {span:?} are held and no {waiting_for} has come for them"),
            TrackerError::Odometry(cause) => write!(f, "{cause}"),
        }
    }
}

impl std::error::Error for TrackerError {}

enum Phase<M> {
    Resting {
        /// Held for the odometry, which takes it at the start.
        map: Option<M>,
        window: Vec<ImuSample>,
    },
    Tracking {
        odometry: Box<DepthInertialOdometry<M>>,
        /// Paired samples the odometry has not been given: it takes them up to each depth frame's time.
        pending: VecDeque<ImuSample>,
        started_at: Duration,
    },
}

pub struct PoseTracker<M> {
    settings: TrackerSettings,
    projector: DepthProjector,
    pairer: ImuPairer,
    phase: Phase<M>,
    last_posed: Option<Duration>,
    paired: Vec<ImuSample>,
    sensor_points: Vec<Vec3>,
}

impl<M: SurfaceMap> PoseTracker<M> {
    /// A tracker for a camera whose depth stream has these intrinsics and sits on the IMU at `mount`.
    pub fn new(intrinsics: PinholeIntrinsics, mount: SensorMount, map: M, settings: TrackerSettings) -> Result<PoseTracker<M>, DepthImageError> {
        Ok(PoseTracker {
            settings: TrackerSettings { odometry: OdometrySettings { sensor_mount: mount, ..settings.odometry }, ..settings },
            projector: DepthProjector::new(intrinsics, settings.depth_stride)?,
            pairer: ImuPairer::default(),
            phase: Phase::Resting { map: Some(map), window: Vec::new() },
            last_posed: None,
            paired: Vec::new(),
            sensor_points: Vec::new(),
        })
    }

    pub fn phase(&self) -> TrackerPhase {
        match &self.phase {
            Phase::Resting { window, .. } => TrackerPhase::Resting { rested: window.first().zip(window.last()).map_or(Duration::ZERO, |(first, last)| last.time - first.time) },
            Phase::Tracking { .. } => TrackerPhase::Tracking,
        }
    }

    /// The odometry, once it has started. Its estimate stands at the last depth frame's time.
    pub fn odometry(&self) -> Option<&DepthInertialOdometry<M>> {
        match &self.phase {
            Phase::Resting { .. } => None,
            Phase::Tracking { odometry, .. } => Some(odometry),
        }
    }

    /// Takes every accelerometer and gyro sample since the previous batch, each stream oldest first.
    ///
    /// While resting, the samples fill a window of the configured length; when it is full, odometry
    /// starts if the body rested through it, and [`TrackerError::NotAtRest`] is returned if it did not.
    /// Samples wait for the other stream to pair with and, once odometry has started, for a depth
    /// frame; [`TrackerError::Backlog`] is returned when those held span more than the settings allow.
    pub fn add_imu(&mut self, accel: &[AccelReading], gyro: &[GyroReading]) -> Result<TrackerPhase, TrackerError> {
        self.paired.clear();
        self.pairer.add_batch(accel, gyro, &mut self.paired).map_err(TrackerError::Imu)?;
        for (held, silent) in [(ImuStream::Gyro, ImuStream::Accelerometer), (ImuStream::Accelerometer, ImuStream::Gyro)] {
            self.within_backlog(self.pairer.held_span(held), Awaited::Imu(silent))?;
        }
        let mut refused = None;
        for &sample in &self.paired {
            match &mut self.phase {
                Phase::Tracking { pending, .. } => pending.push_back(sample),
                Phase::Resting { map, window } => {
                    window.push(sample);
                    if sample.time - window[0].time < self.settings.rest_for {
                        continue;
                    }
                    match StillStart::from_samples(window) {
                        Ok(start) => {
                            let map = map.take().expect("the map is held until odometry starts");
                            let mut odometry = Box::new(DepthInertialOdometry::new(start, map, self.settings.odometry));
                            odometry.add_imu(sample).map_err(TrackerError::Odometry)?;
                            self.phase = Phase::Tracking { odometry, pending: VecDeque::new(), started_at: sample.time };
                            refused = None;
                        }
                        Err(moving) => {
                            window.clear();
                            refused = Some(moving);
                        }
                    }
                }
            }
        }
        if let Phase::Tracking { pending, .. } = &self.phase {
            let span = pending.front().zip(pending.back()).map_or(Duration::ZERO, |(oldest, newest)| newest.time - oldest.time);
            self.within_backlog(span, Awaited::DepthFrame)?;
        }
        refused.map_or(Ok(self.phase()), |moving| Err(TrackerError::NotAtRest(moving)))
    }

    fn within_backlog(&self, span: Duration, waiting_for: Awaited) -> Result<(), TrackerError> {
        if span > self.settings.max_backlog {
            return Err(TrackerError::Backlog { waiting_for, span });
        }
        Ok(())
    }

    /// Takes a depth image taken at `time` on the camera's clock: row major, depth along the optical
    /// axis in millimetres, zero where there is no reading. The IMU samples up to `time` must have
    /// been given first. Samples newer than `time` wait for the next frame.
    pub fn add_depth_frame(&mut self, time: Duration, frame_id: DepthFrameId, depth_mm: &[u16]) -> Result<DepthFrameOutcome, TrackerError> {
        self.projector.fits(depth_mm.len()).map_err(TrackerError::DepthImage)?;
        let Phase::Tracking { odometry, pending, started_at } = &mut self.phase else { return Ok(DepthFrameOutcome::AwaitingStart) };
        if time < *started_at {
            return Ok(DepthFrameOutcome::AwaitingStart);
        }
        if let Some(previous) = self.last_posed.filter(|&previous| time <= previous) {
            return Err(TrackerError::FrameOutOfOrder { at: time, previous });
        }
        for stream in [ImuStream::Accelerometer, ImuStream::Gyro] {
            let latest = self.pairer.latest(stream);
            if latest.is_none_or(|latest| latest + self.settings.max_imu_lag < time) {
                return Err(TrackerError::ImuBehind { frame_at: time, stream, latest });
            }
        }
        self.projector.project(depth_mm, &mut self.sensor_points).map_err(TrackerError::DepthImage)?;

        // Gyro samples up to the frame that the accelerometer has not reached are newer than every
        // paired sample, so they follow the pending ones in time order.
        self.paired.clear();
        self.pairer.pair_held_with_latest_accel(time, &mut self.paired);
        pending.extend(self.paired.iter().copied());
        while let Some(sample) = pending.front().copied().filter(|sample| sample.time <= time) {
            odometry.add_imu(sample).map_err(TrackerError::Odometry)?;
            pending.pop_front();
        }

        let cloud = match odometry.add_cloud(time, &self.sensor_points) {
            Ok(CloudOutcome::StartedMap) => CloudUse::StartedMap,
            Ok(CloudOutcome::Corrected { matched, weakest, .. }) => {
                CloudUse::Corrected { matched_points: matched, weakest: WeakestDirection { direction: odometry.state().levelling() * weakest.direction_world, sigma_m: weakest.sigma_m } }
            }
            Err(OdometryError::NotCorrected(cause)) => CloudUse::NotCorrected(cause),
            Err(other) => return Err(TrackerError::Odometry(other)),
        };
        self.last_posed = Some(time);
        let state = odometry.state();
        let (position_m, orientation) = state.levelled_pose();
        Ok(DepthFrameOutcome::Posed(FramePose {
            time,
            frame_id,
            position_m,
            orientation,
            velocity_m_s: state.levelled_velocity(),
            uncertainty: odometry.filter().levelled_uncertainty(),
            cloud,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::{seeded, ImuErrors, ImuStreams, PinholeDepthCamera, Scene, Walk};
    use crate::surfaces::{PlaneVoxelMap, PlaneVoxelSettings};
    use rand::rngs::SmallRng;

    const FRAME_PERIOD: Duration = Duration::from_micros(32_500);
    const REST_FOR: Duration = Duration::from_millis(500);

    /// A camera standing still in the simulated room, and a tracker fed from it.
    struct Bench {
        tracker: PoseTracker<PlaneVoxelMap>,
        walk: Walk,
        camera: PinholeDepthCamera,
        mount: SensorMount,
        rng: SmallRng,
        imu_through: Option<Duration>,
        depth_mm: Vec<u16>,
    }

    impl Bench {
        fn new() -> Bench {
            let camera = PinholeDepthCamera::d455(212, 120);
            let mount = SensorMount::from_depth_to_imu(PinholeDepthCamera::FACING_FORWARD, [0.0; 3]).expect("a rotation");
            let settings = TrackerSettings { rest_for: REST_FOR, depth_stride: NonZeroUsize::MIN, ..TrackerSettings::d455() };
            let tracker = PoseTracker::new(camera.intrinsics, mount, PlaneVoxelMap::new(PlaneVoxelSettings::indoors()), settings).expect("usable intrinsics");
            Bench { tracker, walk: Walk::on_a_robot(Duration::from_secs(3600)), camera, mount, rng: seeded(11), imu_through: None, depth_mm: Vec::new() }
        }

        fn imu_batch(&mut self, through: Duration) -> (Vec<AccelReading>, Vec<GyroReading>) {
            let batch = ImuStreams::d455().batch(&self.walk, &ImuErrors::consumer(), self.imu_through, through, &mut self.rng);
            self.imu_through = Some(through);
            batch
        }

        fn add_imu(&mut self, through: Duration) -> Result<TrackerPhase, TrackerError> {
            let (accel, gyro) = self.imu_batch(through);
            self.tracker.add_imu(&accel, &gyro)
        }

        fn add_frame(&mut self, time: Duration) -> Result<DepthFrameOutcome, TrackerError> {
            let seconds = time.as_secs_f64();
            self.camera.image(&Scene::living_room(), self.walk.position(seconds), self.walk.rotation(seconds), &self.mount, &mut self.rng, &mut self.depth_mm);
            self.tracker.add_depth_frame(time, DepthFrameId(time.as_micros() as u64), &self.depth_mm)
        }

        /// Feeds IMU batches frame by frame until odometry has started, and returns the time fed through.
        fn rest_until_started(&mut self) -> Duration {
            let mut time = self.imu_through.map_or(Duration::ZERO, |through| through + FRAME_PERIOD);
            while self.add_imu(time).expect("the camera rests") != TrackerPhase::Tracking {
                time += FRAME_PERIOD;
            }
            time
        }

        fn posed(&mut self, time: Duration) -> FramePose {
            self.add_imu(time).expect("samples are in order");
            match self.add_frame(time).expect("the frame is taken") {
                DepthFrameOutcome::Posed(pose) => pose,
                DepthFrameOutcome::AwaitingStart => panic!("no pose at {time:?}"),
            }
        }
    }

    #[test]
    fn depth_frames_taken_before_the_start_get_no_pose() {
        let mut bench = Bench::new();
        assert_eq!(bench.add_imu(FRAME_PERIOD), Ok(TrackerPhase::Resting { rested: Duration::from_micros(27_500) }));
        assert_eq!(bench.add_frame(FRAME_PERIOD), Ok(DepthFrameOutcome::AwaitingStart));
        let started = bench.rest_until_started();
        assert_eq!(bench.add_frame(started - FRAME_PERIOD), Ok(DepthFrameOutcome::AwaitingStart));
    }

    #[test]
    fn odometry_starts_once_the_body_has_rested_for_the_configured_time() {
        let mut bench = Bench::new();
        let started = bench.rest_until_started();
        assert!(started >= REST_FOR && started < REST_FOR + 2 * FRAME_PERIOD, "{started:?}");
        assert!(bench.tracker.odometry().is_some());
    }

    #[test]
    fn a_window_in_which_the_body_moved_is_refused_and_the_next_still_window_starts_odometry() {
        let mut bench = Bench::new();
        let mut time = Duration::ZERO;
        let refusal = loop {
            let (accel, mut gyro) = bench.imu_batch(time);
            for reading in gyro.iter_mut().filter(|reading| reading.time < Duration::from_millis(300)) {
                reading.rad_s.z += 0.5 * (40.0 * reading.time.as_secs_f64()).sin();
            }
            match bench.tracker.add_imu(&accel, &gyro) {
                Ok(phase) => assert!(matches!(phase, TrackerPhase::Resting { .. }), "{phase:?}"),
                Err(refusal) => break refusal,
            }
            time += FRAME_PERIOD;
        };
        assert!(matches!(refusal, TrackerError::NotAtRest(StillStartError::Moving { gyro_spread_rad_s, .. }) if gyro_spread_rad_s > 0.1), "{refusal:?}");
        assert!(time >= REST_FOR && bench.tracker.odometry().is_none());
        let refused_at = time;
        loop {
            time += FRAME_PERIOD;
            if bench.add_imu(time).expect("the camera now rests") == TrackerPhase::Tracking {
                break;
            }
        }
        // A whole new window, to within the batch that completes it.
        assert!(time - refused_at > REST_FOR - FRAME_PERIOD && time - refused_at < REST_FOR + FRAME_PERIOD, "started {:?} after the refusal", time - refused_at);
    }

    #[test]
    fn the_first_frame_starts_the_map_and_the_frames_after_it_are_corrected() {
        let mut bench = Bench::new();
        let started = bench.rest_until_started();
        let first = bench.posed(started + FRAME_PERIOD);
        assert_eq!((first.cloud, first.corrected(), first.matched_points(), first.weakest()), (CloudUse::StartedMap, false, 0, None));
        let second = bench.posed(started + 2 * FRAME_PERIOD);
        assert!(second.corrected() && second.matched_points() > 50, "{second:?}");
        let weakest = second.weakest().expect("a corrected frame has a weakest direction");
        assert!((weakest.direction.length() - 1.0).abs() < 1e-9 && weakest.sigma_m > 0.0 && weakest.sigma_m < 0.05, "{weakest:?}");
        assert_eq!((second.time, second.frame_id), (started + 2 * FRAME_PERIOD, DepthFrameId((started + 2 * FRAME_PERIOD).as_micros() as u64)));
        assert!(second.position_m.length() < 0.01 && second.velocity_m_s.length() < 0.05, "{second:?}");
        let sigma = second.uncertainty;
        assert!(sigma.position_sigma_m.max_element() < 0.02 && sigma.tilt_sigma_rad < 0.02 && sigma.heading_sigma_rad < 0.02, "{sigma:?}");
    }

    #[test]
    fn a_frame_with_no_readings_gets_the_predicted_pose_and_the_next_frame_is_corrected() {
        let mut bench = Bench::new();
        let started = bench.rest_until_started();
        bench.posed(started + FRAME_PERIOD);
        bench.add_imu(started + 2 * FRAME_PERIOD).expect("samples are in order");
        let blank = bench.tracker.add_depth_frame(started + 2 * FRAME_PERIOD, DepthFrameId(2), &vec![0u16; 212 * 120]).expect("a blank frame is still a frame");
        let DepthFrameOutcome::Posed(predicted) = blank else { panic!("{blank:?}") };
        assert!(matches!(predicted.cloud, CloudUse::NotCorrected(CorrectionError::TooFewRows { rows: 0, .. })), "{predicted:?}");
        assert_eq!((predicted.corrected(), predicted.matched_points(), predicted.weakest()), (false, 0, None));
        assert!(bench.posed(started + 3 * FRAME_PERIOD).corrected());
    }

    #[test]
    fn a_frame_not_newer_than_the_last_posed_frame_is_refused() {
        let mut bench = Bench::new();
        let started = bench.rest_until_started();
        let posed_at = started + 2 * FRAME_PERIOD;
        bench.posed(started + FRAME_PERIOD);
        bench.posed(posed_at);
        for stale in [posed_at, started + FRAME_PERIOD] {
            assert_eq!(bench.add_frame(stale), Err(TrackerError::FrameOutOfOrder { at: stale, previous: posed_at }));
        }
        assert!(bench.posed(started + 3 * FRAME_PERIOD).time > posed_at);
    }

    #[test]
    fn pose_times_strictly_increase() {
        let mut bench = Bench::new();
        let started = bench.rest_until_started();
        let times: Vec<Duration> = (1..=10).map(|frame| bench.posed(started + frame * FRAME_PERIOD).time).collect();
        assert!(times.windows(2).all(|pair| pair[0] < pair[1]), "{times:?}");
    }

    #[test]
    fn a_frame_ahead_of_the_imu_is_refused() {
        let mut bench = Bench::new();
        let started = bench.rest_until_started();
        bench.posed(started + FRAME_PERIOD);
        let ahead = started + 2 * FRAME_PERIOD;
        let refusal = bench.add_frame(ahead).expect_err("no IMU sample since the last frame");
        assert!(matches!(refusal, TrackerError::ImuBehind { frame_at, stream: ImuStream::Accelerometer, latest: Some(latest) } if frame_at == ahead && latest <= started + FRAME_PERIOD), "{refusal:?}");
        assert!(bench.posed(ahead).corrected(), "the frame is taken once its IMU samples are in");
    }

    #[test]
    fn an_imu_stream_that_stays_silent_is_refused_once_the_other_stream_has_waited_past_the_bound() {
        let max_backlog = TrackerSettings::d455().max_backlog;
        for silent in [ImuStream::Accelerometer, ImuStream::Gyro] {
            let mut bench = Bench::new();
            let mut time = Duration::ZERO;
            let refusal = loop {
                let (mut accel, mut gyro) = bench.imu_batch(time);
                match silent {
                    ImuStream::Accelerometer => accel.clear(),
                    ImuStream::Gyro => gyro.clear(),
                }
                match bench.tracker.add_imu(&accel, &gyro) {
                    Ok(phase) => assert_eq!(phase, TrackerPhase::Resting { rested: Duration::ZERO }),
                    Err(refusal) => break refusal,
                }
                time += FRAME_PERIOD;
            };
            assert!(matches!(refusal, TrackerError::Backlog { waiting_for: Awaited::Imu(stream), span } if stream == silent && span > max_backlog), "{refusal:?}");
            assert!(time > max_backlog && time < max_backlog + 2 * FRAME_PERIOD, "refused at {time:?}");
        }
    }

    #[test]
    fn imu_samples_that_no_depth_frame_comes_for_are_refused_once_they_span_more_than_the_bound() {
        let max_backlog = TrackerSettings::d455().max_backlog;
        let mut bench = Bench::new();
        let started = bench.rest_until_started();
        bench.posed(started + FRAME_PERIOD);
        let mut time = started + FRAME_PERIOD;
        let refusal = loop {
            time += FRAME_PERIOD;
            if let Err(refusal) = bench.add_imu(time) {
                break refusal;
            }
        };
        assert!(matches!(refusal, TrackerError::Backlog { waiting_for: Awaited::DepthFrame, span } if span > max_backlog), "{refusal:?}");
        let waited = time - (started + FRAME_PERIOD);
        assert!(waited > max_backlog && waited < max_backlog + 2 * FRAME_PERIOD, "refused after {waited:?}");
        // A frame every period keeps the samples held well under the bound.
        let mut bench = Bench::new();
        let started = bench.rest_until_started();
        for frame in 1..=40 {
            bench.posed(started + frame * FRAME_PERIOD);
        }
    }

    #[test]
    fn imu_samples_newer_than_a_frame_wait_for_the_next_frame() {
        let mut bench = Bench::new();
        let started = bench.rest_until_started();
        bench.posed(started + FRAME_PERIOD);
        // The batch runs most of a frame past the frame it arrives with.
        let frame_at = started + 2 * FRAME_PERIOD;
        bench.add_imu(frame_at + Duration::from_millis(25)).expect("samples are in order");
        let DepthFrameOutcome::Posed(pose) = bench.add_frame(frame_at).expect("the frame is taken") else { panic!("no pose") };
        assert!(pose.time == frame_at && pose.corrected(), "{pose:?}");
        assert!(bench.posed(started + 3 * FRAME_PERIOD).corrected());
    }

    #[test]
    fn an_image_of_another_size_is_refused_at_any_time() {
        let mut bench = Bench::new();
        let refused = Err(TrackerError::DepthImage(DepthImageError::WrongSize { readings: 5, width: 212, height: 120 }));
        assert_eq!(bench.tracker.add_depth_frame(Duration::ZERO, DepthFrameId(0), &[0; 5]), refused);
        let started = bench.rest_until_started();
        assert_eq!(bench.tracker.add_depth_frame(started + FRAME_PERIOD, DepthFrameId(1), &[0; 5]), refused);
    }
}
