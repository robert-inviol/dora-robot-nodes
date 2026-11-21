//! The odometry loop: IMU samples move the estimate on, clouds correct it and grow the map.

use std::time::Duration;

use glam::{DQuat, DVec3, Vec3};

use crate::association::{measure, AssociationSettings, BodyPoint, RangeNoise};
use crate::filter::{DepthScaleError, LevelledUncertainty, PositionPerDepthScale, MeasurementUpdate, MeasurementUpdateError, MeasurementUpdateSettings, ErrorStateFilter, ImuNoise, InitialUncertainty, PoseInformation, State, DegenerateTranslation};
use crate::surfaces::{MapPoint, SurfaceMap};
use crate::downsampling::{SensorExtrinsics, Downsampler, DownsamplingSettings};

/// One reading of the IMU, on the camera's clock.
#[derive(Clone, Copy, Debug)]
pub struct ImuSample {
    pub time: Duration,
    pub gyro_rad_s: DVec3,
    pub accel_m_s2: DVec3,
}

#[derive(Clone, Copy, Debug)]
pub struct OdometrySettings {
    pub gravity_m_s2: f64,
    pub imu_noise: ImuNoise,
    pub start_uncertainty: InitialUncertainty,
    pub sensor_extrinsics: SensorExtrinsics,
    pub downsampling: DownsamplingSettings,
    pub association: AssociationSettings,
    pub measurement_update: MeasurementUpdateSettings,
    /// The map keeps a cube of this half side about the body.
    pub map_half_extent_m: f32,
    /// The map is trimmed once the body has moved this far since the last trim.
    pub map_trim_after_m: f32,
    /// One sigma of each part of the depth scale error that a cloud's readings share and the map's readings do not:
    /// how far the camera's depth may have changed since it laid down the surfaces it sees. It is not estimated;
    /// [`DepthInertialOdometry::levelled_uncertainty`] adds what it does to the position.
    pub depth_scale_change_sigma: DepthScaleError,
}

impl OdometrySettings {
    /// Settings for an Intel RealSense D455 indoors.
    pub fn d455() -> OdometrySettings {
        OdometrySettings {
            gravity_m_s2: 9.81,
            imu_noise: ImuNoise {
                gyro_rad_s_per_sqrt_hz: 2e-3,
                accel_m_s2_per_sqrt_hz: 2e-2,
                gyro_bias_walk_rad_s2_per_sqrt_hz: 2e-5,
                accel_bias_walk_m_s3_per_sqrt_hz: 5e-4,
            },
            start_uncertainty: InitialUncertainty {
                position_m: 1e-3,
                orientation_rad: 1e-3,
                levelling_rad: 2e-3,
                velocity_m_s: 0.01,
                gyro_bias_rad_s: 5e-3,
                accel_bias_m_s2: 0.1,
            },
            sensor_extrinsics: SensorExtrinsics::AT_BODY,
            downsampling: DownsamplingSettings { voxel_m: 0.1, min_range_m: 0.4, max_range_m: 6.0, min_readings: 4, flat_within_m: 0.003, flat_noise_sigmas: 1.5, facing_known_below: 0.3 },
            association: AssociationSettings { gate_sigmas: 4.0, huber_sigmas: 2.0, min_facing_agreement: 0.8, noise: RangeNoise { base_m: 0.005, per_square_m: 0.005 } },
            measurement_update: MeasurementUpdateSettings { max_iterations: 4, settled_m: 1e-4, settled_rad: 1e-4, min_rows: 50 },
            map_half_extent_m: 8.0,
            map_trim_after_m: 1.0,
            // The D455's depth grows as it warms up after its stream starts, more on the left of the image than on
            // the right (see the README, "Depth warm-up").
            depth_scale_change_sigma: DepthScaleError { uniform: 0.0085, across: 0.004, down: 0.0013 },
        }
    }
}

/// The body held still: enough to find which way is up and what the gyro reads at rest.
#[derive(Clone, Copy, Debug)]
pub struct StaticInitialization {
    pub state: State,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StaticInitializationError {
    NoSamples,
    /// The IMU was not at rest: the largest spread of any gyro axis, and of any accelerometer axis.
    Moving { gyro_spread_rad_s: f64, accel_spread_m_s2: f64 },
    /// The gyro read a steady turn: the size of its mean rate.
    Rotating { mean_gyro_rad_s: f64 },
    /// The accelerometer's mean reading is not the size of gravity, so it does not measure gravity
    /// alone.
    NotGravity { mean_accel_m_s2: f64, gravity_m_s2: f64 },
}

impl std::fmt::Display for StaticInitializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StaticInitializationError::NoSamples => write!(f, "no IMU samples to start from"),
            StaticInitializationError::Moving { gyro_spread_rad_s, accel_spread_m_s2 } => {
                write!(f, "the IMU was moving at the start: gyro spread {gyro_spread_rad_s:.4} rad/s, accelerometer spread {accel_spread_m_s2:.3} m/s2")
            }
            StaticInitializationError::Rotating { mean_gyro_rad_s } => write!(f, "the IMU was turning steadily at the start: mean gyro rate {mean_gyro_rad_s:.4} rad/s"),
            StaticInitializationError::NotGravity { mean_accel_m_s2, gravity_m_s2 } => {
                write!(f, "the accelerometer read {mean_accel_m_s2:.3} m/s2 at the start, not gravity's {gravity_m_s2:.3} m/s2")
            }
        }
    }
}

impl std::error::Error for StaticInitializationError {}

impl StaticInitialization {
    const MAX_GYRO_SPREAD_RAD_S: f64 = 0.02;
    const MAX_ACCEL_SPREAD_M_S2: f64 = 0.3;
    /// The mean gyro rate at rest is the gyro's bias, which is well under this for a calibrated IMU;
    /// a steady turn shows as a larger mean with no spread.
    const MAX_MEAN_GYRO_RAD_S: f64 = 0.05;
    /// At rest the accelerometer's mean reading is gravity, off only by its bias and scale error.
    const MAX_GRAVITY_ERROR_M_S2: f64 = 0.5;

    /// A starting state at the world origin with zero heading, from samples taken at rest under
    /// gravity of `gravity_m_s2` (static initialization). The world's z is where the accelerometer
    /// says up is, which its bias throws off; the state's `up` starts there and is corrected as the
    /// filter learns the bias.
    ///
    /// The samples are taken as stationary when neither stream spreads, the mean gyro rate is no more
    /// than a bias, and the mean acceleration is the size of gravity.
    pub fn from_samples(samples: &[ImuSample], gravity_m_s2: f64) -> Result<StaticInitialization, StaticInitializationError> {
        if samples.is_empty() {
            return Err(StaticInitializationError::NoSamples);
        }
        let count = samples.len() as f64;
        let mean = |pick: fn(&ImuSample) -> DVec3| samples.iter().map(pick).sum::<DVec3>() / count;
        let spread = |pick: fn(&ImuSample) -> DVec3, centre: DVec3| {
            let variance = samples.iter().map(|s| (pick(s) - centre) * (pick(s) - centre)).sum::<DVec3>() / count;
            variance.max_element().sqrt()
        };
        let (gyro, accel) = (mean(|s| s.gyro_rad_s), mean(|s| s.accel_m_s2));
        let (gyro_spread_rad_s, accel_spread_m_s2) = (spread(|s| s.gyro_rad_s, gyro), spread(|s| s.accel_m_s2, accel));
        if gyro_spread_rad_s > Self::MAX_GYRO_SPREAD_RAD_S || accel_spread_m_s2 > Self::MAX_ACCEL_SPREAD_M_S2 {
            return Err(StaticInitializationError::Moving { gyro_spread_rad_s, accel_spread_m_s2 });
        }
        if gyro.length() > Self::MAX_MEAN_GYRO_RAD_S {
            return Err(StaticInitializationError::Rotating { mean_gyro_rad_s: gyro.length() });
        }
        if (accel.length() - gravity_m_s2).abs() > Self::MAX_GRAVITY_ERROR_M_S2 {
            return Err(StaticInitializationError::NotGravity { mean_accel_m_s2: accel.length(), gravity_m_s2 });
        }
        // At rest the accelerometer reads the reaction to gravity, which points up.
        let rotation = DQuat::from_rotation_arc(accel.normalize(), DVec3::Z);
        Ok(StaticInitialization { state: State { position: DVec3::ZERO, rotation, velocity: DVec3::ZERO, gyro_bias: gyro, accel_bias: DVec3::ZERO, up: DVec3::Z } })
    }
}

/// What one cloud did to the estimate.
#[derive(Clone, Copy, Debug)]
pub enum CloudOutcome {
    /// The first cloud: it became the map.
    StartedMap,
    Corrected {
        iterations: usize,
        /// Points that found a surface in the map.
        matched: usize,
        /// Points of the downsampled cloud.
        offered: usize,
        /// Root mean square of the matched points' residuals, in sigmas.
        rms_sigmas: f64,
        /// The direction the cloud pinned down least. A sigma far above the others' centimetres means
        /// the scene left that direction to the IMU.
        degenerate: DegenerateTranslation,
    },
}

/// How a cloud's points fared, whether or not the cloud corrected the estimate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CloudCounts {
    /// Points of the cloud as given, before downsampling.
    pub given: usize,
    /// Points of the downsampled cloud.
    pub downsampled: usize,
    /// Downsampled points that landed where the map holds a surface facing their way, in the last linearisation.
    pub found_surface: usize,
    /// Of those, the points whose innovation passed the gate: the cloud's rows in the measurement update.
    pub within_gate: usize,
    /// The fewest rows a measurement update takes.
    pub needed: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OdometryError {
    /// A sample or cloud older than the estimate.
    OutOfOrder { at: Duration, estimate_at: Duration },
    /// A cloud before any IMU sample.
    NoImuYet,
    /// The cloud could not correct the estimate, which keeps the IMU's prediction; the cloud is not
    /// added to the map.
    NotCorrected(MeasurementUpdateError),
}

impl std::fmt::Display for OdometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OdometryError::OutOfOrder { at, estimate_at } => write!(f, "input at {at:?} is older than the estimate at {estimate_at:?}"),
            OdometryError::NoImuYet => write!(f, "a cloud arrived before any IMU sample"),
            OdometryError::NotCorrected(cause) => write!(f, "the cloud did not correct the estimate: {cause}"),
        }
    }
}

impl std::error::Error for OdometryError {}

pub struct DepthInertialOdometry<M> {
    filter: ErrorStateFilter,
    map: M,
    settings: OdometrySettings,
    last_imu: Option<ImuSample>,
    estimate_at: Duration,
    /// When a cloud last corrected the estimate, or the first IMU sample came.
    corrected_at: Option<Duration>,
    trimmed_at: DVec3,
    downsampler: Downsampler,
    downsampled: Vec<BodyPoint>,
    world_points: Vec<MapPoint>,
    last_cloud: CloudCounts,
    /// How far each part of a depth scale error shared by every depth reading moved the position at the last
    /// measurement update, per unit of that part, in the world frame.
    position_shift_per_depth_scale: PositionPerDepthScale,
}

impl<M: SurfaceMap> DepthInertialOdometry<M> {
    pub fn new(start: StaticInitialization, map: M, settings: OdometrySettings) -> DepthInertialOdometry<M> {
        let filter = ErrorStateFilter::new(start.state, settings.start_uncertainty, settings.imu_noise, settings.gravity_m_s2);
        DepthInertialOdometry {
            filter,
            map,
            settings,
            last_imu: None,
            estimate_at: Duration::ZERO,
            corrected_at: None,
            trimmed_at: start.state.position,
            downsampler: Downsampler::default(),
            downsampled: Vec::new(),
            world_points: Vec::new(),
            last_cloud: CloudCounts::default(),
            position_shift_per_depth_scale: PositionPerDepthScale::zeros(),
        }
    }

    /// How the points of the last cloud given to [`Self::add_cloud`] fared; all zero before the first.
    pub fn last_cloud(&self) -> CloudCounts {
        self.last_cloud
    }

    /// The estimate in the world frame, which is fixed where the start put it and is level only as
    /// far as the start was. [`State::levelled_pose`] gives the pose in a frame whose z is up.
    pub fn state(&self) -> &State {
        self.filter.state()
    }

    pub fn filter(&self) -> &ErrorStateFilter {
        &self.filter
    }

    /// One-sigma uncertainty of the levelled pose against the room for a camera at rest: the filter's own, which is
    /// against the map, with the position shift that a change of the camera's depth scale since it laid down the
    /// map would cause ([`OdometrySettings::depth_scale_change_sigma`]). The map's drift while the camera travels is
    /// not in it.
    pub fn levelled_uncertainty(&self) -> LevelledUncertainty {
        let against_map = self.filter.levelled_uncertainty();
        let levelling = self.filter.state().levelling();
        let sigmas = self.settings.depth_scale_change_sigma.as_vector();
        let mut variance = DVec3::ZERO;
        for (part, column) in self.position_shift_per_depth_scale.column_iter().enumerate() {
            let shift = levelling * DVec3::new(column[0], column[1], column[2]) * sigmas[part];
            variance += shift * shift;
        }
        let position_sigma_m = (against_map.position_sigma_m * against_map.position_sigma_m + variance).map(f64::sqrt);
        LevelledUncertainty { position_sigma_m, ..against_map }
    }

    pub fn map(&self) -> &M {
        &self.map
    }

    /// How long the IMU alone has carried the estimate (dead reckoning): the time since a cloud last
    /// corrected it, or since the static initialization when none has. Zero straight after a
    /// measurement update.
    pub fn dead_reckoning_for(&self) -> Duration {
        self.estimate_at.saturating_sub(self.corrected_at.unwrap_or(self.estimate_at))
    }

    /// Moves the estimate on to this sample's time.
    pub fn add_imu(&mut self, sample: ImuSample) -> Result<(), OdometryError> {
        if let Some(last) = self.last_imu {
            if sample.time < self.estimate_at {
                return Err(OdometryError::OutOfOrder { at: sample.time, estimate_at: self.estimate_at });
            }
            let seconds = (sample.time - self.estimate_at).as_secs_f64();
            self.filter.predict(0.5 * (last.gyro_rad_s + sample.gyro_rad_s), 0.5 * (last.accel_m_s2 + sample.accel_m_s2), seconds);
        }
        self.estimate_at = sample.time;
        self.corrected_at.get_or_insert(sample.time);
        self.last_imu = Some(sample);
        Ok(())
    }

    /// Corrects the estimate with a cloud of sensor-frame points taken at `time`, then adds it to the map.
    pub fn add_cloud(&mut self, time: Duration, sensor_points: &[Vec3]) -> Result<CloudOutcome, OdometryError> {
        self.add_cloud_with(time, sensor_points, |_| PoseInformation::default())
    }

    /// As [`Self::add_cloud`], with another sensor's point-to-plane measurements joining the measurement update.
    pub fn add_cloud_with(&mut self, time: Duration, sensor_points: &[Vec3], others: impl Fn(&State) -> PoseInformation) -> Result<CloudOutcome, OdometryError> {
        let last = self.last_imu.ok_or(OdometryError::NoImuYet)?;
        if time < self.estimate_at {
            return Err(OdometryError::OutOfOrder { at: time, estimate_at: self.estimate_at });
        }
        self.filter.predict(last.gyro_rad_s, last.accel_m_s2, (time - self.estimate_at).as_secs_f64());
        self.estimate_at = time;
        self.downsampler.downsample(sensor_points, &self.settings.sensor_extrinsics, &self.settings.downsampling, &self.settings.association.noise, &mut self.downsampled);
        self.last_cloud = CloudCounts { given: sensor_points.len(), downsampled: self.downsampled.len(), needed: self.settings.measurement_update.min_rows, ..CloudCounts::default() };

        let outcome = if self.map.is_empty() {
            CloudOutcome::StartedMap
        } else {
            let (map, downsampled, association, sensor) = (&self.map, &self.downsampled, &self.settings.association, &self.settings.sensor_extrinsics);
            let predicted_pose_covariance = self.filter.pose_covariance();
            let last_cloud = &mut self.last_cloud;
            let MeasurementUpdate { iterations, information } = self
                .filter
                .update(&self.settings.measurement_update, |state| {
                    let cloud = measure(downsampled, sensor, map, state, &predicted_pose_covariance, association);
                    (last_cloud.found_surface, last_cloud.within_gate) = (cloud.found_surface, cloud.information.rows);
                    cloud.information.merge(others(state))
                })
                .map_err(OdometryError::NotCorrected)?;
            self.corrected_at = Some(time);
            self.position_shift_per_depth_scale = information.position_shift_per_depth_scale();
            CloudOutcome::Corrected {
                iterations,
                matched: information.rows,
                offered: self.downsampled.len(),
                rms_sigmas: (information.squared_sigmas / information.rows as f64).sqrt(),
                degenerate: information.degenerate_translation(),
            }
        };

        let state = *self.filter.state();
        self.world_points.clear();
        self.world_points.extend(self.downsampled.iter().map(|point| MapPoint { position: state.to_world(point.position.as_dvec3()).as_vec3(), sigma_m: point.sigma_m }));
        self.map.insert(&self.world_points);
        if (state.position - self.trimmed_at).length() > self.settings.map_trim_after_m as f64 {
            self.map.keep_within(state.position.as_vec3(), self.settings.map_half_extent_m);
            self.trimmed_at = state.position;
        }
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resting(count: usize, accel: DVec3, gyro: DVec3) -> Vec<ImuSample> {
        (0..count)
            .map(|i| {
                let wobble = 1e-3 * ((i % 7) as f64 - 3.0);
                ImuSample { time: Duration::from_micros(2500 * i as u64), gyro_rad_s: gyro + DVec3::splat(wobble * 0.1), accel_m_s2: accel + DVec3::splat(wobble) }
            })
            .collect()
    }

    #[test]
    fn a_still_start_finds_which_way_is_up_and_the_gyro_bias() {
        let tilt = DQuat::from_euler(glam::EulerRot::ZYX, 0.0, 0.15, -0.1);
        let bias = DVec3::new(0.003, -0.002, 0.001);
        let start = StaticInitialization::from_samples(&resting(400, tilt.conjugate() * DVec3::new(0.0, 0.0, 9.81), bias), 9.81).expect("the samples are at rest");
        let up = start.state.rotation * (tilt.conjugate() * DVec3::Z);
        assert!((up - DVec3::Z).length() < 1e-3, "{up}");
        assert!((start.state.gyro_bias - bias).length() < 1e-4, "{}", start.state.gyro_bias);
    }

    #[test]
    fn a_start_while_turning_is_refused() {
        let mut samples = resting(400, DVec3::new(0.0, 0.0, 9.81), DVec3::ZERO);
        for (i, sample) in samples.iter_mut().enumerate() {
            sample.gyro_rad_s.z = 0.5 * (i as f64 * 0.05).sin();
        }
        assert!(matches!(StaticInitialization::from_samples(&samples, 9.81), Err(StaticInitializationError::Moving { .. })));
    }

    #[test]
    fn a_start_during_a_steady_turn_is_refused() {
        // A turn about up at a constant rate: the gyro does not spread and the accelerometer reads
        // gravity alone.
        let samples = resting(400, DVec3::new(0.0, 0.0, 9.81), DVec3::new(0.0, 0.0, 0.3));
        assert!(matches!(StaticInitialization::from_samples(&samples, 9.81), Err(StaticInitializationError::Rotating { mean_gyro_rad_s }) if (mean_gyro_rad_s - 0.3).abs() < 1e-3));
    }

    #[test]
    fn a_start_whose_accelerometer_does_not_read_gravity_is_refused() {
        let samples = resting(400, DVec3::new(0.0, 0.0, 10.6), DVec3::ZERO);
        assert!(matches!(StaticInitialization::from_samples(&samples, 9.81), Err(StaticInitializationError::NotGravity { .. })));
    }
}
