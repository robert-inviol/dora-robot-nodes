//! The node's settings: a TOML file named by `MAP_LOCALIZER_CONFIG`, defaults from the library. Every key is optional
//! except `camera_in_imu_m` in an `[odometry]` section. `map-localizer.example.toml` lists every key with its default.

use std::path::{Path, PathBuf};

use map_localizer::cloud::CloudSettings;
use glam::Vec3;
use map_localizer::gyro::GyroSettings;
use map_localizer::map_to_odom::{CameraMount, MapToOdomSettings};
use map_localizer::{HeadingWithoutGyro, LocalizerSettings};
use serde::Deserialize;

pub const CONFIG_VARIABLE: &str = "MAP_LOCALIZER_CONFIG";

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("reading the settings file {path}: {source}")]
    Read { path: PathBuf, source: std::io::Error },
    #[error("parsing the settings file {path}: {source}")]
    Parse { path: PathBuf, source: toml::de::Error },
    #[error("the settings file {path} sets imu = \"absent\", which needs heading_without_gyro = \"search\"")]
    ImuAbsentNeedsSearch { path: PathBuf },
    #[error("the settings file {path} has an [odometry] section and imu = \"absent\"; odometry is joined to keyframes on the camera clock, which only the imu messages give")]
    OdometryNeedsImu { path: PathBuf },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CloudSection {
    pub stride_px: usize,
    pub voxel_m: f32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SubmapSection {
    pub radius_m: f32,
    pub max_keyframes: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrackingSection {
    pub accept_fitness: f32,
    pub max_step_m: f32,
    pub gyro_yaw_weight: f64,
    pub max_gyro_disagreement_deg: f32,
    pub min_points: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoopSection {
    pub skip_latest_keyframes: usize,
    pub radius_m: f32,
    pub attempt_every_keyframes: usize,
    pub min_fitness: f32,
    pub max_rmse_m: f32,
    pub max_correction_m: f32,
    pub max_correction_deg: f32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RelocaliseSection {
    pub radius_m: f32,
    pub min_fitness: f32,
    pub restart_after_lost: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GraphSection {
    pub optimise_interval_s: f64,
    pub version_move_m: f32,
    pub version_turn_deg: f32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GyroSection {
    pub buffer_s: f64,
    pub min_still_s: f64,
    pub heading_without_gyro: HeadingWithoutGyroSetting,
    pub imu: ImuStream,
}

/// The robot's odometry, joined to keyframes to send `map_to_odom`. Without this section the node takes no odometry.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OdometrySection {
    /// The colour camera's optical centre in the IMU's coordinates (x right, y down, z forward), in metres.
    pub camera_in_imu_m: [f32; 3],
    #[serde(default = "OdometrySection::default_retention_s")]
    pub retention_s: f64,
    #[serde(default = "OdometrySection::default_window_keyframes")]
    pub window_keyframes: usize,
    #[serde(default = "OdometrySection::default_min_sigma_xy_m")]
    pub min_sigma_xy_m: f32,
    #[serde(default = "OdometrySection::default_min_sigma_z_m")]
    pub min_sigma_z_m: f32,
    #[serde(default = "OdometrySection::default_min_sigma_yaw_deg")]
    pub min_sigma_yaw_deg: f32,
}

impl OdometrySection {
    fn default_retention_s() -> f64 {
        OdometrySettings::DEFAULT_RETENTION_NS as f64 * 1e-9
    }

    fn default_window_keyframes() -> usize {
        MapToOdomSettings::default().window_keyframes
    }

    fn default_min_sigma_xy_m() -> f32 {
        MapToOdomSettings::default().min_sigma_xy_m
    }

    fn default_min_sigma_z_m() -> f32 {
        MapToOdomSettings::default().min_sigma_z_m
    }

    fn default_min_sigma_yaw_deg() -> f32 {
        MapToOdomSettings::default().min_sigma_yaw_rad.to_degrees()
    }

    fn settings(&self) -> OdometrySettings {
        OdometrySettings {
            mount: CameraMount { camera_in_imu_m: Vec3::from_array(self.camera_in_imu_m) },
            retention_ns: (self.retention_s * 1e9).round() as i64,
            estimator: MapToOdomSettings {
                window_keyframes: self.window_keyframes,
                min_sigma_xy_m: self.min_sigma_xy_m,
                min_sigma_z_m: self.min_sigma_z_m,
                min_sigma_yaw_rad: self.min_sigma_yaw_deg.to_radians(),
            },
        }
    }
}

/// How the node turns odometry into `map_to_odom`.
#[derive(Clone, Debug)]
pub struct OdometrySettings {
    pub mount: CameraMount,
    /// Odometry poses, and registered keyframes waiting for odometry to reach them, are kept this long.
    pub retention_ns: i64,
    pub estimator: MapToOdomSettings,
}

impl OdometrySettings {
    pub const DEFAULT_RETENTION_NS: i64 = 10_000_000_000;
}

/// Whether the dataflow feeds the node an IMU stream. A keyframe's camera time comes from its imu message when it
/// is present, and from the keyframe's own capture time when it is absent (recordings made without an IMU).
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ImuStream {
    #[default]
    Present,
    Absent,
}

/// `HeadingWithoutGyro` as written in the settings file.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HeadingWithoutGyroSetting {
    Refuse,
    Search,
}

impl From<HeadingWithoutGyro> for HeadingWithoutGyroSetting {
    fn from(choice: HeadingWithoutGyro) -> Self {
        match choice {
            HeadingWithoutGyro::Refuse => HeadingWithoutGyroSetting::Refuse,
            HeadingWithoutGyro::Search => HeadingWithoutGyroSetting::Search,
        }
    }
}

impl From<HeadingWithoutGyroSetting> for HeadingWithoutGyro {
    fn from(setting: HeadingWithoutGyroSetting) -> Self {
        match setting {
            HeadingWithoutGyroSetting::Refuse => HeadingWithoutGyro::Refuse,
            HeadingWithoutGyroSetting::Search => HeadingWithoutGyro::Search,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NodeConfig {
    pub cloud: CloudSection,
    pub submap: SubmapSection,
    pub tracking: TrackingSection,
    pub loops: LoopSection,
    pub relocalise: RelocaliseSection,
    pub graph: GraphSection,
    pub gyro: GyroSection,
    pub odometry: Option<OdometrySection>,
}

impl Default for CloudSection {
    fn default() -> Self {
        let d = CloudSettings::default();
        CloudSection { stride_px: d.stride_px, voxel_m: d.voxel_m }
    }
}

impl Default for SubmapSection {
    fn default() -> Self {
        let d = LocalizerSettings::default().submap;
        SubmapSection { radius_m: d.radius_m, max_keyframes: d.max_keyframes }
    }
}

impl Default for TrackingSection {
    fn default() -> Self {
        let d = LocalizerSettings::default().tracking;
        TrackingSection {
            accept_fitness: d.accept_fitness,
            max_step_m: d.max_step_m,
            gyro_yaw_weight: d.gyro_yaw_weight,
            max_gyro_disagreement_deg: d.max_gyro_disagreement_deg,
            min_points: d.min_points,
        }
    }
}

impl Default for LoopSection {
    fn default() -> Self {
        let d = LocalizerSettings::default().loops;
        LoopSection {
            skip_latest_keyframes: d.skip_latest_keyframes,
            radius_m: d.radius_m,
            attempt_every_keyframes: d.attempt_every_keyframes,
            min_fitness: d.min_fitness,
            max_rmse_m: d.max_rmse_m,
            max_correction_m: d.max_correction_m,
            max_correction_deg: d.max_correction_deg,
        }
    }
}

impl Default for RelocaliseSection {
    fn default() -> Self {
        let d = LocalizerSettings::default().relocalise;
        RelocaliseSection { radius_m: d.radius_m, min_fitness: d.min_fitness, restart_after_lost: d.restart_after_lost }
    }
}

impl Default for GraphSection {
    fn default() -> Self {
        let d = LocalizerSettings::default().graph;
        GraphSection { optimise_interval_s: d.optimise_interval_ns as f64 * 1e-9, version_move_m: d.version_move_m, version_turn_deg: d.version_turn_deg }
    }
}

impl Default for GyroSection {
    fn default() -> Self {
        let d = GyroSettings::default();
        GyroSection {
            buffer_s: d.buffer_ns as f64 * 1e-9,
            min_still_s: d.still.min_still_ns as f64 * 1e-9,
            heading_without_gyro: HeadingWithoutGyro::default().into(),
            imu: ImuStream::default(),
        }
    }
}

/// Everything the node is built from.
#[derive(Clone, Debug)]
pub struct NodeSettings {
    pub localizer: LocalizerSettings,
    pub cloud: CloudSettings,
    pub gyro: GyroSettings,
    pub heading_without_gyro: HeadingWithoutGyro,
    pub imu: ImuStream,
    /// `None`: the dataflow wires no odometry.
    pub odometry: Option<OdometrySettings>,
}

impl Default for NodeSettings {
    fn default() -> Self {
        NodeConfig::default().settings()
    }
}

impl NodeConfig {
    pub fn read(path: &Path) -> Result<NodeConfig, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read { path: path.to_owned(), source })?;
        let config: NodeConfig = toml::from_str(&text).map_err(|source| ConfigError::Parse { path: path.to_owned(), source })?;
        if config.gyro.imu == ImuStream::Absent && config.gyro.heading_without_gyro != HeadingWithoutGyroSetting::Search {
            return Err(ConfigError::ImuAbsentNeedsSearch { path: path.to_owned() });
        }
        if config.odometry.is_some() && config.gyro.imu == ImuStream::Absent {
            return Err(ConfigError::OdometryNeedsImu { path: path.to_owned() });
        }
        Ok(config)
    }

    pub fn settings(&self) -> NodeSettings {
        let mut localizer = LocalizerSettings::default();
        localizer.submap.radius_m = self.submap.radius_m;
        localizer.submap.max_keyframes = self.submap.max_keyframes;
        localizer.submap.voxel_m = self.cloud.voxel_m;
        localizer.tracking.accept_fitness = self.tracking.accept_fitness;
        localizer.tracking.max_step_m = self.tracking.max_step_m;
        localizer.tracking.gyro_yaw_weight = self.tracking.gyro_yaw_weight;
        localizer.tracking.max_gyro_disagreement_deg = self.tracking.max_gyro_disagreement_deg;
        localizer.tracking.min_points = self.tracking.min_points;
        localizer.loops.skip_latest_keyframes = self.loops.skip_latest_keyframes;
        localizer.loops.radius_m = self.loops.radius_m;
        localizer.loops.attempt_every_keyframes = self.loops.attempt_every_keyframes;
        localizer.loops.min_fitness = self.loops.min_fitness;
        localizer.loops.max_rmse_m = self.loops.max_rmse_m;
        localizer.loops.max_correction_m = self.loops.max_correction_m;
        localizer.loops.max_correction_deg = self.loops.max_correction_deg;
        localizer.relocalise.radius_m = self.relocalise.radius_m;
        localizer.relocalise.min_fitness = self.relocalise.min_fitness;
        localizer.relocalise.restart_after_lost = self.relocalise.restart_after_lost;
        localizer.graph.optimise_interval_ns = (self.graph.optimise_interval_s * 1e9).round() as i64;
        localizer.graph.version_move_m = self.graph.version_move_m;
        localizer.graph.version_turn_deg = self.graph.version_turn_deg;
        let cloud = CloudSettings { stride_px: self.cloud.stride_px, voxel_m: self.cloud.voxel_m, ..CloudSettings::default() };
        let mut gyro = GyroSettings { buffer_ns: (self.gyro.buffer_s * 1e9).round() as i64, ..GyroSettings::default() };
        gyro.still.min_still_ns = (self.gyro.min_still_s * 1e9).round() as i64;
        NodeSettings {
            localizer,
            cloud,
            gyro,
            heading_without_gyro: self.gyro.heading_without_gyro.into(),
            imu: self.gyro.imu,
            odometry: self.odometry.as_ref().map(OdometrySection::settings),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("map-localizer.example.toml")
    }

    #[test]
    fn the_example_file_holds_the_defaults() {
        let config = NodeConfig::read(&example()).unwrap();
        let (read, defaults) = (config.settings(), NodeSettings::default());
        assert_eq!(format!("{:?}", read.localizer), format!("{:?}", defaults.localizer));
        assert_eq!(format!("{:?}", read.cloud), format!("{:?}", defaults.cloud));
        assert_eq!(format!("{:?}", read.gyro), format!("{:?}", defaults.gyro));
        assert_eq!(read.heading_without_gyro, HeadingWithoutGyro::Refuse);
        assert_eq!(read.imu, ImuStream::Present);
        assert!(read.odometry.is_none(), "the example wires no odometry");
    }

    #[test]
    fn an_odometry_section_needs_the_camera_offset_and_defaults_the_rest() {
        let config = written("[odometry]\ncamera_in_imu_m = [0.03, 0.0, -0.01]\n").unwrap();
        let odometry = config.settings().odometry.unwrap();
        assert_eq!(odometry.mount.camera_in_imu_m, Vec3::new(0.03, 0.0, -0.01));
        assert_eq!(odometry.retention_ns, OdometrySettings::DEFAULT_RETENTION_NS);
        assert_eq!(format!("{:?}", odometry.estimator), format!("{:?}", MapToOdomSettings::default()));
        let error = toml::from_str::<NodeConfig>("[odometry]\nretention_s = 5.0\n").unwrap_err();
        assert!(error.to_string().contains("camera_in_imu_m"), "{error}");
    }

    #[test]
    fn odometry_without_the_imu_is_refused() {
        let refused = written("[gyro]\nimu = \"absent\"\nheading_without_gyro = \"search\"\n[odometry]\ncamera_in_imu_m = [0.0, 0.0, 0.0]\n");
        assert!(matches!(refused, Err(ConfigError::OdometryNeedsImu { .. })));
    }

    fn written(text: &str) -> Result<NodeConfig, ConfigError> {
        let path = std::env::temp_dir().join(format!("map-localizer-{}-{}.toml", std::process::id(), text.len()));
        std::fs::write(&path, text).unwrap();
        let read = NodeConfig::read(&path);
        std::fs::remove_file(&path).unwrap();
        read
    }

    #[test]
    fn an_absent_imu_is_accepted_with_heading_search() {
        let config = written("[gyro]\nimu = \"absent\"\nheading_without_gyro = \"search\"\n").unwrap();
        assert_eq!(config.settings().imu, ImuStream::Absent);
    }

    #[test]
    fn an_absent_imu_without_heading_search_is_refused() {
        let refused = written("[gyro]\nimu = \"absent\"\n");
        assert!(matches!(refused, Err(ConfigError::ImuAbsentNeedsSearch { .. })));
    }

    #[test]
    fn heading_search_is_chosen_by_name() {
        let config: NodeConfig = toml::from_str("[gyro]\nheading_without_gyro = \"search\"\nmin_still_s = 0.5\n").unwrap();
        let settings = config.settings();
        assert_eq!(settings.heading_without_gyro, HeadingWithoutGyro::Search);
        assert_eq!(settings.gyro.still.min_still_ns, 500_000_000);
        assert!(toml::from_str::<NodeConfig>("[gyro]\nheading_without_gyro = \"guess\"\n").is_err());
    }

    #[test]
    fn a_key_overrides_its_default_and_leaves_the_rest() {
        let config: NodeConfig = toml::from_str("[graph]\nversion_move_m = 0.1\n").unwrap();
        let settings = config.settings();
        assert_eq!(settings.localizer.graph.version_move_m, 0.1);
        assert_eq!(settings.localizer.graph.version_turn_deg, 1.0);
    }

    #[test]
    fn an_unknown_key_is_rejected() {
        let error = toml::from_str::<NodeConfig>("[graph]\nversion_move = 0.1\n").unwrap_err();
        assert!(error.to_string().contains("version_move"), "{error}");
    }
}
