//! dora node that tracks the camera's pose from its depth frames and IMU samples.
//!
//! Inputs `depth`, `imu` and `calibration`, as `dora-realsense` sends them; output `odom_pose`, one for every depth
//! frame tracked once odometry has started.

use dora_depth_inertial_odometry::{CalibrationOutcome, DepthFrame, DepthOutcome, FrameOutcome, ImuOutcome, NodeSettings, OdometryNode, StartProgress};
use dora_node_api::arrow::array::{Array, AsArray};
use dora_node_api::arrow::datatypes::UInt16Type;
use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters, Parameter};
use depth_inertial_odometry::tracker::DepthFrameId;
use eyre::{bail, eyre, Result, WrapErr};
use perception_wire::camera::decode_camera_calibration;
use perception_wire::localization::CameraNanos;
use perception_wire::odometry::{encode_odom_pose, OdomPose};
use perception_wire::{decode_imu, MessageContext};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::EnvFilter;

const DEPTH_INPUT: &str = "depth";
const IMU_INPUT: &str = "imu";
const CALIBRATION_INPUT: &str = "calibration";
const ODOM_POSE_OUTPUT: &str = "odom_pose";
/// Poses between the progress lines this node logs.
const LOG_EVERY_POSES: u32 = 300;

fn integer(metadata: &MetadataParameters, name: &str) -> Result<i64> {
    match metadata.get(name) {
        Some(Parameter::Integer(value)) => Ok(*value),
        other => bail!("depth metadata {name} is {other:?}, not an integer"),
    }
}

fn unsigned<T: TryFrom<i64>>(metadata: &MetadataParameters, name: &str) -> Result<T> {
    let value = integer(metadata, name)?;
    T::try_from(value).map_err(|_| eyre!("depth metadata {name} is {value}, which is out of range"))
}

fn depth_frame<'a>(metadata: &MetadataParameters, data: &'a dyn Array) -> Result<DepthFrame<'a>> {
    let depth_mm = data.as_primitive_opt::<UInt16Type>().ok_or_else(|| eyre!("depth is {:?}, not u16", data.data_type()))?.values();
    Ok(DepthFrame {
        frame_id: DepthFrameId(unsigned(metadata, "frame_id")?),
        time: CameraNanos(integer(metadata, "camera_ns")?),
        width: unsigned(metadata, "width")?,
        height: unsigned(metadata, "height")?,
        depth_mm,
    })
}

/// Sends poses and says how the tracking goes.
struct Publisher {
    node: DoraNode,
    output: DataId,
    told_of_missing_calibration: bool,
    poses: u32,
    corrected: u32,
}

impl Publisher {
    fn dropped_before_calibration(&mut self, input: &str) {
        if !self.told_of_missing_calibration {
            tracing::warn!(input, "no calibration has arrived: dropping every imu message and depth frame until one does");
            self.told_of_missing_calibration = true;
        }
    }

    fn start_progress(&self, start: StartProgress) {
        match start {
            // One refusal for every rest time the body moved through.
            StartProgress::NotAtRest(moving) => tracing::warn!(%moving, "odometry has not started: hold the camera still"),
            StartProgress::Started => tracing::info!("the camera rested: odometry started"),
            StartProgress::Resting { .. } | StartProgress::Tracking => {}
        }
    }

    fn frame(&mut self, frame: FrameOutcome) -> Result<()> {
        match frame {
            FrameOutcome::BeforeStart => Ok(()),
            FrameOutcome::Posed(pose) => self.send(&pose),
        }
    }

    fn send(&mut self, pose: &OdomPose) -> Result<()> {
        self.node.send_output(self.output.clone(), MetadataParameters::default(), encode_odom_pose(pose))?;
        self.poses += 1;
        self.corrected += u32::from(pose.corrected);
        if self.poses == LOG_EVERY_POSES {
            tracing::info!(
                depth_frame_id = pose.depth_frame_id,
                corrected = self.corrected,
                poses = self.poses,
                position_m = %pose.position_m,
                position_sigma_m = %pose.position_sigma_m,
                "odom poses sent"
            );
            (self.poses, self.corrected) = (0, 0);
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    // RUST_LOG picks what is logged; without it, info and above.
    let filter = EnvFilter::builder().with_default_directive(LevelFilter::INFO.into()).from_env()?;
    tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr).init();
    let settings = NodeSettings::d455();
    tracing::info!(rest_for = ?settings.tracker.rest_for, depth_stride = settings.tracker.depth_stride, "depth inertial odometry starting");
    let mut odometry = OdometryNode::new(settings);
    let (node, mut events) = DoraNode::init_from_env()?;
    let mut publisher = Publisher { node, output: DataId::from(ODOM_POSE_OUTPUT.to_owned()), told_of_missing_calibration: false, poses: 0, corrected: 0 };
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } => match id.as_str() {
                CALIBRATION_INPUT => match odometry.on_calibration(&decode_camera_calibration(data.as_ref()).for_message(CALIBRATION_INPUT)?)? {
                    CalibrationOutcome::TrackerBuilt(calibration) => tracing::info!(?calibration, "calibration arrived: waiting for the camera to rest"),
                    CalibrationOutcome::Unchanged => {}
                },
                IMU_INPUT => match odometry.on_imu(&decode_imu(data.as_ref()).for_message(IMU_INPUT)?)? {
                    ImuOutcome::BeforeCalibration => publisher.dropped_before_calibration(IMU_INPUT),
                    ImuOutcome::Fed { start, held_frame } => {
                        publisher.start_progress(start);
                        if let Some(frame) = held_frame {
                            publisher.frame(frame)?;
                        }
                    }
                },
                DEPTH_INPUT => match odometry.on_depth(depth_frame(&metadata.parameters, data.as_ref()).wrap_err("reading a depth frame")?)? {
                    DepthOutcome::BeforeCalibration => publisher.dropped_before_calibration(DEPTH_INPUT),
                    DepthOutcome::HeldForImu { replaced: Some(replaced) } => {
                        tracing::warn!(frame_id = replaced.0, "depth frame dropped: a newer frame arrived before the imu message leading up to it")
                    }
                    DepthOutcome::HeldForImu { replaced: None } => {}
                    DepthOutcome::Tracked(frame) => publisher.frame(frame)?,
                },
                other => bail!("unknown input {other}"),
            },
            Event::Stop(_) => break,
            _ => {}
        }
    }
    Ok(())
}
