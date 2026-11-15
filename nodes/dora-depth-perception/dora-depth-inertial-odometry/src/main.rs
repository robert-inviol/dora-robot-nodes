//! dora node that tracks the camera's pose from its depth frames and IMU samples.
//!
//! Inputs `depth`, `imu` and `calibration`, as `dora-realsense` sends them, and `person_poses` from the pose model,
//! whose people are cut out of the depth; output `odom_pose`, one for every depth frame tracked once odometry has
//! started, except the frame on which odometry is found lost.

use dora_depth_inertial_odometry::{CalibrationOutcome, DepthFrame, DepthOutcome, FrameOutcome, ImuOutcome, NodeSettings, OdometryNode, StartProgress};
use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters};
use eyre::{bail, Result, WrapErr};
use perception_wire::camera::decode_camera_calibration;
use perception_wire::odometry::{encode_odom_pose, OdomPose};
use perception_wire::people::decode_person_poses;
use perception_wire::{decode_imu, MessageContext};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::EnvFilter;

const DEPTH_INPUT: &str = "depth";
const IMU_INPUT: &str = "imu";
const CALIBRATION_INPUT: &str = "calibration";
const PERSON_POSES_INPUT: &str = "person_poses";
const ODOM_POSE_OUTPUT: &str = "odom_pose";
/// Poses between the progress lines this node logs.
const LOG_EVERY_POSES: u32 = 300;

/// Sends poses and says how the tracking goes.
struct Publisher {
    node: DoraNode,
    output: DataId,
    told_of_missing_calibration: bool,
    poses: u32,
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
            // One refusal for every static initialization window the body moved through.
            StartProgress::NotStationary(moving) => tracing::warn!(%moving, "odometry has not started: hold the camera still"),
            StartProgress::Started => tracing::info!("the camera was stationary: odometry started"),
            StartProgress::StaticInitialization { .. } | StartProgress::Tracking => {}
        }
    }

    fn frame(&mut self, frame: FrameOutcome, odometry: &mut OdometryNode) -> Result<()> {
        match frame {
            FrameOutcome::BeforeStart => Ok(()),
            FrameOutcome::Posed(pose) => self.send(&pose, odometry),
            FrameOutcome::Lost { uncorrected_for, largest_position_sigma_m, last_second } => {
                tracing::warn!(?uncorrected_for, largest_position_sigma_m, %last_second, "odometry lost: no depth frame corrected the pose before its uncertainty passed the bound; hold the camera still to start again");
                Ok(())
            }
        }
    }

    fn send(&mut self, pose: &OdomPose, odometry: &mut OdometryNode) -> Result<()> {
        self.node.send_output(self.output.clone(), MetadataParameters::default(), encode_odom_pose(pose))?;
        self.poses += 1;
        if self.poses == LOG_EVERY_POSES {
            let frames = odometry.take_progress();
            tracing::info!(
                depth_frame_id = pose.depth_frame_id,
                poses = self.poses,
                corrected = frames.corrected,
                with_people = frames.with_people,
                position_m = %pose.position_m,
                position_sigma_m = %pose.position_sigma_m,
                %frames,
                "odom poses sent"
            );
            self.poses = 0;
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    // RUST_LOG picks what is logged; without it, info and above.
    let filter = EnvFilter::builder().with_default_directive(LevelFilter::INFO.into()).from_env()?;
    tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr).init();
    let settings = NodeSettings::d455();
    tracing::info!(stationary_for = ?settings.tracker.stationary_for, depth_stride = settings.tracker.depth_stride, "depth inertial odometry starting");
    let mut odometry = OdometryNode::new(settings);
    let (node, mut events) = DoraNode::init_from_env()?;
    let output = DataId::from(ODOM_POSE_OUTPUT.to_owned());
    let mut publisher = Publisher { node, output, told_of_missing_calibration: false, poses: 0 };
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
                            publisher.frame(frame, &mut odometry)?;
                        }
                    }
                },
                DEPTH_INPUT => match odometry.on_depth(DepthFrame::from_input(&metadata.parameters, data.as_ref()).wrap_err("reading a depth frame")?)? {
                    DepthOutcome::BeforeCalibration => publisher.dropped_before_calibration(DEPTH_INPUT),
                    DepthOutcome::HeldForImu { replaced: Some(replaced) } => {
                        tracing::warn!(frame_id = replaced.0, "depth frame dropped: a newer frame arrived before the imu message leading up to it")
                    }
                    DepthOutcome::HeldForImu { replaced: None } => {}
                    DepthOutcome::Tracked(frame) => publisher.frame(frame, &mut odometry)?,
                },
                PERSON_POSES_INPUT => odometry.on_person_poses(&decode_person_poses(data.as_ref()).for_message(PERSON_POSES_INPUT)?)?,
                other => bail!("unknown input {other}"),
            },
            Event::Stop(_) => break,
            _ => {}
        }
    }
    Ok(())
}
