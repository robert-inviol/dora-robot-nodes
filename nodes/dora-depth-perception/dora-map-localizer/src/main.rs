//! dora node that localizes keyframes in the map and sends each keyframe's pose, and with odometry the map to
//! odom correction.
//!
//! Inputs `keyframe`, `imu` and, with an `[odometry]` section in the settings, `odom_pose` and the camera's
//! `calibration`; outputs `keyframe_pose` and, with odometry, `map_to_odom`. An `odom_pose` message without that
//! section stops the node.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use dora_map_localizer::config::{ImuStream, NodeConfig, NodeSettings, CONFIG_VARIABLE};
use dora_map_localizer::{GyroGap, Handled, LocalizerNode};
use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters, Parameter};
use eyre::{bail, Result};
use map_localizer::HeadingWithoutGyro;
use perception_wire::localization::{encode_keyframe_pose, encode_pose_graph_update, encode_map_to_odom};
use perception_wire::camera::decode_camera_calibration;
use perception_wire::odometry::decode_odom_pose;
use perception_wire::{decode_imu, decode_keyframe, MessageContext};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::EnvFilter;

const KEYFRAME_INPUT: &str = "keyframe";
const IMU_INPUT: &str = "imu";
const ODOM_POSE_INPUT: &str = "odom_pose";
const CALIBRATION_INPUT: &str = "calibration";
const KEYFRAME_POSE_OUTPUT: &str = "keyframe_pose";
const MAP_TO_ODOM_OUTPUT: &str = "map_to_odom";
const KEYFRAME_POSES_MOVED_OUTPUT: &str = "pose_graph_update";

struct Outputs {
    keyframe_pose: DataId,
    map_to_odom: DataId,
    pose_graph_update: DataId,
}

fn settings() -> Result<NodeSettings> {
    match std::env::var_os(CONFIG_VARIABLE) {
        Some(path) => Ok(NodeConfig::read(&PathBuf::from(path))?.settings()),
        None => Ok(NodeSettings::default()),
    }
}

fn log_gyro_gap(gap: &GyroGap) {
    let (frame_id, problem, gyro_gaps) = (gap.frame_id.0, &gap.problem, gap.count);
    match gap.handled_by {
        HeadingWithoutGyro::Refuse => {
            tracing::error!(frame_id, %problem, gyro_gaps, "keyframe lost: the gyro cannot give its turn and heading_without_gyro is refuse")
        }
        HeadingWithoutGyro::Search => {
            tracing::warn!(frame_id, %problem, gyro_gaps, "the gyro cannot give this keyframe's turn; searching every heading (heading_without_gyro is search)")
        }
    }
}

/// Milliseconds to a tenth, for the log.
fn ms(duration: Duration) -> f64 {
    (duration.as_secs_f64() * 1e4).round() / 10.0
}

fn send(node: &mut DoraNode, outputs: &Outputs, handled: Handled, started: Instant) -> Result<()> {
    if let Some(dropped) = handled.dropped {
        tracing::warn!(frame_id = dropped.0, "keyframe replaced before its imu message arrived");
    }
    if let Some(gap) = &handled.gyro_gap {
        log_gyro_gap(gap);
    }
    for unjoined in &handled.unjoined {
        tracing::warn!(frame_id = unjoined.frame_id.0, problem = %unjoined.problem, "keyframe left out of map_to_odom: no odometry at its camera time");
    }
    if let Some(restart) = handled.odometry_restarted {
        tracing::warn!(
            previous_start_camera_ns = restart.previous_start.0,
            start_camera_ns = restart.start.0,
            "odometry started again in a new odom frame; map_to_odom waits for a keyframe in it"
        );
    }
    if let Some(time) = handled.odometry_before_calibration {
        tracing::warn!(camera_ns = time.0, "odometry pose left out: no calibration has said where the colour camera sits on the IMU yet");
    }
    if let Some(time) = handled.headingless_odometry {
        tracing::warn!(camera_ns = time.0, "odometry pose left out: the camera faces too near straight up or down to have a heading");
    }
    if let Some(estimate) = handled.estimate {
        let (fitness, inlier_rmse_m) = estimate.registration.map_or((None, None), |r| (Some(r.fitness), Some(r.inlier_rmse_m)));
        let work = handled.keyframe_work.expect("a keyframe estimate comes with the node's work on it");
        let stages = estimate.timings;
        let localized = started.elapsed();
        let odometry_step_m = work.odometry_step.as_ref().ok().map(|step| step.translation.length());
        let no_odometry_step = work.odometry_step.as_ref().err().map(ToString::to_string);
        tracing::info!(
            frame_id = estimate.frame_id.0,
            tracking = estimate.tracking.wire_name(),
            selection = ?estimate.selection,
            ?fitness,
            ?inlier_rmse_m,
            x_m = estimate.pose.translation.x,
            y_m = estimate.pose.translation.y,
            yaw_deg = estimate.pose.yaw_rad.to_degrees(),
            submap = estimate.submap.len(),
            ?odometry_step_m,
            ?no_odometry_step,
            cloud_points = work.cloud_points,
            cloud_ms = ms(work.cloud),
            tracking_target_ms = ms(stages.tracking_target),
            seed_trials_ms = ms(stages.tracking_seed_trials),
            refinement_ms = ms(stages.tracking_refinement),
            loop_closure_ms = ms(stages.loop_closure),
            relocalization_ms = ms(stages.relocalization),
            optimization_ms = ms(stages.pose_graph_optimization),
            map_to_odom_ms = ms(work.map_to_odom),
            localize_ms = ms(localized),
            "keyframe localized"
        );
        let mut parameters = MetadataParameters::default();
        parameters.insert("localize_ms".to_owned(), Parameter::Float(localized.as_secs_f64() * 1e3));
        node.send_output(outputs.keyframe_pose.clone(), parameters, encode_keyframe_pose(&estimate.to_message()))?;
    }
    if let Some(moved) = handled.moved {
        tracing::info!(map_version = moved.map_version.0, keyframes = moved.keyframes.len(), "optimization moved keyframes already sent");
        node.send_output(outputs.pose_graph_update.clone(), MetadataParameters::default(), encode_pose_graph_update(&moved))?;
    }
    if let Some(correction) = handled.map_to_odom {
        node.send_output(outputs.map_to_odom.clone(), MetadataParameters::default(), encode_map_to_odom(&correction))?;
    }
    Ok(())
}

fn main() -> Result<()> {
    // RUST_LOG picks what is logged; without it, info and above.
    let filter = EnvFilter::builder().with_default_directive(LevelFilter::INFO.into()).from_env()?;
    tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr).init();
    let settings = settings()?;
    tracing::info!(
        heading_without_gyro = ?settings.heading_without_gyro,
        imu = ?settings.imu,
        odometry = settings.odometry.is_some(),
        "map localizer starting"
    );
    let mut localizer = LocalizerNode::new(settings);
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let outputs = Outputs { keyframe_pose: DataId::from(KEYFRAME_POSE_OUTPUT.to_owned()), map_to_odom: DataId::from(MAP_TO_ODOM_OUTPUT.to_owned()),
        pose_graph_update: DataId::from(KEYFRAME_POSES_MOVED_OUTPUT.to_owned()),
    };
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, .. } => {
                let started = Instant::now();
                let handled = match id.as_str() {
                    KEYFRAME_INPUT => localizer.on_keyframe(decode_keyframe(data.as_ref()).for_message(KEYFRAME_INPUT)?),
                    IMU_INPUT if localizer.imu_stream() == ImuStream::Absent => {
                        bail!("an {IMU_INPUT} message arrived but the settings say imu = \"absent\"")
                    }
                    IMU_INPUT => localizer.on_imu(&decode_imu(data.as_ref()).for_message(IMU_INPUT)?)?,
                    CALIBRATION_INPUT => {
                        localizer.on_calibration(&decode_camera_calibration(data.as_ref()).for_message(CALIBRATION_INPUT)?)?;
                        continue;
                    }
                    ODOM_POSE_INPUT => localizer.on_odom_pose(&decode_odom_pose(data.as_ref()).for_message(ODOM_POSE_INPUT)?)?,
                    other => bail!("unknown input {other}"),
                };
                send(&mut node, &outputs, handled, started)?;
            }
            Event::Stop(_) => break,
            _ => {}
        }
    }
    Ok(())
}
