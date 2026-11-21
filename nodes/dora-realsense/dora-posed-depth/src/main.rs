//! dora node: sends every third depth frame of dora-realsense, downsampled for mapping, at the odometry's pose of
//! the depth camera for exactly that frame, as `posed_depth`, with the people of the pose model's `person_poses` cut
//! out.

use std::time::{Duration, Instant};

use dora_node_api::arrow::array::{Array, AsArray};
use dora_node_api::arrow::datatypes::UInt16Type;
use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters, Parameter};
use dora_posed_depth::{PosedDepthJoiner, PosedDepthSettings};
use eyre::{bail, eyre, Context, Result};
use perception_wire::camera::decode_camera_calibration;
use perception_wire::odometry::decode_odom_pose;
use perception_wire::people::decode_person_poses;
use perception_wire::posed_depth::{encode_posed_depth_frame, PosedDepthFrame};
use shared_memory_outputs::SharedMemoryOutputs;

const DEPTH_INPUT: &str = "depth";
const ODOM_INPUT: &str = "odom_pose";
const CALIBRATION_INPUT: &str = "calibration";
const PERSON_POSES_INPUT: &str = "person_poses";
const POSED_DEPTH_OUTPUT: &str = "posed_depth";
/// The most shared-memory samples of the output its receivers may hold unreleased: three seconds of frames.
const UNRELEASED_SAMPLES_LIMIT: usize = 30;
const LOG_EVERY: Duration = Duration::from_secs(30);

fn integer(metadata: &MetadataParameters, name: &str) -> Result<i64> {
    match metadata.get(name) {
        Some(Parameter::Integer(value)) => Ok(*value),
        other => bail!("depth metadata {name} is {other:?}, not an integer"),
    }
}

fn send(node: &mut DoraNode, outputs: &mut SharedMemoryOutputs, frames: Vec<PosedDepthFrame>) -> Result<()> {
    let output = DataId::from(POSED_DEPTH_OUTPUT.to_owned());
    for frame in frames {
        outputs.send_array(node, &output, MetadataParameters::default(), encode_posed_depth_frame(&frame))?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let mut outputs = SharedMemoryOutputs::new(&node, UNRELEASED_SAMPLES_LIMIT)?;
    let mut joiner = PosedDepthJoiner::new(PosedDepthSettings::DEFAULT);
    let mut logged_at = Instant::now();
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } => {
                let posed = match id.as_str() {
                    CALIBRATION_INPUT => {
                        joiner.on_calibration(decode_camera_calibration(data.as_ref()).wrap_err("reading the calibration")?)?;
                        Vec::new()
                    }
                    DEPTH_INPUT => {
                        let parameters = &metadata.parameters;
                        let depth_mm = data.as_primitive_opt::<UInt16Type>().ok_or_else(|| eyre!("depth is {:?}, not u16", data.data_type()))?;
                        joiner.on_depth(
                            integer(parameters, "frame_id")? as u64,
                            integer(parameters, "camera_ns")?,
                            integer(parameters, "width")? as usize,
                            integer(parameters, "height")? as usize,
                            depth_mm.values(),
                        )?
                    }
                    PERSON_POSES_INPUT => {
                        joiner.on_person_poses(&decode_person_poses(data.as_ref()).wrap_err("reading person poses")?)?;
                        Vec::new()
                    }
                    ODOM_INPUT => joiner.on_odom_pose(decode_odom_pose(data.as_ref()).wrap_err("reading an odom pose")?)?,
                    other => bail!("unexpected input {other}"),
                };
                send(&mut node, &mut outputs, posed)?;
                if logged_at.elapsed() >= LOG_EVERY {
                    let t = joiner.take_tally();
                    eprintln!(
                        "[posed-depth] {:.1} frames/s sent; {} with people cut out, {} without their pose, {} dropped waiting, {} before the calibration, {} odometry restarts",
                        t.sent as f64 / logged_at.elapsed().as_secs_f64(),
                        t.with_people,
                        t.without_pose,
                        t.dropped_waiting,
                        t.before_calibration,
                        t.odometry_restarts
                    );
                    logged_at = Instant::now();
                }
            }
            Event::Stop(_) => break,
            // The daemon has gone or the connection to it broke; nothing more will arrive.
            Event::Error(error) => bail!("dora reported an error: {error}"),
            _ => {}
        }
    }
    Ok(())
}
