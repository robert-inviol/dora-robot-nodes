//! A serial lidar as a dora node: publishes every turn of the head as a `scan`.

mod clock;
mod delta2a_turns;
mod ld06_turns;
mod link;
mod model;
mod turn;
mod wire;

use std::time::Instant;

use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters};
use eyre::WrapErr;

use link::LidarLink;
use model::LidarModel;

const SERIAL_DEVICE_VARIABLE: &str = "LIDAR_SERIAL_DEVICE";
const MODEL_VARIABLE: &str = "LIDAR_MODEL";
const TICK_INPUT: &str = "tick";
const SCAN_OUTPUT: &str = "scan";

fn main() -> eyre::Result<()> {
    let device = std::env::var(SERIAL_DEVICE_VARIABLE)
        .wrap_err_with(|| format!("{SERIAL_DEVICE_VARIABLE} is not set"))?;
    let model: LidarModel = std::env::var(MODEL_VARIABLE)
        .wrap_err_with(|| format!("{MODEL_VARIABLE} is not set"))?
        .parse()?;
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let scan_output = DataId::from(SCAN_OUTPUT.to_owned());
    let mut link = LidarLink::new(device, model, Instant::now());

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, .. } if id.as_str() == TICK_INPUT => {
                for turn in link.poll(Instant::now()) {
                    node.send_output(
                        scan_output.clone(),
                        MetadataParameters::default(),
                        wire::scan_message(&turn),
                    )?;
                }
            }
            Event::Stop(_) => break,
            _ => {}
        }
    }
    Ok(())
}
