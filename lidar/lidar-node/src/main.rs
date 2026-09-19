//! The Delta-2A lidar as a dora node: publishes every revolution of the head as a `scan`.

mod link;
mod wire;

use std::time::Instant;

use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event};
use eyre::WrapErr;

use link::LidarLink;

const SERIAL_DEVICE_VARIABLE: &str = "LIDAR_SERIAL_DEVICE";
const TICK_INPUT: &str = "tick";
const SCAN_OUTPUT: &str = "scan";

fn main() -> eyre::Result<()> {
    let device = std::env::var(SERIAL_DEVICE_VARIABLE)
        .wrap_err_with(|| format!("{SERIAL_DEVICE_VARIABLE} is not set"))?;
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let scan_output = DataId::from(SCAN_OUTPUT.to_owned());
    let mut link = LidarLink::new(device, Instant::now());

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, .. } if id.as_str() == TICK_INPUT => {
                for revolution in link.poll(Instant::now()) {
                    node.send_output(
                        scan_output.clone(),
                        wire::scan_parameters(&revolution),
                        wire::scan_rows(&revolution),
                    )?;
                }
            }
            Event::Stop(_) => break,
            _ => {}
        }
    }
    Ok(())
}
