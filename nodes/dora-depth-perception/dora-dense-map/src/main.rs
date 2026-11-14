//! Keyframes, their poses and object updates in; free space, topology and keep-out zones out.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

use dora_dense_map::{settings_from_environment, DenseMapNode, Input};
use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters};
use eyre::{Context, Result};
use perception_wire::navigation::UnixNanos;
use tracing_subscriber::EnvFilter;

fn unix_now() -> Result<UnixNanos> {
    let since_epoch = SystemTime::now().duration_since(UNIX_EPOCH).wrap_err("the clock is before 1970")?;
    Ok(UnixNanos(i64::try_from(since_epoch.as_nanos()).wrap_err("the clock is past 2262")?))
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env()).with_writer(std::io::stderr).init();
    let mut dense_map = DenseMapNode::new(settings_from_environment()?)?;
    let (mut node, mut events) = DoraNode::init_from_env()?;
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, .. } => {
                let input = Input::from_id(id.as_str())?;
                for (output, message) in dense_map.on_input(input, data.as_ref(), Instant::now(), unix_now()?)? {
                    node.send_output(DataId::from(output.id().to_owned()), MetadataParameters::default(), message)?;
                }
            }
            Event::Stop(_) => break,
            _ => {}
        }
    }
    Ok(())
}
