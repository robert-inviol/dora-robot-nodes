//! dora node that turns segments into persistent object updates.

use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters};
use dora_object_map::{settings_from_environment, Input, ObjectMapNode, OBJECT_UPDATE_OUTPUT};
use eyre::Result;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    // RUST_LOG picks what is logged; without it, every identity decision at info and above.
    let filter = EnvFilter::builder().with_default_directive(LevelFilter::INFO.into()).from_env()?;
    tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr).init();
    let settings = settings_from_environment()?;
    tracing::info!(?settings, "object map starting");
    let mut object_map = ObjectMapNode::new(settings);
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let output = DataId::from(OBJECT_UPDATE_OUTPUT.to_owned());

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, .. } => {
                for update in object_map.on_input(Input::from_id(id.as_str())?, data.as_ref())? {
                    node.send_output(output.clone(), MetadataParameters::default(), update)?;
                }
            }
            Event::Stop(_) => break,
            _ => {}
        }
    }
    Ok(())
}
