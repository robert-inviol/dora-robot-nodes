//! The spatial agent as a dora node: `space_question` in, `space_answer` and `nav_goal` out; `goal_status`,
//! `topology`, `free_space`, `object_update` and `keyframe_pose` go to the world store the agent answers from.

use std::path::PathBuf;

use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters};
use dora_world_agent::node::{SpatialNode, NAV_GOAL_OUTPUT, SPACE_ANSWER_OUTPUT};
use dora_world_agent::world::SqliteWorld;
use eyre::{eyre, Result, WrapErr};
use perception_wire::navigation::{encode_nav_goal, encode_space_answer};
use world_agent::config::SpatialConfig;

const CONFIG_VARIABLE: &str = "SPATIAL_AGENT_CONFIG";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,rmcp=warn".into()))
        .with_writer(std::io::stderr)
        .init();
    let path = PathBuf::from(std::env::var_os(CONFIG_VARIABLE).ok_or_else(|| eyre!("{CONFIG_VARIABLE} is not set"))?);
    let config = SpatialConfig::load(&path)?;
    let world = SqliteWorld::open(&config.world_store)
        .wrap_err_with(|| format!("opening the world store at {}", config.world_store.path.display()))?;
    let (spatial, mut outputs) = SpatialNode::start(&config, world).await?;
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let nav_goal_output = DataId::from(NAV_GOAL_OUTPUT.to_owned());
    let answer_output = DataId::from(SPACE_ANSWER_OUTPUT.to_owned());
    loop {
        tokio::select! {
            event = events.recv_async() => match event {
                Some(Event::Input { id, data, .. }) => spatial.on_input(id.as_str(), data.as_ref()).await?,
                Some(Event::Stop(cause)) => {
                    tracing::info!(?cause, "stopping");
                    break;
                }
                Some(other) => tracing::debug!(?other, "dora event"),
                None => break,
            },
            Some(sent) = outputs.goals.recv() => {
                let goal = spatial.on_goal(sent)?;
                node.send_output(nav_goal_output.clone(), MetadataParameters::default(), encode_nav_goal(&goal))?;
            }
            Some(answer) = outputs.answers.recv() => {
                node.send_output(answer_output.clone(), MetadataParameters::default(), encode_space_answer(&answer))?;
            }
        }
    }
    spatial.shutdown().await
}
