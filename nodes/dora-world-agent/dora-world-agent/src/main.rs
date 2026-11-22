//! The spatial agent as a dora node: `spatial_query` in, `spatial_answer` and `nav_goal` out; `goal_status`,
//! `topology`, `occupancy_grid`, `object_update`, `keyframe_pose` and `people_tracks` go to the world store the agent
//! answers from, people transformed into the map frame with `map_to_odom`.

use std::path::PathBuf;

use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters};
use dora_world_agent::node::{SpatialNode, NAV_GOAL_OUTPUT, SPATIAL_ANSWER_OUTPUT};
use dora_world_agent::world::SqliteWorld;
use eyre::{eyre, Result, WrapErr};
use object_descriptions::gemini::{GeminiApiKey, GEMINI_API_KEY_VARIABLE};
use perception_wire::navigation::{encode_nav_goal, encode_spatial_answer};
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
    let gemini_key = GeminiApiKey::from_variable(std::env::var_os(GEMINI_API_KEY_VARIABLE));
    let (spatial, mut outputs) = SpatialNode::start(&config, world, gemini_key).await?;
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let nav_goal_output = DataId::from(NAV_GOAL_OUTPUT.to_owned());
    let answer_output = DataId::from(SPATIAL_ANSWER_OUTPUT.to_owned());
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
                node.send_output(answer_output.clone(), MetadataParameters::default(), encode_spatial_answer(&answer))?;
            }
        }
    }
    spatial.shutdown().await
}
