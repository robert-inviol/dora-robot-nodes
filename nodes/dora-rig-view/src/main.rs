//! dora node serving the rig view page on `RIG_VIEW_PORT`.
//!
//! Inputs `keyframe`, `keyframe_pose`, `keyframe_poses_moved`, `odom_pose`, `map_to_odom`, `people_tracks`,
//! `surface_patches`, `object_update`, `topology`, `layout` (a structure or scene message of the geometry node) and a
//! dora timer `tick`, on which the robot and the people go out. Every input is
//! optional except `tick`; an input the node does not know stops it.

use std::future::IntoFuture;
use std::str::FromStr;

use dora_node_api::arrow::array::Array;
use dora_node_api::{DoraNode, Event};
use dora_rig_view::cloud::CloudSettings;
use dora_rig_view::scene::{Scene, SceneSettings};
use dora_rig_view::server::{router, SharedScene};
use dora_rig_view::view::Outgoing;
use eyre::{bail, eyre, Result, WrapErr};
use perception_wire::localization::{decode_keyframe_pose, decode_keyframe_poses_moved, decode_map_to_odom};
use perception_wire::objects::decode_object_update;
use perception_wire::odometry::decode_odom_pose;
use perception_wire::people::decode_people_tracks;
use perception_wire::scene_layout::decode_scene_layout;
use perception_wire::space_map::{decode_surface_patches, decode_topology};
use perception_wire::{decode_keyframe, MessageContext};

const PORT_VARIABLE: &str = "RIG_VIEW_PORT";
const STRIDE_VARIABLE: &str = "RIG_VIEW_STRIDE_PX";
const MAX_RANGE_VARIABLE: &str = "RIG_VIEW_MAX_RANGE_M";
const MAX_KEYFRAMES_VARIABLE: &str = "RIG_VIEW_MAX_KEYFRAMES";

fn required<T: FromStr>(name: &str) -> Result<T> {
    let value = std::env::var(name).map_err(|_| eyre!("{name} is not set"))?;
    value.parse().map_err(|_| eyre!("{name} is {value:?}, which does not parse"))
}

fn optional<T: FromStr>(name: &str, default: T) -> Result<T> {
    match std::env::var(name) {
        Ok(value) => value.parse().map_err(|_| eyre!("{name} is {value:?}, which does not parse")),
        Err(_) => Ok(default),
    }
}

fn settings() -> Result<SceneSettings> {
    let defaults = SceneSettings::DEFAULT;
    let cloud = CloudSettings {
        stride_px: optional(STRIDE_VARIABLE, defaults.cloud.stride_px)?,
        max_range_m: optional(MAX_RANGE_VARIABLE, defaults.cloud.max_range_m)?,
        ..defaults.cloud
    }
    .checked()?;
    let max_keyframes = optional(MAX_KEYFRAMES_VARIABLE, defaults.max_keyframes)?;
    if max_keyframes == 0 {
        bail!("{MAX_KEYFRAMES_VARIABLE} must keep at least one keyframe");
    }
    Ok(SceneSettings { cloud, max_keyframes })
}

fn on_input(scene: &mut Scene, input: &str, data: &dyn Array) -> Result<Vec<Outgoing>> {
    Ok(match input {
        "keyframe" => scene.on_keyframe(&decode_keyframe(data).for_message(input)?)?,
        "keyframe_pose" => scene.on_keyframe_pose(&decode_keyframe_pose(data).wrap_err("decoding a keyframe_pose message")?),
        "keyframe_poses_moved" => scene.on_keyframe_poses_moved(&decode_keyframe_poses_moved(data).for_message(input)?),
        "odom_pose" => {
            scene.on_odom_pose(&decode_odom_pose(data).for_message(input)?);
            Vec::new()
        }
        "map_to_odom" => {
            scene.on_map_to_odom(&decode_map_to_odom(data).wrap_err("decoding a map_to_odom message")?);
            Vec::new()
        }
        "people_tracks" => {
            scene.on_people_tracks(&decode_people_tracks(data).for_message(input)?);
            Vec::new()
        }
        "surface_patches" => {
            scene.on_surface_patches(&decode_surface_patches(data).wrap_err("decoding a surface_patches message")?)?
        }
        "object_update" => scene.on_object_update(&decode_object_update(data).for_message(input)?),
        "layout" => scene.on_scene_layout(decode_scene_layout(data).for_message(input)?),
        "topology" => scene.on_topology(&decode_topology(data).wrap_err("decoding a topology message")?),
        "tick" => scene.on_tick(),
        other => bail!("unknown input {other}"),
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with_writer(std::io::stderr)
        .init();
    let port: u16 = required(PORT_VARIABLE)?;
    let shared = SharedScene::new(Scene::new(settings()?));
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await.wrap_err_with(|| format!("listening on port {port}"))?;
    tracing::info!(port, "serving the rig view");
    let mut server = tokio::spawn(axum::serve(listener, router(shared.clone())).into_future());
    let (_node, mut events) = DoraNode::init_from_env()?;
    loop {
        tokio::select! {
            event = events.recv_async() => match event {
                Some(Event::Input { id, data, .. }) => shared.apply(|scene| on_input(scene, id.as_str(), data.as_ref()))?,
                Some(Event::Stop(cause)) => {
                    tracing::info!(?cause, "stopping");
                    return Ok(());
                }
                Some(other) => tracing::debug!(?other, "dora event"),
                None => return Ok(()),
            },
            stopped = &mut server => bail!("the web server stopped: {stopped:?}"),
        }
    }
}
