//! dora node serving the rig view page on `RIG_VIEW_ADDRESS`, port `RIG_VIEW_PORT`.
//!
//! Inputs `keyframe`, `keyframe_pose`, `pose_graph_update`, `odom_pose`, `map_to_odom`, `people_tracks`,
//! `mesh_blocks`, `object_update`, `topology` and `room_layout` (a room_layout or scene message of the plane_segmentation node).
//! Every input is optional; an input the node does not know, or an error from dora, stops it. Output
//! `mesh_snapshot_request` asks the dense map for its whole surface when a surface message was missed or the node
//! joined the dense map's run late.

use std::future::IntoFuture;
use std::net::IpAddr;
use std::str::FromStr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use dora_node_api::arrow::array::Array;
use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters};
use dora_rig_view::cloud::CloudSettings;
use dora_rig_view::rig::{RigInput, RigView, RigViewError, RigViewSettings};
use eyre::{bail, eyre, Result, WrapErr};
use perception_wire::localization::{decode_keyframe_pose, decode_pose_graph_update, decode_map_to_odom};
use perception_wire::objects::decode_object_update;
use perception_wire::odometry::decode_odom_pose;
use perception_wire::people::decode_people_tracks;
use perception_wire::room_layout::decode_room_layout;
use perception_wire::navigation::UnixNanos;
use perception_wire::space_map::{decode_mesh_blocks, decode_topology, encode_mesh_snapshot_request, MeshSnapshotRequest};
use perception_wire::{decode_keyframe, MessageContext};

const ADDRESS_VARIABLE: &str = "RIG_VIEW_ADDRESS";
const PORT_VARIABLE: &str = "RIG_VIEW_PORT";
const STRIDE_VARIABLE: &str = "RIG_VIEW_STRIDE_PX";
const MAX_RANGE_VARIABLE: &str = "RIG_VIEW_MAX_RANGE_M";
const MAX_KEYFRAMES_VARIABLE: &str = "RIG_VIEW_MAX_KEYFRAMES";
const SNAPSHOT_REQUEST_OUTPUT: &str = "mesh_snapshot_request";

fn unix_now() -> Result<UnixNanos> {
    let since_epoch = SystemTime::now().duration_since(UNIX_EPOCH).wrap_err("the clock is before 1970")?;
    Ok(UnixNanos(i64::try_from(since_epoch.as_nanos()).wrap_err("the clock is past 2262")?))
}

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

fn settings() -> Result<RigViewSettings> {
    let defaults = RigViewSettings::DEFAULT;
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
    Ok(RigViewSettings { cloud, max_keyframes })
}

fn decode(input: &str, data: &dyn Array) -> Result<RigInput> {
    Ok(match input {
        "keyframe" => RigInput::Keyframe(decode_keyframe(data).for_message(input)?),
        "keyframe_pose" => RigInput::KeyframePose(decode_keyframe_pose(data).wrap_err("decoding a keyframe_pose message")?),
        "pose_graph_update" => RigInput::PoseGraphUpdate(decode_pose_graph_update(data).for_message(input)?),
        "odom_pose" => RigInput::OdomPose(decode_odom_pose(data).for_message(input)?),
        "map_to_odom" => RigInput::MapToOdom(decode_map_to_odom(data).wrap_err("decoding a map_to_odom message")?),
        "people_tracks" => RigInput::PeopleTracks(decode_people_tracks(data).for_message(input)?),
        "mesh_blocks" => {
            RigInput::MeshBlocks(decode_mesh_blocks(data).wrap_err("decoding a mesh_blocks message")?)
        }
        "object_update" => RigInput::ObjectUpdate(decode_object_update(data).for_message(input)?),
        "room_layout" => RigInput::RoomLayout(decode_room_layout(data).for_message(input)?),
        "topology" => RigInput::Topology(decode_topology(data).wrap_err("decoding a topology message")?),
        other => bail!("unknown input {other}"),
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with_writer(std::io::stderr)
        .init();
    let address: IpAddr = required(ADDRESS_VARIABLE)?;
    let port: u16 = required(PORT_VARIABLE)?;
    let (mut rig_view, mut background) = RigView::start(settings()?);
    let listener =
        tokio::net::TcpListener::bind((address, port)).await.wrap_err_with(|| format!("listening on {address} port {port}"))?;
    tracing::info!(%address, port, "serving the rig view");
    let mut server = tokio::spawn(axum::serve(listener, rig_view.router()).into_future());
    let (mut node, mut events) = DoraNode::init_from_env()?;
    loop {
        tokio::select! {
            event = events.recv_async() => match event {
                Some(Event::Input { id, data, .. }) => match rig_view.on_input(decode(id.as_str(), data.as_ref())?, Instant::now()) {
                    Ok(None) => {}
                    Ok(Some(reason)) => {
                        tracing::info!(?reason, "asking the dense map for a mesh snapshot");
                        let request = encode_mesh_snapshot_request(&MeshSnapshotRequest { requested: unix_now()? });
                        node.send_output(DataId::from(SNAPSHOT_REQUEST_OUTPUT.to_owned()), MetadataParameters::default(), request)?;
                    }
                    Err(RigViewError::BackgroundStopped) => bail!("the keyframe and surface work stopped: {:?}", (&mut background).await),
                    Err(error) => return Err(error.into()),
                },
                Some(Event::Stop(cause)) => {
                    tracing::info!(?cause, "stopping");
                    return Ok(());
                }
                Some(Event::Error(error)) => bail!("dora reported an error: {error}"),
                Some(other) => tracing::debug!(?other, "dora event"),
                None => return Ok(()),
            },
            stopped = &mut server => bail!("the web server stopped: {stopped:?}"),
            stopped = &mut background => bail!("the keyframe and surface work stopped: {stopped:?}"),
        }
    }
}
