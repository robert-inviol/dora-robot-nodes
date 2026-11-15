//! Keyframes, their poses, posed depth frames, the localizer's correction and object updates in; the occupancy grid, topology,
//! keep-out zones and mesh blocks out. A mesh snapshot request is answered with the whole surface as a
//! snapshot, the next message of the mesh_blocks sequence.

use std::path::Path;
use std::time::Instant;

use dense_map::{
    Arrival, BlockTsdf, DenseMapSettings, DenseMapper, DepthKeyframe, Tsdf, KeyframePose, MapOutput, ObjectUpdate,
    PosedDepthArrival, TickReport, Unfused, BLOCK_EDGE_VOXELS,
};
use dora_node_api::arrow::array::{Array, StructArray};
use eyre::{bail, Context, Result};
use glam::Vec3;
use perception_wire::localization::{decode_keyframe_pose, decode_pose_graph_update, decode_map_to_odom, Tracking};
use perception_wire::navigation::UnixNanos;
use perception_wire::objects::decode_object_update;
use perception_wire::posed_depth::decode_posed_depth_frame;
use perception_wire::space_map::{
    decode_mesh_snapshot_request, encode_occupancy_grid, encode_keep_out, encode_mesh_blocks, encode_topology,
};
use perception_wire::{decode_keyframe, MessageContext};

/// The environment variable naming the settings file; without it the settings are the defaults.
pub const SETTINGS_VARIABLE: &str = "DENSE_MAP_SETTINGS";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    Keyframe,
    KeyframePose,
    /// Keyframes already placed that the localizer's optimization moved.
    PoseGraphUpdate,
    ObjectUpdate,
    /// Depth frames at their odometry pose, fused as they arrive once the localizer has corrected their odom frame.
    PosedDepth,
    /// The localizer's correction from the odom frame to the map.
    MapToOdom,
    /// A timer: collects and starts background rebuilds, meshing and slicing, and sends what is ready.
    Tick,
    /// A receiver of mesh_blocks that missed a message or joined late, asking for the whole surface.
    MeshSnapshotRequest,
}

impl Input {
    pub fn from_id(id: &str) -> Result<Input> {
        Ok(match id {
            "keyframe" => Input::Keyframe,
            "keyframe_pose" => Input::KeyframePose,
            "pose_graph_update" => Input::PoseGraphUpdate,
            "object_update" => Input::ObjectUpdate,
            "posed_depth" => Input::PosedDepth,
            "map_to_odom" => Input::MapToOdom,
            "tick" => Input::Tick,
            "mesh_snapshot_request" => Input::MeshSnapshotRequest,
            other => bail!("the dense map has no input {other:?}"),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    OccupancyGrid,
    Topology,
    KeepOut,
    MeshBlocks,
}

impl Output {
    pub fn id(self) -> &'static str {
        match self {
            Output::OccupancyGrid => "occupancy_grid",
            Output::Topology => "topology",
            Output::KeepOut => "keep_out",
            Output::MeshBlocks => "mesh_blocks",
        }
    }
}

pub fn read_settings(path: &Path) -> Result<DenseMapSettings> {
    let text = std::fs::read_to_string(path).wrap_err_with(|| format!("reading the dense map settings {}", path.display()))?;
    let settings: DenseMapSettings =
        toml::from_str(&text).wrap_err_with(|| format!("parsing the dense map settings {}", path.display()))?;
    settings.validate().wrap_err_with(|| format!("checking the dense map settings {}", path.display()))?;
    Ok(settings)
}

pub fn settings_from_environment() -> Result<DenseMapSettings> {
    match std::env::var_os(SETTINGS_VARIABLE) {
        Some(path) => read_settings(Path::new(&path)),
        None => Ok(DenseMapSettings::default()),
    }
}

pub struct DenseMapNode {
    mapper: DenseMapper<BlockTsdf>,
    tally: MeshBlockTally,
    /// Why posed depth frames are not being fused, while they are not.
    unfused: Option<Unfused>,
    /// Surface snapshots sent on request in this run.
    snapshots: u64,
}

/// How often each block's surface was sent over a minute, logged so that blocks whose surface keeps changing can be
/// found.
struct MeshBlockTally {
    started: Option<Instant>,
    sends: std::collections::HashMap<[i32; 3], u32>,
    emptied: u32,
}

impl MeshBlockTally {
    const PERIOD: std::time::Duration = std::time::Duration::from_secs(60);
    const SHOWN: usize = 5;

    fn count(&mut self, surface: &perception_wire::space_map::MeshBlocks, now: Instant) {
        self.started.get_or_insert(now);
        for mesh_block in &surface.blocks {
            *self.sends.entry(mesh_block.block.to_array()).or_default() += 1;
            self.emptied += u32::from(mesh_block.triangles.is_empty());
        }
    }

    /// Logs the minute's sends once a minute has passed, and starts the next minute.
    fn log_when_due(&mut self, now: Instant, block_m: f32) {
        let Some(started) = self.started.filter(|started| now.duration_since(*started) >= Self::PERIOD) else { return };
        let mut busiest: Vec<([i32; 3], u32)> = self.sends.iter().map(|(block, sends)| (*block, *sends)).collect();
        busiest.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let shown: Vec<String> = busiest
            .iter()
            .take(Self::SHOWN)
            .map(|(block, sends)| {
                let centre = block.map(|b| (b as f32 + 0.5) * block_m);
                format!("{sends}x at ({:.1}, {:.1}, {:.1}) m", centre[0], centre[1], centre[2])
            })
            .collect();
        let sent: u32 = self.sends.values().sum();
        let seconds = now.duration_since(started).as_secs();
        tracing::info!(seconds, sent, blocks = self.sends.len(), emptied = self.emptied, busiest = %shown.join(", "), "mesh blocks sent");
        *self = MeshBlockTally { started: Some(now), sends: Default::default(), emptied: 0 };
    }
}

impl DenseMapNode {
    /// A node whose mesh_blocks messages carry `run_started`, the time the node started.
    pub fn new(settings: DenseMapSettings, run_started: UnixNanos) -> Result<DenseMapNode> {
        settings.validate()?;
        let field = BlockTsdf::new(settings.tsdf);
        let tally = MeshBlockTally { started: None, sends: Default::default(), emptied: 0 };
        Ok(DenseMapNode { mapper: DenseMapper::new(settings, field, run_started), tally, unfused: None, snapshots: 0 })
    }

    pub fn mapper(&self) -> &DenseMapper<BlockTsdf> {
        &self.mapper
    }

    /// The messages one input leads to.
    pub fn on_input(&mut self, input: Input, data: &dyn Array, now: Instant, now_unix: UnixNanos) -> Result<Vec<(Output, StructArray)>> {
        match input {
            Input::Keyframe => {
                let keyframe = decode_keyframe(data).for_message("keyframe")?;
                log_arrival(self.mapper.on_keyframe(DepthKeyframe::from_wire(&keyframe)));
                Ok(Vec::new())
            }
            Input::KeyframePose => {
                let pose = decode_keyframe_pose(data).wrap_err("decoding a keyframe_pose message")?;
                log_arrival(self.mapper.on_keyframe_pose(KeyframePose::from(&pose)));
                Ok(Vec::new())
            }
            Input::PoseGraphUpdate => {
                let moved = decode_pose_graph_update(data).wrap_err("decoding a pose_graph_update message")?;
                let poses = moved.keyframes.iter().map(|keyframe| KeyframePose {
                    frame_id: keyframe.frame_id,
                    map_version: moved.map_version,
                    translation: Vec3::new(keyframe.pose.x_m, keyframe.pose.y_m, keyframe.pose.z_m),
                    yaw_rad: keyframe.pose.yaw_rad,
                    tracking: Tracking::Tracking,
                });
                for arrival in self.mapper.on_pose_graph_update(poses) {
                    log_arrival(arrival);
                }
                Ok(Vec::new())
            }
            Input::PosedDepth => {
                let frame = decode_posed_depth_frame(data).for_message("posed_depth")?;
                let arrival = self.mapper.on_posed_depth(&frame);
                self.log_fusion(arrival);
                Ok(Vec::new())
            }
            Input::MapToOdom => {
                let correction = decode_map_to_odom(data).wrap_err("decoding a map_to_odom message")?;
                self.mapper.on_map_to_odom(correction);
                Ok(Vec::new())
            }
            Input::ObjectUpdate => {
                let message = decode_object_update(data).wrap_err("decoding an object_update message")?;
                let update = ObjectUpdate::try_from(&message)?;
                Ok(self.mapper.on_object_update(&update).map(|k| (Output::KeepOut, encode_keep_out(&k))).into_iter().collect())
            }
            Input::Tick => {
                let ticked = self.mapper.on_tick(now, now_unix)?;
                Ok(self.messages(ticked, now))
            }
            Input::MeshSnapshotRequest => {
                let request = decode_mesh_snapshot_request(data).wrap_err("decoding a mesh_snapshot_request message")?;
                let snapshot = self.mapper.mesh_snapshot(now_unix);
                self.snapshots += 1;
                tracing::info!(
                    blocks = snapshot.blocks.len(),
                    sequence = snapshot.sequence.0,
                    snapshots = self.snapshots,
                    requested_unix_ns = request.requested.0,
                    "sent the whole surface as a snapshot on request"
                );
                Ok(vec![(Output::MeshBlocks, encode_mesh_blocks(&snapshot))])
            }
        }
    }
}

impl DenseMapNode {
    /// As a tick, waiting for the background work it collects and starts, so that what it starts is sent now.
    pub fn on_tick_settled(&mut self, now: Instant, now_unix: UnixNanos) -> Result<Vec<(Output, StructArray)>> {
        let ticked = self.mapper.on_tick_settled(now, now_unix)?;
        Ok(self.messages(ticked, now))
    }

    fn messages(&mut self, (outputs, report): (Vec<MapOutput>, TickReport), now: Instant) -> Vec<(Output, StructArray)> {
        log_tick(&report);
        let block_m = self.mapper.field().voxel_m() * BLOCK_EDGE_VOXELS as f32;
        let messages = outputs
            .into_iter()
            .map(|output| match output {
                MapOutput::OccupancyGrid(occupancy_grid) => (Output::OccupancyGrid, encode_occupancy_grid(&occupancy_grid)),
                MapOutput::Topology(topology) => (Output::Topology, encode_topology(&topology)),
                MapOutput::MeshBlocks(surface) => {
                    self.tally.count(&surface, now);
                    (Output::MeshBlocks, encode_mesh_blocks(&surface))
                }
            })
            .collect();
        self.tally.log_when_due(now, block_m);
        messages
    }

    /// Logs when posed depth frames stop being fused, and why, and when fusing resumes.
    fn log_fusion(&mut self, arrival: PosedDepthArrival) {
        match (arrival, self.unfused) {
            (PosedDepthArrival::Fused, Some(_)) => {
                tracing::info!("fusing posed depth frames again");
                self.unfused = None;
            }
            (PosedDepthArrival::Fused, None) => {}
            (PosedDepthArrival::Unfused(why), before) if before != Some(why) => {
                match why {
                    Unfused::NoCorrection => tracing::warn!("posed depth frames are not fused until the localizer corrects their odom frame"),
                    Unfused::OtherOdomFrame { frame, correction } => tracing::warn!(
                        frame_odom_started_ns = frame.0,
                        correction_odom_started_ns = correction.0,
                        "posed depth frames are not fused: the localizer's correction is for another odom frame"
                    ),
                }
                self.unfused = Some(why);
            }
            (PosedDepthArrival::Unfused(_), _) => {}
        }
    }
}

fn log_arrival(arrival: Arrival) {
    match arrival {
        Arrival::Fused(frame) => tracing::debug!("fused keyframe {}", frame.0),
        Arrival::Waiting | Arrival::AlreadyAtPose(_) => {}
        Arrival::WaitingDropped(frame) => tracing::warn!("keyframe or pose {} waited too long for its partner and is dropped", frame.0),
        Arrival::Skipped(frame) => tracing::info!("keyframe {} is not fused: the localizer lost track", frame.0),
        Arrival::Moved(frame) => tracing::debug!("keyframe {} moved; its blocks will be rebuilt", frame.0),
        Arrival::NotHeld(frame) => tracing::debug!("keyframe {} moved but the map holds nothing of it", frame.0),
    }
}

fn log_tick(report: &TickReport) {
    if let Some(rebuild) = report.rebuild_started {
        tracing::info!(blocks = rebuild.blocks, "rebuilding the blocks of moved keyframes for map version {}", rebuild.map_version.0);
    }
    if let Some(rebuild) = report.rebuilt {
        tracing::info!(blocks = rebuild.blocks, "rebuilt the blocks of moved keyframes for map version {}", rebuild.map_version.0);
    }
    if report.let_go > 0 {
        tracing::debug!("let go of {} kept keyframes", report.let_go);
    }
    if report.no_floor {
        tracing::info!("no occupancy grid yet: too little floor seen");
    }
    if let Some(time) = report.slice_time {
        tracing::debug!("the occupancy grid and topology in {:.1} ms", time.as_secs_f64() * 1e3);
    }
    if let Some(time) = report.mesh_time {
        tracing::debug!("surface meshed in {:.1} ms", time.as_secs_f64() * 1e3);
    }
}
