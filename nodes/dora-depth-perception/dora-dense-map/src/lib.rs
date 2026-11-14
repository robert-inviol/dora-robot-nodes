//! Keyframes, their poses and object updates in; free space, topology and keep-out zones out.

use std::path::Path;
use std::time::Instant;

use dense_map::{
    Arrival, BlockTsdf, DenseMapSettings, DenseMapper, DepthKeyframe, KeyframePose, MapOutput, ObjectUpdate, TickReport,
};
use dora_node_api::arrow::array::{Array, StructArray};
use eyre::{bail, Context, Result};
use perception_wire::localization::decode_keyframe_pose;
use perception_wire::navigation::UnixNanos;
use perception_wire::objects::decode_object_update;
use perception_wire::space_map::{encode_free_space, encode_keep_out, encode_topology};
use perception_wire::{decode_keyframe, MessageContext};

/// The environment variable naming the settings file; without it the settings are the defaults.
pub const SETTINGS_VARIABLE: &str = "DENSE_MAP_SETTINGS";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    Keyframe,
    KeyframePose,
    ObjectUpdate,
    /// A timer: lands background rebuilds and slices the map when it is due.
    Tick,
}

impl Input {
    pub fn from_id(id: &str) -> Result<Input> {
        Ok(match id {
            "keyframe" => Input::Keyframe,
            "keyframe_pose" => Input::KeyframePose,
            "object_update" => Input::ObjectUpdate,
            "tick" => Input::Tick,
            other => bail!("the dense map has no input {other:?}"),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    FreeSpace,
    Topology,
    KeepOut,
}

impl Output {
    pub fn id(self) -> &'static str {
        match self {
            Output::FreeSpace => "free_space",
            Output::Topology => "topology",
            Output::KeepOut => "keep_out",
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
}

impl DenseMapNode {
    pub fn new(settings: DenseMapSettings) -> Result<DenseMapNode> {
        settings.validate()?;
        let field = BlockTsdf::new(settings.tsdf);
        Ok(DenseMapNode { mapper: DenseMapper::new(settings, field) })
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
            Input::ObjectUpdate => {
                let message = decode_object_update(data).wrap_err("decoding an object_update message")?;
                let update = ObjectUpdate::try_from(&message)?;
                Ok(self.mapper.on_object_update(&update).map(|k| (Output::KeepOut, encode_keep_out(&k))).into_iter().collect())
            }
            Input::Tick => {
                let (outputs, report) = self.mapper.on_tick(now, now_unix)?;
                log_tick(&report);
                Ok(outputs
                    .into_iter()
                    .map(|output| match output {
                        MapOutput::FreeSpace(free_space) => (Output::FreeSpace, encode_free_space(&free_space)),
                        MapOutput::Topology(topology) => (Output::Topology, encode_topology(&topology)),
                    })
                    .collect())
            }
        }
    }
}

fn log_arrival(arrival: Arrival) {
    match arrival {
        Arrival::Integrated(frame) => tracing::debug!("integrated keyframe {}", frame.0),
        Arrival::Waiting | Arrival::AlreadyPlaced(_) => {}
        Arrival::WaitingDropped(frame) => tracing::warn!("keyframe or pose {} waited too long for its partner and is dropped", frame.0),
        Arrival::Skipped(frame) => tracing::info!("keyframe {} is not integrated: the localizer lost track", frame.0),
        Arrival::Moved(frame) => tracing::debug!("keyframe {} moved; the map will be rebuilt", frame.0),
    }
}

fn log_tick(report: &TickReport) {
    if let Some(version) = report.rebuild_started {
        tracing::info!("rebuilding the map for map version {}", version.0);
    }
    if let Some(version) = report.rebuilt {
        tracing::info!("rebuilt the map for map version {}", version.0);
    }
    if report.no_floor {
        tracing::info!("no free space yet: too little floor seen");
    }
    if let Some(time) = report.slice_time {
        tracing::debug!("free space and topology in {:.1} ms", time.as_secs_f64() * 1e3);
    }
}
