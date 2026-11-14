//! Keyframes, their poses and their segments in; object updates out.

use std::path::Path;

use eyre::{bail, Context, Result};
use glam::Vec3;
use object_map::{FrameJoiner, ObjectMap, ObjectMapSettings, Placement};
use dora_node_api::arrow::array::{Array, StructArray};
use perception_wire::localization::{decode_keyframe_pose, KeyframePose, Tracking};
use perception_wire::objects::encode_object_update;
use perception_wire::{decode_keyframe, decode_segments, MessageContext};
use room_map::Pose;

/// The environment variable naming the settings file; without it the settings are the defaults.
pub const SETTINGS_VARIABLE: &str = "OBJECT_MAP_SETTINGS";

pub const OBJECT_UPDATE_OUTPUT: &str = "object_update";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    Keyframe,
    KeyframePose,
    Segments,
}

impl Input {
    pub fn from_id(id: &str) -> Result<Input> {
        Ok(match id {
            "keyframe" => Input::Keyframe,
            "keyframe_pose" => Input::KeyframePose,
            "segments" => Input::Segments,
            other => bail!("the object map has no input {other:?}"),
        })
    }
}

pub fn read_settings(path: &Path) -> Result<ObjectMapSettings> {
    let text = std::fs::read_to_string(path).wrap_err_with(|| format!("reading the object map settings {}", path.display()))?;
    toml::from_str(&text).wrap_err_with(|| format!("parsing the object map settings {}", path.display()))
}

pub fn settings_from_environment() -> Result<ObjectMapSettings> {
    match std::env::var_os(SETTINGS_VARIABLE) {
        Some(path) => read_settings(Path::new(&path)),
        None => Ok(ObjectMapSettings::default()),
    }
}

fn placement(pose: &KeyframePose) -> Placement {
    match pose.tracking {
        Tracking::Lost => Placement::Lost,
        Tracking::Tracking | Tracking::Relocalised => {
            let p = pose.pose;
            Placement::Placed {
                pose: Pose { yaw_rad: p.yaw_rad, translation: Vec3::new(p.x_m, p.y_m, p.z_m) },
                map_version: pose.map_version,
            }
        }
    }
}

pub struct ObjectMapNode {
    joiner: FrameJoiner,
    map: ObjectMap,
}

impl ObjectMapNode {
    pub fn new(settings: ObjectMapSettings) -> ObjectMapNode {
        ObjectMapNode { joiner: FrameJoiner::new(settings.pending_frames), map: ObjectMap::new(settings) }
    }

    /// The object updates, as `object_update` messages, that one input leads to.
    pub fn on_input(&mut self, input: Input, data: &dyn Array) -> Result<Vec<StructArray>> {
        let ready = match input {
            Input::Keyframe => self.joiner.add_keyframe(decode_keyframe(data).for_message("keyframe")?)?,
            Input::Segments => self.joiner.add_segments(decode_segments(data).for_message("segments")?)?,
            Input::KeyframePose => {
                let pose = decode_keyframe_pose(data).wrap_err("decoding a keyframe_pose message")?;
                self.joiner.add_placement(pose.frame_id.0, placement(&pose))?
            }
        };
        let Some(frame) = ready else { return Ok(Vec::new()) };
        Ok(self.map.observe(&frame)?.iter().map(encode_object_update).collect())
    }
}
