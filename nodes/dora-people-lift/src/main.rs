//! dora node: joins each frame's `person_poses` with that frame's raw `depth` and the camera's `calibration`, and
//! publishes `people_tracks` for every pose frame, even an empty one.

use std::collections::{BTreeSet, VecDeque};
use std::time::Instant;

use dora_node_api::arrow::array::{Array, AsArray};
use dora_node_api::arrow::buffer::ScalarBuffer;
use dora_node_api::arrow::datatypes::UInt16Type;
use dora_node_api::{DoraNode, Event, MetadataParameters, Parameter};
use dora_people_lift::depth::DepthImage;
use dora_people_lift::tracks::{DepthFrame, PeopleTracker, TrackSettings};
use eyre::{bail, eyre, Context, Result};
use perception_wire::camera::decode_camera_calibration;
use perception_wire::people::{decode_person_poses, encode_people_tracks};

const POSES_INPUT: &str = "person_poses";
const DEPTH_INPUT: &str = "depth";
const CALIBRATION_INPUT: &str = "calibration";
const TRACKS_OUTPUT: &str = "people_tracks";
/// Depth frames kept while their pose frames are still on the Hailo; the pose model runs a few frames behind.
const DEPTH_FRAMES_KEPT: usize = 15;
/// Pose frames between the progress lines this node logs.
const LOG_EVERY_FRAMES: u32 = 30;

struct HeldDepth {
    frame_id: u64,
    width: usize,
    height: usize,
    depth_mm: ScalarBuffer<u16>,
    gravity_reaction: Option<[f32; 3]>,
}

fn integer(metadata: &MetadataParameters, name: &str) -> Result<i64> {
    match metadata.get(name) {
        Some(Parameter::Integer(value)) => Ok(*value),
        other => bail!("depth metadata {name} is {other:?}, not an integer"),
    }
}

fn hold_depth(metadata: &MetadataParameters, data: &dyn Array) -> Result<HeldDepth> {
    let (width, height) = (integer(metadata, "width")? as usize, integer(metadata, "height")? as usize);
    let depth_mm = data.as_primitive_opt::<UInt16Type>().ok_or_else(|| eyre!("depth is {:?}, not u16", data.data_type()))?.values().clone();
    if depth_mm.len() != width * height {
        bail!("depth has {} values for {width}x{height}", depth_mm.len());
    }
    let gravity_reaction = match metadata.get("gravity_reaction") {
        Some(Parameter::ListFloat(values)) if values.len() == 3 => Some([values[0] as f32, values[1] as f32, values[2] as f32]),
        None => None,
        other => bail!("depth metadata gravity_reaction is {other:?}, not three floats"),
    };
    Ok(HeldDepth { frame_id: integer(metadata, "frame_id")? as u64, width, height, depth_mm, gravity_reaction })
}

fn main() -> Result<()> {
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let mut tracker = PeopleTracker::new(TrackSettings::default());
    let mut calibration = None;
    let mut depths: VecDeque<HeldDepth> = VecDeque::with_capacity(DEPTH_FRAMES_KEPT + 1);
    let (mut logged_at, mut frames, mut with_depth, mut placed) = (Instant::now(), 0u32, 0u32, 0u32);
    let mut track_ids = BTreeSet::new();
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } => match id.as_str() {
                CALIBRATION_INPUT => calibration = Some(decode_camera_calibration(data.as_ref()).wrap_err("reading the calibration")?),
                DEPTH_INPUT => {
                    depths.push_back(hold_depth(&metadata.parameters, data.as_ref()).wrap_err("reading a depth frame")?);
                    if depths.len() > DEPTH_FRAMES_KEPT {
                        depths.pop_front();
                    }
                }
                POSES_INPUT => {
                    let poses = decode_person_poses(data.as_ref())?;
                    let held = depths.iter().find(|d| d.frame_id == poses.frame_id);
                    let depth = held.map(|d| DepthFrame {
                        image: DepthImage { width: d.width, height: d.height, depth_mm: &d.depth_mm },
                        gravity_reaction: d.gravity_reaction,
                    });
                    let tracks = tracker.update(&poses, depth.as_ref(), calibration.as_ref());
                    let mut parameters = MetadataParameters::default();
                    parameters.insert("depth_found".to_owned(), Parameter::Bool(held.is_some()));
                    node.send_output(TRACKS_OUTPUT.to_owned().into(), parameters, encode_people_tracks(&tracks))?;
                    frames += 1;
                    with_depth += held.is_some() as u32;
                    placed += tracks.tracks.iter().filter(|t| t.position.is_some()).count() as u32;
                    track_ids.extend(tracks.tracks.iter().map(|t| t.track_id));
                    if frames == LOG_EVERY_FRAMES {
                        let first = tracks.tracks.first().map(|t| format!(", track {} {:?} at {:?}", t.track_id, t.state, t.position));
                        eprintln!(
                            "[people-lift] {:.1} pose frames/s, {with_depth}/{frames} with depth, {placed} people placed, track ids {track_ids:?}{}",
                            frames as f64 / logged_at.elapsed().as_secs_f64(),
                            first.unwrap_or_default()
                        );
                        (logged_at, frames, with_depth, placed) = (Instant::now(), 0, 0, 0);
                        track_ids.clear();
                    }
                }
                other => bail!("unexpected input {other}"),
            },
            Event::Stop(_) => break,
            // The daemon has gone or the connection to it broke; nothing more will arrive.
            Event::Error(error) => bail!("dora reported an error: {error}"),
            _ => {}
        }
    }
    Ok(())
}
