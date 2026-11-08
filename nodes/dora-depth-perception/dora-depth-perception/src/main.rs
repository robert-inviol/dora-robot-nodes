//! Keyframes and their segments in, scenes out.
//!
//! A `structure` scene (planes and room, no objects) goes out as soon as a keyframe arrives, for anything
//! that must not wait on segmentation. A full `scene` follows when that keyframe's segments arrive.

use std::collections::VecDeque;
use std::time::Instant;

use depth_perception::{
    find_planes, infer_room, object_footprint, sample_point_cloud, DepthImage, FloorFrame, ObjectMask, PlaneFit,
    PlaneSettings, RangeLimits, Room, RoomError, RoomSettings,
};
use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters, Parameter};
use eyre::Result;
use perception_wire::{decode_keyframe, decode_segments, encode_scene, Keyframe, MessageContext, Scene, SceneObject};
use rayon::prelude::*;

const KEYFRAME_INPUT: &str = "keyframe";
const SEGMENTS_INPUT: &str = "segments";
const STRUCTURE_OUTPUT: &str = "structure";
const SCENE_OUTPUT: &str = "scene";
/// Segmentation lags keyframes by seconds; this many recent keyframes wait for their segments.
const AWAITING_SEGMENTS: usize = 8;
/// Every second pixel each way keeps planes accurate at a quarter of the work.
const CLOUD_STRIDE_PX: usize = 2;

struct Analysed {
    keyframe: Keyframe,
    depth_m: Vec<f32>,
    floor: FloorFrame,
    fit: PlaneFit,
    room: Result<Room, RoomError>,
}

fn analyse(keyframe: Keyframe) -> Analysed {
    let depth_m = keyframe.depth_m();
    let floor = FloorFrame::from_gravity_reaction(keyframe.gravity_reaction);
    let depth = DepthImage::new(keyframe.width, keyframe.height, &depth_m);
    let cloud = sample_point_cloud(&depth, &keyframe.intrinsics, &floor, RangeLimits::D455, CLOUD_STRIDE_PX);
    let fit = find_planes(&cloud.points, &PlaneSettings::default());
    let room = infer_room(&fit.planes, &cloud.points, &RoomSettings::default());
    Analysed { keyframe, depth_m, floor, fit, room }
}

fn scene<'a>(analysed: &'a Analysed, objects: &'a [SceneObject<'a>]) -> Scene<'a> {
    let (camera_forward, camera_up) = analysed.floor.camera_forward_and_up();
    Scene {
        frame_id: analysed.keyframe.frame_id,
        captured_unix_ns: analysed.keyframe.captured_unix_ns,
        camera_forward,
        camera_up,
        planes: &analysed.fit.planes,
        room: &analysed.room,
        objects,
    }
}

fn timing(stage: &str, started: Instant) -> MetadataParameters {
    let mut parameters = MetadataParameters::default();
    parameters.insert(format!("{stage}_ms"), Parameter::Float(started.elapsed().as_secs_f64() * 1e3));
    parameters
}

fn main() -> Result<()> {
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let structure_output = DataId::from(STRUCTURE_OUTPUT.to_owned());
    let scene_output = DataId::from(SCENE_OUTPUT.to_owned());
    let mut recent: VecDeque<Analysed> = VecDeque::with_capacity(AWAITING_SEGMENTS);

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, .. } if id.as_str() == KEYFRAME_INPUT => {
                let started = Instant::now();
                let analysed = analyse(decode_keyframe(data.as_ref()).for_message(KEYFRAME_INPUT)?);
                node.send_output(structure_output.clone(), timing("structure", started), encode_scene(&scene(&analysed, &[])))?;
                if recent.len() == AWAITING_SEGMENTS {
                    recent.pop_front();
                }
                recent.push_back(analysed);
            }
            Event::Input { id, data, .. } if id.as_str() == SEGMENTS_INPUT => {
                let started = Instant::now();
                let segments = decode_segments(data.as_ref()).for_message(SEGMENTS_INPUT)?;
                let Some(analysed) = recent.iter().find(|a| a.keyframe.frame_id == segments.frame_id) else {
                    eprintln!("segments for frame {} arrived after its keyframe was dropped", segments.frame_id);
                    continue;
                };
                let k = &analysed.keyframe;
                if (segments.width, segments.height) != (k.width, k.height) {
                    eyre::bail!("segments for frame {} are {}x{}, the keyframe {}x{}", k.frame_id, segments.width, segments.height, k.width, k.height);
                }
                let depth = DepthImage::new(k.width, k.height, &analysed.depth_m);
                let footprints: Vec<_> = (0..segments.labels.len())
                    .into_par_iter()
                    .map(|index| {
                        let pixels: Vec<bool> = segments.segment_ids.iter().map(|&id| id == index as i16).collect();
                        let mask = ObjectMask { width: k.width, height: k.height, pixels: &pixels };
                        object_footprint(&mask, &depth, &k.intrinsics, &analysed.floor, RangeLimits::D455, &Default::default())
                    })
                    .collect();
                let objects: Vec<SceneObject> = footprints
                    .iter()
                    .enumerate()
                    .filter_map(|(index, footprint)| {
                        footprint.as_ref().map(|footprint| SceneObject {
                            label: &segments.labels[index],
                            segment_index: index as u16,
                            score: segments.scores[index],
                            footprint,
                        })
                    })
                    .collect();
                node.send_output(scene_output.clone(), timing("objects", started), encode_scene(&scene(analysed, &objects)))?;
            }
            Event::Stop(_) => break,
            _ => {}
        }
    }
    Ok(())
}
