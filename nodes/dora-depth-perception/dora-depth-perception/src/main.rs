//! Keyframes and their instance masks in, scenes out.
//!
//! A `room_layout` scene (planes and room, no objects) goes out as soon as a keyframe arrives, for anything
//! that must not wait on segmentation. A full `scene` follows when that keyframe's instance masks arrive.

use std::collections::VecDeque;
use std::time::Instant;

use depth_perception::{
    find_planes, infer_room, object_footprint, sample_point_cloud, DepthImage, FloorFrame, ObjectMask, PlaneFit,
    PlaneSettings, RangeLimits, Room, RoomError, RoomSettings,
};
use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters, Parameter};
use eyre::Result;
use perception_wire::{decode_keyframe, decode_instance_masks, encode_scene, Keyframe, MessageContext, Scene, SceneObject};
use rayon::prelude::*;

const KEYFRAME_INPUT: &str = "keyframe";
const INSTANCE_MASKS_INPUT: &str = "instance_masks";
const ROOM_LAYOUT_OUTPUT: &str = "room_layout";
const SCENE_OUTPUT: &str = "scene";
/// Segmentation lags keyframes by seconds; this many recent keyframes wait for their instance masks.
const AWAITING_INSTANCE_MASKS: usize = 8;
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
    let floor = FloorFrame::from_specific_force(keyframe.specific_force);
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
    let room_layout_output = DataId::from(ROOM_LAYOUT_OUTPUT.to_owned());
    let scene_output = DataId::from(SCENE_OUTPUT.to_owned());
    let mut recent: VecDeque<Analysed> = VecDeque::with_capacity(AWAITING_INSTANCE_MASKS);

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, .. } if id.as_str() == KEYFRAME_INPUT => {
                let started = Instant::now();
                let analysed = analyse(decode_keyframe(data.as_ref()).for_message(KEYFRAME_INPUT)?);
                node.send_output(room_layout_output.clone(), timing("room_layout", started), encode_scene(&scene(&analysed, &[])))?;
                if recent.len() == AWAITING_INSTANCE_MASKS {
                    recent.pop_front();
                }
                recent.push_back(analysed);
            }
            Event::Input { id, data, .. } if id.as_str() == INSTANCE_MASKS_INPUT => {
                let started = Instant::now();
                let instance_masks = decode_instance_masks(data.as_ref()).for_message(INSTANCE_MASKS_INPUT)?;
                let Some(analysed) = recent.iter().find(|a| a.keyframe.frame_id == instance_masks.frame_id) else {
                    eprintln!("instance masks for frame {} arrived after its keyframe was dropped", instance_masks.frame_id);
                    continue;
                };
                let k = &analysed.keyframe;
                if (instance_masks.width, instance_masks.height) != (k.width, k.height) {
                    eyre::bail!("instance masks for frame {} are {}x{}, the keyframe {}x{}", k.frame_id, instance_masks.width, instance_masks.height, k.width, k.height);
                }
                let depth = DepthImage::new(k.width, k.height, &analysed.depth_m);
                let footprints: Vec<_> = (0..instance_masks.labels.len())
                    .into_par_iter()
                    .map(|index| {
                        let pixels: Vec<bool> = instance_masks.instance_ids.iter().map(|&id| id == index as i16).collect();
                        let mask = ObjectMask { width: k.width, height: k.height, pixels: &pixels };
                        object_footprint(&mask, &depth, &k.intrinsics, &analysed.floor, RangeLimits::D455, &Default::default())
                    })
                    .collect();
                let objects: Vec<SceneObject> = footprints
                    .iter()
                    .enumerate()
                    .filter_map(|(index, footprint)| {
                        footprint.as_ref().map(|footprint| SceneObject {
                            label: &instance_masks.labels[index],
                            instance_index: index as u16,
                            score: instance_masks.scores[index],
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
