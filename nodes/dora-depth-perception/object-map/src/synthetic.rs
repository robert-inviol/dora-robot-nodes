//! Observed frames of a rendered box room with labelled box objects, with the masks a perfect segmenter would
//! give, for tests.

use depth_perception::synthetic::{Cuboid, SyntheticRoom};
use depth_perception::CameraIntrinsics;
use dora_node_api::arrow::buffer::ScalarBuffer;
use glam::Vec3;
use perception_wire::navigation::MapVersion;
use perception_wire::{Keyframe, Segments, NO_SEGMENT};
use room_map::Pose;

use crate::crop::encode_rgb_jpeg;
use crate::ObservedFrame;

#[derive(Clone, Debug)]
pub struct LabelledBox {
    pub label: String,
    pub score: f32,
    pub cuboid: Cuboid,
}

impl LabelledBox {
    pub fn new(label: &str, min: Vec3, max: Vec3) -> LabelledBox {
        LabelledBox { label: label.to_owned(), score: 0.9, cuboid: Cuboid { min, max } }
    }
}

/// A 3.5 m x 5 m x 2.4 m room, its floor 1 m below the map origin, seen at 640x360.
#[derive(Clone, Debug)]
pub struct SyntheticScene {
    pub room: Cuboid,
    pub objects: Vec<LabelledBox>,
}

pub const WIDTH: usize = 640;
pub const HEIGHT: usize = 360;
pub const INTRINSICS: CameraIntrinsics = CameraIntrinsics { fx: 250.0, fy: 250.0, ppx: 320.0, ppy: 180.0 };
pub const FLOOR_Z_M: f32 = -1.0;

impl SyntheticScene {
    pub fn living_room(objects: Vec<LabelledBox>) -> SyntheticScene {
        SyntheticScene { room: Cuboid { min: Vec3::new(-1.5, -1.0, FLOOR_Z_M), max: Vec3::new(2.0, 4.0, 1.4) }, objects }
    }

    /// The frame a level camera at `position`, turned `heading_rad` anticlockwise from facing +y, sees.
    pub fn observe(&self, frame_id: u64, captured_unix_ns: i64, position: Vec3, heading_rad: f32) -> ObservedFrame {
        let room = SyntheticRoom {
            room: self.room,
            objects: self.objects.iter().map(|o| o.cuboid).collect(),
            width: WIDTH,
            height: HEIGHT,
            intrinsics: INTRINSICS,
        };
        let rendered = room.render_from(position, heading_rad);
        let depth_mm: Vec<u16> = rendered.depth_m.iter().map(|&d| (d * 1000.0).round().clamp(0.0, 65535.0) as u16).collect();
        let segment_ids: Vec<i16> = rendered.object_ids.iter().map(|id| id.map_or(NO_SEGMENT, |i| i as i16)).collect();
        let colour: Vec<u8> = rendered.object_ids.iter().flat_map(|id| colour_of(*id)).collect();
        let keyframe = Keyframe {
            frame_id,
            captured_unix_ns,
            width: WIDTH,
            height: HEIGHT,
            intrinsics: INTRINSICS,
            gravity_reaction: rendered.accel_camera,
            depth_mm: ScalarBuffer::from(depth_mm),
            color_jpeg: encode_rgb_jpeg(WIDTH, HEIGHT, &colour, 90).expect("a synthetic image encodes"),
        };
        let segments = Segments {
            frame_id,
            width: WIDTH,
            height: HEIGHT,
            labels: self.objects.iter().map(|o| o.label.clone()).collect(),
            scores: self.objects.iter().map(|o| o.score).collect(),
            segment_ids: ScalarBuffer::from(segment_ids),
        };
        ObservedFrame { keyframe, segments, pose: Pose { yaw_rad: heading_rad, translation: position }, map_version: MapVersion(1) }
    }
}

fn colour_of(object: Option<usize>) -> [u8; 3] {
    match object {
        None => [128, 128, 128],
        Some(index) => [[200, 40, 40], [40, 200, 40], [40, 40, 200], [200, 200, 40]][index % 4],
    }
}
