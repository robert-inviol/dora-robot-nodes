//! What the page receives over its WebSocket: keyframe clouds as binary messages, everything else as JSON text
//! messages tagged by `type`. Positions are in the map frame (z up, metres) unless a message says otherwise.

use std::sync::Arc;

use bytes::Bytes;
use serde::Serialize;

/// One message to every page, cheap to clone for each of them.
#[derive(Clone, Debug, PartialEq)]
pub enum Outgoing {
    Binary(Bytes),
    Json(Arc<str>),
}

impl From<&Update> for Outgoing {
    fn from(update: &Update) -> Outgoing {
        Outgoing::Json(serde_json::to_string(update).expect("view updates serialize").into())
    }
}

/// A keyframe's floor frame in the map: turned by `yaw_rad` about z, then moved by `x_m`, `y_m`, `z_m`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct FloorPose {
    pub x_m: f32,
    pub y_m: f32,
    pub z_m: f32,
    pub yaw_rad: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NoticeTopic {
    /// While set, the robot is not drawn.
    Robot,
    People,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PersonView {
    pub track_id: u64,
    /// tentative, confirmed or lost.
    pub state: &'static str,
    pub torso: Option<[f32; 3]>,
    /// The fitted skeleton's joints in COCO-17 order; null where the fit has none.
    pub joints: Vec<Option<[f32; 3]>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RoomView {
    pub room_id: u64,
    pub polygon: Vec<[f32; 2]>,
    pub floor_m: f32,
    pub ceiling_m: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DoorwayView {
    pub position: [f32; 2],
    pub width_m: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlaceView {
    pub place_id: u64,
    pub room_id: u64,
    pub position: [f32; 2],
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Update {
    /// The first message on every connection: forget everything drawn so far.
    Reset,
    /// Where a keyframe's cloud goes; a keyframe the localizer lost is not drawn.
    KeyframePose { frame_id: u64, pose: FloorPose, placed: bool },
    /// The keyframe's cloud is no longer kept; remove it.
    KeyframeDropped { frame_id: u64 },
    /// The IMU body in the map, and its orientation as a quaternion (x, y, z, w) from body axes (x right, y down,
    /// z forward) into the map.
    Robot { position: [f32; 3], orientation: [f32; 4] },
    /// Every track of the latest people frame; an empty list clears them.
    People { tracks: Vec<PersonView> },
    Object {
        object_id: u64,
        label: String,
        confidence: f32,
        centre: [f32; 3],
        footprint: [[f32; 2]; 4],
        bottom_m: f32,
        top_m: f32,
        /// The cubes holding the object's points: cube (x, y, z) spans `[x, x + 1) * voxel_m` along x, and
        /// likewise along y and z.
        voxel_m: f32,
        voxels: Vec<[i32; 3]>,
    },
    ObjectGone { object_id: u64 },
    /// The whole latest topology; it replaces the one before.
    Topology { rooms: Vec<RoomView>, doorways: Vec<DoorwayView>, places: Vec<PlaceView>, edges: Vec<[u64; 2]> },
    /// Why a layer is not drawn, or null once it is again.
    Notice { topic: NoticeTopic, text: Option<String> },
}

/// Millimetre precision is all the page draws, and it keeps the JSON short.
pub fn mm(value: f32) -> f32 {
    (value * 1000.0).round() / 1000.0
}

pub fn mm3(value: glam::Vec3) -> [f32; 3] {
    [mm(value.x), mm(value.y), mm(value.z)]
}
