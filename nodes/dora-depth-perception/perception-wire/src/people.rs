//! The people messages of lib/dora-rig-messages/dora_rig_messages/people.py: 2D poses from the pose model, and
//! tracks placed in 3D. Pixel coordinates are in the colour image, origin top left; keypoints are the 17 COCO
//! body points in COCO order.

use dora_node_api::arrow::array::{Array, StructArray};
use dora_node_api::arrow::datatypes::{DataType, Fields};
use eyre::{ensure, Result};
use serde::{Deserialize, Serialize};

use crate::serde_wire::{decode_row, encode_row, floats_struct, list_of, optional, required};

pub const KEYPOINT_COUNT: usize = 17;

/// Stable across frames for as long as the tracker follows the same person.
pub type TrackId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PixelBox {
    pub x_px: f32,
    pub y_px: f32,
    pub w_px: f32,
    pub h_px: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Keypoint {
    pub x_px: f32,
    pub y_px: f32,
    pub score: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x_m: f32,
    pub y_m: f32,
    pub z_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Velocity {
    pub vx_m_s: f32,
    pub vy_m_s: f32,
    pub vz_m_s: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersonPose {
    /// None until the tracker has numbered the person.
    #[serde(default)]
    pub track_id: Option<TrackId>,
    pub score: f32,
    #[serde(rename = "box")]
    pub bbox: PixelBox,
    pub keypoints: Vec<Keypoint>,
}

/// Every person the pose model found in one colour frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersonPoses {
    pub frame_id: u64,
    pub captured_camera_ns: i64,
    pub width: u16,
    pub height: u16,
    pub people: Vec<PersonPose>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrackState {
    Tentative,
    Confirmed,
    /// The tracker has stopped seeing the person; the position is where they were last seen.
    Lost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PositionFrame {
    /// The colour camera: x right, y down, z forward.
    Camera,
    /// Odometry: origin where the robot started, z up.
    Odom,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersonTrack {
    pub track_id: TrackId,
    pub state: TrackState,
    pub score: f32,
    #[serde(rename = "box")]
    pub bbox: PixelBox,
    pub keypoints: Vec<Keypoint>,
    /// The torso centre, when depth gives one.
    #[serde(default)]
    pub position: Option<Point>,
    #[serde(default)]
    pub velocity: Option<Velocity>,
    /// In keypoint order; None where depth has no reading or the keypoint is not trusted.
    pub keypoints_3d: Vec<Option<Point>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PeopleTracks {
    pub frame_id: u64,
    pub captured_camera_ns: i64,
    pub width: u16,
    pub height: u16,
    pub position_frame: PositionFrame,
    /// The accelerometer at rest in camera coordinates, as in a keyframe; None once positions are in odom.
    #[serde(default)]
    pub gravity_reaction: Option<[f32; 3]>,
    pub tracks: Vec<PersonTrack>,
}

fn pixel_box_type() -> DataType {
    floats_struct(&["x_px", "y_px", "w_px", "h_px"])
}

fn keypoints_type() -> DataType {
    list_of(floats_struct(&["x_px", "y_px", "score"]), false)
}

fn point_type() -> DataType {
    floats_struct(&["x_m", "y_m", "z_m"])
}

pub fn person_poses_fields() -> Fields {
    let person = DataType::Struct(Fields::from(vec![
        optional("track_id", DataType::UInt64),
        required("score", DataType::Float32),
        required("box", pixel_box_type()),
        required("keypoints", keypoints_type()),
    ]));
    Fields::from(vec![
        required("frame_id", DataType::UInt64),
        required("captured_camera_ns", DataType::Int64),
        required("width", DataType::UInt16),
        required("height", DataType::UInt16),
        required("people", list_of(person, false)),
    ])
}

pub fn people_tracks_fields() -> Fields {
    let track = DataType::Struct(Fields::from(vec![
        required("track_id", DataType::UInt64),
        required("state", DataType::Utf8),
        required("score", DataType::Float32),
        required("box", pixel_box_type()),
        required("keypoints", keypoints_type()),
        optional("position", point_type()),
        optional("velocity", floats_struct(&["vx_m_s", "vy_m_s", "vz_m_s"])),
        required("keypoints_3d", list_of(point_type(), true)),
    ]));
    Fields::from(vec![
        required("frame_id", DataType::UInt64),
        required("captured_camera_ns", DataType::Int64),
        required("width", DataType::UInt16),
        required("height", DataType::UInt16),
        required("position_frame", DataType::Utf8),
        optional("gravity_reaction", list_of(DataType::Float32, false)),
        required("tracks", list_of(track, false)),
    ])
}

pub fn encode_person_poses(poses: &PersonPoses) -> StructArray {
    encode_row(poses, person_poses_fields())
}

pub fn decode_person_poses(array: &dyn Array) -> Result<PersonPoses> {
    let poses: PersonPoses = decode_row(array, &person_poses_fields(), "person poses")?;
    for person in &poses.people {
        ensure!(person.keypoints.len() == KEYPOINT_COUNT, "a person has {} keypoints, not {KEYPOINT_COUNT}", person.keypoints.len());
    }
    Ok(poses)
}

pub fn encode_people_tracks(tracks: &PeopleTracks) -> StructArray {
    encode_row(tracks, people_tracks_fields())
}

pub fn decode_people_tracks(array: &dyn Array) -> Result<PeopleTracks> {
    decode_row(array, &people_tracks_fields(), "people tracks")
}
