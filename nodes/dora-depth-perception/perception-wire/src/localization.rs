//! The localization messages of lib/dora-rig-messages/dora_rig_messages/localization.py, as single-row Arrow
//! structs. Field names, types and nullability must match that module exactly; the tests read its golden fixtures.
//!
//! The map frame is z up, gravity aligned, metres. A keyframe pose is applied to that keyframe's gravity-aligned
//! floor frame: a floor-frame point p is at R_z(yaw) p + (x, y, z) in the map, yaw anticlockwise seen from above.

use std::str::FromStr;
use std::sync::Arc;

use dora_node_api::arrow::array::{Array, ArrayRef, AsArray, Float32Array, Int64Array, StringArray, StructArray, UInt64Array};
use dora_node_api::arrow::datatypes::{DataType, Fields, Float32Type, Int64Type, UInt64Type};

use serde::{Deserialize, Serialize};

use crate::navigation::{MapVersion, UnixNanos, WireError};
use crate::serde_wire::{decode_row, encode_row};
pub use wire_schemas::localization::{keyframe_pose_fields, map_to_odom_fields, pose_graph_update_fields};

/// A keyframe's id, as the camera node numbered it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FrameId(pub u64);

/// A time on the depth camera's own clock, shared by all its sensors, in nanoseconds; not a unix time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CameraNanos(pub i64);

/// How a keyframe's pose was found.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tracking {
    /// Registered against the map around the previous keyframe.
    Tracking,
    /// Registered again by matching stored keyframes after tracking was lost.
    Relocalized,
    /// Starts a new run after tracking was lost: in the map, joined to it only by a dead-reckoning edge, not
    /// registered against it. Its pose is dead reckoning, which later pose graph updates move.
    Restarted,
    /// Not registered and not in the map: dead reckoning from the gyro and the last known position.
    Lost,
}

impl Tracking {
    pub const ALL: [Tracking; 4] = [Tracking::Tracking, Tracking::Relocalized, Tracking::Restarted, Tracking::Lost];

    pub fn wire_name(self) -> &'static str {
        match self {
            Tracking::Tracking => "tracking",
            Tracking::Relocalized => "relocalized",
            Tracking::Restarted => "restarted",
            Tracking::Lost => "lost",
        }
    }
}

impl FromStr for Tracking {
    type Err = WireError;

    fn from_str(name: &str) -> Result<Self, WireError> {
        Tracking::ALL
            .into_iter()
            .find(|state| state.wire_name() == name)
            .ok_or_else(|| WireError::UnknownName { field: "tracking", name: name.to_owned() })
    }
}

/// Whether keyframe selection added a keyframe to the map.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyframeSelection {
    /// The keyframe is a node of the pose graph. A keyframe that restarts tracking after loss is one, reported
    /// restarted.
    Selected,
    /// Localized within the keyframe selection distance of the map keyframe `reference` and not added to the map;
    /// its pose keeps its offset from the reference keyframe's.
    NotSelected { reference: FrameId },
    /// Not localized, and not in the map.
    NotLocalized,
}

impl KeyframeSelection {
    pub fn wire_name(self) -> &'static str {
        match self {
            KeyframeSelection::Selected => "selected",
            KeyframeSelection::NotSelected { .. } => "not_selected",
            KeyframeSelection::NotLocalized => "not_localized",
        }
    }

    fn from_wire(name: &str, reference: Option<FrameId>) -> Result<KeyframeSelection, WireError> {
        let mismatch = |has_reference| WireError::ReferenceDoesNotFitSelection { selection: name.to_owned(), has_reference };
        match (name, reference) {
            ("selected", None) => Ok(KeyframeSelection::Selected),
            ("not_selected", Some(reference)) => Ok(KeyframeSelection::NotSelected { reference }),
            ("not_localized", None) => Ok(KeyframeSelection::NotLocalized),
            ("not_selected", None) => Err(mismatch(false)),
            ("selected" | "not_localized", Some(_)) => Err(mismatch(true)),
            _ => Err(WireError::UnknownName { field: "selection", name: name.to_owned() }),
        }
    }

    /// Exactly the lost keyframes are not localized, and a restarted keyframe is in the map.
    pub fn fits(self, tracking: Tracking) -> bool {
        match tracking {
            Tracking::Lost => self == KeyframeSelection::NotLocalized,
            Tracking::Restarted => self == KeyframeSelection::Selected,
            Tracking::Tracking | Tracking::Relocalized => self != KeyframeSelection::NotLocalized,
        }
    }
}

/// A turn about the vertical then a translation, in metres and radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose4Dof {
    pub x_m: f32,
    pub y_m: f32,
    pub z_m: f32,
    pub yaw_rad: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyframePose {
    pub frame_id: FrameId,
    pub captured: UnixNanos,
    /// The pose graph the pose comes from; it rises whenever optimization moves past keyframes.
    pub map_version: MapVersion,
    /// The keyframe's floor frame in the map.
    pub pose: Pose4Dof,
    pub tracking: Tracking,
    pub selection: KeyframeSelection,
    /// Registration RMS of the inliers; meaningful only when `inlier_fraction` is above zero.
    pub residual_m: f32,
    /// Share of the keyframe's points within the inlier distance of the map surface.
    pub inlier_fraction: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MapToOdom {
    pub map_version: MapVersion,
    /// Odometry poses from this time on are corrected with this transform.
    pub valid_from: CameraNanos,
    /// The odom frame the correction takes points out of, named as odom poses name it: the camera time at which
    /// odometry started. Odom poses of another odom frame cannot be corrected with it.
    pub odom_started: CameraNanos,
    /// p_map = R_z(yaw) p_odom + (x, y, z).
    pub correction: Pose4Dof,
    pub sigma_xy_m: f32,
    pub sigma_z_m: f32,
    pub sigma_yaw_rad: f32,
}

fn checked<'a>(array: &'a dyn Array, fields: Fields, message: &'static str) -> Result<&'a StructArray, WireError> {
    let expected = DataType::Struct(fields);
    if array.data_type() != &expected {
        return Err(WireError::WrongType { message, expected, received: array.data_type().clone() });
    }
    if array.len() != 1 {
        return Err(WireError::NotOneRow { message, rows: array.len() });
    }
    Ok(array.as_struct())
}

fn column<'a>(row: &'a StructArray, name: &str) -> &'a ArrayRef {
    row.column_by_name(name).expect("checked wire type")
}

fn u64_field(row: &StructArray, name: &'static str) -> Result<u64, WireError> {
    let values = column(row, name).as_primitive::<UInt64Type>();
    values.is_valid(0).then(|| values.value(0)).ok_or(WireError::Null { field: name })
}

fn i64_field(row: &StructArray, name: &'static str) -> Result<i64, WireError> {
    let values = column(row, name).as_primitive::<Int64Type>();
    values.is_valid(0).then(|| values.value(0)).ok_or(WireError::Null { field: name })
}

fn f32_field(row: &StructArray, name: &'static str) -> Result<f32, WireError> {
    let values = column(row, name).as_primitive::<Float32Type>();
    values.is_valid(0).then(|| values.value(0)).ok_or(WireError::Null { field: name })
}

fn str_field<'a>(row: &'a StructArray, name: &'static str) -> Result<&'a str, WireError> {
    let values = column(row, name).as_string::<i32>();
    values.is_valid(0).then(|| values.value(0)).ok_or(WireError::Null { field: name })
}

fn pose_4dof(row: &StructArray) -> Result<Pose4Dof, WireError> {
    Ok(Pose4Dof { x_m: f32_field(row, "x_m")?, y_m: f32_field(row, "y_m")?, z_m: f32_field(row, "z_m")?, yaw_rad: f32_field(row, "yaw_rad")? })
}

fn floats(value: f32) -> ArrayRef {
    Arc::new(Float32Array::from(vec![value]))
}

fn pose_columns(pose: &Pose4Dof) -> [ArrayRef; 4] {
    [floats(pose.x_m), floats(pose.y_m), floats(pose.z_m), floats(pose.yaw_rad)]
}

pub fn decode_keyframe_pose(array: &dyn Array) -> Result<KeyframePose, WireError> {
    let row = checked(array, keyframe_pose_fields(), "keyframe pose")?;
    let tracking: Tracking = str_field(row, "tracking")?.parse()?;
    let reference = column(row, "reference_frame_id").as_primitive::<UInt64Type>();
    let reference = reference.is_valid(0).then(|| FrameId(reference.value(0)));
    let selection = KeyframeSelection::from_wire(str_field(row, "selection")?, reference)?;
    if !selection.fits(tracking) {
        return Err(WireError::SelectionDoesNotFitTracking { tracking: tracking.wire_name(), selection: selection.wire_name() });
    }
    Ok(KeyframePose {
        frame_id: FrameId(u64_field(row, "frame_id")?),
        captured: UnixNanos(i64_field(row, "captured_unix_ns")?),
        map_version: MapVersion(u64_field(row, "map_version")?),
        pose: pose_4dof(row)?,
        tracking,
        selection,
        residual_m: f32_field(row, "residual_m")?,
        inlier_fraction: f32_field(row, "inlier_fraction")?,
    })
}

pub fn encode_keyframe_pose(message: &KeyframePose) -> StructArray {
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(vec![message.frame_id.0])),
        Arc::new(Int64Array::from(vec![message.captured.0])),
        Arc::new(UInt64Array::from(vec![message.map_version.0])),
    ];
    columns.extend(pose_columns(&message.pose));
    columns.extend([
        Arc::new(StringArray::from(vec![message.tracking.wire_name()])) as ArrayRef,
        Arc::new(StringArray::from(vec![message.selection.wire_name()])),
        Arc::new(UInt64Array::from(vec![match message.selection {
            KeyframeSelection::NotSelected { reference } => Some(reference.0),
            KeyframeSelection::Selected | KeyframeSelection::NotLocalized => None,
        }])),
        floats(message.residual_m),
        floats(message.inlier_fraction),
    ]);
    StructArray::new(keyframe_pose_fields(), columns, None)
}

pub fn decode_map_to_odom(array: &dyn Array) -> Result<MapToOdom, WireError> {
    let row = checked(array, map_to_odom_fields(), "map to odom")?;
    Ok(MapToOdom {
        map_version: MapVersion(u64_field(row, "map_version")?),
        valid_from: CameraNanos(i64_field(row, "valid_from_camera_ns")?),
        odom_started: CameraNanos(i64_field(row, "odom_started_camera_ns")?),
        correction: pose_4dof(row)?,
        sigma_xy_m: f32_field(row, "sigma_xy_m")?,
        sigma_z_m: f32_field(row, "sigma_z_m")?,
        sigma_yaw_rad: f32_field(row, "sigma_yaw_rad")?,
    })
}

pub fn encode_map_to_odom(message: &MapToOdom) -> StructArray {
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(vec![message.map_version.0])),
        Arc::new(Int64Array::from(vec![message.valid_from.0])),
        Arc::new(Int64Array::from(vec![message.odom_started.0])),
    ];
    columns.extend(pose_columns(&message.correction));
    columns.extend([floats(message.sigma_xy_m), floats(message.sigma_z_m), floats(message.sigma_yaw_rad)]);
    StructArray::new(map_to_odom_fields(), columns, None)
}

/// A keyframe already sent that an optimization moved, at its new pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpdatedKeyframePose {
    pub frame_id: FrameId,
    pub pose: Pose4Dof,
}

/// Every keyframe already sent that one optimization of the pose graph moved, with its new pose. A keyframe's own
/// keyframe_pose is never sent again; this is how its later poses arrive.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseGraphUpdate {
    /// The pose graph the poses come from: the optimization's.
    pub map_version: MapVersion,
    pub keyframes: Vec<UpdatedKeyframePose>,
}

#[derive(Serialize, Deserialize)]
struct UpdatedKeyframePoseRow {
    frame_id: u64,
    x_m: f32,
    y_m: f32,
    z_m: f32,
    yaw_rad: f32,
}

#[derive(Serialize, Deserialize)]
struct PoseGraphUpdateRow {
    map_version: u64,
    keyframes: Vec<UpdatedKeyframePoseRow>,
}

pub fn decode_pose_graph_update(array: &dyn Array) -> eyre::Result<PoseGraphUpdate> {
    let row: PoseGraphUpdateRow = decode_row(array, &pose_graph_update_fields(), "keyframe poses moved")?;
    Ok(PoseGraphUpdate {
        map_version: MapVersion(row.map_version),
        keyframes: row
            .keyframes
            .into_iter()
            .map(|k| UpdatedKeyframePose { frame_id: FrameId(k.frame_id), pose: Pose4Dof { x_m: k.x_m, y_m: k.y_m, z_m: k.z_m, yaw_rad: k.yaw_rad } })
            .collect(),
    })
}

pub fn encode_pose_graph_update(message: &PoseGraphUpdate) -> StructArray {
    let row = PoseGraphUpdateRow {
        map_version: message.map_version.0,
        keyframes: message
            .keyframes
            .iter()
            .map(|k| UpdatedKeyframePoseRow { frame_id: k.frame_id.0, x_m: k.pose.x_m, y_m: k.pose.y_m, z_m: k.pose.z_m, yaw_rad: k.pose.yaw_rad })
            .collect(),
    };
    encode_row(&row, pose_graph_update_fields())
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use dora_node_api::arrow::ipc::reader::FileReader;

    use super::*;

    fn fixture(name: &str) -> StructArray {
        let path = format!("{}/../../../lib/dora-rig-messages/dora_rig_messages/fixtures/{name}.arrow", env!("CARGO_MANIFEST_DIR"));
        let mut reader = FileReader::try_new(File::open(&path).unwrap_or_else(|e| panic!("{path}: {e}")), None).unwrap();
        StructArray::from(reader.next().unwrap().unwrap())
    }

    #[test]
    fn the_tracking_keyframe_pose_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("keyframe_pose_tracking");
        let message = decode_keyframe_pose(&golden).unwrap();
        assert_eq!(
            message,
            KeyframePose {
                frame_id: FrameId(42),
                captured: UnixNanos(1_762_560_000_123_456_789),
                map_version: MapVersion(7),
                pose: Pose4Dof { x_m: 1.25, y_m: -0.5, z_m: 0.0625, yaw_rad: 0.75 },
                tracking: Tracking::Tracking,
                selection: KeyframeSelection::Selected,
                residual_m: 0.015625,
                inlier_fraction: 0.875,
            }
        );
        assert_eq!(encode_keyframe_pose(&message), golden);
    }

    #[test]
    fn the_lost_keyframe_pose_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("keyframe_pose_lost");
        let message = decode_keyframe_pose(&golden).unwrap();
        assert_eq!((message.frame_id, message.tracking, message.inlier_fraction), (FrameId(43), Tracking::Lost, 0.0));
        assert_eq!(encode_keyframe_pose(&message), golden);
    }

    #[test]
    fn the_not_selected_keyframe_pose_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("keyframe_pose_not_selected");
        let message = decode_keyframe_pose(&golden).unwrap();
        assert_eq!((message.frame_id, message.selection), (FrameId(44), KeyframeSelection::NotSelected { reference: FrameId(42) }));
        assert_eq!(encode_keyframe_pose(&message), golden);
        let lost = decode_keyframe_pose(&fixture("keyframe_pose_lost")).unwrap();
        assert_eq!(lost.selection, KeyframeSelection::NotLocalized);
    }

    #[test]
    fn a_selection_that_does_not_fit_its_tracking_state_is_rejected() {
        let mut message = decode_keyframe_pose(&fixture("keyframe_pose_tracking")).unwrap();
        message.selection = KeyframeSelection::NotLocalized;
        let error = decode_keyframe_pose(&encode_keyframe_pose(&message)).unwrap_err();
        assert!(matches!(error, WireError::SelectionDoesNotFitTracking { tracking: "tracking", selection: "not_localized" }), "{error}");
        message.tracking = Tracking::Restarted;
        message.selection = KeyframeSelection::NotSelected { reference: FrameId(3) };
        assert!(decode_keyframe_pose(&encode_keyframe_pose(&message)).is_err());
    }

    #[test]
    fn a_reference_frame_id_that_does_not_fit_the_selection_is_rejected() {
        let golden = fixture("keyframe_pose_tracking");
        let with_reference = golden
            .fields()
            .iter()
            .zip(golden.columns())
            .map(|(field, column)| match field.name().as_str() {
                "reference_frame_id" => Arc::new(UInt64Array::from(vec![Some(3)])) as ArrayRef,
                _ => column.clone(),
            })
            .collect();
        let error = decode_keyframe_pose(&StructArray::new(golden.fields().clone(), with_reference, None)).unwrap_err();
        assert_eq!(error, WireError::ReferenceDoesNotFitSelection { selection: "selected".to_owned(), has_reference: true });
    }

    #[test]
    fn the_restarted_keyframe_pose_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("keyframe_pose_restarted");
        let message = decode_keyframe_pose(&golden).unwrap();
        assert_eq!((message.frame_id, message.tracking, message.map_version), (FrameId(47), Tracking::Restarted, MapVersion(8)));
        assert_eq!(encode_keyframe_pose(&message), golden);
    }

    #[test]
    fn the_map_to_odom_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("map_to_odom");
        let message = decode_map_to_odom(&golden).unwrap();
        assert_eq!(
            message,
            MapToOdom {
                map_version: MapVersion(7),
                valid_from: CameraNanos(1_762_560_000_100_000_000),
                odom_started: CameraNanos(1_762_559_990_000_000_000),
                correction: Pose4Dof { x_m: 0.5, y_m: -0.25, z_m: 0.0, yaw_rad: -0.125 },
                sigma_xy_m: 0.03125,
                sigma_z_m: 0.015625,
                sigma_yaw_rad: 0.0078125,
            }
        );
        assert_eq!(encode_map_to_odom(&message), golden);
    }

    #[test]
    fn every_tracking_state_survives_the_trip() {
        let mut message = decode_keyframe_pose(&fixture("keyframe_pose_tracking")).unwrap();
        for tracking in Tracking::ALL {
            message.tracking = tracking;
            message.selection = match tracking {
                Tracking::Lost => KeyframeSelection::NotLocalized,
                _ => KeyframeSelection::Selected,
            };
            assert_eq!(decode_keyframe_pose(&encode_keyframe_pose(&message)).unwrap(), message);
        }
    }

    #[test]
    fn an_unknown_tracking_state_is_rejected() {
        assert_eq!("drifting".parse::<Tracking>(), Err(WireError::UnknownName { field: "tracking", name: "drifting".to_owned() }));
    }

    #[test]
    fn a_map_to_odom_of_another_wire_type_is_rejected() {
        let error = decode_map_to_odom(&fixture("keyframe_pose_tracking")).unwrap_err();
        assert!(matches!(error, WireError::WrongType { message: "map to odom", .. }), "{error}");
    }

    #[test]
    fn the_pose_graph_update_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("pose_graph_update");
        let message = decode_pose_graph_update(&golden).unwrap();
        assert_eq!(
            message,
            PoseGraphUpdate {
                map_version: MapVersion(8),
                keyframes: vec![
                    UpdatedKeyframePose { frame_id: FrameId(40), pose: Pose4Dof { x_m: 1.0, y_m: -0.5, z_m: 0.0625, yaw_rad: 0.5 } },
                    UpdatedKeyframePose { frame_id: FrameId(42), pose: Pose4Dof { x_m: 1.3125, y_m: -0.4375, z_m: 0.0625, yaw_rad: 0.8125 } },
                ],
            }
        );
        assert_eq!(encode_pose_graph_update(&message), golden);
    }
}
