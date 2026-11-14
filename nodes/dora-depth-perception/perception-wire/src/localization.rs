//! The localization messages of lib/dora-rig-messages/dora_rig_messages/localization.py, as single-row Arrow
//! structs. Field names, types and nullability must match that module exactly; the tests read its golden fixtures.
//!
//! The map frame is z up, gravity aligned, metres. A keyframe pose is applied to that keyframe's gravity-aligned
//! floor frame: a floor-frame point p is at R_z(yaw) p + (x, y, z) in the map, yaw anticlockwise seen from above.

use std::str::FromStr;
use std::sync::Arc;

use dora_node_api::arrow::array::{Array, ArrayRef, AsArray, Float32Array, Int64Array, StringArray, StructArray, UInt64Array};
use dora_node_api::arrow::datatypes::{DataType, Field, Fields, Float32Type, Int64Type, UInt64Type};

use crate::navigation::{MapVersion, UnixNanos, WireError};

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
    /// Placed again by matching stored keyframes after tracking was lost.
    Relocalised,
    /// Not registered: dead reckoning from the gyro and the last known position.
    Lost,
}

impl Tracking {
    pub const ALL: [Tracking; 3] = [Tracking::Tracking, Tracking::Relocalised, Tracking::Lost];

    pub fn wire_name(self) -> &'static str {
        match self {
            Tracking::Tracking => "tracking",
            Tracking::Relocalised => "relocalised",
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

/// A turn about the vertical then a translation, in metres and radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose4 {
    pub x_m: f32,
    pub y_m: f32,
    pub z_m: f32,
    pub yaw_rad: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyframePose {
    pub frame_id: FrameId,
    pub captured: UnixNanos,
    /// The pose graph the pose comes from; it rises whenever optimisation moves past keyframes.
    pub map_version: MapVersion,
    /// The keyframe's floor frame in the map.
    pub pose: Pose4,
    pub tracking: Tracking,
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
    /// p_map = R_z(yaw) p_odom + (x, y, z).
    pub correction: Pose4,
    pub sigma_xy_m: f32,
    pub sigma_z_m: f32,
    pub sigma_yaw_rad: f32,
}

fn field(name: &str, data_type: DataType) -> Field {
    Field::new(name, data_type, true)
}

const POSE_FIELDS: [&str; 4] = ["x_m", "y_m", "z_m", "yaw_rad"];

pub fn keyframe_pose_fields() -> Fields {
    let mut fields = vec![
        field("frame_id", DataType::UInt64),
        field("captured_unix_ns", DataType::Int64),
        field("map_version", DataType::UInt64),
    ];
    fields.extend(POSE_FIELDS.map(|name| field(name, DataType::Float32)));
    fields.extend([
        field("tracking", DataType::Utf8),
        field("residual_m", DataType::Float32),
        field("inlier_fraction", DataType::Float32),
    ]);
    Fields::from(fields)
}

pub fn map_to_odom_fields() -> Fields {
    let mut fields = vec![field("map_version", DataType::UInt64), field("valid_from_camera_ns", DataType::Int64)];
    fields.extend(POSE_FIELDS.map(|name| field(name, DataType::Float32)));
    fields.extend(["sigma_xy_m", "sigma_z_m", "sigma_yaw_rad"].map(|name| field(name, DataType::Float32)));
    Fields::from(fields)
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

fn pose4(row: &StructArray) -> Result<Pose4, WireError> {
    Ok(Pose4 { x_m: f32_field(row, "x_m")?, y_m: f32_field(row, "y_m")?, z_m: f32_field(row, "z_m")?, yaw_rad: f32_field(row, "yaw_rad")? })
}

fn floats(value: f32) -> ArrayRef {
    Arc::new(Float32Array::from(vec![value]))
}

fn pose_columns(pose: &Pose4) -> [ArrayRef; 4] {
    [floats(pose.x_m), floats(pose.y_m), floats(pose.z_m), floats(pose.yaw_rad)]
}

pub fn decode_keyframe_pose(array: &dyn Array) -> Result<KeyframePose, WireError> {
    let row = checked(array, keyframe_pose_fields(), "keyframe pose")?;
    Ok(KeyframePose {
        frame_id: FrameId(u64_field(row, "frame_id")?),
        captured: UnixNanos(i64_field(row, "captured_unix_ns")?),
        map_version: MapVersion(u64_field(row, "map_version")?),
        pose: pose4(row)?,
        tracking: str_field(row, "tracking")?.parse()?,
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
        correction: pose4(row)?,
        sigma_xy_m: f32_field(row, "sigma_xy_m")?,
        sigma_z_m: f32_field(row, "sigma_z_m")?,
        sigma_yaw_rad: f32_field(row, "sigma_yaw_rad")?,
    })
}

pub fn encode_map_to_odom(message: &MapToOdom) -> StructArray {
    let mut columns: Vec<ArrayRef> =
        vec![Arc::new(UInt64Array::from(vec![message.map_version.0])), Arc::new(Int64Array::from(vec![message.valid_from.0]))];
    columns.extend(pose_columns(&message.correction));
    columns.extend([floats(message.sigma_xy_m), floats(message.sigma_z_m), floats(message.sigma_yaw_rad)]);
    StructArray::new(map_to_odom_fields(), columns, None)
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
                pose: Pose4 { x_m: 1.25, y_m: -0.5, z_m: 0.0625, yaw_rad: 0.75 },
                tracking: Tracking::Tracking,
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
    fn the_map_to_odom_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("map_to_odom");
        let message = decode_map_to_odom(&golden).unwrap();
        assert_eq!(
            message,
            MapToOdom {
                map_version: MapVersion(7),
                valid_from: CameraNanos(1_762_560_000_100_000_000),
                correction: Pose4 { x_m: 0.5, y_m: -0.25, z_m: 0.0, yaw_rad: -0.125 },
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
}
