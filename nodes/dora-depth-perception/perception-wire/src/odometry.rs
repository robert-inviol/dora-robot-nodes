//! The odom pose message of lib/dora-rig-messages/dora_rig_messages/odometry.py, as a single-row Arrow struct.
//! Field names, types and nullability must match that module exactly; the tests read its golden fixtures.

use std::sync::Arc;

use dora_node_api::arrow::array::{Array, ArrayRef, AsArray, BooleanArray, Float32Array, Int64Array, ListArray, StructArray, UInt32Array, UInt64Array};
use dora_node_api::arrow::buffer::{NullBuffer, OffsetBuffer};
use dora_node_api::arrow::datatypes::{DataType, Float32Type, Int64Type, UInt32Type, UInt64Type};
use eyre::{bail, Result};
use glam::{Quat, Vec3};

use super::{checked, floats, item, list_values, one_list, scalar};
use wire_schemas::odometry::{odom_pose_fields, DEGENERATE_DIRECTION, DEGENERATE_SIGMA_M};

/// The direction of travel a depth cloud pinned down least, in the odom frame and either way round, and the one
/// sigma the cloud alone leaves along it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DegenerateDirection {
    pub direction: Vec3,
    pub sigma_m: f32,
}

/// Where odometry says the robot is at one depth frame.
///
/// The odom frame is fixed where the IMU was during static initialization, with z up; it drifts against the real room
/// as the robot travels. The body frame is the IMU's own axes (on the D455: x right, y down, z forward).
///
/// `camera_ns` strictly increases from one pose to the next. A consumer that wants the pose at another camera time
/// interpolates between the two rows nearest to it on `camera_ns`.
///
/// `odom_started_camera_ns` names the odom frame: every pose of one odom frame carries the same value, and
/// odometry that starts again carries a later one. Poses with different values are in different odom frames and
/// cannot be combined.
///
/// The sigmas say how well the pose is known against the odometry's own map, and leave out that map's drift. The
/// position sigmas also cover the depth camera's warm-up: they add how far the depth clouds would move the position
/// under a change of the depth's scale, at one sigma of half the change a D455's depth goes through over the first
/// minutes of a stream. The tilt and heading sigmas are the filter's alone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OdomPose {
    /// The time the pose is for, on the camera's own clock in nanoseconds (not a unix time): its depth frame's.
    pub camera_ns: i64,
    /// The camera time at which odometry started and fixed the odom frame this pose is in.
    pub odom_started_camera_ns: i64,
    pub depth_frame_id: u64,
    /// The IMU body's position in the odom frame.
    pub position_m: Vec3,
    /// Unit quaternion that turns a body-frame vector into the odom frame.
    pub orientation: Quat,
    /// In the odom frame.
    pub velocity_m_s: Vec3,
    /// One sigma along the odom frame's x, y and z.
    pub position_sigma_m: Vec3,
    /// One sigma of the lean against gravity, about the level axis where it is known worst.
    pub tilt_sigma_rad: f32,
    /// One sigma of the turn about up.
    pub heading_sigma_rad: f32,
    /// Whether the depth frame's cloud corrected the pose. When false the pose is the IMU's prediction.
    pub corrected: bool,
    /// Cloud points that found a surface in the map; 0 when the cloud did not correct the pose.
    pub matched_points: u32,
    /// Absent when the cloud did not correct the pose.
    pub degenerate: Option<DegenerateDirection>,
}

fn column<'a>(row: &'a StructArray, name: &str) -> &'a ArrayRef {
    row.column_by_name(name).expect("checked wire type")
}

fn values<const N: usize>(row: &StructArray, name: &str) -> Result<[f32; N]> {
    let values = list_values::<Float32Type>(row, name);
    match <[f32; N]>::try_from(&values[..]) {
        Ok(values) => Ok(values),
        Err(_) => bail!("odom pose {name} has {} values, not {N}", values.len()),
    }
}

pub fn decode_odom_pose(array: &dyn Array) -> Result<OdomPose> {
    let row = checked(array, &odom_pose_fields(), "odom pose")?;
    if row.is_null(0) {
        bail!("odom pose row is null");
    }
    let present = |name: &str| !column(row, name).is_null(0);
    if let Some(missing) = odom_pose_fields().iter().map(|field| field.name()).find(|name| ![DEGENERATE_DIRECTION, DEGENERATE_SIGMA_M].contains(&name.as_str()) && !present(name)) {
        bail!("odom pose is missing {missing}");
    }
    let degenerate = match (present(DEGENERATE_DIRECTION), present(DEGENERATE_SIGMA_M)) {
        (true, true) => Some(DegenerateDirection { direction: Vec3::from_array(values(row, DEGENERATE_DIRECTION)?), sigma_m: scalar::<Float32Type>(row, DEGENERATE_SIGMA_M) }),
        (false, false) => None,
        (true, false) | (false, true) => bail!("odom pose has one of {DEGENERATE_DIRECTION} and {DEGENERATE_SIGMA_M} without the other"),
    };
    Ok(OdomPose {
        camera_ns: scalar::<Int64Type>(row, "camera_ns"),
        odom_started_camera_ns: scalar::<Int64Type>(row, "odom_started_camera_ns"),
        depth_frame_id: scalar::<UInt64Type>(row, "depth_frame_id"),
        position_m: Vec3::from_array(values(row, "position_m")?),
        orientation: Quat::from_array(values(row, "orientation_xyzw")?),
        velocity_m_s: Vec3::from_array(values(row, "velocity_m_s")?),
        position_sigma_m: Vec3::from_array(values(row, "position_sigma_m")?),
        tilt_sigma_rad: scalar::<Float32Type>(row, "tilt_sigma_rad"),
        heading_sigma_rad: scalar::<Float32Type>(row, "heading_sigma_rad"),
        corrected: column(row, "corrected").as_boolean().value(0),
        matched_points: scalar::<UInt32Type>(row, "matched_points"),
        degenerate,
    })
}

pub fn encode_odom_pose(pose: &OdomPose) -> StructArray {
    let degenerate_direction = match pose.degenerate {
        Some(degenerate) => one_list(floats(degenerate.direction.to_array())),
        None => Arc::new(ListArray::new(item(DataType::Float32), OffsetBuffer::from_lengths([0]), floats([]), Some(NullBuffer::new_null(1)))),
    };
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![pose.camera_ns])),
        Arc::new(Int64Array::from(vec![pose.odom_started_camera_ns])),
        Arc::new(UInt64Array::from(vec![pose.depth_frame_id])),
        one_list(floats(pose.position_m.to_array())),
        one_list(floats(pose.orientation.to_array())),
        one_list(floats(pose.velocity_m_s.to_array())),
        one_list(floats(pose.position_sigma_m.to_array())),
        floats([pose.tilt_sigma_rad]),
        floats([pose.heading_sigma_rad]),
        Arc::new(BooleanArray::from(vec![pose.corrected])),
        Arc::new(UInt32Array::from(vec![pose.matched_points])),
        degenerate_direction,
        Arc::new(Float32Array::from(vec![pose.degenerate.map(|degenerate| degenerate.sigma_m)])),
    ];
    StructArray::new(odom_pose_fields(), columns, None)
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

    /// A row with one column swapped for another of the same type.
    fn with_column(row: &StructArray, name: &str, replacement: ArrayRef) -> StructArray {
        let columns = row.fields().iter().zip(row.columns()).map(|(field, column)| if field.name() == name { replacement.clone() } else { column.clone() }).collect();
        StructArray::new(row.fields().clone(), columns, None)
    }

    #[test]
    fn the_odom_pose_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("odom_pose");
        let pose = decode_odom_pose(&golden).unwrap();
        let expected = OdomPose {
            camera_ns: 1_762_560_000_100_000_000,
            odom_started_camera_ns: 1_762_559_990_000_000_000,
            depth_frame_id: 42,
            position_m: Vec3::new(1.25, -0.5, 0.25),
            orientation: Quat::from_xyzw(0.5, -0.5, 0.5, 0.5),
            velocity_m_s: Vec3::new(0.375, 0.0, -0.0625),
            position_sigma_m: Vec3::new(0.015625, 0.03125, 0.0078125),
            tilt_sigma_rad: 0.00390625,
            heading_sigma_rad: 0.0078125,
            corrected: true,
            matched_points: 1234,
            degenerate: Some(DegenerateDirection { direction: Vec3::Y, sigma_m: 0.125 }),
        };
        assert_eq!(pose, expected);
        assert_eq!(encode_odom_pose(&pose), golden);
    }

    #[test]
    fn a_predicted_odom_pose_has_no_degenerate_direction_and_encodes_back_to_its_fixture() {
        let golden = fixture("odom_pose_predicted");
        let pose = decode_odom_pose(&golden).unwrap();
        assert_eq!((pose.camera_ns, pose.odom_started_camera_ns, pose.depth_frame_id), (1_762_560_000_133_000_000, 1_762_559_990_000_000_000, 43));
        assert_eq!((pose.corrected, pose.matched_points, pose.degenerate), (false, 0, None));
        assert_eq!(pose.orientation, Quat::from_xyzw(0.0, 0.0, 1.0, 0.0));
        assert_eq!(encode_odom_pose(&pose), golden);
    }

    #[test]
    fn an_odom_pose_vector_of_the_wrong_length_is_rejected() {
        let short = with_column(&fixture("odom_pose"), "orientation_xyzw", one_list(floats([0.0, 0.0, 1.0])));
        let error = decode_odom_pose(&short).err().unwrap();
        assert!(error.to_string().contains("orientation_xyzw has 3 values, not 4"), "{error}");
    }

    #[test]
    fn an_odom_pose_with_half_of_its_degenerate_direction_is_rejected() {
        let without_sigma = with_column(&fixture("odom_pose"), DEGENERATE_SIGMA_M, Arc::new(Float32Array::from(vec![None])));
        let without_direction = with_column(&fixture("odom_pose_predicted"), DEGENERATE_SIGMA_M, floats([0.125]));
        for half in [without_sigma, without_direction] {
            let error = decode_odom_pose(&half).err().unwrap();
            assert!(error.to_string().contains("one of degenerate_direction and degenerate_sigma_m without the other"), "{error}");
        }
    }

    #[test]
    fn an_odom_pose_missing_a_required_field_is_rejected() {
        let unstamped = with_column(&fixture("odom_pose"), "camera_ns", Arc::new(Int64Array::from(vec![None])));
        let error = decode_odom_pose(&unstamped).err().unwrap();
        assert!(error.to_string().contains("missing camera_ns"), "{error}");
    }
}
