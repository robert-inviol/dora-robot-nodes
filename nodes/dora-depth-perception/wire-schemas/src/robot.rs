//! The messages of drive.py, operator.py, status.py and scan.py: driving the robot and the lidar's scan.

use arrow_schema::{DataType, Fields};

use crate::{field, list_of, items};

pub fn drive_demand_fields() -> Fields {
    Fields::from(vec![field("forward", DataType::Float32), field("turn", DataType::Float32)])
}

pub fn operator_command_fields() -> Fields {
    Fields::from(vec![field("mode", DataType::Utf8), field("throttle", DataType::Float32), field("steer", DataType::Float32)])
}

pub fn pilot_status_fields() -> Fields {
    Fields::from(vec![field("mode", DataType::Utf8), field("locked_track_id", DataType::Int32)])
}

pub fn driver_status_fields() -> Fields {
    Fields::from(vec![field("armed", DataType::Boolean), field("left_pct", DataType::Int8), field("right_pct", DataType::Int8)])
}

/// The bearings and ranges run in step, one item per reading; only a range can be null.
pub fn scan_fields() -> Fields {
    Fields::from(vec![
        field("spin_rev_per_s", DataType::Float32),
        field("bearing_deg", items(DataType::Float32)),
        field("range_m", list_of(DataType::Float32, true)),
    ])
}
