//! The `scan` output: one row per sample of a revolution, in bearing order.

use std::sync::Arc;

use delta2a::Revolution;
use dora_node_api::arrow::array::{ArrayRef, Float32Array, StructArray};
use dora_node_api::arrow::datatypes::{DataType, Field};
use dora_node_api::{MetadataParameters, Parameter};

/// Degrees clockwise from the lidar's zero mark, seen from above.
pub const BEARING_COLUMN: &str = "bearing_deg";
/// Null where nothing came back.
pub const RANGE_COLUMN: &str = "range_m";
pub const SPIN_RATE_PARAMETER: &str = "spin_rev_per_s";

pub fn scan_rows(revolution: &Revolution) -> StructArray {
    let bearings: Float32Array = revolution
        .samples
        .iter()
        .map(|sample| Some(sample.bearing.degrees()))
        .collect();
    let ranges: Float32Array = revolution
        .samples
        .iter()
        .map(|sample| sample.range.map(|range| range.metres()))
        .collect();
    StructArray::from(vec![
        (
            Arc::new(Field::new(BEARING_COLUMN, DataType::Float32, false)),
            Arc::new(bearings) as ArrayRef,
        ),
        (
            Arc::new(Field::new(RANGE_COLUMN, DataType::Float32, true)),
            Arc::new(ranges) as ArrayRef,
        ),
    ])
}

pub fn scan_parameters(revolution: &Revolution) -> MetadataParameters {
    let spin_rate = f64::from(revolution.spin_rate.revolutions_per_second());
    MetadataParameters::from([(SPIN_RATE_PARAMETER.to_owned(), Parameter::Float(spin_rate))])
}

#[cfg(test)]
mod tests {
    use delta2a::{Frame, FrameDecoder, RevolutionAssembler};
    use dora_node_api::arrow::array::Array;

    use super::*;

    const DESK_CAPTURE: &[u8] = include_bytes!("../../delta2a/tests/fixtures/desk_capture.bin");

    fn first_whole_revolution() -> Revolution {
        let mut assembler = RevolutionAssembler::new();
        FrameDecoder::new()
            .push(DESK_CAPTURE)
            .into_iter()
            .filter_map(|frame| match frame {
                Frame::Scan(sector) => assembler.push(sector),
                Frame::SpeedFault(_) => None,
            })
            .nth(1)
            .expect("the capture holds two whole revolutions")
    }

    #[test]
    fn a_revolution_becomes_one_row_per_sample_with_null_for_a_miss() {
        let revolution = first_whole_revolution();

        let rows = scan_rows(&revolution);

        assert_eq!(rows.len(), revolution.samples.len());
        let ranges = rows.column_by_name(RANGE_COLUMN).unwrap();
        let misses = revolution
            .samples
            .iter()
            .filter(|sample| sample.range.is_none())
            .count();
        assert!(misses > 0);
        assert_eq!(ranges.null_count(), misses);
        assert_eq!(rows.column_by_name(BEARING_COLUMN).unwrap().null_count(), 0);
    }

    #[test]
    fn the_spin_rate_travels_as_a_parameter() {
        let revolution = first_whole_revolution();

        let parameters = scan_parameters(&revolution);

        let Some(Parameter::Float(spin_rate)) = parameters.get(SPIN_RATE_PARAMETER) else {
            panic!("no spin rate in {parameters:?}");
        };
        assert!((7.5..=8.5).contains(spin_rate), "{spin_rate}");
    }
}
