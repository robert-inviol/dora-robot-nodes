//! The `scan` output: one revolution as the single-row message that lib/messages defines as `Scan`.

use std::sync::Arc;

use delta2a::Revolution;
use dora_node_api::arrow::array::{Array, ArrayRef, Float32Array, ListArray, StructArray};
use dora_node_api::arrow::buffer::OffsetBuffer;
use dora_node_api::arrow::datatypes::{DataType, Field};

const SPIN_RATE_FIELD: &str = "spin_rev_per_s";
/// Degrees clockwise from the lidar's zero mark, seen from above.
const BEARINGS_FIELD: &str = "bearing_deg";
/// In step with the bearings, null where nothing came back.
const RANGES_FIELD: &str = "range_m";
const LIST_ITEM: &str = "item";

/// A reader compares the whole type, so every nullable flag here is part of the contract.
pub fn scan_message(revolution: &Revolution) -> StructArray {
    let spin_rate = Float32Array::from(vec![revolution.spin_rate.revolutions_per_second()]);
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
    let bearings = one_list(bearings, false);
    let ranges = one_list(ranges, true);
    StructArray::from(vec![
        (
            Arc::new(Field::new(SPIN_RATE_FIELD, DataType::Float32, true)),
            Arc::new(spin_rate) as ArrayRef,
        ),
        (
            Arc::new(Field::new(
                BEARINGS_FIELD,
                bearings.data_type().clone(),
                true,
            )),
            Arc::new(bearings) as ArrayRef,
        ),
        (
            Arc::new(Field::new(RANGES_FIELD, ranges.data_type().clone(), true)),
            Arc::new(ranges) as ArrayRef,
        ),
    ])
}

/// A list column holding the one list of a single-row message.
fn one_list(items: Float32Array, items_nullable: bool) -> ListArray {
    let item = Arc::new(Field::new(LIST_ITEM, DataType::Float32, items_nullable));
    let offsets = OffsetBuffer::from_lengths([items.len()]);
    ListArray::new(item, offsets, Arc::new(items), None)
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use delta2a::{
        Bearing, Frame, FrameDecoder, Range, RangeSample, RevolutionAssembler, SpinRate,
    };
    use dora_node_api::arrow::ipc::reader::FileReader;

    use super::*;

    const SCAN_FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../lib/messages/messages/fixtures/scan.arrow"
    );
    const DESK_CAPTURE: &[u8] = include_bytes!("../../delta2a/tests/fixtures/desk_capture.bin");

    /// The same example that lib/messages/messages/fixtures.py writes as `scan`.
    fn fixture_example() -> Revolution {
        let readings = [
            (0.0, Some(0.5)),
            (90.0, None),
            (180.0, Some(3.25)),
            (292.5, Some(11.75)),
        ];
        Revolution {
            spin_rate: SpinRate::from_revolutions_per_second(8.0),
            samples: readings
                .into_iter()
                .map(|(degrees, metres)| RangeSample {
                    bearing: Bearing::from_degrees(degrees),
                    range: metres.map(Range::from_metres),
                })
                .collect(),
        }
    }

    fn golden_scan() -> StructArray {
        let file = File::open(SCAN_FIXTURE).expect("the scan fixture is in lib/messages");
        let mut batches = FileReader::try_new(file, None).expect("the fixture is an Arrow file");
        let batch = batches
            .next()
            .expect("the fixture holds one record batch")
            .expect("the batch can be read");
        StructArray::from(batch)
    }

    #[test]
    fn a_revolution_is_written_exactly_as_the_golden_fixture() {
        let written = scan_message(&fixture_example());
        let golden = golden_scan();

        assert_eq!(written.data_type(), golden.data_type());
        assert_eq!(written, golden);
    }

    #[test]
    fn a_real_revolution_is_one_row_with_a_null_for_every_miss() {
        let mut assembler = RevolutionAssembler::new();
        let revolution = FrameDecoder::new()
            .push(DESK_CAPTURE)
            .into_iter()
            .filter_map(|frame| match frame {
                Frame::Scan(sector) => assembler.push(sector),
                Frame::SpeedFault(_) => None,
            })
            .nth(1)
            .expect("the capture holds two whole revolutions");
        let misses = revolution
            .samples
            .iter()
            .filter(|sample| sample.range.is_none())
            .count();

        let message = scan_message(&revolution);

        assert_eq!(message.len(), 1);
        let ranges = message.column_by_name(RANGES_FIELD).unwrap();
        let ranges = ranges
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .value(0);
        assert_eq!(ranges.len(), revolution.samples.len());
        assert!(misses > 0);
        assert_eq!(ranges.null_count(), misses);
    }
}
