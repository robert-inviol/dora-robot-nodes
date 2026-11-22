//! The `scan` output: one turn as the single-row message that lib/dora-rig-messages defines as `Scan`, of the wire
//! type `wire_schemas::robot::scan_fields`.

use std::sync::Arc;

use dora_node_api::arrow::array::{
    ArrayRef, ArrowPrimitiveType, Float32Array, Int64Array, ListArray, PrimitiveArray, StructArray,
    UInt8Array,
};
use dora_node_api::arrow::buffer::{NullBuffer, OffsetBuffer};
use dora_node_api::arrow::datatypes::Field;
use wire_schemas::robot::scan_fields;

use crate::turn::{Reading, Readings, Turn};

const LIST_ITEM: &str = "item";

pub fn scan_message(turn: &Turn) -> StructArray {
    let (readings, intensities): (Vec<Reading>, Option<UInt8Array>) = match &turn.readings {
        Readings::RangeOnly(readings) => (readings.clone(), None),
        Readings::WithIntensity(pairs) => (
            pairs.iter().map(|(reading, _)| *reading).collect(),
            Some(
                pairs
                    .iter()
                    .map(|(_, intensity)| Some(*intensity))
                    .collect(),
            ),
        ),
    };
    let bearings: Float32Array = readings
        .iter()
        .map(|reading| Some(reading.bearing_deg))
        .collect();
    let ranges: Float32Array = readings.iter().map(|reading| reading.range_m).collect();
    let intensity_list = match intensities {
        Some(intensities) => one_list(intensities, false),
        None => null_list(UInt8Array::from(Vec::<u8>::new())),
    };
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Float32Array::from(vec![turn.spin_rev_per_s])),
        Arc::new(Int64Array::from(vec![turn.first_reading_at.0])),
        Arc::new(Int64Array::from(vec![turn.last_reading_at.0])),
        Arc::new(one_list(bearings, false)),
        Arc::new(one_list(ranges, true)),
        Arc::new(intensity_list),
    ];
    StructArray::new(scan_fields(), columns, None)
}

/// A list column holding the one list of a single-row message.
fn one_list<T: ArrowPrimitiveType>(items: PrimitiveArray<T>, items_nullable: bool) -> ListArray {
    let item = Arc::new(Field::new(LIST_ITEM, T::DATA_TYPE, items_nullable));
    let offsets = OffsetBuffer::from_lengths([items.len()]);
    ListArray::new(item, offsets, Arc::new(items), None)
}

/// A list column whose one row is null, of never-null items.
fn null_list<T: ArrowPrimitiveType>(no_items: PrimitiveArray<T>) -> ListArray {
    let item = Arc::new(Field::new(LIST_ITEM, T::DATA_TYPE, false));
    let offsets = OffsetBuffer::from_lengths([0]);
    ListArray::new(
        item,
        offsets,
        Arc::new(no_items),
        Some(NullBuffer::new_null(1)),
    )
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use dora_node_api::arrow::array::Array;
    use dora_node_api::arrow::ipc::reader::FileReader;

    use super::*;
    use crate::delta2a_turns::Delta2aTurns;
    use crate::ld06_turns::Ld06Turns;
    use crate::turn::{TurnDecoder, UnixNanos};

    const FIXTURE_DIR: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../lib/dora-rig-messages/dora_rig_messages/fixtures"
    );
    const LD06_DESK_CAPTURE: &[u8] = include_bytes!("../../ld06/tests/fixtures/desk_capture.bin");
    const DELTA2A_DESK_CAPTURE: &[u8] =
        include_bytes!("../../delta2a/tests/fixtures/desk_capture.bin");
    const START: UnixNanos = UnixNanos(1_762_560_000_000_000_000);

    fn readings(pairs: &[(f32, Option<f32>)]) -> Vec<Reading> {
        pairs
            .iter()
            .map(|&(bearing_deg, range_m)| Reading {
                bearing_deg,
                range_m,
            })
            .collect()
    }

    /// The examples lib/dora-rig-messages/dora_rig_messages/fixtures.py writes as `scan` and `scan_without_intensity`.
    fn scan_example() -> Turn {
        let readings = readings(&[
            (0.0, Some(0.5)),
            (90.0, None),
            (180.0, Some(3.25)),
            (292.5, Some(11.75)),
        ]);
        Turn {
            spin_rev_per_s: 10.0,
            first_reading_at: UnixNanos(1_762_560_000_100_000_000),
            last_reading_at: UnixNanos(1_762_560_000_197_500_000),
            readings: Readings::WithIntensity(readings.into_iter().zip([212, 0, 87, 9]).collect()),
        }
    }

    fn scan_without_intensity_example() -> Turn {
        Turn {
            spin_rev_per_s: 8.0,
            first_reading_at: UnixNanos(1_762_560_000_100_000_000),
            last_reading_at: UnixNanos(1_762_560_000_225_000_000),
            readings: Readings::RangeOnly(readings(&[(0.0, Some(0.5)), (90.0, None)])),
        }
    }

    fn golden(name: &str) -> StructArray {
        let path = format!("{FIXTURE_DIR}/{name}.arrow");
        let file = File::open(&path).unwrap_or_else(|error| panic!("cannot open {path}: {error}"));
        let mut batches = FileReader::try_new(file, None).expect("the fixture is an Arrow file");
        let batch = batches
            .next()
            .expect("the fixture holds one record batch")
            .expect("the batch can be read");
        StructArray::from(batch)
    }

    fn list_column<'a>(message: &'a StructArray, name: &str) -> &'a ListArray {
        message
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
    }

    /// Feeds a capture in 20 ms reads, as the node's tick does, with the bytes arriving at the LD06's rate of a
    /// 47-byte packet every 2.64 ms.
    fn turns_of(mut decoder: impl TurnDecoder, capture: &[u8]) -> Vec<Turn> {
        const BYTES_PER_TICK: usize = 356;
        const TICK_NS: i64 = 20_000_000;
        capture
            .chunks(BYTES_PER_TICK)
            .enumerate()
            .flat_map(|(tick, bytes)| {
                decoder.push(bytes, UnixNanos(START.0 + tick as i64 * TICK_NS))
            })
            .collect()
    }

    #[test]
    fn a_turn_with_intensities_is_written_exactly_as_the_golden_fixture() {
        let written = scan_message(&scan_example());

        assert_eq!(written.data_type(), golden("scan").data_type());
        assert_eq!(written, golden("scan"));
    }

    #[test]
    fn a_turn_without_intensities_is_written_exactly_as_the_golden_fixture() {
        let written = scan_message(&scan_without_intensity_example());

        assert_eq!(written, golden("scan_without_intensity"));
    }

    #[test]
    fn a_real_ld06_turn_is_one_row_with_an_intensity_for_every_reading() {
        let turn = turns_of(Ld06Turns::new(), LD06_DESK_CAPTURE).remove(1);

        let message = scan_message(&turn);

        assert_eq!(message.len(), 1);
        let bearings = list_column(&message, "bearing_deg").value(0);
        let intensities = list_column(&message, "intensity");
        assert!(intensities.is_valid(0));
        assert_eq!(intensities.value(0).len(), bearings.len());
    }

    #[test]
    fn a_real_ld06_turn_spans_one_turn_of_the_head_in_time() {
        let turns = turns_of(Ld06Turns::new(), LD06_DESK_CAPTURE);

        for turn in &turns[1..] {
            let span_ms = (turn.last_reading_at.0 - turn.first_reading_at.0) / 1_000_000;
            assert!((90..=105).contains(&span_ms), "a turn spans {span_ms} ms");
        }
        for pair in turns.windows(2) {
            assert!(pair[1].first_reading_at >= pair[0].last_reading_at);
        }
    }

    #[test]
    fn a_real_delta2a_turn_is_one_row_with_a_null_for_every_miss_and_no_intensities() {
        let turn = turns_of(Delta2aTurns::new(), DELTA2A_DESK_CAPTURE).remove(1);
        let Readings::RangeOnly(readings) = &turn.readings else {
            panic!("the Delta-2A reports no intensities");
        };
        let misses = readings
            .iter()
            .filter(|reading| reading.range_m.is_none())
            .count();

        let message = scan_message(&turn);

        let ranges = list_column(&message, "range_m").value(0);
        assert_eq!(ranges.len(), readings.len());
        assert!(misses > 0);
        assert_eq!(ranges.null_count(), misses);
        assert!(list_column(&message, "intensity").is_null(0));
        assert!(turn.last_reading_at > turn.first_reading_at);
    }
}
