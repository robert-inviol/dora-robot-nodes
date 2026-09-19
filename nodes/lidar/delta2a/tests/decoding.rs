use delta2a::{Frame, FrameDecoder, Revolution, RevolutionAssembler, ScanSector};

/// 0.3 s off a Delta-2A on a desk: 45 whole frames, the first starting at 157.5 degrees.
const DESK_CAPTURE: &[u8] = include_bytes!("fixtures/desk_capture.bin");
const DESK_CAPTURE_FRAMES: usize = 45;
const DESK_SAMPLES_PER_SECTOR: usize = 38;
const SECTORS_PER_REVOLUTION: usize = 16;

/// The speed fault example in the manufacturer's protocol document.
const SPEED_FAULT_FRAME: [u8; 11] = [
    0xAA, 0x00, 0x09, 0x00, 0x61, 0xAE, 0x00, 0x01, 0x69, 0x02, 0x2C,
];

fn sectors(frames: Vec<Frame>) -> Vec<ScanSector> {
    frames
        .into_iter()
        .map(|frame| match frame {
            Frame::Scan(sector) => sector,
            other => panic!("expected a scan sector, got {other:?}"),
        })
        .collect()
}

fn revolutions(sectors: Vec<ScanSector>) -> Vec<Revolution> {
    let mut assembler = RevolutionAssembler::new();
    sectors
        .into_iter()
        .filter_map(|sector| assembler.push(sector))
        .collect()
}

#[test]
fn every_frame_of_a_clean_capture_is_decoded() {
    let mut decoder = FrameDecoder::new();

    let frames = decoder.push(DESK_CAPTURE);

    assert_eq!(frames.len(), DESK_CAPTURE_FRAMES);
    assert_eq!(decoder.discarded_bytes(), 0);
}

#[test]
fn sectors_follow_each_other_in_steps_of_22_5_degrees() {
    let starts: Vec<f32> = sectors(FrameDecoder::new().push(DESK_CAPTURE))
        .iter()
        .map(|sector| sector.start.degrees())
        .collect();

    assert_eq!(starts[..4], [157.5, 180.0, 202.5, 225.0]);
    for pair in starts.windows(2) {
        assert_eq!((pair[1] - pair[0]).rem_euclid(360.0), 22.5);
    }
}

#[test]
fn the_samples_of_a_sector_are_spread_evenly_across_it() {
    let sector = sectors(FrameDecoder::new().push(DESK_CAPTURE)).remove(0);

    assert_eq!(sector.samples.len(), DESK_SAMPLES_PER_SECTOR);
    assert_eq!(sector.samples[0].bearing, sector.start);
    let step = 22.5 / DESK_SAMPLES_PER_SECTOR as f32;
    let last = sector.samples.last().unwrap().bearing.degrees();
    assert!(
        (last - (sector.start.degrees() + 22.5 - step)).abs() < 1e-3,
        "last sample at {last}"
    );
}

#[test]
fn ranges_are_in_metres_and_misses_have_none() {
    let samples: Vec<_> = sectors(FrameDecoder::new().push(DESK_CAPTURE))
        .into_iter()
        .flat_map(|sector| sector.samples)
        .collect();

    let ranges: Vec<f32> = samples
        .iter()
        .filter_map(|sample| sample.range)
        .map(|range| range.metres())
        .collect();
    assert!(
        ranges.len() < samples.len(),
        "the desk capture has directions with no return"
    );
    let nearest = ranges.iter().copied().fold(f32::INFINITY, f32::min);
    let farthest = ranges.iter().copied().fold(0.0, f32::max);
    // The desk capture spans a wall half a metre away and a return from beyond the rated 8 m.
    assert!((0.45..0.50).contains(&nearest), "nearest {nearest}");
    assert!((11.7..11.8).contains(&farthest), "farthest {farthest}");
}

#[test]
fn the_spin_rate_is_reported_in_revolutions_per_second() {
    let sector = sectors(FrameDecoder::new().push(DESK_CAPTURE)).remove(0);

    assert!(
        (7.5..=8.5).contains(&sector.spin_rate.revolutions_per_second()),
        "{:?}",
        sector.spin_rate
    );
}

#[test]
fn a_stream_delivered_one_byte_at_a_time_decodes_the_same() {
    let mut decoder = FrameDecoder::new();

    let frames: Vec<Frame> = DESK_CAPTURE
        .iter()
        .flat_map(|&byte| decoder.push(&[byte]))
        .collect();

    assert_eq!(frames, FrameDecoder::new().push(DESK_CAPTURE));
}

#[test]
fn noise_before_a_frame_is_skipped_and_counted() {
    let mut stream = vec![0x00, 0xAA, 0x13, 0x37];
    stream.extend_from_slice(DESK_CAPTURE);
    let mut decoder = FrameDecoder::new();

    let frames = decoder.push(&stream);

    assert_eq!(frames.len(), DESK_CAPTURE_FRAMES);
    assert_eq!(decoder.discarded_bytes(), 4);
}

#[test]
fn a_frame_that_fails_its_checksum_is_dropped_and_the_next_one_still_decodes() {
    let mut stream = DESK_CAPTURE.to_vec();
    stream[20] ^= 0xFF;
    let mut decoder = FrameDecoder::new();

    let frames = decoder.push(&stream);

    assert_eq!(frames.len(), DESK_CAPTURE_FRAMES - 1);
    assert!(decoder.discarded_bytes() > 0);
}

#[test]
fn a_speed_fault_frame_carries_the_spin_rate() {
    let frames = FrameDecoder::new().push(&SPEED_FAULT_FRAME);

    let [Frame::SpeedFault(spin_rate)] = frames[..] else {
        panic!("expected one speed fault, got {frames:?}");
    };
    assert_eq!(spin_rate.revolutions_per_second(), 5.25);
}

#[test]
fn a_revolution_is_handed_over_when_the_next_one_starts() {
    let revolutions = revolutions(sectors(FrameDecoder::new().push(DESK_CAPTURE)));

    let sample_counts: Vec<usize> = revolutions
        .iter()
        .map(|revolution| revolution.samples.len())
        .collect();
    let whole = SECTORS_PER_REVOLUTION * DESK_SAMPLES_PER_SECTOR;
    // The capture starts part way through a turn and ends part way through another.
    assert_eq!(sample_counts, [9 * DESK_SAMPLES_PER_SECTOR, whole, whole]);
}

#[test]
fn the_samples_of_a_whole_revolution_run_once_around_in_bearing_order() {
    let revolution = revolutions(sectors(FrameDecoder::new().push(DESK_CAPTURE))).remove(1);

    let bearings: Vec<f32> = revolution
        .samples
        .iter()
        .map(|sample| sample.bearing.degrees())
        .collect();
    assert_eq!(bearings[0], 0.0);
    assert!(
        bearings.windows(2).all(|pair| pair[0] < pair[1]),
        "bearings are not increasing"
    );
    assert!(*bearings.last().unwrap() < 360.0);
}
