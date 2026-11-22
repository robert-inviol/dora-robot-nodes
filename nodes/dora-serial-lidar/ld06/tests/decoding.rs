use ld06::{Packet, PacketDecoder, Revolution, RevolutionAssembler, READINGS_PER_PACKET};

/// 1.3 s off an LD06 on a desk, read from the moment the port opened: a packet, the 10-byte tail of a packet whose
/// start the port missed, 509 whole packets, then the start of the next.
const DESK_CAPTURE: &[u8] = include_bytes!("fixtures/desk_capture.bin");
const DESK_CAPTURE_PACKETS: usize = 510;
const DESK_CAPTURE_STRAY_BYTES: u64 = 10;
const PACKET_BYTES: usize = 47;
/// Times the bearing passes the zero mark in the capture.
const DESK_CAPTURE_ZERO_CROSSINGS: usize = 13;

fn desk_packets() -> Vec<Packet> {
    PacketDecoder::new().push(DESK_CAPTURE)
}

/// Each revolution is tagged with the indices of the packets its first and last reading came from.
fn desk_revolutions() -> Vec<Revolution<usize>> {
    let mut assembler = RevolutionAssembler::new();
    desk_packets()
        .into_iter()
        .enumerate()
        .filter_map(|(index, packet)| assembler.push(packet, index))
        .collect()
}

#[test]
fn every_packet_of_a_clean_capture_is_decoded() {
    let mut decoder = PacketDecoder::new();

    let packets = decoder.push(DESK_CAPTURE);

    assert_eq!(packets.len(), DESK_CAPTURE_PACKETS);
    assert_eq!(decoder.discarded_bytes(), DESK_CAPTURE_STRAY_BYTES);
}

#[test]
fn the_first_packet_reads_as_its_bytes_say() {
    let packet = desk_packets().remove(0);

    assert_eq!(packet.spin_rate.revolutions_per_second(), 3604.0 / 360.0);
    assert_eq!(packet.stamp.milliseconds, 2947);
    assert_eq!(packet.readings.len(), READINGS_PER_PACKET);
    assert_eq!(packet.readings[0].bearing.degrees(), 6.5);
    assert!((packet.readings[11].bearing.degrees() - 15.18).abs() < 1e-3);
    assert_eq!(
        packet.readings[0].range.map(|range| range.metres()),
        Some(4.88)
    );
    assert_eq!(packet.readings[0].intensity.0, 225);
}

#[test]
fn a_stream_chopped_into_single_bytes_decodes_the_same() {
    let mut decoder = PacketDecoder::new();

    let packets: Vec<Packet> = DESK_CAPTURE
        .iter()
        .flat_map(|&byte| decoder.push(&[byte]))
        .collect();

    assert_eq!(packets, desk_packets());
}

#[test]
fn a_corrupted_packet_is_dropped_and_its_neighbours_kept() {
    let mut corrupted = DESK_CAPTURE.to_vec();
    let fourth_packet_start = 3 * PACKET_BYTES + DESK_CAPTURE_STRAY_BYTES as usize;
    corrupted[fourth_packet_start + 10] ^= 0xFF;
    let mut decoder = PacketDecoder::new();

    let packets = decoder.push(&corrupted);

    assert_eq!(packets.len(), DESK_CAPTURE_PACKETS - 1);
    assert_eq!(packets[3], desk_packets()[4]);
    assert_eq!(
        decoder.discarded_bytes(),
        DESK_CAPTURE_STRAY_BYTES + PACKET_BYTES as u64
    );
}

#[test]
fn a_turn_ends_each_time_the_bearing_passes_the_zero_mark() {
    assert_eq!(desk_revolutions().len(), DESK_CAPTURE_ZERO_CROSSINGS);
}

#[test]
fn a_whole_turn_runs_clockwise_once_round() {
    let revolutions = desk_revolutions();

    for revolution in &revolutions[1..] {
        let bearings: Vec<f32> = revolution
            .readings
            .iter()
            .map(|reading| reading.bearing.degrees())
            .collect();
        assert!(bearings.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(bearings[0] < 1.0, "a turn starts at {}", bearings[0]);
        assert!(*bearings.last().unwrap() > 359.0);
        assert!(
            (440..=470).contains(&revolution.readings.len()),
            "{} readings",
            revolution.readings.len()
        );
    }
}

#[test]
fn a_turn_carries_the_times_of_the_packets_its_first_and_last_reading_came_from() {
    let revolutions = desk_revolutions();

    for pair in revolutions.windows(2) {
        // A packet that straddles the zero mark ends one turn and starts the next.
        assert!(pair[1].first_reading_at - pair[0].last_reading_at <= 1);
        assert!(pair[1].last_reading_at - pair[1].first_reading_at >= 36);
    }
}
