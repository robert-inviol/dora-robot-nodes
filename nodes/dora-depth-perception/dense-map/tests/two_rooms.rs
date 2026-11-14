//! The dense map as the node runs it, on a rendered walk through two rooms joined by a door.

use std::time::{Duration, Instant};

use dense_map::synthetic::{keyframe_at, look_around, two_rooms};
use dense_map::{Arrival, BlockTsdf, DenseMapSettings, DenseMapper, FrameId, KeyframePose, MapOutput};
use glam::Vec2;
use perception_wire::navigation::{MapVersion, UnixNanos};
use perception_wire::space_map::{Clearance, FreeSpace, RoomId, Topology};

const SPOTS: [(f32, f32); 10] =
    [(1.0, 1.0), (3.0, 1.0), (2.0, 2.0), (1.0, 3.0), (3.0, 3.0), (5.0, 1.0), (7.0, 1.0), (6.0, 2.0), (5.0, 3.0), (7.0, 3.0)];

fn walk() -> Vec<(Vec2, f32)> {
    look_around(&SPOTS.map(|(x, y)| Vec2::new(x, y)), 6)
}

fn outputs(outputs: Vec<MapOutput>) -> (FreeSpace, Option<Topology>) {
    let mut free_space = None;
    let mut topology = None;
    for output in outputs {
        match output {
            MapOutput::FreeSpace(f) => free_space = Some(f),
            MapOutput::Topology(t) => topology = Some(t),
        }
    }
    (free_space.expect("free space"), topology)
}

fn room_at(topology: &Topology, point: Vec2) -> RoomId {
    topology
        .places
        .iter()
        .min_by(|a, b| a.position.distance(point).total_cmp(&b.position.distance(point)))
        .expect("places")
        .room_id
}

#[test]
fn a_walk_through_two_rooms_maps_both_and_the_door_between_and_a_rebuild_keeps_their_ids() {
    let house = two_rooms();
    let mut mapper = DenseMapper::new(DenseMapSettings::default(), BlockTsdf::new(Default::default()));
    let start = Instant::now();
    let mut poses: Vec<KeyframePose> = Vec::new();
    for (i, (position, heading)) in walk().into_iter().enumerate() {
        let (keyframe, pose) = keyframe_at(&house, i as u64, position, heading, MapVersion(1));
        if i % 2 == 0 {
            assert_eq!(mapper.on_keyframe(keyframe), Arrival::Waiting);
            assert_eq!(mapper.on_keyframe_pose(pose), Arrival::Integrated(FrameId(i as u64)));
        } else {
            assert_eq!(mapper.on_keyframe_pose(pose), Arrival::Waiting, "a pose ahead of its keyframe");
            assert_eq!(mapper.on_keyframe(keyframe), Arrival::Integrated(FrameId(i as u64)));
        }
        poses.push(pose);
    }

    let (free_space, topology) = outputs(mapper.on_tick(start, UnixNanos(1)).unwrap().0);
    assert_eq!(free_space.map_version, MapVersion(1));
    let cell = |x: f32, y: f32| {
        let (column, row) = free_space.cell_at(Vec2::new(x, y)).expect("inside the grid");
        free_space.cell(column, row)
    };
    assert!(matches!(cell(2.0, 2.0), Clearance::Free { mm } if mm > 1500), "{:?}", cell(2.0, 2.0));
    assert_eq!(cell(4.05, 0.5), Clearance::Occupied, "the wall");
    assert!(matches!(cell(4.05, 2.0), Clearance::Free { .. }), "the door");

    let topology = topology.expect("a first topology");
    assert_eq!(topology.rooms.len(), 2, "{:?}", topology.rooms);
    assert_eq!(topology.doorways.len(), 1, "{:?}", topology.doorways);
    let door = topology.doorways[0];
    assert!(door.position.distance(Vec2::new(4.05, 2.0)) < 0.2, "{door:?}");
    assert!((door.width_m - 0.8).abs() < 0.15, "{door:?}");
    for room in &topology.rooms {
        assert!((room.floor_m + 0.3).abs() < 0.025, "{room:?}");
        let ceiling = room.ceiling_m.expect("a ceiling seen");
        assert!((ceiling - 2.1).abs() < 0.025, "{room:?}");
    }
    let (left, right) = (room_at(&topology, Vec2::new(2.0, 2.0)), room_at(&topology, Vec2::new(6.0, 2.0)));
    assert_ne!(left, right);

    let (nothing_new, _) = mapper.on_tick(start + Duration::from_secs(2), UnixNanos(2)).unwrap();
    assert!(nothing_new.is_empty(), "nothing changed, nothing sent");

    let nudge = Vec2::new(0.02, -0.01);
    for pose in &poses {
        let moved = KeyframePose { map_version: MapVersion(2), translation: pose.translation + nudge.extend(0.0), ..*pose };
        assert_eq!(mapper.on_keyframe_pose(moved), Arrival::Moved(pose.frame_id));
    }
    let (_, report) = mapper.on_tick(start + Duration::from_secs(3), UnixNanos(3)).unwrap();
    assert_eq!(report.rebuild_started, Some(MapVersion(2)));
    let (outputs_after, report) = mapper.on_tick_after_rebuild(start + Duration::from_secs(4), UnixNanos(4)).unwrap();
    assert_eq!(report.rebuilt, Some(MapVersion(2)));
    let (free_space, topology) = outputs(outputs_after);
    assert_eq!(free_space.map_version, MapVersion(2));
    let topology = topology.expect("a topology for the new map version");
    assert_eq!(topology.map_version, MapVersion(2));
    assert_eq!(topology.rooms.len(), 2);
    assert_eq!(room_at(&topology, Vec2::new(2.0, 2.0)), left);
    assert_eq!(room_at(&topology, Vec2::new(6.0, 2.0)), right);
}

#[test]
fn a_keyframe_whose_pose_is_lost_is_not_integrated() {
    let house = two_rooms();
    let mut mapper = DenseMapper::new(DenseMapSettings::default(), BlockTsdf::new(Default::default()));
    let (keyframe, pose) = keyframe_at(&house, 9, Vec2::new(2.0, 2.0), 0.0, MapVersion(1));
    mapper.on_keyframe(keyframe);
    let lost = KeyframePose { tracking: dense_map::Tracking::Lost, ..pose };
    assert_eq!(mapper.on_keyframe_pose(lost), Arrival::Skipped(FrameId(9)));
    assert_eq!(mapper.kept_keyframes(), 0);
}

#[test]
fn keyframes_waiting_beyond_the_buffer_drop_the_oldest() {
    let house = two_rooms();
    let mut settings = DenseMapSettings::default();
    settings.buffers.max_awaiting_pose = 2;
    let mut mapper = DenseMapper::new(settings, BlockTsdf::new(Default::default()));
    let arrivals: Vec<Arrival> =
        (0..3).map(|i| mapper.on_keyframe(keyframe_at(&house, i, Vec2::new(2.0, 2.0), 0.0, MapVersion(1)).0)).collect();
    assert_eq!(arrivals, vec![Arrival::Waiting, Arrival::Waiting, Arrival::WaitingDropped(FrameId(0))]);
}
