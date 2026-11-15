//! The dense map as the node runs it, on a rendered walk through two rooms joined by a door.

use std::time::{Duration, Instant};

use dense_map::synthetic::{keyframe_at, look_around, two_rooms};
use dense_map::{Arrival, BlockTsdf, DenseMapSettings, DenseMapper, FrameId, KeyframePose, MapOutput};
use glam::Vec2;
use perception_wire::navigation::{MapVersion, UnixNanos};
use perception_wire::space_map::{Clearance, OccupancyGrid, RoomId, MeshBlocks, Topology};

const SPOTS: [(f32, f32); 10] =
    [(1.0, 1.0), (3.0, 1.0), (2.0, 2.0), (1.0, 3.0), (3.0, 3.0), (5.0, 1.0), (7.0, 1.0), (6.0, 2.0), (5.0, 3.0), (7.0, 3.0)];

fn walk() -> Vec<(Vec2, f32)> {
    look_around(&SPOTS.map(|(x, y)| Vec2::new(x, y)), 6)
}

fn outputs(outputs: Vec<MapOutput>) -> (OccupancyGrid, Option<Topology>, Option<MeshBlocks>) {
    let mut occupancy_grid = None;
    let mut topology = None;
    let mut surface = None;
    for output in outputs {
        match output {
            MapOutput::OccupancyGrid(f) => occupancy_grid = Some(f),
            MapOutput::Topology(t) => topology = Some(t),
            MapOutput::MeshBlocks(s) => surface = Some(s),
        }
    }
    (occupancy_grid.expect("occupancy grid"), topology, surface)
}

/// How many of the surface's vertices lie within 3 cm of the height `z_m`.
fn vertices_at_height(surface: &MeshBlocks, z_m: f32) -> usize {
    surface.blocks.iter().flat_map(|p| &p.positions_m).filter(|v| (v.z - z_m).abs() < 0.03).count()
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
    let mut mapper = DenseMapper::new(DenseMapSettings::default(), BlockTsdf::new(Default::default()), UnixNanos(0));
    let start = Instant::now();
    let mut poses: Vec<KeyframePose> = Vec::new();
    for (i, (position, heading)) in walk().into_iter().enumerate() {
        let (keyframe, pose) = keyframe_at(&house, i as u64, position, heading, MapVersion(1));
        if i % 2 == 0 {
            assert_eq!(mapper.on_keyframe(keyframe), Arrival::Waiting);
            assert_eq!(mapper.on_keyframe_pose(pose), Arrival::Fused(FrameId(i as u64)));
        } else {
            assert_eq!(mapper.on_keyframe_pose(pose), Arrival::Waiting, "a pose ahead of its keyframe");
            assert_eq!(mapper.on_keyframe(keyframe), Arrival::Fused(FrameId(i as u64)));
        }
        poses.push(pose);
    }

    let (occupancy_grid, topology, surface) = outputs(mapper.on_tick_settled(start, UnixNanos(1)).unwrap().0);
    assert_eq!(occupancy_grid.map_version, MapVersion(1));
    let surface = surface.expect("the first surface");
    assert!(surface.blocks.iter().all(|p| !p.triangles.is_empty()), "a first surface has no emptied blocks");
    assert!(vertices_at_height(&surface, -0.3) > 1000, "the floor");
    assert!(vertices_at_height(&surface, 2.1) > 1000, "the ceiling");
    let cell = |x: f32, y: f32| {
        let (column, row) = occupancy_grid.cell_at(Vec2::new(x, y)).expect("inside the grid");
        occupancy_grid.cell(column, row)
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

    let (nothing_new, _) = mapper.on_tick_settled(start + Duration::from_secs(2), UnixNanos(2)).unwrap();
    assert!(nothing_new.is_empty(), "nothing changed, nothing sent");

    let nudge = Vec2::new(0.02, -0.01);
    for pose in &poses {
        let moved = KeyframePose { map_version: MapVersion(2), translation: pose.translation + nudge.extend(0.0), ..*pose };
        assert_eq!(mapper.on_keyframe_pose(moved), Arrival::Moved(pose.frame_id));
    }
    // Past each block's resend period, so the moved surface is sent again as soon as it is rebuilt.
    let (outputs_after, report) = mapper.on_tick_settled(start + Duration::from_secs(6), UnixNanos(6)).unwrap();
    assert_eq!(report.rebuild_started.map(|r| r.map_version), Some(MapVersion(2)));
    assert_eq!(report.rebuilt.map(|r| r.map_version), Some(MapVersion(2)));
    let (occupancy_grid, topology, surface) = outputs(outputs_after);
    assert_eq!(occupancy_grid.map_version, MapVersion(2));
    let surface = surface.expect("the moved walls are meshed again");
    assert_eq!(surface.map_version, MapVersion(2));
    let topology = topology.expect("a topology for the new map version");
    assert_eq!(topology.map_version, MapVersion(2));
    assert_eq!(topology.rooms.len(), 2);
    assert_eq!(room_at(&topology, Vec2::new(2.0, 2.0)), left);
    assert_eq!(room_at(&topology, Vec2::new(6.0, 2.0)), right);
}

#[test]
fn a_keyframe_whose_pose_is_lost_is_not_fused() {
    let house = two_rooms();
    let mut mapper = DenseMapper::new(DenseMapSettings::default(), BlockTsdf::new(Default::default()), UnixNanos(0));
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
    let mut mapper = DenseMapper::new(settings, BlockTsdf::new(Default::default()), UnixNanos(0));
    let arrivals: Vec<Arrival> =
        (0..3).map(|i| mapper.on_keyframe(keyframe_at(&house, i, Vec2::new(2.0, 2.0), 0.0, MapVersion(1)).0)).collect();
    assert_eq!(arrivals, vec![Arrival::Waiting, Arrival::Waiting, Arrival::WaitingDropped(FrameId(0))]);
}
