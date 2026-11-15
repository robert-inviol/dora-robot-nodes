//! The node's handling of its messages, without dora: encoded keyframes, poses and object updates in, decoded
//! the occupancy grid, topology and keep-out zones out.

use std::time::{Duration, Instant};

use dense_map::synthetic::{keyframe_at, look_around, two_rooms};
use dense_map::DenseMapSettings;
use dora_dense_map::{read_settings, DenseMapNode, Input, Output};
use dora_node_api::arrow::array::StructArray;
use dora_node_api::arrow::buffer::ScalarBuffer;
use glam::{Vec2, Vec3};
use perception_wire::localization::{encode_keyframe_pose, encode_pose_graph_update, FrameId, KeyframePose, PoseGraphUpdate, UpdatedKeyframePose, Pose4Dof, Tracking};
use perception_wire::navigation::{MapVersion, ObjectId, UnixNanos};
use perception_wire::objects::{encode_object_update, ObjectExtent, ObjectUpdate, ObjectUpdateKind};
use perception_wire::space_map::{
    decode_occupancy_grid, decode_keep_out, decode_mesh_blocks, decode_topology, encode_mesh_snapshot_request, MeshBlock,
    MeshBlocks, MeshBlocksKind, MeshSequence, MeshSnapshotRequest,
};
use perception_wire::{encode_keyframe, Keyframe};

fn messages(node: &mut DenseMapNode, input: Input, data: StructArray, now: Instant) -> Vec<(Output, StructArray)> {
    node.on_input(input, &data, now, UnixNanos(1_762_560_100_000_000_000)).unwrap()
}

fn glass_door(kind: ObjectUpdateKind) -> ObjectUpdate {
    ObjectUpdate {
        kind,
        object_id: ObjectId(42),
        merged_into: None,
        label: "glass door".to_owned(),
        score: 0.875,
        confidence: 0.75,
        seen_count: 4,
        frame_id: perception_wire::localization::FrameId(3),
        captured: UnixNanos(1_762_560_000_000_000_000),
        map_version: MapVersion(1),
        extent: ObjectExtent {
            centre_m: Vec3::new(4.05, 2.0, 0.7),
            footprint_m: [Vec2::new(4.0, 1.6), Vec2::new(4.1, 1.6), Vec2::new(4.1, 2.4), Vec2::new(4.0, 2.4)],
            long_m: 0.8,
            short_m: 0.1,
            height_m: 2.0,
            top_m: 1.7,
        },
        crop_jpeg: None,
        voxel_m: 0.05,
        voxels: vec![glam::IVec3::new(100, 16, 9)],
    }
}

/// Fuses a walk through the two rooms, sixty keyframes at ten spots.
fn walk_through_two_rooms(node: &mut DenseMapNode, start: Instant) {
    let house = two_rooms();
    let spots = [(1.0, 1.0), (3.0, 1.0), (2.0, 2.0), (1.0, 3.0), (3.0, 3.0), (5.0, 1.0), (7.0, 1.0), (6.0, 2.0), (5.0, 3.0), (7.0, 3.0)];
    for (i, (position, heading)) in look_around(&spots.map(|(x, y)| Vec2::new(x, y)), 6).into_iter().enumerate() {
        let (keyframe, pose) = keyframe_at(&house, i as u64, position, heading, MapVersion(1));
        let wire_keyframe = Keyframe {
            frame_id: keyframe.frame_id.0,
            captured_unix_ns: keyframe.captured.0,
            width: keyframe.width,
            height: keyframe.height,
            intrinsics: keyframe.intrinsics,
            specific_force: keyframe.specific_force,
            depth_mm: ScalarBuffer::from(keyframe.depth_mm.to_vec()),
            color_jpeg: vec![0xFF, 0xD8, 0xFF, 0xD9],
        };
        let wire_pose = KeyframePose {
            frame_id: pose.frame_id,
            captured: keyframe.captured,
            map_version: pose.map_version,
            pose: Pose4Dof { x_m: pose.translation.x, y_m: pose.translation.y, z_m: pose.translation.z, yaw_rad: pose.yaw_rad },
            tracking: Tracking::Tracking,
            residual_m: 0.01,
            inlier_fraction: 0.9,
        };
        assert!(messages(node, Input::Keyframe, encode_keyframe(&wire_keyframe), start).is_empty());
        assert!(messages(node, Input::KeyframePose, encode_keyframe_pose(&wire_pose), start).is_empty());
    }
    assert_eq!(node.mapper().kept_keyframes(), 60);
}

#[test]
fn a_walk_through_two_rooms_comes_out_as_occupancy_grid_and_topology_and_a_glass_door_as_a_keep_out_zone() {
    let mut node = DenseMapNode::new(DenseMapSettings::default(), UnixNanos(1_762_560_000_000_000_000)).unwrap();
    let start = Instant::now();
    walk_through_two_rooms(&mut node, start);

    let out = node.on_tick_settled(start, UnixNanos(1_762_560_100_000_000_000)).unwrap();
    let kinds: Vec<Output> = out.iter().map(|(o, _)| *o).collect();
    assert_eq!(kinds, vec![Output::MeshBlocks, Output::OccupancyGrid, Output::Topology]);
    let surface = decode_mesh_blocks(&out[0].1).unwrap();
    assert_eq!((surface.map_version, surface.voxel_m, surface.block_voxels), (MapVersion(1), 0.05, 8));
    assert_eq!((surface.run_started, surface.sequence), (UnixNanos(1_762_560_000_000_000_000), MeshSequence(0)));
    assert!(surface.blocks.len() > 100, "{} mesh blocks", surface.blocks.len());
    let occupancy_grid = decode_occupancy_grid(&out[1].1).unwrap();
    assert_eq!((occupancy_grid.map_version, occupancy_grid.resolution_m), (MapVersion(1), 0.05));
    let topology = decode_topology(&out[2].1).unwrap();
    assert_eq!((topology.rooms.len(), topology.doorways.len()), (2, 1));

    let early = node.on_tick_settled(start + Duration::from_millis(100), UnixNanos(1_762_560_100_100_000_000)).unwrap();
    assert!(early.is_empty());

    let out = messages(&mut node, Input::ObjectUpdate, encode_object_update(&glass_door(ObjectUpdateKind::Appeared)), start);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].0, Output::KeepOut);
    let keep_out = decode_keep_out(&out[0].1).unwrap();
    assert_eq!(keep_out.zones.len(), 1);
    assert_eq!((keep_out.zones[0].object_id, keep_out.zones[0].label.as_str()), (ObjectId(42), "glass door"));
    assert!(messages(&mut node, Input::ObjectUpdate, encode_object_update(&glass_door(ObjectUpdateKind::Updated)), start).is_empty());
    let out = messages(&mut node, Input::ObjectUpdate, encode_object_update(&glass_door(ObjectUpdateKind::Disappeared)), start);
    assert!(decode_keep_out(&out[0].1).unwrap().zones.is_empty());

    // The first keyframe moves, in the same map version: the move alone has its blocks rebuilt with it in its new place.
    assert!(!node.mapper().is_rebuilding());
    let moved = PoseGraphUpdate {
        map_version: MapVersion(1),
        keyframes: vec![UpdatedKeyframePose { frame_id: FrameId(0), pose: Pose4Dof { x_m: 1.25, y_m: 1.0, z_m: 0.0, yaw_rad: 0.0 } }],
    };
    assert!(messages(&mut node, Input::PoseGraphUpdate, encode_pose_graph_update(&moved), start).is_empty());
    messages(&mut node, Input::Tick, StructArray::new_empty_fields(1, None), start + Duration::from_secs(1));
    assert!(node.mapper().is_rebuilding(), "a moved keyframe sets off a rebuild");
    let out = node.on_tick_settled(start + Duration::from_secs(7), UnixNanos(1_762_560_107_000_000_000)).unwrap();
    let sequences: Vec<MeshSequence> =
        out.iter().filter(|(o, _)| *o == Output::MeshBlocks).map(|(_, m)| decode_mesh_blocks(m).unwrap().sequence).collect();
    assert_eq!(sequences, vec![MeshSequence(1)], "the rebuilt blocks go out as the next message of the run");
}

/// The surface a receiver holds after taking these messages in order.
fn received_surface(messages: &[MeshBlocks]) -> std::collections::BTreeMap<[i32; 3], MeshBlock> {
    let mut held = std::collections::BTreeMap::new();
    for message in messages {
        if message.kind == MeshBlocksKind::Snapshot {
            held.clear();
        }
        for mesh_block in &message.blocks {
            match mesh_block.triangles.is_empty() {
                true => held.remove(&mesh_block.block.to_array()),
                false => held.insert(mesh_block.block.to_array(), mesh_block.clone()),
            };
        }
    }
    held
}

fn surface_messages(out: &[(Output, StructArray)]) -> Vec<MeshBlocks> {
    out.iter().filter(|(o, _)| *o == Output::MeshBlocks).map(|(_, m)| decode_mesh_blocks(m).unwrap()).collect()
}

#[test]
fn a_snapshot_holds_every_meshed_block_and_keeps_the_sequence_contiguous() {
    let mut node = DenseMapNode::new(DenseMapSettings::default(), UnixNanos(1_762_560_000_000_000_000)).unwrap();
    let start = Instant::now();
    walk_through_two_rooms(&mut node, start);
    let mut sent = surface_messages(&node.on_tick_settled(start, UnixNanos(1_762_560_100_000_000_000)).unwrap());
    let moved = PoseGraphUpdate {
        map_version: MapVersion(2),
        keyframes: vec![UpdatedKeyframePose { frame_id: FrameId(0), pose: Pose4Dof { x_m: 1.25, y_m: 1.0, z_m: 0.0, yaw_rad: 0.0 } }],
    };
    messages(&mut node, Input::PoseGraphUpdate, encode_pose_graph_update(&moved), start);
    sent.extend(surface_messages(&node.on_tick_settled(start + Duration::from_secs(7), UnixNanos(1_762_560_107_000_000_000)).unwrap()));
    assert_eq!(sent.iter().map(|m| m.sequence).collect::<Vec<_>>(), [MeshSequence(0), MeshSequence(1)]);

    let request = encode_mesh_snapshot_request(&MeshSnapshotRequest { requested: UnixNanos(1_762_560_108_000_000_000) });
    let snapshot = surface_messages(&messages(&mut node, Input::MeshSnapshotRequest, request, start + Duration::from_secs(8)));
    assert_eq!(snapshot.len(), 1);
    let snapshot = &snapshot[0];
    assert_eq!((snapshot.kind, snapshot.sequence, snapshot.map_version), (MeshBlocksKind::Snapshot, MeshSequence(2), MapVersion(2)));
    assert_eq!(snapshot.run_started, sent[0].run_started);
    assert!(snapshot.blocks.iter().all(|p| !p.triangles.is_empty()), "a snapshot holds only blocks with surface");
    let every_meshed_block: Vec<&MeshBlock> = node.mapper().sent_surface().collect();
    assert_eq!(snapshot.blocks.iter().collect::<Vec<_>>(), every_meshed_block, "every meshed block, in block order");
    assert_eq!(received_surface(std::slice::from_ref(snapshot)), received_surface(&sent), "the snapshot alone is what the messages built");

    let moved_back = PoseGraphUpdate {
        map_version: MapVersion(3),
        keyframes: vec![UpdatedKeyframePose { frame_id: FrameId(0), pose: Pose4Dof { x_m: 1.0, y_m: 1.0, z_m: 0.0, yaw_rad: 0.0 } }],
    };
    messages(&mut node, Input::PoseGraphUpdate, encode_pose_graph_update(&moved_back), start);
    let after = surface_messages(&node.on_tick_settled(start + Duration::from_secs(14), UnixNanos(1_762_560_114_000_000_000)).unwrap());
    assert_eq!(after.iter().map(|m| (m.kind, m.sequence)).collect::<Vec<_>>(), [(MeshBlocksKind::Changes, MeshSequence(3))]);
}

#[test]
fn an_unknown_input_is_refused() {
    assert!(Input::from_id("odom_pose").is_err());
}

#[test]
fn the_example_settings_file_reads_back_as_the_defaults() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("dense-map-settings.toml");
    assert_eq!(read_settings(&path).unwrap(), DenseMapSettings::default());
}
