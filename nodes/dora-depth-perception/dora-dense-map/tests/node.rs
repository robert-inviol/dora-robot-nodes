//! The node's handling of its messages, without dora: encoded keyframes, poses and object updates in, decoded
//! free space, topology and keep-out zones out.

use std::time::{Duration, Instant};

use dense_map::synthetic::{keyframe_at, look_around, two_rooms};
use dense_map::DenseMapSettings;
use dora_dense_map::{read_settings, DenseMapNode, Input, Output};
use dora_node_api::arrow::array::StructArray;
use dora_node_api::arrow::buffer::ScalarBuffer;
use glam::{Vec2, Vec3};
use perception_wire::localization::{encode_keyframe_pose, KeyframePose, Pose4, Tracking};
use perception_wire::navigation::{MapVersion, ObjectId, UnixNanos};
use perception_wire::objects::{encode_object_update, ObjectExtent, ObjectUpdate, ObjectUpdateKind};
use perception_wire::space_map::{decode_free_space, decode_keep_out, decode_topology};
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
    }
}

#[test]
fn a_walk_through_two_rooms_comes_out_as_free_space_and_topology_and_a_glass_door_as_a_keep_out_zone() {
    let house = two_rooms();
    let mut node = DenseMapNode::new(DenseMapSettings::default()).unwrap();
    let start = Instant::now();
    let spots = [(1.0, 1.0), (3.0, 1.0), (2.0, 2.0), (1.0, 3.0), (3.0, 3.0), (5.0, 1.0), (7.0, 1.0), (6.0, 2.0), (5.0, 3.0), (7.0, 3.0)];
    for (i, (position, heading)) in look_around(&spots.map(|(x, y)| Vec2::new(x, y)), 6).into_iter().enumerate() {
        let (keyframe, pose) = keyframe_at(&house, i as u64, position, heading, MapVersion(1));
        let wire_keyframe = Keyframe {
            frame_id: keyframe.frame_id.0,
            captured_unix_ns: keyframe.captured.0,
            width: keyframe.width,
            height: keyframe.height,
            intrinsics: keyframe.intrinsics,
            gravity_reaction: keyframe.gravity_reaction,
            depth_mm: ScalarBuffer::from(keyframe.depth_mm.to_vec()),
            color_jpeg: vec![0xFF, 0xD8, 0xFF, 0xD9],
        };
        let wire_pose = KeyframePose {
            frame_id: pose.frame_id,
            captured: keyframe.captured,
            map_version: pose.map_version,
            pose: Pose4 { x_m: pose.translation.x, y_m: pose.translation.y, z_m: pose.translation.z, yaw_rad: pose.yaw_rad },
            tracking: Tracking::Tracking,
            residual_m: 0.01,
            inlier_fraction: 0.9,
        };
        assert!(messages(&mut node, Input::Keyframe, encode_keyframe(&wire_keyframe), start).is_empty());
        assert!(messages(&mut node, Input::KeyframePose, encode_keyframe_pose(&wire_pose), start).is_empty());
    }
    assert_eq!(node.mapper().kept_keyframes(), 60);

    let out = messages(&mut node, Input::Tick, StructArray::new_empty_fields(1, None), start);
    let kinds: Vec<Output> = out.iter().map(|(o, _)| *o).collect();
    assert_eq!(kinds, vec![Output::FreeSpace, Output::Topology]);
    let free_space = decode_free_space(&out[0].1).unwrap();
    assert_eq!((free_space.map_version, free_space.resolution_m), (MapVersion(1), 0.05));
    let topology = decode_topology(&out[1].1).unwrap();
    assert_eq!((topology.rooms.len(), topology.doorways.len()), (2, 1));

    let early = messages(&mut node, Input::Tick, StructArray::new_empty_fields(1, None), start + Duration::from_millis(250));
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
