//! The node's message handling, fed encoded keyframes, poses and segments of a rendered room.

use std::path::PathBuf;

use dora_object_map::{read_settings, Input, ObjectMapNode};
use glam::{Vec2, Vec3};
use object_map::synthetic::{LabelledBox, SyntheticScene};
use object_map::ObservedFrame;
use object_map::ObjectMapSettings;
use perception_wire::localization::{encode_keyframe_pose, FrameId, KeyframePose, Pose4, Tracking};
use perception_wire::navigation::{MapVersion, UnixNanos};
use perception_wire::objects::{decode_object_update, ObjectUpdate, ObjectUpdateKind};
use perception_wire::{encode_keyframe, encode_segments};

/// Mask erosion and point count for the synthetic camera's resolution, as an operator would set them.
const SETTINGS_TOML: &str = "
[detection]
erosion_px = 3
min_points = 100
";

fn settings() -> ObjectMapSettings {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("object-map-settings.toml");
    std::fs::write(&path, SETTINGS_TOML).unwrap();
    read_settings(&path).unwrap()
}

fn scene() -> SyntheticScene {
    SyntheticScene::living_room(vec![
        LabelledBox::new("box", Vec3::new(0.0, 2.0, -1.0), Vec3::new(0.6, 2.4, -0.5)),
        LabelledBox::new("bin", Vec3::new(1.1, 1.6, -1.0), Vec3::new(1.5, 2.0, -0.6)),
    ])
}

fn keyframe_pose(frame: &ObservedFrame, tracking: Tracking) -> KeyframePose {
    let t = frame.pose.translation;
    KeyframePose {
        frame_id: FrameId(frame.keyframe.frame_id),
        captured: UnixNanos(frame.keyframe.captured_unix_ns),
        map_version: frame.map_version,
        pose: Pose4 { x_m: t.x, y_m: t.y, z_m: t.z, yaw_rad: frame.pose.yaw_rad },
        tracking,
        residual_m: 0.01,
        inlier_fraction: 0.9,
    }
}

/// Sends the frame's three messages, segments first, then the pose, then the keyframe.
fn send(node: &mut ObjectMapNode, frame: &ObservedFrame, tracking: Tracking) -> Vec<ObjectUpdate> {
    let mut outputs = node.on_input(Input::Segments, &encode_segments(&frame.segments)).unwrap();
    outputs.extend(node.on_input(Input::KeyframePose, &encode_keyframe_pose(&keyframe_pose(frame, tracking))).unwrap());
    outputs.extend(node.on_input(Input::Keyframe, &encode_keyframe(&frame.keyframe)).unwrap());
    outputs.iter().map(|o| decode_object_update(o).unwrap()).collect()
}

const POSES: [(Vec2, f32); 3] = [(Vec2::new(0.0, 0.0), 0.0), (Vec2::new(0.3, 0.2), 0.15), (Vec2::new(-0.3, 0.3), -0.15)];

fn observe(scene: &SyntheticScene, frame_id: u64, (position, heading): (Vec2, f32)) -> ObservedFrame {
    scene.observe(frame_id, 1_762_560_000_000_000_000 + frame_id as i64 * 1_000_000_000, position.extend(0.0), heading)
}

#[test]
fn three_placed_frames_make_each_object_appear_once() {
    let scene = scene();
    let mut node = ObjectMapNode::new(settings());
    let mut updates = Vec::new();
    for (k, &pose) in POSES.iter().enumerate() {
        updates.extend(send(&mut node, &observe(&scene, k as u64 + 1, pose), Tracking::Tracking));
    }
    let mut appeared: Vec<&str> = updates.iter().filter(|u| u.kind == ObjectUpdateKind::Appeared).map(|u| u.label.as_str()).collect();
    appeared.sort();
    assert_eq!(appeared, ["bin", "box"], "{updates:#?}");
    assert!(updates.iter().all(|u| u.frame_id == FrameId(3) && u.map_version == MapVersion(1) && u.crop_jpeg.is_some()));
}

#[test]
fn frames_the_localizer_lost_do_not_count() {
    let scene = scene();
    let mut node = ObjectMapNode::new(settings());
    let mut updates = send(&mut node, &observe(&scene, 1, POSES[0]), Tracking::Lost);
    updates.extend(send(&mut node, &observe(&scene, 2, POSES[1]), Tracking::Relocalised));
    updates.extend(send(&mut node, &observe(&scene, 3, POSES[2]), Tracking::Tracking));
    assert!(updates.is_empty(), "only two frames were placed: {updates:#?}");
}

#[test]
fn an_unknown_setting_is_rejected() {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("object-map-bad-settings.toml");
    std::fs::write(&path, "[association]\nmin_overlap = 0.3\nmax_overlap = 0.9\n").unwrap();
    let error = format!("{:#}", read_settings(&path).err().unwrap());
    assert!(error.contains("max_overlap"), "{error}");
}

#[test]
fn an_unknown_input_is_rejected() {
    assert!(Input::from_id("scene").is_err());
}

#[test]
fn the_shipped_settings_file_holds_the_defaults() {
    let shipped = read_settings(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("object-map-settings.toml")).unwrap();
    assert_eq!(shipped, ObjectMapSettings::default());
}
