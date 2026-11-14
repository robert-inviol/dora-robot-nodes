//! The object map fed rendered keyframes of a room whose objects are known exactly.

use glam::{Vec2, Vec3};
use object_map::synthetic::{LabelledBox, SyntheticScene};
use object_map::{ObjectMap, ObjectMapSettings};
use perception_wire::objects::{ObjectId, ObjectUpdate, ObjectUpdateKind};

const SECOND_NS: i64 = 1_000_000_000;
const START_NS: i64 = 1_762_560_000 * SECOND_NS;

fn box_at(label: &str, min_xy: Vec2, max_xy: Vec2, top_m: f32) -> LabelledBox {
    LabelledBox::new(label, min_xy.extend(-1.0), max_xy.extend(top_m))
}

fn the_box() -> LabelledBox {
    box_at("box", Vec2::new(0.0, 2.0), Vec2::new(0.6, 2.4), -0.5)
}

fn living_room() -> SyntheticScene {
    SyntheticScene::living_room(vec![
        the_box(),
        box_at("table", Vec2::new(-1.0, 2.6), Vec2::new(-0.3, 3.2), -0.55),
        box_at("bin", Vec2::new(1.1, 1.6), Vec2::new(1.5, 2.0), -0.6),
    ])
}

/// The default settings, with the mask erosion and point count scaled to the synthetic camera's resolution.
fn settings() -> ObjectMapSettings {
    let mut settings = ObjectMapSettings::default();
    settings.detection.erosion_px = 3;
    settings.detection.min_points = 100;
    settings
}

/// Camera poses a short walk apart, all looking into the room.
const WALK: [(Vec3, f32); 6] = [
    (Vec3::new(0.0, 0.0, 0.0), 0.0),
    (Vec3::new(0.3, 0.2, 0.0), 0.15),
    (Vec3::new(-0.3, 0.3, 0.0), -0.15),
    (Vec3::new(0.2, 0.5, 0.05), 0.1),
    (Vec3::new(-0.2, -0.2, -0.05), -0.1),
    (Vec3::new(0.0, 0.4, 0.0), 0.0),
];

/// Feeds the scene from each pose in turn, one keyframe a second.
struct Walker {
    map: ObjectMap,
    frame_id: u64,
}

impl Walker {
    fn new() -> Walker {
        let _ = tracing_subscriber::fmt().with_test_writer().with_max_level(tracing::Level::DEBUG).try_init();
        Walker { map: ObjectMap::new(settings()), frame_id: 0 }
    }

    fn see(&mut self, scene: &SyntheticScene, (position, heading): (Vec3, f32)) -> Vec<ObjectUpdate> {
        self.frame_id += 1;
        let captured = START_NS + self.frame_id as i64 * SECOND_NS;
        self.map.observe(&scene.observe(self.frame_id, captured, position, heading)).unwrap()
    }

    fn walk(&mut self, scene: &SyntheticScene, poses: &[(Vec3, f32)]) -> Vec<ObjectUpdate> {
        poses.iter().flat_map(|&pose| self.see(scene, pose)).collect()
    }
}

fn of_kind(updates: &[ObjectUpdate], kind: ObjectUpdateKind) -> Vec<&ObjectUpdate> {
    updates.iter().filter(|u| u.kind == kind).collect()
}

fn labelled<'a>(updates: &'a [ObjectUpdate], kind: ObjectUpdateKind, label: &str) -> &'a ObjectUpdate {
    let found: Vec<_> = updates.iter().filter(|u| u.kind == kind && u.label == label).collect();
    assert_eq!(found.len(), 1, "expected one {kind:?} {label}, got {found:#?}");
    found[0]
}

#[test]
fn each_object_seen_from_several_poses_keeps_one_id_and_its_true_extent() {
    let scene = living_room();
    let mut walker = Walker::new();
    let updates = walker.walk(&scene, &WALK);

    let appeared = of_kind(&updates, ObjectUpdateKind::Appeared);
    let mut labels: Vec<&str> = appeared.iter().map(|u| u.label.as_str()).collect();
    labels.sort();
    assert_eq!(labels, ["bin", "box", "table"], "{appeared:#?}");
    let ids: Vec<ObjectId> = appeared.iter().map(|u| u.object_id).collect();
    for update in of_kind(&updates, ObjectUpdateKind::Updated) {
        assert!(ids.contains(&update.object_id), "an update for an object that never appeared: {update:#?}");
    }
    assert!(of_kind(&updates, ObjectUpdateKind::Disappeared).is_empty() && of_kind(&updates, ObjectUpdateKind::Merged).is_empty());
    assert_eq!(walker.map.confirmed_ids().len(), 3);

    let last_of = |label: &str| updates.iter().rev().find(|u| u.label == label).unwrap();
    // Only the faces the walk saw are mapped, so the footprint lies within the true one and spans its front.
    for truth in &scene.objects {
        let extent = &last_of(&truth.label).extent;
        let (min, max) = (truth.cuboid.min.truncate() - 0.1, truth.cuboid.max.truncate() + 0.1);
        assert!(extent.footprint_m.iter().all(|c| c.cmpge(min).all() && c.cmple(max).all()), "{}: {extent:?}", truth.label);
        assert!((extent.top_m - truth.cuboid.max.z).abs() < 0.06, "{}: {extent:?}", truth.label);
        let front_width = truth.cuboid.max.x - truth.cuboid.min.x;
        assert!((extent.long_m - front_width).abs() < 0.15, "{}: {extent:?}", truth.label);
    }
}

#[test]
fn a_candidate_is_confirmed_on_its_third_sighting() {
    let scene = living_room();
    let mut walker = Walker::new();
    assert!(walker.walk(&scene, &WALK[..2]).is_empty());
    assert_eq!((walker.map.candidate_count(), walker.map.confirmed_ids().len()), (3, 0));

    let third = walker.see(&scene, WALK[2]);
    assert_eq!(of_kind(&third, ObjectUpdateKind::Appeared).len(), 3, "{third:#?}");
    assert!(third.iter().all(|u| u.seen_count == 3 && u.confidence == 1.0));
    assert_eq!(walker.map.candidate_count(), 0);
}

#[test]
fn a_candidate_not_confirmed_within_the_window_is_dropped() {
    let scene = living_room();
    let mut map = ObjectMap::new(settings());
    let window_s = settings().confirmation.window_s as i64;
    let updates: Vec<ObjectUpdate> = (0..3)
        .flat_map(|k| map.observe(&scene.observe(k + 1, START_NS + k as i64 * (window_s + 10) * SECOND_NS, Vec3::ZERO, 0.0)).unwrap())
        .collect();
    assert!(updates.is_empty(), "{updates:#?}");
    assert!(map.confirmed_ids().is_empty());
}

#[test]
fn an_object_taken_away_disappears_and_found_elsewhere_has_moved() {
    let mut scene = living_room();
    let mut walker = Walker::new();
    let appeared = walker.walk(&scene, &WALK[..3]);
    let box_id = labelled(&appeared, ObjectUpdateKind::Appeared, "box").object_id;

    scene.objects.retain(|o| o.label != "box");
    let first_two = walker.walk(&scene, &WALK[3..5]);
    assert!(of_kind(&first_two, ObjectUpdateKind::Disappeared).is_empty(), "{first_two:#?}");
    let third = walker.see(&scene, WALK[5]);
    let gone = labelled(&third, ObjectUpdateKind::Disappeared, "box");
    assert_eq!(gone.object_id, box_id);
    assert_eq!(of_kind(&third, ObjectUpdateKind::Disappeared).len(), 1, "only the box went: {third:#?}");

    scene.objects.push(box_at("box", Vec2::new(-0.1, 2.4), Vec2::new(0.5, 2.8), -0.7));
    let found = walker.walk(&scene, &WALK[..3]);
    let moved = labelled(&found, ObjectUpdateKind::Moved, "box");
    assert_eq!(moved.object_id, box_id);
    assert!((moved.extent.centre_m.truncate() - Vec2::new(0.2, 2.6)).length() < 0.1, "{moved:#?}");
    assert!(of_kind(&found, ObjectUpdateKind::Appeared).is_empty(), "{found:#?}");
    assert!(moved.crop_jpeg.is_some(), "a moved object carries its view from the new place");
}

#[test]
fn an_object_hidden_behind_another_is_not_missed() {
    let mut scene = living_room();
    let mut walker = Walker::new();
    walker.walk(&scene, &WALK[..3]);
    scene.objects.push(box_at("screen", Vec2::new(-0.5, 1.2), Vec2::new(1.2, 1.3), 1.0));
    let updates = walker.walk(&scene, &[WALK[0]; 5]);
    assert!(of_kind(&updates, ObjectUpdateKind::Disappeared).is_empty(), "{updates:#?}");
}

#[test]
fn the_crop_is_sent_when_appearing_and_again_only_for_a_clearly_better_view() {
    let scene = SyntheticScene::living_room(vec![the_box()]);
    let mut walker = Walker::new();
    let far = (Vec3::new(0.3, -0.8, 0.0), 0.0);
    let near = (Vec3::new(0.3, 0.9, 0.0), 0.0);

    let appeared = walker.walk(&scene, &[far, far, far]);
    let far_crop = appeared[0].crop_jpeg.as_ref().expect("an appearance carries the best view");
    assert_eq!(appeared[0].kind, ObjectUpdateKind::Appeared);

    let closer = walker.see(&scene, near);
    let near_crop = closer[0].crop_jpeg.as_ref().expect("a much larger view replaces the best one");
    let size = |jpeg: &[u8]| {
        let mut decoder = jpeg_decoder::Decoder::new(jpeg);
        decoder.read_info().unwrap();
        let info = decoder.info().unwrap();
        info.width as usize * info.height as usize
    };
    assert!(size(near_crop) > 2 * size(far_crop));

    let again_far = walker.see(&scene, far);
    assert_eq!(again_far[0].kind, ObjectUpdateKind::Updated);
    assert_eq!(again_far[0].crop_jpeg, None);
}
