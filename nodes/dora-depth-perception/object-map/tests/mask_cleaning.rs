//! Masks cleaned in 3D before they place an object, on rendered keyframes whose masks and depth are spoiled the ways
//! real ones are.

use dora_node_api::arrow::buffer::ScalarBuffer;
use glam::{Vec2, Vec3};
use object_map::detection::{detect, Detection, FrameGeometry};
use object_map::synthetic::{LabelledBox, SyntheticScene, WIDTH};
use object_map::voxels::VoxelGrid;
use object_map::{ObjectMapSettings, ObservedFrame};

const VOXEL_M: f32 = 0.05;

fn box_at(label: &str, min_xy: Vec2, max_xy: Vec2, bottom_m: f32, top_m: f32) -> LabelledBox {
    LabelledBox::new(label, min_xy.extend(bottom_m), max_xy.extend(top_m))
}

/// The default settings, with the mask erosion and point count scaled to the synthetic camera's resolution.
fn settings() -> ObjectMapSettings {
    let mut settings = ObjectMapSettings::default();
    settings.detection.erosion_px = 3;
    settings.detection.min_points = 100;
    settings
}

/// The frame from the origin looking along +y, with its mask ids and depth changed by `spoil`.
fn spoiled(scene: &SyntheticScene, spoil: impl Fn(usize, &mut i16, &mut u16)) -> ObservedFrame {
    let mut frame = scene.observe(1, 0, Vec3::ZERO, 0.0);
    let mut ids: Vec<i16> = frame.instance_masks.instance_ids.to_vec();
    let mut depth: Vec<u16> = frame.keyframe.depth_mm.to_vec();
    for (pixel, (id, mm)) in ids.iter_mut().zip(depth.iter_mut()).enumerate() {
        spoil(pixel, id, mm);
    }
    frame.instance_masks.instance_ids = ScalarBuffer::from(ids);
    frame.keyframe.depth_mm = ScalarBuffer::from(depth);
    frame
}

fn detection_of(frame: &ObservedFrame, label: &str) -> Detection {
    let settings = settings();
    let detections = detect(frame, &FrameGeometry::of(frame), VoxelGrid { size_m: VOXEL_M }, &settings.detection);
    detections.into_iter().find(|d| d.label.text() == label).unwrap_or_else(|| panic!("no {label} detected"))
}

/// The smallest and largest corner of the detection's voxels, in metres.
fn extent(detection: &Detection) -> (Vec3, Vec3) {
    let cells = detection.voxels.sorted_cells();
    let low = cells.iter().fold(Vec3::splat(f32::MAX), |m, c| m.min(c.as_vec3() * VOXEL_M));
    let high = cells.iter().fold(Vec3::splat(f32::MIN), |m, c| m.max((c.as_vec3() + 1.0) * VOXEL_M));
    (low, high)
}

fn mask_pixels(frame: &ObservedFrame, index: i16) -> Vec<usize> {
    frame.instance_masks.instance_ids.iter().enumerate().filter(|(_, &id)| id == index).map(|(pixel, _)| pixel).collect()
}

/// A box and, 15 cm to its right at the same depth, a crate.
fn box_and_crate() -> SyntheticScene {
    SyntheticScene::living_room(vec![
        box_at("box", Vec2::new(-0.4, 2.0), Vec2::new(0.0, 2.4), -1.0, -0.5),
        box_at("crate", Vec2::new(0.15, 2.0), Vec2::new(0.55, 2.4), -1.0, -0.5),
    ])
}

#[test]
fn a_clean_mask_is_unchanged() {
    let frame = spoiled(&box_and_crate(), |_, _, _| {});
    let detection = detection_of(&frame, "box");
    assert_eq!((detection.cleaning.outliers, detection.cleaning.dropped_clusters.len()), (0, 0), "{:?}", detection.cleaning);
    assert_eq!(detection.view_pixels, mask_pixels(&frame, 0));
    let (low, high) = extent(&detection);
    assert!(low.x >= -0.45 && high.x <= 0.05, "{low} {high}");
}

#[test]
fn a_mask_that_spills_onto_a_neighbouring_surface_at_similar_depth_loses_the_spill() {
    let crate_columns = |pixel: usize| (pixel % WIDTH) as f32;
    let scene = box_and_crate();
    let clean = scene.observe(1, 0, Vec3::ZERO, 0.0);
    let crate_left = mask_pixels(&clean, 1).iter().map(|&p| crate_columns(p)).fold(f32::MAX, f32::min);
    let spill = |pixel: usize, id: &mut i16, _: &mut u16| {
        if *id == 1 && crate_columns(pixel) < crate_left + 8.0 {
            *id = 0;
        }
    };
    let frame = spoiled(&scene, spill);
    let detection = detection_of(&frame, "box");

    let (low, high) = extent(&detection);
    assert!(low.x >= -0.45 && high.x <= 0.05, "the box's voxels stay on the box: {low} {high}");
    assert_eq!(detection.cleaning.dropped_clusters.len(), 1, "{:?}", detection.cleaning);
    let spilled: Vec<usize> = mask_pixels(&frame, 0).into_iter().filter(|p| !mask_pixels(&clean, 0).contains(p)).collect();
    assert!(!spilled.is_empty());
    assert!(spilled.iter().all(|&p| !detection.shows(p)), "the view is cut without the spill");
    assert_eq!(detection.cleaning.large_dropped_clusters(0.2).count(), 0, "a strip of the crate is not a second object");
}

#[test]
fn two_objects_under_one_mask_keep_the_larger_and_report_the_other_as_a_large_dropped_cluster() {
    let scene = SyntheticScene::living_room(vec![
        box_at("box", Vec2::new(-0.5, 2.0), Vec2::new(0.0, 2.4), -1.0, -0.4),
        box_at("crate", Vec2::new(0.15, 2.0), Vec2::new(0.45, 2.4), -1.0, -0.5),
    ]);
    let frame = spoiled(&scene, |_, id, _| {
        if *id == 1 {
            *id = 0;
        }
    });
    let detection = detection_of(&frame, "box");
    let large: Vec<usize> = detection.cleaning.large_dropped_clusters(0.2).collect();
    assert_eq!(large.len(), 1, "{:?}", detection.cleaning);
    let (low, high) = extent(&detection);
    assert!(low.x >= -0.55 && high.x <= 0.05, "the larger box is kept: {low} {high}");
}

#[test]
fn isolated_flying_pixels_are_dropped() {
    let scene = box_and_crate();
    let fly = |pixel: usize, id: &mut i16, mm: &mut u16| {
        if *id == 0 && pixel % 37 == 0 {
            *mm = if pixel % 2 == 0 { *mm + 600 } else { *mm - 400 };
        }
    };
    let frame = spoiled(&scene, fly);
    let detection = detection_of(&frame, "box");
    let flying = mask_pixels(&frame, 0).iter().filter(|&&p| p % 37 == 0).count();
    assert!(detection.cleaning.outliers + detection.cleaning.dropped_clusters.iter().sum::<usize>() >= flying * 9 / 10, "{:?}", detection.cleaning);
    let (low, high) = extent(&detection);
    assert!(low.y >= 1.95 && high.y <= 2.5, "the box keeps its depth: {low} {high}");
}

#[test]
fn a_thin_object_with_holes_in_its_depth_stays_whole() {
    let scene = SyntheticScene::living_room(vec![
        box_at("lamp", Vec2::new(-0.15, 2.0), Vec2::new(0.15, 2.3), -1.0, -0.95),
        box_at("stem", Vec2::new(-0.02, 2.13), Vec2::new(0.02, 2.17), -0.95, -0.3),
        box_at("shade", Vec2::new(-0.15, 2.0), Vec2::new(0.15, 2.3), -0.3, -0.1),
    ]);
    let holed = |pixel: usize, id: &mut i16, mm: &mut u16| {
        if *id == 1 && (pixel / WIDTH / 6) % 2 == 0 {
            *mm = 0;
        }
        if *id >= 1 {
            *id = 0;
        }
    };
    let frame = spoiled(&scene, holed);
    let detection = detection_of(&frame, "lamp");
    assert!(detection.cleaning.dropped_clusters.is_empty(), "{:?}", detection.cleaning);
    let (low, high) = extent(&detection);
    assert!(low.z < -0.9 && high.z > -0.15, "base and shade are one object: {low} {high}");
}
