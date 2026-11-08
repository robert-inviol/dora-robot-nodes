use depth_perception::synthetic::SyntheticRoom;
use depth_perception::*;
use glam::Vec2;

struct Analysed {
    scene: SyntheticRoom,
    frame: depth_perception::synthetic::SyntheticFrame,
    floor: FloorFrame,
    cloud: PointCloud,
    fit: PlaneFit,
}

fn analyse() -> Analysed {
    let scene = SyntheticRoom::living_room();
    let frame = scene.render();
    let floor = FloorFrame::from_gravity_reaction(frame.accel_camera);
    let depth = DepthImage::new(scene.width, scene.height, &frame.depth_m);
    let cloud = sample_point_cloud(&depth, &scene.intrinsics, &floor, RangeLimits::D455, 2);
    let fit = find_planes(&cloud.points, &PlaneSettings::default());
    Analysed { scene, frame, floor, cloud, fit }
}

fn plane_of(fit: &PlaneFit, kind: SurfaceKind) -> Vec<&Plane> {
    fit.planes.iter().filter(|p| p.kind == kind).collect()
}

#[test]
fn floor_and_ceiling_are_found_at_their_heights() {
    let a = analyse();
    let floor = plane_of(&a.fit, SurfaceKind::Floor);
    let ceiling = plane_of(&a.fit, SurfaceKind::Ceiling);
    assert_eq!(floor.len(), 1, "{:?}", a.fit.planes);
    assert_eq!(ceiling.len(), 1, "{:?}", a.fit.planes);
    assert!((-floor[0].offset_m - a.scene.room.min.z).abs() < 0.02, "floor at {}", -floor[0].offset_m);
    assert!((-ceiling[0].offset_m - a.scene.room.max.z).abs() < 0.02, "ceiling at {}", -ceiling[0].offset_m);
}

#[test]
fn the_three_visible_walls_are_found_upright() {
    let a = analyse();
    let walls = plane_of(&a.fit, SurfaceKind::Wall);
    for expected in [a.scene.room.min.x, a.scene.room.max.x, a.scene.room.max.y] {
        assert!(
            walls.iter().any(|w| w.normal.z == 0.0
                && (w.normal.x.abs() > 0.99 && (-w.offset_m / w.normal.x - expected).abs() < 0.03
                    || w.normal.y.abs() > 0.99 && (-w.offset_m / w.normal.y - expected).abs() < 0.03)),
            "no wall at {expected}: {walls:?}"
        );
    }
}

#[test]
fn plane_points_are_flattened_onto_their_plane() {
    let a = analyse();
    for (point, plane) in a.fit.flattened.iter().zip(&a.fit.point_planes) {
        if let Some(index) = plane {
            assert!(a.fit.planes[*index as usize].signed_distance_m(*point).abs() < 1e-4);
        }
    }
}

#[test]
fn the_room_spans_the_walls_with_the_unseen_side_open() {
    let a = analyse();
    let room = infer_room(&a.fit.planes, &a.cloud.points, &RoomSettings::default()).expect("room");
    let (low, high) = a.scene.plan_extent();
    let extent = room.footprint_m.iter().fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(lo, hi), c| (lo.min(*c), hi.max(*c)));
    assert!((extent.0.x - low.x).abs() < 0.03 && (extent.1.x - high.x).abs() < 0.03, "{extent:?}");
    assert!((extent.1.y - high.y).abs() < 0.03, "{extent:?}");
    assert!((room.ceiling_m - room.floor_m - 2.4).abs() < 0.03);
    let seen = room.edge_walls.iter().filter(|w| w.is_some()).count();
    assert_eq!((seen, room.edge_walls.len() - seen), (3, 1), "{:?}", room.edge_walls);
}

// Erosion shaves a few pixels off every mask edge, so the footprint is the box less a thin rim, and never
// more than the box. The top face is seen at a grazing angle, so its far rim costs the most depth.
#[test]
fn a_box_on_the_floor_gets_its_footprint_less_the_eroded_rim() {
    let a = analyse();
    let mask_pixels: Vec<bool> = a.frame.object_ids.iter().map(|id| *id == Some(0)).collect();
    let mask = ObjectMask { width: a.scene.width, height: a.scene.height, pixels: &mask_pixels };
    let depth = DepthImage::new(a.scene.width, a.scene.height, &a.frame.depth_m);
    let footprint = object_footprint(&mask, &depth, &a.scene.intrinsics, &a.floor, RangeLimits::D455, &Default::default())
        .expect("footprint");
    let (long, short) = footprint.footprint_m;
    assert!((0.54..=0.61).contains(&long), "long side {long}");
    assert!((0.22..=0.41).contains(&short), "short side {short}");
    assert!((footprint.top_above_camera_m + 0.5).abs() < 0.03, "{}", footprint.top_above_camera_m);
    assert!((footprint.height_m - 0.5).abs() < 0.05, "{}", footprint.height_m);
}

#[test]
fn an_empty_mask_has_no_footprint() {
    let a = analyse();
    let mask_pixels = vec![false; a.scene.width * a.scene.height];
    let mask = ObjectMask { width: a.scene.width, height: a.scene.height, pixels: &mask_pixels };
    let depth = DepthImage::new(a.scene.width, a.scene.height, &a.frame.depth_m);
    assert!(object_footprint(&mask, &depth, &a.scene.intrinsics, &a.floor, RangeLimits::D455, &Default::default()).is_none());
}
