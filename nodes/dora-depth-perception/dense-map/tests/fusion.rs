//! Posed depth frames fused at their corrected odometry pose, and the partial rebuild of the blocks of moved
//! keyframes, on rendered frames of two rooms.

use std::time::{Duration, Instant};

use dense_map::synthetic::{d455_intrinsics, keyframe_at, two_rooms};
use dense_map::{
    Arrival, BlockField, BlockTsdf, DenseMapSettings, DenseMapper, DepthKeyframe, Tsdf, FrameId, KeyframePose, MapOutput,
    PosedDepth, PosedDepthArrival, Unfused, BLOCK_EDGE_VOXELS,
};
use depth_perception::synthetic::SyntheticRoom;
use dora_node_api::arrow::buffer::ScalarBuffer;
use glam::{IVec3, Mat3, Quat, Vec2, Vec3};
use perception_wire::camera::StreamIntrinsics;
use perception_wire::localization::{CameraNanos, MapToOdom, Pose4Dof};
use perception_wire::navigation::{MapVersion, UnixNanos};
use perception_wire::posed_depth::PosedDepthFrame;
use perception_wire::space_map::MeshSequence;
use rustc_hash::FxHashSet;

const RUN_STARTED: UnixNanos = UnixNanos(1_762_560_000_000_000_000);

fn mapper(settings: DenseMapSettings) -> DenseMapper<BlockTsdf> {
    let tsdf = settings.tsdf;
    DenseMapper::new(settings, BlockTsdf::new(tsdf), RUN_STARTED)
}

/// Keyframes looking around from two spots in the left room, frame ids from 0.
fn walk(house: &SyntheticRoom) -> Vec<(DepthKeyframe, KeyframePose)> {
    [Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0)]
        .iter()
        .flat_map(|&spot| (0..4).map(move |turn| (spot, turn as f32 * std::f32::consts::FRAC_PI_2)))
        .enumerate()
        .map(|(i, (spot, heading))| keyframe_at(house, i as u64, spot, heading, MapVersion(1)))
        .collect()
}

fn fuse_all(mapper: &mut DenseMapper<BlockTsdf>, keyframes: &[(DepthKeyframe, KeyframePose)]) {
    for (keyframe, pose) in keyframes {
        assert_eq!(mapper.on_keyframe(keyframe.clone()), Arrival::Waiting);
        assert_eq!(mapper.on_keyframe_pose(*pose), Arrival::Fused(pose.frame_id));
    }
}

/// Every voxel of either field holds the same in both.
fn assert_same_field(a: &BlockTsdf, b: &BlockTsdf) {
    let blocks: FxHashSet<IVec3> = a.blocks().into_iter().chain(b.blocks()).collect();
    assert_eq!(a.blocks().len(), b.blocks().len(), "block counts");
    for block in blocks {
        for i in 0..BLOCK_EDGE_VOXELS.pow(3) {
            let local = IVec3::new(i % BLOCK_EDGE_VOXELS, (i / BLOCK_EDGE_VOXELS) % BLOCK_EDGE_VOXELS, i / BLOCK_EDGE_VOXELS.pow(2));
            let voxel = block * BLOCK_EDGE_VOXELS + local;
            assert_eq!(a.voxel(voxel), b.voxel(voxel), "voxel {voxel}");
        }
    }
}

/// The kept keyframes, fused from scratch in frame order at the given poses.
fn from_scratch(settings: &DenseMapSettings, keyframes: &[(DepthKeyframe, KeyframePose)], kept: &[FrameId]) -> BlockTsdf {
    let mut field = BlockTsdf::new(settings.tsdf);
    for (keyframe, pose) in keyframes.iter().filter(|(k, _)| kept.contains(&k.frame_id)) {
        field.fuse_into_footprint(&PosedDepth::new(keyframe, pose));
    }
    field
}

fn moved(pose: &KeyframePose, by: Vec3, version: u64) -> KeyframePose {
    KeyframePose { translation: pose.translation + by, map_version: MapVersion(version), ..*pose }
}

/// A frame of the house rendered from `position` facing `heading_rad`, posed in an odom frame that `correction`
/// carries into the map, so that it is fused where it was rendered.
fn posed_frame(house: &SyntheticRoom, position: Vec3, heading_rad: f32, correction: &MapToOdom, odom_started: CameraNanos) -> PosedDepthFrame {
    let frame = house.render_from(position, heading_rad);
    let level_camera_facing_y = Mat3::from_cols(Vec3::X, Vec3::NEG_Z, Vec3::Y);
    let camera_to_map = Mat3::from_rotation_z(heading_rad) * level_camera_facing_y;
    let c = correction.correction;
    let map_to_odom = Mat3::from_rotation_z(-c.yaw_rad);
    let k = d455_intrinsics();
    PosedDepthFrame {
        frame_id: FrameId(7),
        camera_ns: CameraNanos(odom_started.0 + 1_000),
        odom_started,
        intrinsics: StreamIntrinsics { width: house.width as u16, height: house.height as u16, fx: k.fx, fy: k.fy, ppx: k.ppx, ppy: k.ppy },
        camera_position_m: map_to_odom * (position - Vec3::new(c.x_m, c.y_m, c.z_m)),
        camera_orientation: Quat::from_mat3(&(map_to_odom * camera_to_map)),
        depth_mm: ScalarBuffer::from(frame.depth_m.iter().map(|d| (d * 1000.0).round().min(u16::MAX as f32) as u16).collect::<Vec<_>>()),
    }
}

fn correction(odom_started: CameraNanos) -> MapToOdom {
    MapToOdom {
        map_version: MapVersion(3),
        valid_from: CameraNanos(odom_started.0 + 500),
        odom_started,
        correction: Pose4Dof { x_m: 1.0, y_m: -0.5, z_m: 0.125, yaw_rad: 0.5 },
        sigma_xy_m: 0.02,
        sigma_z_m: 0.01,
        sigma_yaw_rad: 0.01,
    }
}

#[test]
fn a_posed_depth_frame_is_fused_where_its_corrected_pose_puts_it() {
    let house = two_rooms();
    let mut mapper = mapper(DenseMapSettings::default());
    let odom = CameraNanos(5_000);
    mapper.on_map_to_odom(correction(odom));
    let frame = posed_frame(&house, Vec3::new(2.0, 1.0, 0.0), 0.0, &correction(odom), odom);
    assert_eq!(mapper.on_posed_depth(&frame), PosedDepthArrival::Fused);
    let wall: Vec<_> = mapper.field().surface().into_iter().filter(|p| p.normal.dot(Vec3::NEG_Y) > 0.9).collect();
    assert!(wall.len() > 500, "{} points on the far wall", wall.len());
    for point in &wall {
        assert!((point.position.y - 4.0).abs() < 0.03, "the far wall is at y = 4: {point:?}");
    }
    let start = Instant::now();
    let (outputs, _) = mapper.on_tick_settled(start, UnixNanos(1)).unwrap();
    let surface = outputs.iter().find_map(|o| if let MapOutput::MeshBlocks(s) = o { Some(s) } else { None }).expect("a surface");
    assert_eq!(surface.map_version, MapVersion(3), "the correction's map version");
}

#[test]
fn posed_depth_frames_wait_for_a_correction_of_their_own_odom_frame() {
    let house = two_rooms();
    let mut mapper = mapper(DenseMapSettings::default());
    let frame = posed_frame(&house, Vec3::new(2.0, 1.0, 0.0), 0.0, &correction(CameraNanos(9_000)), CameraNanos(9_000));
    assert_eq!(mapper.on_posed_depth(&frame), PosedDepthArrival::Unfused(Unfused::NoCorrection));
    mapper.on_map_to_odom(correction(CameraNanos(5_000)));
    assert_eq!(
        mapper.on_posed_depth(&frame),
        PosedDepthArrival::Unfused(Unfused::OtherOdomFrame { frame: CameraNanos(9_000), correction: CameraNanos(5_000) })
    );
    assert!(mapper.field().blocks().is_empty(), "nothing fused");
    mapper.on_map_to_odom(correction(CameraNanos(9_000)));
    assert_eq!(mapper.on_posed_depth(&frame), PosedDepthArrival::Fused);
}

#[test]
fn a_moved_keyframe_rebuilds_only_its_blocks_and_the_map_matches_the_kept_keyframes_fused_from_scratch() {
    let house = two_rooms();
    let settings = DenseMapSettings::default();
    let mut mapper = mapper(settings.clone());
    let mut keyframes = walk(&house);
    fuse_all(&mut mapper, &keyframes);
    mapper.on_tick_settled(Instant::now(), UnixNanos(1)).unwrap();

    let (keyframe, pose) = keyframes[2].clone();
    let new_pose = moved(&pose, Vec3::new(0.3, 0.2, 0.0), 2);
    let old_blocks: FxHashSet<IVec3> = mapper.footprint(pose.frame_id).expect("kept").iter().copied().collect();
    let new_blocks = BlockTsdf::new(settings.tsdf).footprint(&PosedDepth::new(&keyframe, &new_pose));
    let expected: FxHashSet<IVec3> = old_blocks.union(&new_blocks).copied().collect();
    assert_eq!(mapper.on_pose_graph_update([new_pose]), vec![Arrival::Moved(pose.frame_id)]);
    assert_eq!(mapper.blocks_to_rebuild(), &expected);
    assert!(expected.len() < mapper.field().blocks().len(), "a part of the map");

    let (_, report) = mapper.on_tick_settled(Instant::now(), UnixNanos(2)).unwrap();
    assert_eq!(report.rebuilt.map(|r| (r.map_version, r.blocks)), Some((MapVersion(2), expected.len())));
    keyframes[2].1 = new_pose;
    assert_same_field(mapper.field(), &from_scratch(&settings, &keyframes, &mapper.kept_frame_ids()));
}

#[test]
fn no_rebuild_starts_for_poses_that_did_not_move() {
    let house = two_rooms();
    let mut mapper = mapper(DenseMapSettings::default());
    let keyframes = walk(&house);
    fuse_all(&mut mapper, &keyframes);
    let unmoved: Vec<KeyframePose> = keyframes.iter().map(|(_, pose)| moved(pose, Vec3::new(0.001, 0.0, 0.0), 2)).collect();
    let arrivals = mapper.on_pose_graph_update(unmoved);
    assert!(arrivals.iter().all(|a| matches!(a, Arrival::AlreadyAtPose(_))), "{arrivals:?}");
    assert_eq!(mapper.on_keyframe_pose(keyframes[0].1), Arrival::AlreadyAtPose(FrameId(0)));
    let (_, report) = mapper.on_tick_settled(Instant::now(), UnixNanos(1)).unwrap();
    assert_eq!((report.rebuild_started, report.rebuilt), (None, None));
}

#[test]
fn keyframes_placed_during_a_rebuild_are_in_its_blocks_and_start_no_other() {
    let house = two_rooms();
    let settings = DenseMapSettings::default();
    let mut mapper = mapper(settings.clone());
    let mut keyframes = walk(&house);
    let later = keyframes.split_off(6);
    fuse_all(&mut mapper, &keyframes);
    let new_pose = moved(&keyframes[1].1, Vec3::new(0.0, 0.25, 0.0), 2);
    mapper.on_pose_graph_update([new_pose]);
    keyframes[1].1 = new_pose;
    let start = Instant::now();
    let (_, report) = mapper.on_tick(start, UnixNanos(1)).unwrap();
    assert!(report.rebuild_started.is_some() && mapper.is_rebuilding());

    // Placed while the rebuild runs, in a later map version.
    let later: Vec<(DepthKeyframe, KeyframePose)> =
        later.into_iter().map(|(keyframe, pose)| (keyframe, KeyframePose { map_version: MapVersion(3), ..pose })).collect();
    fuse_all(&mut mapper, &later);
    let (_, report) = mapper.on_tick_settled(start + Duration::from_secs(1), UnixNanos(2)).unwrap();
    assert_eq!((report.rebuilt.map(|r| r.map_version), report.rebuild_started), (Some(MapVersion(2)), None));
    let (_, report) = mapper.on_tick_settled(start + Duration::from_secs(2), UnixNanos(3)).unwrap();
    assert_eq!((report.rebuilt, report.rebuild_started), (None, None), "no rebuild chained");
    keyframes.extend(later);
    assert_same_field(mapper.field(), &from_scratch(&settings, &keyframes, &mapper.kept_frame_ids()));
}

#[test]
fn kept_keyframes_stay_within_the_cap_and_a_moved_one_let_go_rebuilds_its_old_blocks() {
    let house = two_rooms();
    let mut settings = DenseMapSettings::default();
    settings.keyframes.max_kept = 5;
    let mut mapper = mapper(settings.clone());
    let keyframes = walk(&house);
    fuse_all(&mut mapper, &keyframes);
    assert_eq!(mapper.kept_keyframes(), 5);
    let (_, report) = mapper.on_tick_settled(Instant::now(), UnixNanos(1)).unwrap();
    assert_eq!(report.let_go, keyframes.len() - 5);

    let let_go = keyframes.iter().map(|(k, _)| k.frame_id).find(|id| !mapper.kept_frame_ids().contains(id)).expect("one let go");
    let (keyframe, pose) = &keyframes[let_go.0 as usize];
    let old_blocks = BlockTsdf::new(settings.tsdf).footprint(&PosedDepth::new(keyframe, pose));
    assert_eq!(mapper.on_pose_graph_update([moved(pose, Vec3::X * 0.5, 2)]), vec![Arrival::Moved(let_go)]);
    assert_eq!(mapper.blocks_to_rebuild(), &old_blocks, "the blocks it was fused into, and none at its new pose");
    mapper.on_tick_settled(Instant::now(), UnixNanos(2)).unwrap();
    assert_eq!(mapper.on_pose_graph_update([moved(pose, Vec3::X, 3)]), vec![Arrival::NotHeld(let_go)], "rebuilt without it once");
}

#[test]
fn mesh_blocks_messages_number_the_run_from_zero_without_a_gap() {
    let house = two_rooms();
    let mut mapper = mapper(DenseMapSettings::default());
    let start = Instant::now();
    let mut sequences = Vec::new();
    for (round, (keyframe, pose)) in walk(&house).into_iter().enumerate() {
        fuse_all(&mut mapper, &[(keyframe, pose)]);
        let (outputs, _) = mapper.on_tick_settled(start + Duration::from_secs(10 * round as u64), UnixNanos(round as i64)).unwrap();
        for output in outputs {
            if let MapOutput::MeshBlocks(surface) = output {
                assert_eq!(surface.run_started, RUN_STARTED);
                sequences.push(surface.sequence);
            }
        }
    }
    assert!(sequences.len() > 4, "{sequences:?}");
    assert_eq!(sequences, (0..sequences.len() as u64).map(MeshSequence).collect::<Vec<_>>());
}
