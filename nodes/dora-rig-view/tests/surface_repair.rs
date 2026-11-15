//! The rig view following the dense map's surface messages, fed by the dense map itself: a missed message asks for
//! exactly one snapshot, the snapshot repairs the surface to the dense map's, and a restarted dense map starts the
//! surface again.

use std::time::{Duration, Instant};

use bytes::Bytes;
use dense_map::synthetic::{keyframe_at, look_around, two_rooms};
use dense_map::{BlockTsdf, DenseMapSettings, DenseMapper, MapOutput};
use dora_rig_view::rig::{RigInput, RigView, RigViewSettings};
use dora_rig_view::surface::{encode_mesh_blocks, BlockGrid, MeshCache};
use dora_rig_view::surface_stream::{SnapshotRequestReason, Untrusted};
use glam::Vec2;
use perception_wire::navigation::{MapVersion, UnixNanos};
use perception_wire::space_map::{MeshBlocks, MeshSequence};

const GRID: BlockGrid = BlockGrid { voxel_m: 0.05, block_voxels: 8 };

fn dense_map(run_started: UnixNanos) -> DenseMapper<BlockTsdf> {
    let settings = DenseMapSettings::default();
    DenseMapper::new(settings.clone(), BlockTsdf::new(settings.tsdf), run_started)
}

/// Fuses six keyframes turning on the spot at each of `spots`, then meshes them: one surface message per spot.
fn walk(mapper: &mut DenseMapper<BlockTsdf>, spots: &[(f32, f32)], start: Instant) -> Vec<MeshBlocks> {
    let house = two_rooms();
    let mut sent = Vec::new();
    for (i, &(x, y)) in spots.iter().enumerate() {
        for (turn, (position, heading)) in look_around(&[Vec2::new(x, y)], 6).into_iter().enumerate() {
            let (keyframe, pose) = keyframe_at(&house, (i * 6 + turn) as u64, position, heading, MapVersion(1));
            mapper.on_keyframe(keyframe);
            mapper.on_keyframe_pose(pose);
        }
        let now = start + Duration::from_secs(i as u64 + 1);
        let (outputs, _) = mapper.on_tick_settled(now, UnixNanos(i as i64)).unwrap();
        sent.extend(outputs.into_iter().filter_map(|output| match output {
            MapOutput::MeshBlocks(surface) => Some(surface),
            MapOutput::OccupancyGrid(_) | MapOutput::Topology(_) => None,
        }));
    }
    sent
}

/// The dense map's whole surface as the rig view encodes it for a page.
fn dense_map_surface(mapper: &DenseMapper<BlockTsdf>) -> Bytes {
    encode_mesh_blocks(GRID, mapper.sent_surface()).unwrap()
}

/// Waits for the rig view's background thread to have meshed what it was given into the surface `expected`.
async fn surface_becomes(rig_view: &RigView, expected: &Bytes) -> bool {
    for _ in 0..500 {
        if rig_view.whole_surface().as_ref() == Some(expected) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    false
}

fn feed(rig_view: &mut RigView, surface: MeshBlocks, now: Instant) -> Option<SnapshotRequestReason> {
    rig_view.on_input(RigInput::MeshBlocks(surface), now).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_contiguous_stream_is_followed_without_a_request() {
    let start = Instant::now();
    let mut mapper = dense_map(UnixNanos(1_000));
    let sent = walk(&mut mapper, &[(1.0, 1.0), (3.0, 3.0), (6.0, 2.0)], start);
    assert_eq!(sent.iter().map(|m| m.sequence).collect::<Vec<_>>(), [MeshSequence(0), MeshSequence(1), MeshSequence(2)]);
    let (mut rig_view, _background) = RigView::start(RigViewSettings::DEFAULT);
    for surface in sent {
        assert_eq!(feed(&mut rig_view, surface, start), None);
    }
    assert!(surface_becomes(&rig_view, &dense_map_surface(&mapper)).await, "the rig view holds the dense map's surface");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_dropped_message_asks_once_for_the_snapshot_that_repairs_the_surface() {
    let start = Instant::now();
    let mut mapper = dense_map(UnixNanos(1_000));
    let sent = walk(&mut mapper, &[(1.0, 1.0), (3.0, 3.0), (6.0, 2.0), (7.0, 3.0)], start);
    let (mut rig_view, _background) = RigView::start(RigViewSettings::DEFAULT);
    let missed = Untrusted::MissedMessages { expected: MeshSequence(1), came: MeshSequence(2) };
    let mut without_the_dropped = MeshCache::default();
    for (surface, request) in [
        (&sent[0], None),
        (&sent[2], Some(SnapshotRequestReason::Untrusted(missed))),
        (&sent[3], None),
    ] {
        without_the_dropped.apply(GRID, &surface.blocks).unwrap();
        assert_eq!(feed(&mut rig_view, surface.clone(), start), request, "one request at a time, at sequence {}", surface.sequence);
    }
    let held = encode_mesh_blocks(GRID, without_the_dropped.blocks()).unwrap();
    assert!(surface_becomes(&rig_view, &held).await, "the surface held is still drawn, with the mesh blocks after the gap");
    assert_ne!(held, dense_map_surface(&mapper), "the dropped message left the surface wrong");

    let snapshot = mapper.mesh_snapshot(UnixNanos(9));
    assert_eq!(snapshot.sequence, MeshSequence(4), "the snapshot is the next message of the run");
    assert_eq!(feed(&mut rig_view, snapshot, start + Duration::from_secs(1)), None);
    assert!(surface_becomes(&rig_view, &dense_map_surface(&mapper)).await, "the snapshot repaired the surface");
    assert_eq!(rig_view.on_input(RigInput::MapToOdom(correction()), start + Duration::from_secs(60)).unwrap(), None, "nothing is awaited");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_restarted_dense_map_replaces_the_surface_and_a_late_start_asks_for_a_snapshot() {
    let start = Instant::now();
    let mut first_run = dense_map(UnixNanos(1_000));
    let (mut rig_view, _background) = RigView::start(RigViewSettings::DEFAULT);
    for surface in walk(&mut first_run, &[(1.0, 1.0), (3.0, 3.0)], start) {
        feed(&mut rig_view, surface, start);
    }
    assert!(surface_becomes(&rig_view, &dense_map_surface(&first_run)).await);

    let mut second_run = dense_map(UnixNanos(2_000));
    let mut sent = walk(&mut second_run, &[(6.0, 2.0), (7.0, 3.0)], start).into_iter();
    assert_eq!(feed(&mut rig_view, sent.next().unwrap(), start), None, "the new run starts at 0: nothing is missing");
    assert_eq!(feed(&mut rig_view, sent.next().unwrap(), start), None);
    assert!(surface_becomes(&rig_view, &dense_map_surface(&second_run)).await, "only the new run's blocks are held");

    let mut third_run = dense_map(UnixNanos(3_000));
    let mut sent = walk(&mut third_run, &[(1.0, 3.0), (5.0, 1.0)], start).into_iter();
    let _before_the_rig_view_listened = sent.next().unwrap();
    let joined = Untrusted::JoinedLate { first_seen: MeshSequence(1) };
    assert_eq!(feed(&mut rig_view, sent.next().unwrap(), start), Some(SnapshotRequestReason::Untrusted(joined)));
    let overdue = start + dora_rig_view::surface_stream::SNAPSHOT_REQUEST_DEADLINE;
    assert_eq!(rig_view.on_input(RigInput::MapToOdom(correction()), overdue).unwrap(), Some(SnapshotRequestReason::DeadlinePassed));
    feed(&mut rig_view, third_run.mesh_snapshot(UnixNanos(9)), overdue);
    assert!(surface_becomes(&rig_view, &dense_map_surface(&third_run)).await);
}

fn correction() -> perception_wire::localization::MapToOdom {
    use perception_wire::localization::{CameraNanos, MapToOdom, Pose4Dof};
    MapToOdom {
        map_version: MapVersion(1),
        valid_from: CameraNanos(0),
        odom_started: CameraNanos(0),
        correction: Pose4Dof { x_m: 0.0, y_m: 0.0, z_m: 0.0, yaw_rad: 0.0 },
        sigma_xy_m: 0.01,
        sigma_z_m: 0.01,
        sigma_yaw_rad: 0.01,
    }
}
