//! The dense map as a node runs it.
//!
//! Two kinds of depth are fused. Posed depth frames, about ten a second, are fused as they arrive at their odometry
//! pose carried into the map by the localizer's correction valid at their capture time; they are fused once and not
//! kept. Keyframes wait for the localizer's pose (or the pose for its keyframe), are fused, and are kept, at most one
//! per pose cell and at most a set number in all, as the durable record the map is rebuilt from.
//!
//! When the localizer moves keyframes, only the blocks those keyframes were fused into at their old pose and are
//! fused into at their new pose are rebuilt, on a background thread, from the kept keyframes that touch them. Posed
//! depth fused into a rebuilt block is gone from it, since it carried the uncorrected pose. Moves that arrive during a
//! rebuild wait for it to land and start the next one; nothing else starts a rebuild.
//!
//! The surface mesh of the blocks that changed, and the occupancy grid and topology, are made on a background thread from a
//! snapshot of the field, at most once a period each, so that input is never held up by them. The surface of every
//! mesh block as last sent is kept, so that a receiver that missed a message can be sent the whole surface as a
//! snapshot, numbered in the same sequence as the changes.

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use glam::IVec3;
use perception_wire::localization::{CameraNanos, MapToOdom};
use perception_wire::navigation::{MapVersion, UnixNanos};
use perception_wire::posed_depth::PosedDepthFrame;
use perception_wire::space_map::{OccupancyGrid, KeepOut, MeshBlock, MeshBlocks, MeshBlocksKind, MeshSequence, Topology};
use rustc_hash::FxHashSet;

use crate::derived::{DerivedMaps, Sliced};
use crate::keep_out::{KeepOutZones, ObjectUpdate};
use crate::kept::{Footprint, KeptKeyframes, Rebuildable};
use crate::keyframe::{DepthKeyframe, FrameId, KeyframePose, PosedDepth, Tracking};
use crate::settings::DenseMapSettings;
use crate::block_mesher::MeshPass;
use crate::tsdf::{BlockField, BLOCK_EDGE_VOXELS};

#[derive(Debug, thiserror::Error)]
pub enum MapperError {
    #[error("the background rebuild of map {0} panicked")]
    RebuildPanicked(MapVersion),
    #[error("the background meshing and slicing panicked")]
    DerivingPanicked,
}

/// What happened to a keyframe or pose on arrival.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arrival {
    Fused(FrameId),
    /// Held until its pose (or keyframe) arrives.
    Waiting,
    /// Held, pushing out the oldest one waiting, which is lost.
    WaitingDropped(FrameId),
    /// A lost pose: the keyframe is not fused.
    Skipped(FrameId),
    /// A new pose for a keyframe already fused; its blocks will be rebuilt.
    Moved(FrameId),
    /// The pose a keyframe already fused already has, one too close to it to rebuild for, or a lost pose.
    AlreadyAtPose(FrameId),
    /// A moved keyframe the map holds nothing of: never fused, or let go too long ago to be remembered.
    NotHeld(FrameId),
}

/// Why a posed depth frame was not fused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unfused {
    /// No correction from the odom frame to the map has arrived yet.
    NoCorrection,
    /// The latest correction is for another odom frame.
    OtherOdomFrame { frame: CameraNanos, correction: CameraNanos },
    /// The frame was captured before the oldest correction held for its odom frame became valid.
    BeforeCorrections { captured: CameraNanos, oldest_valid_from: CameraNanos },
}

/// What happened to a posed depth frame on arrival.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PosedDepthArrival {
    Fused,
    Unfused(Unfused),
}

/// What a tick produced.
#[derive(Clone, Debug, PartialEq)]
pub enum MapOutput {
    OccupancyGrid(OccupancyGrid),
    Topology(Topology),
    MeshBlocks(MeshBlocks),
}

/// A rebuild of the blocks moved keyframes touch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RebuildReport {
    pub map_version: MapVersion,
    pub blocks: usize,
}

/// What a tick did besides its outputs, for the node to log.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TickReport {
    pub rebuilt: Option<RebuildReport>,
    pub rebuild_started: Option<RebuildReport>,
    /// Keyframes let go since the last tick.
    pub let_go: usize,
    /// Set when there was something new to slice but no floor seen yet.
    pub no_floor: bool,
    pub slice_time: Option<Duration>,
    pub mesh_time: Option<Duration>,
    /// Blocks meshed in the pass collected whose surface stayed within the resend rule of the one sent, so were not
    /// sent again.
    pub mesh_blocks_kept_as_sent: usize,
}

struct Rebuild<F> {
    thread: JoinHandle<F>,
    version: MapVersion,
    blocks: FxHashSet<IVec3>,
    /// The kept keyframes the rebuild fuses; ones kept later are fused into its blocks when it lands.
    fused: FxHashSet<FrameId>,
}

/// One meshing and slicing pass on a snapshot of the field.
struct Derivation {
    mesh: Option<(MeshPass, Duration)>,
    slice: Option<(Option<Sliced>, Duration)>,
}

struct Deriving {
    thread: JoinHandle<(DerivedMaps, Derivation)>,
    map_version: MapVersion,
    built: UnixNanos,
}

pub struct DenseMapper<F: BlockField> {
    settings: DenseMapSettings,
    field: F,
    /// The latest map version of the poses and corrections fused.
    map_version: Option<MapVersion>,
    awaiting_pose: VecDeque<DepthKeyframe>,
    early_poses: VecDeque<KeyframePose>,
    kept: KeptKeyframes,
    /// The latest corrections of the latest correction's odom frame, oldest valid first.
    corrections: VecDeque<MapToOdom>,
    /// Blocks to rebuild once the running rebuild lands, and the map version of the moves that asked for them.
    to_rebuild: FxHashSet<IVec3>,
    to_rebuild_version: Option<MapVersion>,
    rebuild: Option<Rebuild<F>>,
    let_go: usize,
    keep_out: KeepOutZones,
    /// Idle between passes; the running pass holds it otherwise.
    derived: Option<DerivedMaps>,
    deriving: Option<Deriving>,
    /// Blocks whose surface may have changed since it was last meshed.
    unmeshed: FxHashSet<IVec3>,
    changed_since_slice: bool,
    last_mesh: Option<Instant>,
    last_slice: Option<Instant>,
    run_started: UnixNanos,
    next_sequence: MeshSequence,
    /// Every mesh block's surface as last sent, for blocks that have one: what a receiver that missed no message
    /// holds.
    sent_surface: BTreeMap<[i32; 3], MeshBlock>,
    /// The map version of the surface message last sent; the localizer's first version, 0, before any.
    surface_version: MapVersion,
}

impl<F: BlockField> DenseMapper<F> {
    /// A map whose mesh_blocks messages carry `run_started`, the time this run of the map started.
    pub fn new(settings: DenseMapSettings, field: F, run_started: UnixNanos) -> DenseMapper<F> {
        DenseMapper {
            keep_out: KeepOutZones::new(&settings.keep_out),
            derived: Some(DerivedMaps::new(&settings)),
            kept: KeptKeyframes::new(settings.keyframes),
            settings,
            field,
            map_version: None,
            awaiting_pose: VecDeque::new(),
            early_poses: VecDeque::new(),
            corrections: VecDeque::new(),
            to_rebuild: FxHashSet::default(),
            to_rebuild_version: None,
            rebuild: None,
            let_go: 0,
            deriving: None,
            unmeshed: FxHashSet::default(),
            changed_since_slice: false,
            last_mesh: None,
            last_slice: None,
            run_started,
            next_sequence: MeshSequence(0),
            sent_surface: BTreeMap::new(),
            surface_version: MapVersion(0),
        }
    }

    /// Every mesh block's surface as last sent, in block order: what a receiver that missed no surface message holds.
    pub fn sent_surface(&self) -> impl ExactSizeIterator<Item = &MeshBlock> {
        self.sent_surface.values()
    }

    /// The whole surface as last sent, as a snapshot that replaces every block a receiver holds, numbered next in
    /// the run's sequence.
    pub fn mesh_snapshot(&mut self, now_unix: UnixNanos) -> MeshBlocks {
        let mesh_blocks = self.sent_surface.values().cloned().collect();
        self.surface_message(MeshBlocksKind::Snapshot, self.surface_version, now_unix, mesh_blocks)
    }

    fn surface_message(&mut self, kind: MeshBlocksKind, map_version: MapVersion, built: UnixNanos, blocks: Vec<MeshBlock>) -> MeshBlocks {
        let sequence = self.next_sequence;
        self.next_sequence = sequence.next();
        MeshBlocks {
            map_version,
            built,
            run_started: self.run_started,
            sequence,
            kind,
            voxel_m: self.field.voxel_m(),
            block_voxels: BLOCK_EDGE_VOXELS as u32,
            blocks,
        }
    }

    pub fn field(&self) -> &F {
        &self.field
    }

    pub fn kept_keyframes(&self) -> usize {
        self.kept.len()
    }

    pub fn kept_frame_ids(&self) -> Vec<FrameId> {
        self.kept.frame_ids().collect()
    }

    pub fn is_rebuilding(&self) -> bool {
        self.rebuild.is_some()
    }

    /// The blocks waiting for a rebuild to start.
    pub fn blocks_to_rebuild(&self) -> &FxHashSet<IVec3> {
        &self.to_rebuild
    }

    /// The blocks a kept keyframe was fused into.
    pub fn footprint(&self, frame_id: FrameId) -> Option<&Footprint> {
        self.kept.keyframe(frame_id).map(|(_, _, footprint)| footprint)
    }

    pub fn on_keyframe(&mut self, keyframe: DepthKeyframe) -> Arrival {
        if let Some(at) = self.early_poses.iter().position(|p| p.frame_id == keyframe.frame_id) {
            let pose = self.early_poses.remove(at).expect("found above");
            return self.fuse_keyframe(keyframe, pose);
        }
        self.awaiting_pose.push_back(keyframe);
        if self.awaiting_pose.len() > self.settings.buffers.max_awaiting_pose {
            let dropped = self.awaiting_pose.pop_front().expect("over the limit");
            return Arrival::WaitingDropped(dropped.frame_id);
        }
        Arrival::Waiting
    }

    /// A keyframe's pose: it fuses the keyframe, or moves it when it is already fused.
    pub fn on_keyframe_pose(&mut self, pose: KeyframePose) -> Arrival {
        if self.kept.contains(pose.frame_id) {
            return match pose.tracking {
                Tracking::Tracking | Tracking::Relocalized | Tracking::Restarted => self.move_keyframe(pose),
                Tracking::Lost => Arrival::AlreadyAtPose(pose.frame_id),
            };
        }
        if let Some(at) = self.awaiting_pose.iter().position(|k| k.frame_id == pose.frame_id) {
            let keyframe = self.awaiting_pose.remove(at).expect("found above");
            return self.fuse_keyframe(keyframe, pose);
        }
        if let Some(footprint) = self.kept.take_forgotten(pose.frame_id) {
            return self.queue_rebuild(pose.frame_id, pose.map_version, &footprint, &[]);
        }
        self.early_poses.push_back(pose);
        if self.early_poses.len() > self.settings.buffers.max_awaiting_pose {
            let dropped = self.early_poses.pop_front().expect("over the limit");
            return Arrival::WaitingDropped(dropped.frame_id);
        }
        Arrival::Waiting
    }

    /// Keyframes an optimization of the pose graph moved, each to its new pose.
    pub fn on_pose_graph_update(&mut self, moved: impl IntoIterator<Item = KeyframePose>) -> Vec<Arrival> {
        moved
            .into_iter()
            .map(|pose| {
                if self.kept.contains(pose.frame_id) {
                    return self.move_keyframe(pose);
                }
                if let Some(footprint) = self.kept.take_forgotten(pose.frame_id) {
                    return self.queue_rebuild(pose.frame_id, pose.map_version, &footprint, &[]);
                }
                if let Some(early) = self.early_poses.iter_mut().find(|p| p.frame_id == pose.frame_id) {
                    *early = pose;
                    return Arrival::Waiting;
                }
                Arrival::NotHeld(pose.frame_id)
            })
            .collect()
    }

    /// Holds a correction from the odom frame to the map, in order of the time it is valid from, replacing one
    /// valid from the same time; a correction of a new odom frame lets go of the old frame's, and one of an older odom
    /// frame than those held is let go itself, as no frame of it will be fused again. At most
    /// `buffers.max_corrections` are held, the oldest let go first.
    pub fn on_map_to_odom(&mut self, correction: MapToOdom) {
        if self.corrections.back().is_some_and(|latest| latest.odom_started != correction.odom_started) {
            if correction.odom_started.0 < self.corrections.back().expect("checked above").odom_started.0 {
                return;
            }
            self.corrections.clear();
        }
        match self.corrections.binary_search_by_key(&correction.valid_from.0, |held| held.valid_from.0) {
            Ok(at) => self.corrections[at] = correction,
            Err(at) => self.corrections.insert(at, correction),
        }
        while self.corrections.len() > self.settings.buffers.max_corrections {
            self.corrections.pop_front();
        }
    }

    /// Fuses a posed depth frame at its odometry pose corrected into the map by the latest correction valid at its
    /// capture time, when the corrections held are for its odom frame and one was valid by then.
    pub fn on_posed_depth(&mut self, frame: &PosedDepthFrame) -> PosedDepthArrival {
        let Some(latest) = self.corrections.back() else { return PosedDepthArrival::Unfused(Unfused::NoCorrection) };
        if latest.odom_started != frame.odom_started {
            return PosedDepthArrival::Unfused(Unfused::OtherOdomFrame { frame: frame.odom_started, correction: latest.odom_started });
        }
        let Some(correction) = self.corrections.iter().rev().find(|held| held.valid_from.0 <= frame.camera_ns.0).copied() else {
            let oldest_valid_from = self.corrections.front().expect("not empty").valid_from;
            return PosedDepthArrival::Unfused(Unfused::BeforeCorrections { captured: frame.camera_ns, oldest_valid_from });
        };
        self.field.fuse_into_footprint(&PosedDepth::corrected(frame, &correction));
        self.note_version(correction.map_version);
        self.changed_since_slice = true;
        PosedDepthArrival::Fused
    }

    pub fn on_object_update(&mut self, update: &ObjectUpdate) -> Option<KeepOut> {
        self.keep_out.apply(update)
    }

    fn note_version(&mut self, version: MapVersion) {
        self.map_version = Some(self.map_version.map_or(version, |v| v.max(version)));
    }

    fn fuse_keyframe(&mut self, keyframe: DepthKeyframe, pose: KeyframePose) -> Arrival {
        let frame_id = keyframe.frame_id;
        match pose.tracking {
            // A restarted keyframe is a map keyframe: its pose is dead reckoning, and the pose graph updates that tie
            // its run to the map move it, as they move any keyframe fused.
            Tracking::Tracking | Tracking::Relocalized | Tracking::Restarted => {}
            Tracking::Lost => return Arrival::Skipped(frame_id),
        }
        // The localizer places keyframes in order, so any older one still waiting was skipped and never will be.
        self.awaiting_pose.retain(|waiting| waiting.frame_id > frame_id);
        let footprint = self.field.fuse_into_footprint(&PosedDepth::new(&keyframe, &pose));
        self.note_version(pose.map_version);
        self.let_go += self.kept.keep(Arc::new(keyframe), pose, footprint.into_iter().collect()).len();
        self.changed_since_slice = true;
        Arrival::Fused(frame_id)
    }

    fn move_keyframe(&mut self, pose: KeyframePose) -> Arrival {
        let (keyframe, fused_pose, _) = self.kept.keyframe(pose.frame_id).expect("a kept keyframe");
        let k = &self.settings.keyframes;
        let moved_m = fused_pose.translation.distance(pose.translation);
        let turned_rad = (pose.yaw_rad - fused_pose.yaw_rad + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
        if moved_m < k.rebuild_min_move_m && turned_rad.abs() < k.rebuild_min_turn_deg.to_radians() {
            return Arrival::AlreadyAtPose(pose.frame_id);
        }
        let new_footprint: Footprint = self.field.footprint(&PosedDepth::new(keyframe, &pose)).into_iter().collect();
        let (old_footprint, let_go) = self.kept.move_kept(pose.frame_id, pose, new_footprint.clone());
        self.let_go += let_go.len();
        self.queue_rebuild(pose.frame_id, pose.map_version, &old_footprint, &new_footprint)
    }

    fn queue_rebuild(&mut self, frame_id: FrameId, version: MapVersion, old: &[IVec3], new: &[IVec3]) -> Arrival {
        self.to_rebuild.extend(old.iter().chain(new).copied());
        self.to_rebuild_version = Some(self.to_rebuild_version.map_or(version, |v| v.max(version)));
        self.note_version(version);
        Arrival::Moved(frame_id)
    }

    /// Collects finished background work, starts what is due, and returns what is ready to send.
    pub fn on_tick(&mut self, now: Instant, now_unix: UnixNanos) -> Result<(Vec<MapOutput>, TickReport), MapperError> {
        self.tick(now, now_unix, false)
    }

    /// As [`DenseMapper::on_tick`], waiting for each piece of background work to finish, so that what this tick
    /// starts is sent by this tick.
    pub fn on_tick_settled(&mut self, now: Instant, now_unix: UnixNanos) -> Result<(Vec<MapOutput>, TickReport), MapperError> {
        self.tick(now, now_unix, true)
    }

    fn tick(&mut self, now: Instant, now_unix: UnixNanos, wait: bool) -> Result<(Vec<MapOutput>, TickReport), MapperError> {
        let mut report = TickReport { let_go: std::mem::take(&mut self.let_go), ..TickReport::default() };
        let mut outputs = Vec::new();
        self.collect_derived(wait, &mut outputs, &mut report)?;
        if self.rebuild.as_ref().is_some_and(|r| wait || r.thread.is_finished()) {
            report.rebuilt = Some(self.land_rebuild()?);
        }
        if !self.to_rebuild.is_empty() && self.rebuild.is_none() {
            report.rebuild_started = Some(self.start_rebuild());
            if wait {
                report.rebuilt = Some(self.land_rebuild()?);
            }
        }
        self.unmeshed.extend(self.field.take_changed_blocks());
        if self.derived.is_some() {
            self.start_derived(now, now_unix);
            if wait {
                self.collect_derived(true, &mut outputs, &mut report)?;
            }
        }
        Ok((outputs, report))
    }

    fn start_rebuild(&mut self) -> RebuildReport {
        let blocks = std::mem::take(&mut self.to_rebuild);
        let version = self.to_rebuild_version.take().expect("blocks to rebuild come with their map version");
        let sources: Vec<Rebuildable> = self.kept.touching(&blocks);
        let fused = sources.iter().map(|(keyframe, _)| keyframe.frame_id).collect();
        let mut field = self.field.cleared();
        let within = blocks.clone();
        let thread = std::thread::spawn(move || {
            for (keyframe, pose) in &sources {
                field.fuse_within(&PosedDepth::new(keyframe, pose), &within);
            }
            field
        });
        let report = RebuildReport { map_version: version, blocks: blocks.len() };
        self.rebuild = Some(Rebuild { thread, version, blocks, fused });
        report
    }

    fn land_rebuild(&mut self) -> Result<RebuildReport, MapperError> {
        let rebuild = self.rebuild.take().expect("a running rebuild");
        let mut rebuilt = rebuild.thread.join().map_err(|_| MapperError::RebuildPanicked(rebuild.version))?;
        for (keyframe, pose) in self.kept.touching(&rebuild.blocks) {
            if !rebuild.fused.contains(&keyframe.frame_id) {
                rebuilt.fuse_within(&PosedDepth::new(&keyframe, &pose), &rebuild.blocks);
            }
        }
        self.field.replace_blocks(&rebuilt, &rebuild.blocks);
        self.changed_since_slice = true;
        Ok(RebuildReport { map_version: rebuild.version, blocks: rebuild.blocks.len() })
    }

    /// Starts a meshing and slicing pass on a snapshot of the field when either is due.
    fn start_derived(&mut self, now: Instant, now_unix: UnixNanos) {
        let due = |last: Option<Instant>, period_s: f32| last.is_none_or(|last| now.duration_since(last) >= Duration::from_secs_f32(period_s));
        let mesh = !self.unmeshed.is_empty() && due(self.last_mesh, self.settings.mesh.period_s);
        let slice = self.changed_since_slice && due(self.last_slice, self.settings.occupancy_grid.period_s);
        if !(mesh || slice) {
            return;
        }
        let map_version = self.map_version.expect("a field with changes has fused depth");
        let unmeshed = mesh.then(|| std::mem::take(&mut self.unmeshed));
        if mesh {
            self.last_mesh = Some(now);
        }
        if slice {
            self.last_slice = Some(now);
            self.changed_since_slice = false;
        }
        let field = self.field.snapshot();
        let mut derived = self.derived.take().expect("no pass running");
        let thread = std::thread::spawn(move || {
            let mesh = unmeshed.map(|unmeshed| {
                let started = Instant::now();
                (derived.mesh(&field, &unmeshed), started.elapsed())
            });
            let slice = slice.then(|| {
                let started = Instant::now();
                (derived.slice(&field, map_version, now_unix), started.elapsed())
            });
            (derived, Derivation { mesh, slice })
        });
        self.deriving = Some(Deriving { thread, map_version, built: now_unix });
    }

    /// The outputs of a finished meshing and slicing pass, or of the running one when `wait`.
    fn collect_derived(&mut self, wait: bool, outputs: &mut Vec<MapOutput>, report: &mut TickReport) -> Result<(), MapperError> {
        if !self.deriving.as_ref().is_some_and(|d| wait || d.thread.is_finished()) {
            return Ok(());
        }
        let deriving = self.deriving.take().expect("a running pass");
        let (derived, derivation) = deriving.thread.join().map_err(|_| MapperError::DerivingPanicked)?;
        self.derived = Some(derived);
        if let Some((pass, time)) = derivation.mesh {
            report.mesh_time = Some(time);
            report.mesh_blocks_kept_as_sent = pass.kept_as_sent;
            if !pass.blocks.is_empty() {
                for mesh_block in &pass.blocks {
                    match mesh_block.triangles.is_empty() {
                        true => self.sent_surface.remove(&mesh_block.block.to_array()),
                        false => self.sent_surface.insert(mesh_block.block.to_array(), mesh_block.clone()),
                    };
                }
                self.surface_version = deriving.map_version;
                let changes = self.surface_message(MeshBlocksKind::Changes, deriving.map_version, deriving.built, pass.blocks);
                outputs.push(MapOutput::MeshBlocks(changes));
            }
        }
        if let Some((sliced, time)) = derivation.slice {
            match sliced {
                Some(sliced) => {
                    report.slice_time = Some(time);
                    outputs.push(MapOutput::OccupancyGrid(sliced.occupancy_grid));
                    outputs.extend(sliced.topology.map(MapOutput::Topology));
                }
                None => report.no_floor = true,
            }
        }
        Ok(())
    }
}
