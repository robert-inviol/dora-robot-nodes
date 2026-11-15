//! The keyframes the map keeps to rebuild from: at most one per pose cell and at most a set number in all, each with
//! the blocks it was fused into. Keyframes that are let go are remembered by those blocks for a while, so that when
//! one moves the blocks it left behind can be rebuilt without it.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use glam::{IVec3, IVec4};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::keyframe::{DepthKeyframe, FrameId, KeyframePose};
use crate::settings::KeyframeSettings;

/// The blocks a keyframe was fused into at its pose.
pub type Footprint = Arc<[IVec3]>;

/// A cell of keyframe poses: x, y and z in cells of `pose_cell_m`, and yaw in cells of about `pose_cell_yaw_deg`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PoseCell(IVec4);

struct Kept {
    keyframe: Arc<DepthKeyframe>,
    pose: KeyframePose,
    footprint: Footprint,
    cell: PoseCell,
}

/// A kept keyframe as a rebuild uses it.
pub type Rebuildable = (Arc<DepthKeyframe>, KeyframePose);

pub struct KeptKeyframes {
    settings: KeyframeSettings,
    /// Yaw cells round the circle.
    yaw_cells: i32,
    /// Oldest first: frame ids rise with capture time.
    kept: BTreeMap<FrameId, Kept>,
    by_cell: FxHashMap<PoseCell, FrameId>,
    forgotten: FxHashMap<FrameId, Footprint>,
    /// Oldest first.
    forgotten_order: VecDeque<FrameId>,
}

impl KeptKeyframes {
    pub fn new(settings: KeyframeSettings) -> KeptKeyframes {
        KeptKeyframes {
            yaw_cells: (360.0 / settings.pose_cell_yaw_deg).round().max(1.0) as i32,
            settings,
            kept: BTreeMap::new(),
            by_cell: FxHashMap::default(),
            forgotten: FxHashMap::default(),
            forgotten_order: VecDeque::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.kept.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kept.is_empty()
    }

    pub fn contains(&self, frame_id: FrameId) -> bool {
        self.kept.contains_key(&frame_id)
    }

    pub fn frame_ids(&self) -> impl Iterator<Item = FrameId> + '_ {
        self.kept.keys().copied()
    }

    pub fn cell_of(&self, pose: &KeyframePose) -> PoseCell {
        let cell = (pose.translation / self.settings.pose_cell_m).floor().as_ivec3();
        let turns = (pose.yaw_rad.to_degrees() / 360.0).rem_euclid(1.0);
        let yaw = ((turns * self.yaw_cells as f32).floor() as i32).rem_euclid(self.yaw_cells);
        PoseCell(cell.extend(yaw))
    }

    pub fn keyframe(&self, frame_id: FrameId) -> Option<(&Arc<DepthKeyframe>, &KeyframePose, &Footprint)> {
        self.kept.get(&frame_id).map(|k| (&k.keyframe, &k.pose, &k.footprint))
    }

    /// Keeps a fused keyframe, letting go of the one in its pose cell and, past the cap, the oldest redundant one.
    /// Returns the keyframes let go.
    pub fn keep(&mut self, keyframe: Arc<DepthKeyframe>, pose: KeyframePose, footprint: Footprint) -> Vec<FrameId> {
        let frame_id = keyframe.frame_id;
        let cell = self.cell_of(&pose);
        self.kept.insert(frame_id, Kept { keyframe, pose, footprint, cell });
        let mut let_go = self.claim_cell(frame_id, cell);
        while self.kept.len() > self.settings.max_kept {
            let oldest = self.oldest_redundant().unwrap_or_else(|| *self.kept.keys().next().expect("over the cap"));
            self.let_go(oldest);
            let_go.push(oldest);
        }
        let_go
    }

    /// Moves a kept keyframe to a new pose fused into a new footprint, and returns its old footprint and the
    /// keyframes let go because it moved into their pose cell.
    pub fn move_kept(&mut self, frame_id: FrameId, pose: KeyframePose, footprint: Footprint) -> (Footprint, Vec<FrameId>) {
        let cell = self.cell_of(&pose);
        let kept = self.kept.get_mut(&frame_id).expect("a kept keyframe");
        let old_cell = kept.cell;
        kept.pose = pose;
        kept.cell = cell;
        let old = std::mem::replace(&mut kept.footprint, footprint);
        if self.by_cell.get(&old_cell) == Some(&frame_id) {
            self.by_cell.remove(&old_cell);
        }
        (old, self.claim_cell(frame_id, cell))
    }

    /// The footprint of a keyframe let go, which it gives up: once its blocks are rebuilt nothing of it is left.
    pub fn take_forgotten(&mut self, frame_id: FrameId) -> Option<Footprint> {
        let footprint = self.forgotten.remove(&frame_id)?;
        self.forgotten_order.retain(|id| *id != frame_id);
        Some(footprint)
    }

    /// Every kept keyframe fused into any of `blocks`, oldest first.
    pub fn touching(&self, blocks: &FxHashSet<IVec3>) -> Vec<Rebuildable> {
        self.kept
            .values()
            .filter(|k| k.footprint.iter().any(|block| blocks.contains(block)))
            .map(|k| (k.keyframe.clone(), k.pose))
            .collect()
    }

    /// Makes `frame_id` the one keyframe of `cell`, letting go of the older one there.
    fn claim_cell(&mut self, frame_id: FrameId, cell: PoseCell) -> Vec<FrameId> {
        match self.by_cell.insert(cell, frame_id) {
            Some(other) if other != frame_id => {
                let older = other.min(frame_id);
                self.by_cell.insert(cell, other.max(frame_id));
                self.let_go(older);
                vec![older]
            }
            _ => Vec::new(),
        }
    }

    /// The oldest kept keyframe with another kept keyframe in a pose cell one step away in any of x, y, z and yaw.
    fn oldest_redundant(&self) -> Option<FrameId> {
        self.kept.iter().find(|(_, kept)| self.has_neighbour(kept.cell)).map(|(id, _)| *id)
    }

    fn has_neighbour(&self, PoseCell(cell): PoseCell) -> bool {
        let steps = [-1, 0, 1];
        steps.iter().any(|&dx| {
            steps.iter().any(|&dy| {
                steps.iter().any(|&dz| {
                    steps.iter().any(|&dyaw| {
                        let yaw = (cell.w + dyaw).rem_euclid(self.yaw_cells);
                        let near = PoseCell(IVec4::new(cell.x + dx, cell.y + dy, cell.z + dz, yaw));
                        near != PoseCell(cell) && self.by_cell.contains_key(&near)
                    })
                })
            })
        })
    }

    fn let_go(&mut self, frame_id: FrameId) {
        let kept = self.kept.remove(&frame_id).expect("a kept keyframe");
        if self.by_cell.get(&kept.cell) == Some(&frame_id) {
            self.by_cell.remove(&kept.cell);
        }
        self.forgotten.insert(frame_id, kept.footprint);
        self.forgotten_order.push_back(frame_id);
        while self.forgotten_order.len() > self.settings.max_forgotten_footprints {
            let oldest = self.forgotten_order.pop_front().expect("over the cap");
            self.forgotten.remove(&oldest);
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;
    use perception_wire::navigation::{MapVersion, UnixNanos};

    use super::*;
    use crate::keyframe::Tracking;

    fn keyframe(id: u64) -> Arc<DepthKeyframe> {
        Arc::new(DepthKeyframe {
            frame_id: FrameId(id),
            captured: UnixNanos(id as i64),
            width: 1,
            height: 1,
            intrinsics: depth_perception::CameraIntrinsics { fx: 1.0, fy: 1.0, ppx: 0.0, ppy: 0.0 },
            specific_force: Vec3::new(0.0, -9.81, 0.0),
            depth_mm: Arc::from(vec![1000u16]),
        })
    }

    fn pose(id: u64, x: f32, yaw_deg: f32) -> KeyframePose {
        KeyframePose { frame_id: FrameId(id), map_version: MapVersion(1), translation: Vec3::new(x, 0.1, 0.1), yaw_rad: yaw_deg.to_radians(), tracking: Tracking::Tracking }
    }

    fn footprint(x: i32) -> Footprint {
        Arc::from(vec![IVec3::new(x, 0, 0)])
    }

    fn settings(max_kept: usize) -> KeyframeSettings {
        KeyframeSettings { max_kept, max_forgotten_footprints: 2, ..KeyframeSettings::default() }
    }

    #[test]
    fn a_keyframe_in_an_occupied_pose_cell_replaces_the_older_one() {
        let mut kept = KeptKeyframes::new(settings(10));
        assert!(kept.keep(keyframe(1), pose(1, 0.05, 1.0), footprint(1)).is_empty());
        assert_eq!(kept.keep(keyframe(2), pose(2, 0.1, 5.0), footprint(2)), vec![FrameId(1)]);
        assert_eq!(kept.frame_ids().collect::<Vec<_>>(), vec![FrameId(2)]);
        assert_eq!(kept.take_forgotten(FrameId(1)).as_deref(), Some(&[IVec3::new(1, 0, 0)][..]));
        assert!(kept.keep(keyframe(3), pose(3, 0.1, 50.0), footprint(3)).is_empty(), "another yaw cell");
    }

    #[test]
    fn past_the_cap_the_oldest_keyframe_with_a_neighbour_goes() {
        let mut kept = KeptKeyframes::new(settings(3));
        kept.keep(keyframe(1), pose(1, 0.1, 0.0), footprint(1));
        kept.keep(keyframe(2), pose(2, 5.0, 0.0), footprint(2));
        kept.keep(keyframe(3), pose(3, 5.35, 0.0), footprint(3));
        assert_eq!(kept.keep(keyframe(4), pose(4, 9.0, 0.0), footprint(4)), vec![FrameId(2)], "1 and 4 stand alone; 2 is next to 3");
        assert_eq!(kept.len(), 3);
        assert_eq!(kept.keep(keyframe(5), pose(5, 20.0, 0.0), footprint(5)), vec![FrameId(1)], "none redundant: the oldest");
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn yaw_cells_wrap_round_the_circle() {
        let kept = KeptKeyframes::new(settings(3));
        assert_eq!(kept.cell_of(&pose(1, 0.1, 359.0)), kept.cell_of(&pose(1, 0.1, -1.0)));
        assert_ne!(kept.cell_of(&pose(1, 0.1, 359.0)), kept.cell_of(&pose(1, 0.1, 1.0)));
    }

    #[test]
    fn forgotten_footprints_are_bounded() {
        let mut kept = KeptKeyframes::new(settings(1));
        for id in 1..=4 {
            kept.keep(keyframe(id), pose(id, id as f32 * 10.0, 0.0), footprint(id as i32));
        }
        assert!(kept.take_forgotten(FrameId(1)).is_none(), "beyond the remembered footprints");
        assert!(kept.take_forgotten(FrameId(2)).is_some() && kept.take_forgotten(FrameId(3)).is_some());
    }

    #[test]
    fn a_moved_keyframe_hands_back_its_old_footprint_and_takes_its_new_cell() {
        let mut kept = KeptKeyframes::new(settings(10));
        kept.keep(keyframe(1), pose(1, 0.1, 0.0), footprint(1));
        kept.keep(keyframe(2), pose(2, 3.0, 0.0), footprint(2));
        let (old, let_go) = kept.move_kept(FrameId(1), pose(1, 3.05, 0.0), footprint(7));
        assert_eq!(&old[..], &[IVec3::new(1, 0, 0)]);
        assert_eq!(let_go, vec![FrameId(1)], "it moved into the newer keyframe's cell");
        let blocks: FxHashSet<IVec3> = [IVec3::new(2, 0, 0)].into_iter().collect();
        assert_eq!(kept.touching(&blocks).iter().map(|(k, _)| k.frame_id).collect::<Vec<_>>(), vec![FrameId(2)]);
    }
}
