//! Keyframes, their poses and their instance masks arrive separately and in any order; a frame is ready once all
//! three are in.

use std::collections::VecDeque;

use eyre::{bail, Result};
use perception_wire::navigation::MapVersion;
use perception_wire::{Keyframe, InstanceMasks};
use room_map::Pose;
use tracing::{debug, info, warn};

use crate::ObservedFrame;

/// Where the localizer put a keyframe, if it registered it against the map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KeyframeMapPose {
    Registered { pose: Pose, map_version: MapVersion },
    /// The localizer did not register the keyframe against the map: the keyframe has no trustworthy pose.
    NotRegistered,
}

#[derive(Default)]
struct Pending {
    frame_id: u64,
    keyframe: Option<Keyframe>,
    map_pose: Option<KeyframeMapPose>,
    instance_masks: Option<InstanceMasks>,
}

impl Pending {
    fn missing(&self) -> Vec<&'static str> {
        [(self.keyframe.is_none(), "keyframe"), (self.map_pose.is_none(), "pose"), (self.instance_masks.is_none(), "instance masks")]
            .into_iter()
            .filter_map(|(missing, part)| missing.then_some(part))
            .collect()
    }
}

pub struct FrameJoiner {
    capacity: usize,
    pending: VecDeque<Pending>,
}

impl FrameJoiner {
    pub fn new(capacity: usize) -> FrameJoiner {
        assert!(capacity > 0, "the joiner must hold at least one frame");
        FrameJoiner { capacity, pending: VecDeque::with_capacity(capacity) }
    }

    pub fn add_keyframe(&mut self, keyframe: Keyframe) -> Result<Option<ObservedFrame>> {
        let entry = self.entry(keyframe.frame_id);
        entry.keyframe = Some(keyframe);
        self.take_if_complete()
    }

    pub fn add_map_pose(&mut self, frame_id: u64, map_pose: KeyframeMapPose) -> Result<Option<ObservedFrame>> {
        self.entry(frame_id).map_pose = Some(map_pose);
        self.take_if_complete()
    }

    pub fn add_instance_masks(&mut self, instance_masks: InstanceMasks) -> Result<Option<ObservedFrame>> {
        let entry = self.entry(instance_masks.frame_id);
        entry.instance_masks = Some(instance_masks);
        self.take_if_complete()
    }

    fn entry(&mut self, frame_id: u64) -> &mut Pending {
        if let Some(at) = self.pending.iter().position(|p| p.frame_id == frame_id) {
            return &mut self.pending[at];
        }
        if self.pending.len() == self.capacity {
            let dropped = self.pending.pop_front().expect("a full queue has a front");
            // The segmenter skips keyframes as a matter of course; any other part going missing is a fault upstream.
            if dropped.instance_masks.is_some() {
                warn!(frame_id = dropped.frame_id, missing = ?dropped.missing(), "frame dropped before all its parts arrived");
            } else {
                debug!(frame_id = dropped.frame_id, "keyframe dropped: the segmenter skipped it");
            }
        }
        self.pending.push_back(Pending { frame_id, ..Pending::default() });
        self.pending.back_mut().expect("just pushed")
    }

    /// The frame that has all three parts, removed; none when it was lost by the localizer.
    fn take_if_complete(&mut self) -> Result<Option<ObservedFrame>> {
        let Some(at) = self.pending.iter().position(|p| p.missing().is_empty()) else { return Ok(None) };
        let Pending { frame_id, keyframe, map_pose, instance_masks } = self.pending.remove(at).expect("position is in range");
        let (keyframe, instance_masks) = (keyframe.expect("complete"), instance_masks.expect("complete"));
        if (instance_masks.width, instance_masks.height) != (keyframe.width, keyframe.height) {
            bail!(
                "instance masks for frame {frame_id} are {}x{}, the keyframe {}x{}",
                instance_masks.width,
                instance_masks.height,
                keyframe.width,
                keyframe.height
            );
        }
        Ok(match map_pose.expect("complete") {
            KeyframeMapPose::Registered { pose, map_version } => Some(ObservedFrame { keyframe, instance_masks, pose, map_version }),
            KeyframeMapPose::NotRegistered => {
                info!(frame_id, "frame skipped: the localizer did not register it against the map");
                None
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use depth_perception::CameraIntrinsics;
    use dora_node_api::arrow::buffer::ScalarBuffer;
    use glam::Vec3;

    use super::*;

    fn keyframe(frame_id: u64) -> Keyframe {
        Keyframe {
            frame_id,
            captured_unix_ns: 1_000 + frame_id as i64,
            width: 2,
            height: 1,
            intrinsics: CameraIntrinsics { fx: 1.0, fy: 1.0, ppx: 1.0, ppy: 0.5 },
            specific_force: Vec3::new(0.0, -9.81, 0.0),
            depth_mm: ScalarBuffer::from(vec![1000u16, 1000]),
            color_jpeg: Vec::new(),
        }
    }

    fn instance_masks(frame_id: u64, width: usize) -> InstanceMasks {
        InstanceMasks {
            frame_id,
            width,
            height: 1,
            labels: vec!["box".to_owned()],
            scores: vec![0.9],
            instance_ids: ScalarBuffer::from(vec![0i16; width]),
        }
    }

    const REGISTERED: KeyframeMapPose = KeyframeMapPose::Registered { pose: Pose::IDENTITY, map_version: MapVersion(4) };

    #[test]
    fn a_frame_is_ready_once_its_three_parts_are_in_whatever_their_order() {
        let mut joiner = FrameJoiner::new(4);
        assert!(joiner.add_instance_masks(instance_masks(7, 2)).unwrap().is_none());
        assert!(joiner.add_keyframe(keyframe(8)).unwrap().is_none());
        assert!(joiner.add_map_pose(7, REGISTERED).unwrap().is_none());
        let frame = joiner.add_keyframe(keyframe(7)).unwrap().expect("all three parts of frame 7 are in");
        assert_eq!((frame.keyframe.frame_id, frame.instance_masks.frame_id, frame.map_version), (7, 7, MapVersion(4)));
    }

    #[test]
    fn a_frame_the_localizer_lost_is_skipped() {
        let mut joiner = FrameJoiner::new(4);
        joiner.add_keyframe(keyframe(7)).unwrap();
        joiner.add_instance_masks(instance_masks(7, 2)).unwrap();
        assert!(joiner.add_map_pose(7, KeyframeMapPose::NotRegistered).unwrap().is_none());
        assert!(joiner.pending.is_empty());
    }

    #[test]
    fn the_oldest_waiting_frame_is_dropped_when_the_joiner_is_full() {
        let mut joiner = FrameJoiner::new(2);
        for frame_id in 1..=3 {
            joiner.add_keyframe(keyframe(frame_id)).unwrap();
            joiner.add_map_pose(frame_id, REGISTERED).unwrap();
        }
        assert!(joiner.add_instance_masks(instance_masks(1, 2)).unwrap().is_none(), "frame 1 was dropped");
        assert!(joiner.add_instance_masks(instance_masks(3, 2)).unwrap().is_some());
    }

    #[test]
    fn instance_masks_of_another_size_than_their_keyframe_are_an_error() {
        let mut joiner = FrameJoiner::new(4);
        joiner.add_keyframe(keyframe(7)).unwrap();
        joiner.add_map_pose(7, REGISTERED).unwrap();
        let error = joiner.add_instance_masks(instance_masks(7, 3)).err().unwrap();
        assert!(error.to_string().contains("are 3x1, the keyframe 2x1"), "{error}");
    }
}
