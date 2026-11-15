//! The keyframe clouds kept for the page, and their carving: when a keyframe is placed, the points of every other
//! placed cloud that its depth sees through are removed, so whatever has moved away since, such as a person
//! walking through, does not stay in the view.

use std::collections::{HashMap, VecDeque};
use std::io::Write;

use bytes::{BufMut, Bytes, BytesMut};
use depth_perception::{CameraIntrinsics, FloorFrame};
use dora_node_api::arrow::buffer::ScalarBuffer;
use flate2::write::GzEncoder;
use flate2::Compression;
use glam::{Mat3, Vec3};
use perception_wire::localization::{FrameId, Pose4};
use perception_wire::Keyframe;

use crate::cloud::{CloudError, CloudSettings, KeyframeCloud};
use crate::view::{mm, FloorPose, Layer, Outgoing, Update};

/// The first byte of a binary message to the page that removes points from one cloud.
pub const CARVE_MESSAGE: u8 = 3;

/// Keyframes whose pose has not come yet are kept for carving up to this many.
const MAX_UNPLACED_VIEWS: usize = 16;

/// Where the localizer put a keyframe, and whether it is placed at all: a lost keyframe is not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    pub pose: Pose4,
    pub placed: bool,
}

impl Placement {
    pub fn floor_pose(&self) -> FloorPose {
        FloorPose { x_m: mm(self.pose.x_m), y_m: mm(self.pose.y_m), z_m: mm(self.pose.z_m), yaw_rad: self.pose.yaw_rad }
    }

    fn translation(&self) -> Vec3 {
        Vec3::new(self.pose.x_m, self.pose.y_m, self.pose.z_m)
    }
}

/// What a keyframe's depth saw, kept until it has carved the other clouds.
struct DepthView {
    width: usize,
    height: usize,
    intrinsics: CameraIntrinsics,
    floor_to_camera: Mat3,
    depth_mm: ScalarBuffer<u16>,
}

impl DepthView {
    fn of(keyframe: &Keyframe) -> DepthView {
        let floor = FloorFrame::from_gravity_reaction(keyframe.gravity_reaction);
        let camera_to_floor = Mat3::from_cols(floor.to_floor(Vec3::X), floor.to_floor(Vec3::Y), floor.to_floor(Vec3::Z));
        DepthView {
            width: keyframe.width,
            height: keyframe.height,
            intrinsics: keyframe.intrinsics,
            floor_to_camera: camera_to_floor.transpose(),
            depth_mm: keyframe.depth_mm.clone(),
        }
    }

    /// Whether this view's depth at the point lies further than `margin_m` beyond it: the view saw through it.
    fn sees_past(&self, camera: Vec3, settings: &CloudSettings) -> bool {
        if camera.z <= settings.min_range_m {
            return false;
        }
        let k = self.intrinsics;
        let column = (k.fx * camera.x / camera.z + k.ppx + 0.5).floor();
        let row = (k.fy * camera.y / camera.z + k.ppy + 0.5).floor();
        if column < 0.0 || row < 0.0 || column >= self.width as f32 || row >= self.height as f32 {
            return false;
        }
        let depth_mm = self.depth_mm[row as usize * self.width + column as usize];
        depth_mm != 0 && depth_mm as f32 * 0.001 > camera.z + settings.carve_margin_m
    }
}

struct KeptCloud {
    positions_mm: Vec<[i16; 3]>,
    removed: Vec<bool>,
    encoded: Bytes,
}

pub struct CloudStore {
    settings: CloudSettings,
    max_keyframes: usize,
    /// Oldest first.
    order: VecDeque<FrameId>,
    clouds: HashMap<FrameId, KeptCloud>,
    placements: HashMap<FrameId, Placement>,
    unplaced_views: VecDeque<(FrameId, DepthView)>,
}

impl CloudStore {
    pub fn new(settings: CloudSettings, max_keyframes: usize) -> CloudStore {
        CloudStore {
            settings,
            max_keyframes,
            order: VecDeque::new(),
            clouds: HashMap::new(),
            placements: HashMap::new(),
            unplaced_views: VecDeque::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    pub fn placement(&self, frame_id: FrameId) -> Option<Placement> {
        self.placements.get(&frame_id).copied()
    }

    /// How many of a cloud's points are carved away.
    pub fn removed_count(&self, frame_id: FrameId) -> Option<usize> {
        self.clouds.get(&frame_id).map(|cloud| cloud.removed.iter().filter(|r| **r).count())
    }

    /// Every cloud with its placement and the points carved from it, oldest first.
    pub fn snapshot(&self) -> Vec<Outgoing> {
        let mut out = Vec::new();
        for frame_id in &self.order {
            let cloud = &self.clouds[frame_id];
            out.push(Outgoing::binary(Layer::Clouds, cloud.encoded.clone()));
            if let Some(placement) = self.placements.get(frame_id) {
                out.push(placement_update(*frame_id, placement));
            }
            let removed: Vec<u32> = cloud.removed.iter().enumerate().filter(|(_, r)| **r).map(|(i, _)| i as u32).collect();
            if !removed.is_empty() {
                out.push(Outgoing::binary(Layer::Clouds, encode_carve(*frame_id, &removed)));
            }
        }
        out
    }

    pub fn on_keyframe(&mut self, keyframe: &Keyframe) -> Result<Vec<Outgoing>, CloudError> {
        let cloud = KeyframeCloud::from_keyframe(keyframe, &self.settings)?;
        let frame_id = cloud.frame_id;
        let encoded = cloud.encode();
        let mut out = vec![Outgoing::binary(Layer::Clouds, encoded.clone())];
        let removed = vec![false; cloud.positions_mm.len()];
        self.clouds.insert(frame_id, KeptCloud { positions_mm: cloud.positions_mm, removed, encoded });
        self.order.push_back(frame_id);
        while self.order.len() > self.max_keyframes {
            let dropped = self.order.pop_front().expect("over the limit");
            self.clouds.remove(&dropped);
            self.placements.remove(&dropped);
            out.push(Outgoing::json(Layer::Clouds, &Update::KeyframeDropped { frame_id: dropped.0 }));
        }
        let view = DepthView::of(keyframe);
        match self.placements.get(&frame_id).copied() {
            Some(placement) if placement.placed => out.extend(self.carve(frame_id, &view, &placement)),
            Some(_) => {}
            None => {
                self.unplaced_views.push_back((frame_id, view));
                if self.unplaced_views.len() > MAX_UNPLACED_VIEWS {
                    self.unplaced_views.pop_front();
                }
            }
        }
        Ok(out)
    }

    /// Places a keyframe's cloud; a newly placed keyframe still waiting to carve carves the other clouds.
    pub fn place(&mut self, frame_id: FrameId, placement: Placement) -> Vec<Outgoing> {
        self.placements.insert(frame_id, placement);
        let mut out = vec![placement_update(frame_id, &placement)];
        if let Some(at) = self.unplaced_views.iter().position(|(waiting, _)| *waiting == frame_id) {
            let (_, view) = self.unplaced_views.remove(at).expect("found above");
            if placement.placed {
                out.extend(self.carve(frame_id, &view, &placement));
            }
        }
        out
    }

    fn carve(&mut self, seen_from: FrameId, view: &DepthView, at: &Placement) -> Vec<Outgoing> {
        let mut out = Vec::new();
        let (from_yaw, from_position) = (at.pose.yaw_rad, at.translation());
        for frame_id in &self.order {
            let Some(placement) = self.placements.get(frame_id).filter(|p| p.placed && *frame_id != seen_from) else { continue };
            // camera = F^T Rz(-yaw_view) (Rz(yaw_cloud) q + t_cloud - t_view)
            let turn = view.floor_to_camera * Mat3::from_rotation_z(placement.pose.yaw_rad - from_yaw);
            let shift = view.floor_to_camera * (Mat3::from_rotation_z(-from_yaw) * (placement.translation() - from_position));
            let cloud = self.clouds.get_mut(frame_id).expect("every ordered frame has a cloud");
            let mut newly: Vec<u32> = Vec::new();
            for (i, position) in cloud.positions_mm.iter().enumerate() {
                if cloud.removed[i] {
                    continue;
                }
                let point = Vec3::new(position[0] as f32, position[1] as f32, position[2] as f32) * 0.001;
                if view.sees_past(turn * point + shift, &self.settings) {
                    cloud.removed[i] = true;
                    newly.push(i as u32);
                }
            }
            if !newly.is_empty() {
                out.push(Outgoing::binary(Layer::Clouds, encode_carve(*frame_id, &newly)));
            }
        }
        out
    }
}

fn placement_update(frame_id: FrameId, placement: &Placement) -> Outgoing {
    let update = Update::KeyframePose { frame_id: frame_id.0, pose: placement.floor_pose(), placed: placement.placed };
    Outgoing::json(Layer::Clouds, &update)
}

/// `CARVE_MESSAGE`, then gzip of: frame id (u64), count (u32), every removed point's index in the cloud (u32).
/// Little endian.
pub fn encode_carve(frame_id: FrameId, removed: &[u32]) -> Bytes {
    let mut body = BytesMut::with_capacity(12 + 4 * removed.len());
    body.put_u64_le(frame_id.0);
    body.put_u32_le(removed.len() as u32);
    removed.iter().for_each(|&index| body.put_u32_le(index));
    let mut encoder = GzEncoder::new(vec![CARVE_MESSAGE], Compression::fast());
    encoder.write_all(&body).expect("writing to memory");
    Bytes::from(encoder.finish().expect("writing to memory"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud::tests::wall_keyframe;
    use crate::view::Payload;

    fn level(x_m: f32, y_m: f32) -> Placement {
        Placement { pose: Pose4 { x_m, y_m, z_m: 1.0, yaw_rad: 0.0 }, placed: true }
    }

    fn carves(out: &[Outgoing]) -> usize {
        out.iter().filter(|o| matches!(&o.payload, Payload::Binary(b) if b[0] == CARVE_MESSAGE)).count()
    }

    #[test]
    fn a_later_view_through_an_earlier_surface_carves_it_away() {
        let mut store = CloudStore::new(CloudSettings::DEFAULT, 10);
        // A person 1.5 m ahead, then gone: the same camera sees the wall 3 m ahead.
        store.on_keyframe(&wall_keyframe(1, 1500, [200, 0, 0])).unwrap();
        store.place(FrameId(1), level(0.0, 0.0));
        store.on_keyframe(&wall_keyframe(2, 3000, [0, 0, 200])).unwrap();
        let out = store.place(FrameId(2), level(0.0, 0.0));
        assert_eq!(carves(&out), 1);
        assert_eq!(store.removed_count(FrameId(1)), Some(4), "every point of the person: 2 x 2 at stride 4");
        assert_eq!(store.removed_count(FrameId(2)), Some(0), "the wall behind stays");
    }

    #[test]
    fn a_surface_seen_again_where_it_was_is_kept() {
        let mut store = CloudStore::new(CloudSettings::DEFAULT, 10);
        store.on_keyframe(&wall_keyframe(1, 3000, [0; 3])).unwrap();
        store.place(FrameId(1), level(0.0, 0.0));
        store.on_keyframe(&wall_keyframe(2, 3050, [0; 3])).unwrap();
        let out = store.place(FrameId(2), level(0.0, 0.0));
        assert_eq!(carves(&out), 0, "5 cm further is within the margin");
    }

    #[test]
    fn a_view_from_further_back_sees_past_nothing_in_front_of_it() {
        let mut store = CloudStore::new(CloudSettings::DEFAULT, 10);
        store.on_keyframe(&wall_keyframe(1, 2000, [0; 3])).unwrap();
        store.place(FrameId(1), level(0.0, 1.0));
        // From 1 m further back the same wall is 3 m away.
        store.on_keyframe(&wall_keyframe(2, 3000, [0; 3])).unwrap();
        assert_eq!(carves(&store.place(FrameId(2), level(0.0, 0.0))), 0);
    }

    #[test]
    fn a_lost_keyframe_carves_nothing_and_a_late_cloud_carves_once_placed() {
        let mut store = CloudStore::new(CloudSettings::DEFAULT, 10);
        store.on_keyframe(&wall_keyframe(1, 1500, [0; 3])).unwrap();
        store.place(FrameId(1), level(0.0, 0.0));
        store.on_keyframe(&wall_keyframe(2, 3000, [0; 3])).unwrap();
        assert_eq!(carves(&store.place(FrameId(2), Placement { placed: false, ..level(0.0, 0.0) })), 0);
        // Its pose arrives before the keyframe: it carves as the keyframe comes.
        store.place(FrameId(3), level(0.0, 0.0));
        assert_eq!(carves(&store.on_keyframe(&wall_keyframe(3, 3000, [0; 3])).unwrap()), 1);
    }

    #[test]
    fn a_late_page_gets_each_cloud_with_its_placement_and_its_carving() {
        let mut store = CloudStore::new(CloudSettings::DEFAULT, 10);
        store.on_keyframe(&wall_keyframe(1, 1500, [0; 3])).unwrap();
        store.place(FrameId(1), level(0.0, 0.0));
        store.on_keyframe(&wall_keyframe(2, 3000, [0; 3])).unwrap();
        store.place(FrameId(2), level(0.0, 0.0));
        let snapshot = store.snapshot();
        assert_eq!(snapshot.len(), 5, "cloud, pose and carving of the first; cloud and pose of the second");
        assert!(snapshot.iter().all(|o| o.layer == Layer::Clouds));
        assert_eq!(carves(&snapshot), 1);
    }
}
