//! The keyframe clouds kept for the page, and their space carving: when a registered keyframe gets its map pose, the
//! points of every other drawn cloud that its depth sees through are removed, so whatever has moved away since, such
//! as a person walking through, does not stay in the view.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::Write;

use bytes::{BufMut, Bytes, BytesMut};
use depth_perception::{CameraIntrinsics, FloorFrame};
use dora_node_api::arrow::buffer::ScalarBuffer;
use flate2::write::GzEncoder;
use flate2::Compression;
use glam::{Mat3, Vec3};
use perception_wire::localization::{FrameId, KeyframePose, UpdatedKeyframePose, Pose4Dof, Tracking};
use perception_wire::Keyframe;

use crate::cloud::{CloudError, CloudSettings, KeyframeCloud};
use crate::view::{mm, FloorPose, Layer, Outgoing, Update};

/// The first byte of a binary message to the page that removes points from one cloud.
pub const SPACE_CARVING_MESSAGE: u8 = 3;

/// Keyframes whose pose has not come yet are kept for space carving up to this many.
const MAX_VIEWS_AWAITING_POSE: usize = 16;

/// What a keyframe's map pose is good for on the page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapPoseStanding {
    /// Registered against the map (tracking or relocalized): drawn, and its depth carves the other clouds.
    Registered,
    /// Starts a new run after tracking was lost: a map keyframe at a dead-reckoned pose that later pose graph updates
    /// move. Drawn at that pose, but its depth carves nothing, since the pose is not registered against the map.
    Restarted,
    /// Lost: dead reckoning outside the map. Not drawn, and its depth carves nothing.
    Lost,
}

impl MapPoseStanding {
    pub fn of(tracking: Tracking) -> MapPoseStanding {
        match tracking {
            Tracking::Tracking | Tracking::Relocalized => MapPoseStanding::Registered,
            Tracking::Restarted => MapPoseStanding::Restarted,
            Tracking::Lost => MapPoseStanding::Lost,
        }
    }

    /// Whether the keyframe's cloud, and anything drawn at its pose, is shown.
    pub fn drawn(self) -> bool {
        match self {
            MapPoseStanding::Registered | MapPoseStanding::Restarted => true,
            MapPoseStanding::Lost => false,
        }
    }

    /// Whether the keyframe's depth carves the other clouds.
    pub fn carves(self) -> bool {
        match self {
            MapPoseStanding::Registered => true,
            MapPoseStanding::Restarted | MapPoseStanding::Lost => false,
        }
    }
}

/// Where the localizer put a keyframe in the map, and what that pose is good for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyframeMapPose {
    pub pose: Pose4Dof,
    pub standing: MapPoseStanding,
}

impl KeyframeMapPose {
    pub fn of_keyframe_pose(pose: &KeyframePose) -> KeyframeMapPose {
        KeyframeMapPose { pose: pose.pose, standing: MapPoseStanding::of(pose.tracking) }
    }

    /// Where a pose graph update moved a keyframe. It keeps the standing its keyframe pose gave it; a keyframe whose
    /// keyframe pose is no longer held is taken as registered.
    pub fn moved(keyframe: &UpdatedKeyframePose, before: Option<MapPoseStanding>) -> KeyframeMapPose {
        KeyframeMapPose { pose: keyframe.pose, standing: before.unwrap_or(MapPoseStanding::Registered) }
    }

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
        let floor = FloorFrame::from_specific_force(keyframe.specific_force);
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
        depth_mm != 0 && depth_mm as f32 * 0.001 > camera.z + settings.space_carving_margin_m
    }
}

/// A keyframe's cloud, encoded, with the depth it carves with: everything about a keyframe that needs no other
/// keyframe, so it is worked out before the store is touched.
pub struct PreparedKeyframe {
    frame_id: FrameId,
    positions_mm: Vec<[i16; 3]>,
    encoded: Bytes,
    view: DepthView,
}

impl PreparedKeyframe {
    pub fn of(keyframe: &Keyframe, settings: &CloudSettings) -> Result<PreparedKeyframe, CloudError> {
        let cloud = KeyframeCloud::from_keyframe(keyframe, settings)?;
        let encoded = cloud.encode();
        Ok(PreparedKeyframe { frame_id: cloud.frame_id, positions_mm: cloud.positions_mm, encoded, view: DepthView::of(keyframe) })
    }

    pub fn frame_id(&self) -> FrameId {
        self.frame_id
    }
}

/// Where the localizer put each keyframe, for the newest `capacity` frame ids: the same bound as the clouds, so
/// poses of keyframes that never arrive do not pile up.
pub struct KeyframeMapPoses {
    capacity: usize,
    by_frame: BTreeMap<FrameId, KeyframeMapPose>,
}

impl KeyframeMapPoses {
    pub fn new(capacity: usize) -> KeyframeMapPoses {
        KeyframeMapPoses { capacity, by_frame: BTreeMap::new() }
    }

    pub fn insert(&mut self, frame_id: FrameId, map_pose: KeyframeMapPose) {
        self.by_frame.insert(frame_id, map_pose);
        while self.by_frame.len() > self.capacity {
            self.by_frame.pop_first();
        }
    }

    pub fn get(&self, frame_id: FrameId) -> Option<&KeyframeMapPose> {
        self.by_frame.get(&frame_id)
    }

    pub fn remove(&mut self, frame_id: FrameId) {
        self.by_frame.remove(&frame_id);
    }

    /// The map pose a pose graph update moves a keyframe to, keeping the standing it has.
    pub fn moved(&self, keyframe: &UpdatedKeyframePose) -> KeyframeMapPose {
        KeyframeMapPose::moved(keyframe, self.get(keyframe.frame_id).map(|held| held.standing))
    }

    pub fn len(&self) -> usize {
        self.by_frame.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_frame.is_empty()
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
    map_poses: KeyframeMapPoses,
    views_awaiting_pose: VecDeque<(FrameId, DepthView)>,
}

impl CloudStore {
    pub fn new(settings: CloudSettings, max_keyframes: usize) -> CloudStore {
        CloudStore {
            settings,
            max_keyframes,
            order: VecDeque::new(),
            clouds: HashMap::new(),
            map_poses: KeyframeMapPoses::new(max_keyframes),
            views_awaiting_pose: VecDeque::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    pub fn map_pose(&self, frame_id: FrameId) -> Option<KeyframeMapPose> {
        self.map_poses.get(frame_id).copied()
    }

    /// The map pose a pose graph update moves a keyframe's cloud to, keeping the standing it has.
    pub fn moved(&self, keyframe: &UpdatedKeyframePose) -> KeyframeMapPose {
        self.map_poses.moved(keyframe)
    }

    /// How many keyframe poses are kept.
    pub fn map_pose_count(&self) -> usize {
        self.map_poses.len()
    }

    /// How many of a cloud's points are carved away.
    pub fn removed_count(&self, frame_id: FrameId) -> Option<usize> {
        self.clouds.get(&frame_id).map(|cloud| cloud.removed.iter().filter(|r| **r).count())
    }

    /// Every cloud with its map pose and the points carved from it, oldest first.
    pub fn snapshot(&self) -> Vec<Outgoing> {
        let mut out = Vec::new();
        for frame_id in &self.order {
            let cloud = &self.clouds[frame_id];
            out.push(Outgoing::binary(Layer::Clouds, cloud.encoded.clone()));
            if let Some(map_pose) = self.map_poses.get(*frame_id) {
                out.push(map_pose_update(*frame_id, map_pose));
            }
            let removed: Vec<u32> = cloud.removed.iter().enumerate().filter(|(_, r)| **r).map(|(i, _)| i as u32).collect();
            if !removed.is_empty() {
                out.push(Outgoing::binary(Layer::Clouds, encode_space_carving(*frame_id, &removed)));
            }
        }
        out
    }

    /// Keeps a prepared keyframe's cloud, dropping the oldest past the limit, and carves the other clouds with its
    /// depth once it has a registered map pose.
    pub fn on_keyframe(&mut self, prepared: PreparedKeyframe) -> Vec<Outgoing> {
        let PreparedKeyframe { frame_id, positions_mm, encoded, view } = prepared;
        let mut out = vec![Outgoing::binary(Layer::Clouds, encoded.clone())];
        let removed = vec![false; positions_mm.len()];
        self.clouds.insert(frame_id, KeptCloud { positions_mm, removed, encoded });
        self.order.push_back(frame_id);
        while self.order.len() > self.max_keyframes {
            let dropped = self.order.pop_front().expect("over the limit");
            self.clouds.remove(&dropped);
            self.map_poses.remove(dropped);
            out.push(Outgoing::json(Layer::Clouds, &Update::KeyframeDropped { frame_id: dropped.0 }));
        }
        match self.map_poses.get(frame_id).copied() {
            Some(map_pose) if map_pose.standing.carves() => out.extend(self.space_carve(frame_id, &view, &map_pose)),
            Some(_) => {}
            None => {
                self.views_awaiting_pose.push_back((frame_id, view));
                if self.views_awaiting_pose.len() > MAX_VIEWS_AWAITING_POSE {
                    self.views_awaiting_pose.pop_front();
                }
            }
        }
        out
    }

    /// Sets a keyframe cloud's map pose; a keyframe still waiting to carve carves the other clouds once its map pose is
    /// registered.
    pub fn set_map_pose(&mut self, frame_id: FrameId, map_pose: KeyframeMapPose) -> Vec<Outgoing> {
        self.map_poses.insert(frame_id, map_pose);
        let mut out = vec![map_pose_update(frame_id, &map_pose)];
        if let Some(at) = self.views_awaiting_pose.iter().position(|(waiting, _)| *waiting == frame_id) {
            let (_, view) = self.views_awaiting_pose.remove(at).expect("found above");
            if map_pose.standing.carves() {
                out.extend(self.space_carve(frame_id, &view, &map_pose));
            }
        }
        out
    }

    fn space_carve(&mut self, seen_from: FrameId, view: &DepthView, at: &KeyframeMapPose) -> Vec<Outgoing> {
        let mut out = Vec::new();
        let (from_yaw, from_position) = (at.pose.yaw_rad, at.translation());
        for frame_id in &self.order {
            let Some(map_pose) = self.map_poses.get(*frame_id).filter(|p| p.standing.drawn() && *frame_id != seen_from) else { continue };
            // camera = F^T Rz(-yaw_view) (Rz(yaw_cloud) q + t_cloud - t_view)
            let turn = view.floor_to_camera * Mat3::from_rotation_z(map_pose.pose.yaw_rad - from_yaw);
            let shift = view.floor_to_camera * (Mat3::from_rotation_z(-from_yaw) * (map_pose.translation() - from_position));
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
                out.push(Outgoing::binary(Layer::Clouds, encode_space_carving(*frame_id, &newly)));
            }
        }
        out
    }
}

fn map_pose_update(frame_id: FrameId, map_pose: &KeyframeMapPose) -> Outgoing {
    let update = Update::KeyframePose { frame_id: frame_id.0, pose: map_pose.floor_pose(), drawn: map_pose.standing.drawn() };
    Outgoing::json(Layer::Clouds, &update)
}

/// `SPACE_CARVING_MESSAGE`, then gzip of: frame id (u64), count (u32), every removed point's index in the cloud (u32).
/// Little endian.
pub fn encode_space_carving(frame_id: FrameId, removed: &[u32]) -> Bytes {
    let mut body = BytesMut::with_capacity(12 + 4 * removed.len());
    body.put_u64_le(frame_id.0);
    body.put_u32_le(removed.len() as u32);
    removed.iter().for_each(|&index| body.put_u32_le(index));
    let mut encoder = GzEncoder::new(vec![SPACE_CARVING_MESSAGE], Compression::fast());
    encoder.write_all(&body).expect("writing to memory");
    Bytes::from(encoder.finish().expect("writing to memory"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud::tests::wall_keyframe;
    use crate::view::Payload;

    fn level(x_m: f32, y_m: f32) -> KeyframeMapPose {
        KeyframeMapPose { pose: Pose4Dof { x_m, y_m, z_m: 1.0, yaw_rad: 0.0 }, standing: MapPoseStanding::Registered }
    }

    fn prepared(keyframe: &Keyframe) -> PreparedKeyframe {
        PreparedKeyframe::of(keyframe, &CloudSettings::DEFAULT).unwrap()
    }

    fn carves(out: &[Outgoing]) -> usize {
        out.iter().filter(|o| matches!(&o.payload, Payload::Binary(b) if b[0] == SPACE_CARVING_MESSAGE)).count()
    }

    #[test]
    fn a_later_view_through_an_earlier_surface_carves_it_away() {
        let mut store = CloudStore::new(CloudSettings::DEFAULT, 10);
        // A person 1.5 m ahead, then gone: the same camera sees the wall 3 m ahead.
        store.on_keyframe(prepared(&wall_keyframe(1, 1500, [200, 0, 0])));
        store.set_map_pose(FrameId(1), level(0.0, 0.0));
        store.on_keyframe(prepared(&wall_keyframe(2, 3000, [0, 0, 200])));
        let out = store.set_map_pose(FrameId(2), level(0.0, 0.0));
        assert_eq!(carves(&out), 1);
        assert_eq!(store.removed_count(FrameId(1)), Some(4), "every point of the person: 2 x 2 at stride 4");
        assert_eq!(store.removed_count(FrameId(2)), Some(0), "the wall behind stays");
    }

    #[test]
    fn a_surface_seen_again_where_it_was_is_kept() {
        let mut store = CloudStore::new(CloudSettings::DEFAULT, 10);
        store.on_keyframe(prepared(&wall_keyframe(1, 3000, [0; 3])));
        store.set_map_pose(FrameId(1), level(0.0, 0.0));
        store.on_keyframe(prepared(&wall_keyframe(2, 3050, [0; 3])));
        let out = store.set_map_pose(FrameId(2), level(0.0, 0.0));
        assert_eq!(carves(&out), 0, "5 cm further is within the margin");
    }

    #[test]
    fn a_view_from_further_back_sees_past_nothing_in_front_of_it() {
        let mut store = CloudStore::new(CloudSettings::DEFAULT, 10);
        store.on_keyframe(prepared(&wall_keyframe(1, 2000, [0; 3])));
        store.set_map_pose(FrameId(1), level(0.0, 1.0));
        // From 1 m further back the same wall is 3 m away.
        store.on_keyframe(prepared(&wall_keyframe(2, 3000, [0; 3])));
        assert_eq!(carves(&store.set_map_pose(FrameId(2), level(0.0, 0.0))), 0);
    }

    #[test]
    fn a_lost_keyframe_carves_nothing_and_a_late_cloud_carves_once_registered() {
        let mut store = CloudStore::new(CloudSettings::DEFAULT, 10);
        store.on_keyframe(prepared(&wall_keyframe(1, 1500, [0; 3])));
        store.set_map_pose(FrameId(1), level(0.0, 0.0));
        store.on_keyframe(prepared(&wall_keyframe(2, 3000, [0; 3])));
        assert_eq!(carves(&store.set_map_pose(FrameId(2), KeyframeMapPose { standing: MapPoseStanding::Lost, ..level(0.0, 0.0) })), 0);
        // Its pose arrives before the keyframe: it carves as the keyframe comes.
        store.set_map_pose(FrameId(3), level(0.0, 0.0));
        assert_eq!(carves(&store.on_keyframe(prepared(&wall_keyframe(3, 3000, [0; 3])))), 1);
    }

    #[test]
    fn a_restarted_keyframe_is_drawn_carves_nothing_and_keeps_its_standing_when_moved() {
        let mut store = CloudStore::new(CloudSettings::DEFAULT, 10);
        store.on_keyframe(prepared(&wall_keyframe(1, 1500, [0; 3])));
        store.set_map_pose(FrameId(1), level(0.0, 0.0));
        store.on_keyframe(prepared(&wall_keyframe(2, 3000, [0; 3])));
        let restarted = KeyframeMapPose { standing: MapPoseStanding::Restarted, ..level(0.0, 0.0) };
        let out = store.set_map_pose(FrameId(2), restarted);
        assert_eq!(carves(&out), 0, "a dead-reckoned pose carves nothing");
        let sent: serde_json::Value = match &out[0].payload {
            Payload::Json(text) => serde_json::from_str(text).unwrap(),
            other => panic!("a pose update is json: {other:?}"),
        };
        assert_eq!(sent["drawn"], true, "a restarted keyframe's cloud is drawn");
        store.on_keyframe(prepared(&wall_keyframe(3, 3000, [0; 3])));
        assert_eq!(carves(&store.set_map_pose(FrameId(3), level(0.0, 0.0))), 1, "a registered view through the same spot carves");
        let moved = store.map_poses.moved(&UpdatedKeyframePose { frame_id: FrameId(2), pose: level(0.5, 0.0).pose });
        assert_eq!(moved.standing, MapPoseStanding::Restarted, "a move keeps the standing");
    }

    #[test]
    fn a_late_page_gets_each_cloud_with_its_map_pose_and_its_carving() {
        let mut store = CloudStore::new(CloudSettings::DEFAULT, 10);
        store.on_keyframe(prepared(&wall_keyframe(1, 1500, [0; 3])));
        store.set_map_pose(FrameId(1), level(0.0, 0.0));
        store.on_keyframe(prepared(&wall_keyframe(2, 3000, [0; 3])));
        store.set_map_pose(FrameId(2), level(0.0, 0.0));
        let snapshot = store.snapshot();
        assert_eq!(snapshot.len(), 5, "cloud, pose and carving of the first; cloud and pose of the second");
        assert!(snapshot.iter().all(|o| o.layer == Layer::Clouds));
        assert_eq!(carves(&snapshot), 1);
    }

    #[test]
    fn poses_of_keyframes_that_never_arrive_are_kept_only_for_the_newest_frames() {
        let mut store = CloudStore::new(CloudSettings::DEFAULT, 3);
        for frame_id in 1..=10 {
            store.set_map_pose(FrameId(frame_id), level(0.0, 0.0));
        }
        assert_eq!(store.map_pose_count(), 3);
        assert!(store.map_pose(FrameId(7)).is_none());
        assert!(store.map_pose(FrameId(8)).is_some());
    }
}
