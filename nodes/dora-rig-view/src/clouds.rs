//! The keyframe clouds kept for the page, and their space carving.
//!
//! A cloud is drawn only for a keyframe the localizer selected into the map (`KeyframeSelection::Selected` on its
//! keyframe pose), so one cloud per map keyframe. A keyframe it did not select, localized near a map keyframe, is a
//! repeat view: its cloud is neither kept nor sent. So a still camera draws one cloud, and a new keyframe from the
//! same place changes nothing the page shows, but for the points its depth carves away.
//!
//! Space carving: every registered keyframe's depth, repeat views included, votes on each point of every other drawn
//! cloud. A view whose depth at the point lies more than the carving margin beyond it saw through it; one whose
//! depth lies within the margin of it saw it again. A point is removed once more views have seen through it than
//! have seen it, counting the view it came from as one that saw it, so whatever has moved away since, such as a
//! person who walked through, is removed after two views, while depth noise, which sees through a point now and
//! then but sees it most of the time, removes nothing. A removed point stays removed.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::Write;

use bytes::{BufMut, Bytes, BytesMut};
use depth_perception::{CameraIntrinsics, FloorFrame};
use dora_node_api::arrow::buffer::ScalarBuffer;
use flate2::write::GzEncoder;
use flate2::Compression;
use glam::{Mat3, Vec3};
use perception_wire::localization::{FrameId, KeyframePose, KeyframeSelection, UpdatedKeyframePose, Pose4Dof, Tracking};
use perception_wire::Keyframe;

use crate::cloud::{CloudError, CloudSettings, KeyframeCloud};
use crate::view::{mm, FloorPose, Layer, Outgoing, Update};

/// The first byte of a binary message to the page that removes points from one cloud.
pub const SPACE_CARVING_MESSAGE: u8 = 3;

/// Keyframes whose pose has not come yet are held up to this many; past it the oldest is dropped unseen.
const MAX_KEYFRAMES_AWAITING_POSE: usize = 16;

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

/// Where the localizer put a keyframe in the map, what that pose is good for, and whether the localizer selected the
/// keyframe into the map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyframeMapPose {
    pub pose: Pose4Dof,
    pub standing: MapPoseStanding,
    pub selection: KeyframeSelection,
}

impl KeyframeMapPose {
    pub fn of_keyframe_pose(pose: &KeyframePose) -> KeyframeMapPose {
        KeyframeMapPose { pose: pose.pose, standing: MapPoseStanding::of(pose.tracking), selection: pose.selection }
    }

    /// Where a pose graph update moved a keyframe. It keeps the standing and selection its keyframe pose gave it; a
    /// keyframe whose keyframe pose is no longer held is taken as a registered map keyframe, which is what a pose
    /// graph update moves.
    pub fn moved(keyframe: &UpdatedKeyframePose, before: Option<&KeyframeMapPose>) -> KeyframeMapPose {
        let (standing, selection) = before.map_or((MapPoseStanding::Registered, KeyframeSelection::Selected), |held| (held.standing, held.selection));
        KeyframeMapPose { pose: keyframe.pose, standing, selection }
    }

    /// Whether the keyframe's cloud is drawn: it is in the map, so not lost and selected.
    fn new_view(&self) -> bool {
        self.standing.drawn() && self.selection == KeyframeSelection::Selected
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

    /// What this view's depth says about a point in its camera's coordinates.
    fn sighting(&self, camera: Vec3, settings: &CloudSettings) -> Sighting {
        if camera.z <= settings.min_range_m {
            return Sighting::Nothing;
        }
        let k = self.intrinsics;
        let column = (k.fx * camera.x / camera.z + k.ppx + 0.5).floor();
        let row = (k.fy * camera.y / camera.z + k.ppy + 0.5).floor();
        if column < 0.0 || row < 0.0 || column >= self.width as f32 || row >= self.height as f32 {
            return Sighting::Nothing;
        }
        let depth_mm = self.depth_mm[row as usize * self.width + column as usize];
        if depth_mm == 0 {
            return Sighting::Nothing;
        }
        let beyond_m = depth_mm as f32 * 0.001 - camera.z;
        if beyond_m > settings.space_carving_margin_m {
            Sighting::SeenThrough
        } else if beyond_m >= -settings.space_carving_margin_m {
            Sighting::Seen
        } else {
            Sighting::Nothing
        }
    }
}

/// What one view's depth says about one point of a cloud.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sighting {
    /// Its depth lies beyond the point by more than the margin: whatever was there has gone.
    SeenThrough,
    /// Its depth lies within the margin of the point: it is still there.
    Seen,
    /// Out of view, no reading, or something nearer hides it.
    Nothing,
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
        KeyframeMapPose::moved(keyframe, self.get(keyframe.frame_id))
    }

    pub fn len(&self) -> usize {
        self.by_frame.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_frame.is_empty()
    }
}

/// How many views saw a point and how many saw through it.
#[derive(Clone, Copy, Debug, Default)]
struct Votes {
    seen: u16,
    seen_through: u16,
}

impl Votes {
    /// The view the point came from saw it.
    const OWN_VIEW: Votes = Votes { seen: 1, seen_through: 0 };

    fn removed(&self) -> bool {
        self.seen_through > self.seen
    }
}

struct KeptCloud {
    map_pose: KeyframeMapPose,
    positions_mm: Vec<[i16; 3]>,
    votes: Vec<Votes>,
    encoded: Bytes,
}

impl KeptCloud {
    fn removed_indices(&self) -> Vec<u32> {
        self.votes.iter().enumerate().filter(|(_, v)| v.removed()).map(|(i, _)| i as u32).collect()
    }
}

pub struct CloudStore {
    settings: CloudSettings,
    max_keyframes: usize,
    /// The drawn clouds, oldest first.
    order: VecDeque<FrameId>,
    clouds: HashMap<FrameId, KeptCloud>,
    /// Poses of keyframes with no drawn cloud: those still to come, and those placed without one.
    other_poses: KeyframeMapPoses,
    awaiting_pose: VecDeque<PreparedKeyframe>,
    repeat_views: u64,
}

impl CloudStore {
    pub fn new(settings: CloudSettings, max_keyframes: usize) -> CloudStore {
        CloudStore {
            settings,
            max_keyframes,
            order: VecDeque::new(),
            clouds: HashMap::new(),
            other_poses: KeyframeMapPoses::new(max_keyframes),
            awaiting_pose: VecDeque::new(),
            repeat_views: 0,
        }
    }

    /// How many clouds are drawn.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// How many keyframes were repeat views, not selected into the map, whose clouds were not kept.
    pub fn repeat_views(&self) -> u64 {
        self.repeat_views
    }

    pub fn map_pose(&self, frame_id: FrameId) -> Option<KeyframeMapPose> {
        self.clouds.get(&frame_id).map(|cloud| cloud.map_pose).or_else(|| self.other_poses.get(frame_id).copied())
    }

    /// The map pose a pose graph update moves a keyframe to, keeping the standing and selection it has.
    pub fn moved(&self, keyframe: &UpdatedKeyframePose) -> KeyframeMapPose {
        KeyframeMapPose::moved(keyframe, self.map_pose(keyframe.frame_id).as_ref())
    }

    /// How many poses of keyframes without a drawn cloud are kept.
    pub fn other_pose_count(&self) -> usize {
        self.other_poses.len()
    }

    /// How many of a drawn cloud's points are carved away; None for a keyframe whose cloud is not drawn.
    pub fn removed_count(&self, frame_id: FrameId) -> Option<usize> {
        self.clouds.get(&frame_id).map(|cloud| cloud.votes.iter().filter(|v| v.removed()).count())
    }

    /// Every drawn cloud with its map pose and the points carved from it, oldest first.
    pub fn snapshot(&self) -> Vec<Outgoing> {
        let mut out = Vec::new();
        for frame_id in &self.order {
            let cloud = &self.clouds[frame_id];
            out.push(Outgoing::binary(Layer::Clouds, cloud.encoded.clone()));
            out.push(map_pose_update(*frame_id, &cloud.map_pose));
            let removed = cloud.removed_indices();
            if !removed.is_empty() {
                out.push(Outgoing::binary(Layer::Clouds, encode_space_carving(*frame_id, &removed)));
            }
        }
        out
    }

    /// Takes a prepared keyframe: placed at once when its map pose has come, else held until it does.
    pub fn on_keyframe(&mut self, prepared: PreparedKeyframe) -> Vec<Outgoing> {
        if let Some(map_pose) = self.other_poses.get(prepared.frame_id).copied() {
            self.other_poses.remove(prepared.frame_id);
            return self.place(prepared, map_pose);
        }
        self.awaiting_pose.push_back(prepared);
        if self.awaiting_pose.len() > MAX_KEYFRAMES_AWAITING_POSE {
            self.awaiting_pose.pop_front();
        }
        Vec::new()
    }

    /// Sets a keyframe's map pose: a keyframe held for it is placed now; a drawn cloud moves to it.
    pub fn set_map_pose(&mut self, frame_id: FrameId, map_pose: KeyframeMapPose) -> Vec<Outgoing> {
        if let Some(cloud) = self.clouds.get_mut(&frame_id) {
            cloud.map_pose = map_pose;
            return vec![map_pose_update(frame_id, &map_pose)];
        }
        if let Some(at) = self.awaiting_pose.iter().position(|waiting| waiting.frame_id == frame_id) {
            let prepared = self.awaiting_pose.remove(at).expect("found above");
            return self.place(prepared, map_pose);
        }
        self.other_poses.insert(frame_id, map_pose);
        Vec::new()
    }

    /// A keyframe at its map pose: its depth votes on the drawn clouds when it is registered, and its cloud is drawn
    /// when the localizer selected it into the map.
    fn place(&mut self, prepared: PreparedKeyframe, map_pose: KeyframeMapPose) -> Vec<Outgoing> {
        let PreparedKeyframe { frame_id, positions_mm, encoded, view } = prepared;
        let mut out = if map_pose.standing.carves() { self.space_carve(&view, &map_pose) } else { Vec::new() };
        if !map_pose.new_view() {
            self.repeat_views += matches!(map_pose.selection, KeyframeSelection::NotSelected { .. }) as u64;
            self.other_poses.insert(frame_id, map_pose);
            return out;
        }
        out.push(Outgoing::binary(Layer::Clouds, encoded.clone()));
        out.push(map_pose_update(frame_id, &map_pose));
        let votes = vec![Votes::OWN_VIEW; positions_mm.len()];
        self.clouds.insert(frame_id, KeptCloud { map_pose, positions_mm, votes, encoded });
        self.order.push_back(frame_id);
        while self.order.len() > self.max_keyframes {
            let dropped = self.order.pop_front().expect("over the limit");
            self.clouds.remove(&dropped);
            out.push(Outgoing::json(Layer::Clouds, &Update::KeyframeDropped { frame_id: dropped.0 }));
        }
        out
    }

    /// One registered view's votes on every other drawn cloud, and the points they newly remove.
    fn space_carve(&mut self, view: &DepthView, at: &KeyframeMapPose) -> Vec<Outgoing> {
        let mut out = Vec::new();
        let (from_yaw, from_position) = (at.pose.yaw_rad, at.translation());
        for frame_id in &self.order {
            let cloud = self.clouds.get_mut(frame_id).expect("every drawn frame has a cloud");
            let map_pose = cloud.map_pose;
            // camera = F^T Rz(-yaw_view) (Rz(yaw_cloud) q + t_cloud - t_view)
            let turn = view.floor_to_camera * Mat3::from_rotation_z(map_pose.pose.yaw_rad - from_yaw);
            let shift = view.floor_to_camera * (Mat3::from_rotation_z(-from_yaw) * (map_pose.translation() - from_position));
            let mut newly: Vec<u32> = Vec::new();
            for (i, (position, votes)) in cloud.positions_mm.iter().zip(&mut cloud.votes).enumerate() {
                if votes.removed() {
                    continue;
                }
                let point = Vec3::new(position[0] as f32, position[1] as f32, position[2] as f32) * 0.001;
                match view.sighting(turn * point + shift, &self.settings) {
                    Sighting::SeenThrough => votes.seen_through = votes.seen_through.saturating_add(1),
                    Sighting::Seen => votes.seen = votes.seen.saturating_add(1),
                    Sighting::Nothing => continue,
                }
                if votes.removed() {
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
    use depth_perception::CameraIntrinsics;

    use super::*;
    use crate::cloud::tests::wall_keyframe;
    use crate::view::Payload;

    /// A map keyframe: registered, level, and selected into the map.
    fn level(x_m: f32, y_m: f32) -> KeyframeMapPose {
        KeyframeMapPose { pose: Pose4Dof { x_m, y_m, z_m: 1.0, yaw_rad: 0.0 }, standing: MapPoseStanding::Registered, selection: KeyframeSelection::Selected }
    }

    /// A keyframe at the origin the localizer did not select, localized near the map keyframe `reference`.
    fn again(reference: u64) -> KeyframeMapPose {
        KeyframeMapPose { selection: KeyframeSelection::NotSelected { reference: FrameId(reference) }, ..level(0.0, 0.0) }
    }

    fn store() -> CloudStore {
        CloudStore::new(CloudSettings::DEFAULT, 10)
    }

    fn prepared(keyframe: &Keyframe) -> PreparedKeyframe {
        PreparedKeyframe::of(keyframe, &CloudSettings::DEFAULT).unwrap()
    }

    fn count(out: &[Outgoing], kind: u8) -> usize {
        out.iter().filter(|o| matches!(&o.payload, Payload::Binary(b) if b[0] == kind)).count()
    }

    fn carves(out: &[Outgoing]) -> usize {
        count(out, SPACE_CARVING_MESSAGE)
    }

    fn clouds_sent(out: &[Outgoing]) -> usize {
        count(out, crate::cloud::CLOUD_MESSAGE)
    }

    /// A keyframe and its pose, in either order; everything the store sent for them.
    fn view(store: &mut CloudStore, keyframe: &Keyframe, at: KeyframeMapPose) -> Vec<Outgoing> {
        let mut out = store.on_keyframe(prepared(keyframe));
        out.extend(store.set_map_pose(FrameId(keyframe.frame_id), at));
        out
    }

    /// A level camera 64 x 48 px with the depth given, its image grey.
    fn keyframe_with_depth(frame_id: u64, depth_mm: Vec<u16>) -> Keyframe {
        let (width, height) = (64usize, 48usize);
        let mut jpeg = Vec::new();
        jpeg_encoder::Encoder::new(&mut jpeg, 90).encode(&vec![128u8; width * height * 3], width as u16, height as u16, jpeg_encoder::ColorType::Rgb).unwrap();
        Keyframe {
            frame_id,
            captured_unix_ns: 1,
            width,
            height,
            intrinsics: CameraIntrinsics { fx: 32.0, fy: 32.0, ppx: 32.0, ppy: 24.0 },
            specific_force: Vec3::new(0.0, -9.8, 0.0),
            depth_mm: depth_mm.into(),
            color_jpeg: jpeg,
        }
    }

    /// A repeatable stream of normally distributed numbers (xorshift64* with Box-Muller).
    struct Noise(u64);

    impl Noise {
        fn uniform(&mut self) -> f32 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            ((self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 + 0.5) / (1u64 << 24) as f32
        }

        fn normal(&mut self, sigma: f32) -> f32 {
            sigma * (-2.0 * self.uniform().ln()).sqrt() * (std::f32::consts::TAU * self.uniform()).cos()
        }
    }

    /// A wall at 4.5 m with each reading off by 8 cm, as the D455 reads it there, fresh noise each frame.
    fn noisy_far_wall(frame_id: u64, noise: &mut Noise) -> Keyframe {
        keyframe_with_depth(frame_id, (0..64 * 48).map(|_| (4500.0 + 1000.0 * noise.normal(0.08)) as u16).collect())
    }

    #[test]
    fn a_still_camera_draws_one_cloud_that_settles_within_seconds_under_depth_noise() {
        let mut store = store();
        let mut noise = Noise(3);
        let mut sent = Vec::new();
        let mut removed_by_view = Vec::new();
        for frame_id in 1..=120 {
            let at = if frame_id == 1 { level(0.0, 0.0) } else { again(1) };
            sent.extend(view(&mut store, &noisy_far_wall(frame_id, &mut noise), at));
            removed_by_view.push(store.removed_count(FrameId(1)).unwrap());
        }
        assert_eq!((clouds_sent(&sent), store.len(), store.repeat_views()), (1, 1, 119), "one cloud, the rest repeat views");
        let cloud = &store.clouds[&FrameId(1)];
        let points = cloud.votes.len();
        // A single view seeing past a point by the margin would remove it: most points, given 119 views.
        let ever_seen_through = cloud.votes.iter().filter(|v| v.seen_through > 0).count();
        assert!(ever_seen_through * 10 > points * 8, "{ever_seen_through} of {points} points were seen through at least once");
        // The majority removes the readings that noise put well in front of the wall, and then nothing more.
        let (after_10, after_60, after_120) = (removed_by_view[9], removed_by_view[59], removed_by_view[119]);
        assert!(after_120 * 3 < points, "{after_120} of {points} points removed");
        assert!(after_120 - after_60 <= points / 100, "{after_10} removed after 10 views, {after_60} after 60, {after_120} after 120");
    }

    /// A person 1.5 m ahead in the middle columns of a wall at 3 m.
    fn person_before_wall(frame_id: u64) -> Keyframe {
        keyframe_with_depth(frame_id, (0..64 * 48).map(|i| if (24..40).contains(&(i % 64)) { 1500 } else { 3000 }).collect())
    }

    fn wall(frame_id: u64) -> Keyframe {
        keyframe_with_depth(frame_id, vec![3000; 64 * 48])
    }

    #[test]
    fn someone_who_walks_away_is_carved_after_two_views_and_the_wall_stays() {
        let mut store = store();
        view(&mut store, &person_before_wall(1), level(0.0, 0.0));
        let person_points = store.clouds[&FrameId(1)].positions_mm.iter().filter(|p| p[1] == 1500).count();
        assert!(person_points > 0);
        assert_eq!(carves(&view(&mut store, &wall(2), again(1))), 0, "one view through them is not yet a majority");
        let out = view(&mut store, &wall(3), again(1));
        assert_eq!((carves(&out), clouds_sent(&out)), (1, 0), "the second view carves them, and adds no cloud");
        assert_eq!(store.removed_count(FrameId(1)), Some(person_points), "every point of the person, none of the wall");
    }

    #[test]
    fn someone_who_stood_for_a_while_is_carved_once_more_views_saw_through_them_than_saw_them() {
        let mut store = store();
        view(&mut store, &person_before_wall(1), level(0.0, 0.0));
        for frame_id in 2..=4 {
            view(&mut store, &person_before_wall(frame_id), again(1));
        }
        let removed_after = |store: &mut CloudStore, frame_id: u64| {
            view(store, &wall(frame_id), again(1));
            store.removed_count(FrameId(1)).unwrap()
        };
        assert_eq!((5..=8).map(|frame_id| removed_after(&mut store, frame_id)).collect::<Vec<_>>(), [0, 0, 0, 0]);
        assert!(removed_after(&mut store, 9) > 0, "seen in 4 views, seen through in 5");
    }

    #[test]
    fn a_keyframe_the_localizer_selected_is_drawn_and_one_it_did_not_is_not() {
        let mut store = store();
        assert_eq!(clouds_sent(&view(&mut store, &wall(1), level(0.0, 0.0))), 1);
        let near = KeyframeMapPose { pose: level(0.05, 0.0).pose, ..again(1) };
        assert_eq!(clouds_sent(&view(&mut store, &wall(2), near)), 0, "not selected: a repeat view");
        assert_eq!(clouds_sent(&view(&mut store, &wall(3), level(0.2, 0.0))), 1, "selected: a new view");
        let turned = KeyframeMapPose { pose: Pose4Dof { yaw_rad: 10f32.to_radians(), ..level(0.0, 0.0).pose }, ..level(0.0, 0.0) };
        assert_eq!(clouds_sent(&view(&mut store, &wall(4), turned)), 1, "selected at the same place, turned: a new view");
        assert_eq!((store.len(), store.repeat_views()), (3, 1));
    }

    #[test]
    fn a_still_camera_keeps_one_cloud_however_many_repeat_views_come() {
        let mut store = CloudStore::new(CloudSettings::DEFAULT, 3);
        let sent: usize = (1..=40).map(|frame_id| clouds_sent(&view(&mut store, &wall(frame_id), if frame_id == 1 { level(0.0, 0.0) } else { again(1) }))).sum();
        assert_eq!((sent, store.len()), (1, 1), "more repeat views than clouds kept leave the drawn cloud's pose in place");
    }

    #[test]
    fn a_view_from_further_back_sees_past_nothing_in_front_of_it() {
        let mut store = store();
        view(&mut store, &wall_keyframe(1, 2000, [0; 3]), level(0.0, 1.0));
        // From 1 m further back the same wall is 3 m away.
        assert_eq!(carves(&view(&mut store, &wall_keyframe(2, 3000, [0; 3]), level(0.0, 0.0))), 0);
    }

    #[test]
    fn a_lost_keyframe_is_not_drawn_and_carves_nothing_and_a_pose_may_come_first() {
        let mut store = store();
        view(&mut store, &person_before_wall(1), level(0.0, 0.0));
        let lost = KeyframeMapPose { standing: MapPoseStanding::Lost, selection: KeyframeSelection::NotLocalized, ..level(1.0, 0.0) };
        let out = view(&mut store, &wall(2), lost);
        assert!(out.is_empty(), "a lost keyframe sends nothing");
        // Two poses arrive before their keyframes: they carve as their keyframes come.
        store.set_map_pose(FrameId(3), again(1));
        store.set_map_pose(FrameId(4), again(1));
        store.on_keyframe(prepared(&wall(3)));
        assert_eq!(carves(&store.on_keyframe(prepared(&wall(4)))), 1);
    }

    #[test]
    fn a_restarted_keyframe_is_drawn_carves_nothing_and_keeps_its_standing_when_moved() {
        let mut store = store();
        view(&mut store, &person_before_wall(1), level(0.0, 0.0));
        let restarted = KeyframeMapPose { standing: MapPoseStanding::Restarted, ..level(1.0, 0.0) };
        let out = view(&mut store, &wall(2), restarted);
        assert_eq!((carves(&out), clouds_sent(&out)), (0, 1), "drawn, but a dead-reckoned pose carves nothing");
        let sent: serde_json::Value = match &out[1].payload {
            Payload::Json(text) => serde_json::from_str(text).unwrap(),
            other => panic!("a pose update is json: {other:?}"),
        };
        assert_eq!(sent["drawn"], true, "a restarted keyframe's cloud is drawn");
        let moved = store.moved(&UpdatedKeyframePose { frame_id: FrameId(2), pose: level(1.5, 0.0).pose });
        assert_eq!((moved.standing, moved.selection), (MapPoseStanding::Restarted, KeyframeSelection::Selected), "a move keeps the standing");
        assert_eq!(store.set_map_pose(FrameId(2), moved).len(), 1, "a drawn cloud's move goes to the page");
    }

    #[test]
    fn a_late_page_gets_each_cloud_with_its_map_pose_and_its_carving() {
        let mut store = store();
        view(&mut store, &person_before_wall(1), level(0.0, 0.0));
        view(&mut store, &wall(2), again(1));
        view(&mut store, &wall(3), again(1));
        view(&mut store, &wall(4), level(1.0, 0.0));
        let snapshot = store.snapshot();
        assert_eq!(snapshot.len(), 5, "cloud, pose and carving of the first; cloud and pose of the fourth");
        assert!(snapshot.iter().all(|o| o.layer == Layer::Clouds));
        assert_eq!(carves(&snapshot), 1);
    }

    #[test]
    fn poses_of_keyframes_without_a_drawn_cloud_are_kept_only_for_the_newest_frames() {
        let mut store = CloudStore::new(CloudSettings::DEFAULT, 3);
        for frame_id in 1..=10 {
            store.set_map_pose(FrameId(frame_id), level(0.0, 0.0));
        }
        assert_eq!(store.other_pose_count(), 3);
        assert!(store.map_pose(FrameId(7)).is_none());
        assert!(store.map_pose(FrameId(8)).is_some());
    }
}
