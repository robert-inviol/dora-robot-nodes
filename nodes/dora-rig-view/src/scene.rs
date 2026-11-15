//! Everything the page draws, kept so a page that connects late gets the whole scene, and turned into the updates
//! that every connected page receives as messages arrive.
//!
//! Odometry and people tracks are in the odom frame; they are carried into the map with the localizer's latest
//! `map_to_odom`, which applies only to the odom frame in use: a correction made before odometry last started
//! belongs to an earlier frame. The robot and the people are sent on each tick at most, whatever their rate.

use std::collections::{BTreeMap, HashMap, VecDeque};

use bytes::Bytes;
use glam::{Quat, Vec3};
use perception_wire::localization::{FrameId, KeyframePose, KeyframePosesMoved, MapToOdom, Pose4, Tracking};
use perception_wire::objects::{ObjectId, ObjectUpdate, ObjectUpdateKind};
use perception_wire::odometry::OdomPose;
use perception_wire::people::{PeopleTracks, PositionFrame, TrackState};
use perception_wire::space_map::{SurfacePatch, SurfacePatches, Topology};
use perception_wire::Keyframe;

use crate::cloud::{CloudError, CloudSettings, KeyframeCloud};
use crate::surface::{encode_patches, BlockGrid, SurfaceError};
use crate::view::{mm, mm3, DoorwayView, FloorPose, NoticeTopic, Outgoing, PersonView, PlaceView, RoomView, Update};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneSettings {
    pub cloud: CloudSettings,
    /// The newest this many keyframe clouds are kept and drawn.
    pub max_keyframes: usize,
}

impl SceneSettings {
    pub const DEFAULT: SceneSettings = SceneSettings { cloud: CloudSettings::DEFAULT, max_keyframes: 600 };
}

pub struct Scene {
    settings: SceneSettings,
    /// Encoded clouds, oldest first.
    clouds: VecDeque<(FrameId, Bytes)>,
    keyframe_poses: HashMap<FrameId, Update>,
    correction: Option<MapToOdom>,
    odometry: Option<OdomPose>,
    robot_due: bool,
    people: Option<PeopleTracks>,
    people_due: bool,
    objects: BTreeMap<ObjectId, Update>,
    topology: Option<Update>,
    notices: BTreeMap<NoticeTopic, String>,
    surface_grid: Option<BlockGrid>,
    /// The latest surface of every block that has one.
    surfaces: BTreeMap<[i32; 3], SurfacePatch>,
}

impl Scene {
    pub fn new(settings: SceneSettings) -> Scene {
        Scene {
            settings,
            clouds: VecDeque::new(),
            keyframe_poses: HashMap::new(),
            correction: None,
            odometry: None,
            robot_due: false,
            people: None,
            people_due: false,
            objects: BTreeMap::new(),
            topology: None,
            notices: BTreeMap::new(),
            surface_grid: None,
            surfaces: BTreeMap::new(),
        }
    }

    pub fn keyframe_count(&self) -> usize {
        self.clouds.len()
    }

    pub fn surface_block_count(&self) -> usize {
        self.surfaces.len()
    }

    /// Everything a newly connected page needs, starting with a reset.
    pub fn snapshot(&self) -> Vec<Outgoing> {
        let mut out = vec![Outgoing::from(&Update::Reset)];
        for (&topic, text) in &self.notices {
            out.push((&Update::Notice { topic, text: Some(text.clone()) }).into());
        }
        for (frame_id, cloud) in &self.clouds {
            out.push(Outgoing::Binary(cloud.clone()));
            if let Some(pose) = self.keyframe_poses.get(frame_id) {
                out.push(pose.into());
            }
        }
        if let Some(grid) = self.surface_grid {
            let encoded = encode_patches(grid, self.surfaces.values()).expect("every kept patch was encoded once already");
            out.push(Outgoing::Binary(encoded));
        }
        out.extend(self.robot().map(|robot| Outgoing::from(&robot)));
        out.extend(self.people_update().map(|people| Outgoing::from(&people)));
        out.extend(self.objects.values().map(Outgoing::from));
        out.extend(self.topology.iter().map(Outgoing::from));
        out
    }

    pub fn on_surface_patches(&mut self, surface: &SurfacePatches) -> Result<Vec<Outgoing>, SurfaceError> {
        let grid = BlockGrid { voxel_m: surface.voxel_m, block_voxels: surface.block_voxels };
        if let Some(kept) = self.surface_grid.filter(|kept| *kept != grid) {
            return Err(SurfaceError::GridChanged { was: kept, now: grid });
        }
        let encoded = encode_patches(grid, surface.patches.iter())?;
        self.surface_grid = Some(grid);
        for patch in &surface.patches {
            let block = patch.block.to_array();
            if patch.triangles.is_empty() {
                self.surfaces.remove(&block);
            } else {
                self.surfaces.insert(block, patch.clone());
            }
        }
        Ok(vec![Outgoing::Binary(encoded)])
    }

    pub fn on_keyframe(&mut self, keyframe: &Keyframe) -> Result<Vec<Outgoing>, CloudError> {
        let cloud = KeyframeCloud::from_keyframe(keyframe, &self.settings.cloud)?;
        let encoded = cloud.encode();
        let mut out = vec![Outgoing::Binary(encoded.clone())];
        self.clouds.push_back((cloud.frame_id, encoded));
        while self.clouds.len() > self.settings.max_keyframes {
            let (dropped, _) = self.clouds.pop_front().expect("over the limit");
            self.keyframe_poses.remove(&dropped);
            out.push((&Update::KeyframeDropped { frame_id: dropped.0 }).into());
        }
        Ok(out)
    }

    pub fn on_keyframe_pose(&mut self, pose: &KeyframePose) -> Vec<Outgoing> {
        self.place_keyframe(pose.frame_id, pose.pose, pose.tracking != Tracking::Lost)
    }

    pub fn on_keyframe_poses_moved(&mut self, moved: &KeyframePosesMoved) -> Vec<Outgoing> {
        moved.keyframes.iter().flat_map(|keyframe| self.place_keyframe(keyframe.frame_id, keyframe.pose, true)).collect()
    }

    fn place_keyframe(&mut self, frame_id: FrameId, pose: Pose4, placed: bool) -> Vec<Outgoing> {
        let pose = FloorPose { x_m: mm(pose.x_m), y_m: mm(pose.y_m), z_m: mm(pose.z_m), yaw_rad: pose.yaw_rad };
        let update = Update::KeyframePose { frame_id: frame_id.0, pose, placed };
        let out = vec![Outgoing::from(&update)];
        self.keyframe_poses.insert(frame_id, update);
        out
    }

    pub fn on_map_to_odom(&mut self, correction: &MapToOdom) {
        self.correction = Some(*correction);
        self.robot_due = true;
        self.people_due = true;
    }

    pub fn on_odom_pose(&mut self, pose: &OdomPose) {
        self.odometry = Some(*pose);
        self.robot_due = true;
    }

    pub fn on_people_tracks(&mut self, tracks: &PeopleTracks) {
        self.people = Some(tracks.clone());
        self.people_due = true;
    }

    pub fn on_object_update(&mut self, update: &ObjectUpdate) -> Vec<Outgoing> {
        let gone = matches!(update.kind, ObjectUpdateKind::Disappeared | ObjectUpdateKind::Merged);
        if gone {
            self.objects.remove(&update.object_id);
            return vec![(&Update::ObjectGone { object_id: update.object_id.0 }).into()];
        }
        let extent = &update.extent;
        let object = Update::Object {
            object_id: update.object_id.0,
            label: update.label.clone(),
            confidence: mm(update.confidence),
            centre: mm3(extent.centre_m),
            footprint: extent.footprint_m.map(|corner| [mm(corner.x), mm(corner.y)]),
            bottom_m: mm(extent.top_m - extent.height_m),
            top_m: mm(extent.top_m),
            voxel_m: update.voxel_m,
            voxels: update.voxels.iter().map(|v| v.to_array()).collect(),
        };
        let out = vec![Outgoing::from(&object)];
        self.objects.insert(update.object_id, object);
        out
    }

    pub fn on_topology(&mut self, topology: &Topology) -> Vec<Outgoing> {
        let update = Update::Topology {
            rooms: topology
                .rooms
                .iter()
                .map(|room| RoomView {
                    room_id: room.room_id.0,
                    polygon: room.polygon.iter().map(|p| [mm(p.x), mm(p.y)]).collect(),
                    floor_m: mm(room.floor_m),
                    ceiling_m: room.ceiling_m.map(mm),
                })
                .collect(),
            doorways: topology
                .doorways
                .iter()
                .map(|d| DoorwayView { position: [mm(d.position.x), mm(d.position.y)], width_m: mm(d.width_m) })
                .collect(),
            places: topology
                .places
                .iter()
                .map(|p| PlaceView {
                    place_id: p.place_id.0,
                    room_id: p.room_id.0,
                    position: [mm(p.position.x), mm(p.position.y)],
                })
                .collect(),
            edges: topology.edges.iter().map(|e| [e.place_a.0, e.place_b.0]).collect(),
        };
        let out = vec![Outgoing::from(&update)];
        self.topology = Some(update);
        out
    }

    /// The robot and the people, where they changed since the last tick, and any notice that changed with them.
    pub fn on_tick(&mut self) -> Vec<Outgoing> {
        let mut out = Vec::new();
        if std::mem::take(&mut self.robot_due) {
            let notice = self.robot_notice();
            out.extend(self.notice(NoticeTopic::Robot, notice));
            out.extend(self.robot().map(|robot| Outgoing::from(&robot)));
        }
        if std::mem::take(&mut self.people_due) {
            let notice = self.people_notice();
            out.extend(self.notice(NoticeTopic::People, notice));
            out.extend(self.people_update().map(|people| Outgoing::from(&people)));
        }
        out
    }

    /// Sets or clears a topic's notice, with the update that says so when it changed.
    fn notice(&mut self, topic: NoticeTopic, text: Option<String>) -> Option<Outgoing> {
        let changed = self.notices.get(&topic) != text.as_ref();
        match &text {
            Some(text) => self.notices.insert(topic, text.clone()),
            None => self.notices.remove(&topic),
        };
        changed.then(|| (&Update::Notice { topic, text }).into())
    }

    /// The correction for the odom frame in use, if the localizer has made one in it.
    fn current_correction(&self) -> Option<&MapToOdom> {
        let started = self.odometry.as_ref().map(|pose| pose.odom_started_camera_ns);
        self.correction.as_ref().filter(|c| started.is_none_or(|started| c.valid_from.0 >= started))
    }

    fn robot_notice(&self) -> Option<String> {
        match (&self.odometry, &self.correction) {
            (None, _) => Some("no odometry yet".into()),
            (Some(_), None) => Some("waiting for the localizer's first map_to_odom".into()),
            (Some(_), Some(_)) if self.current_correction().is_none() => {
                Some("odometry started again; waiting for map_to_odom in its new frame".into())
            }
            _ => None,
        }
    }

    fn robot(&self) -> Option<Update> {
        let (pose, correction) = (self.odometry.as_ref()?, self.current_correction()?);
        let (turn, shift) = turn_and_shift(&correction.correction);
        Some(Update::Robot {
            position: mm3(turn * pose.position_m + shift),
            orientation: (turn * pose.orientation).normalize().to_array(),
        })
    }

    fn people_notice(&self) -> Option<String> {
        let frame = self.people.as_ref()?;
        if frame.position_frame == PositionFrame::Camera {
            return Some("people tracks are in the camera frame; only odom-frame tracks can be placed".into());
        }
        self.current_correction().is_none().then(|| "people wait for map_to_odom in the odom frame in use".into())
    }

    fn people_update(&self) -> Option<Update> {
        let frame = self.people.as_ref()?;
        let correction = self.current_correction().filter(|_| frame.position_frame == PositionFrame::Odom);
        let Some(correction) = correction else { return Some(Update::People { tracks: Vec::new() }) };
        let (turn, shift) = turn_and_shift(&correction.correction);
        let to_map = |p: &perception_wire::people::Point| mm3(turn * Vec3::new(p.x_m, p.y_m, p.z_m) + shift);
        let tracks = frame
            .tracks
            .iter()
            .map(|track| PersonView {
                track_id: track.track_id,
                state: match track.state {
                    TrackState::Tentative => "tentative",
                    TrackState::Confirmed => "confirmed",
                    TrackState::Lost => "lost",
                },
                torso: track.position.as_ref().map(to_map),
                joints: track.skeleton_3d.iter().map(|joint| joint.as_ref().map(to_map)).collect(),
            })
            .collect();
        Some(Update::People { tracks })
    }
}

/// `p_map = R_z(yaw) p_odom + (x, y, z)`.
fn turn_and_shift(correction: &Pose4) -> (Quat, Vec3) {
    (Quat::from_rotation_z(correction.yaw_rad), Vec3::new(correction.x_m, correction.y_m, correction.z_m))
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};
    use perception_wire::localization::{CameraNanos, MovedKeyframe};
    use perception_wire::navigation::{MapVersion, UnixNanos};
    use perception_wire::objects::ObjectExtent;
    use perception_wire::people::{Keypoint, PersonTrack, PixelBox, Point};
    use serde_json::Value;

    use super::*;
    use crate::cloud::tests::wall_keyframe;

    fn json(outgoing: &[Outgoing]) -> Vec<Value> {
        outgoing
            .iter()
            .filter_map(|o| match o {
                Outgoing::Json(text) => Some(serde_json::from_str(text).unwrap()),
                Outgoing::Binary(_) => None,
            })
            .collect()
    }

    fn types(outgoing: &[Outgoing]) -> Vec<String> {
        outgoing
            .iter()
            .map(|o| match o {
                Outgoing::Json(text) => serde_json::from_str::<Value>(text).unwrap()["type"].as_str().unwrap().to_owned(),
                Outgoing::Binary(bytes) => if bytes[0] == crate::surface::SURFACE_MESSAGE { "surface" } else { "cloud" }.to_owned(),
            })
            .collect()
    }

    fn keyframe_pose(frame_id: u64, x_m: f32, tracking: Tracking) -> KeyframePose {
        KeyframePose {
            frame_id: FrameId(frame_id),
            captured: UnixNanos(1),
            map_version: MapVersion(1),
            pose: Pose4 { x_m, y_m: 0.0, z_m: 0.0, yaw_rad: 0.0 },
            tracking,
            residual_m: 0.01,
            inlier_fraction: 0.9,
        }
    }

    /// The odom frame turned a quarter left and moved 1 m along x in the map, from camera time `valid_from_ns`.
    fn quarter_turn(valid_from_ns: i64) -> MapToOdom {
        MapToOdom {
            map_version: MapVersion(1),
            valid_from: CameraNanos(valid_from_ns),
            correction: Pose4 { x_m: 1.0, y_m: 0.0, z_m: 0.0, yaw_rad: std::f32::consts::FRAC_PI_2 },
            sigma_xy_m: 0.01,
            sigma_z_m: 0.01,
            sigma_yaw_rad: 0.01,
        }
    }

    fn odom_pose(started_ns: i64, camera_ns: i64, position_m: Vec3) -> OdomPose {
        OdomPose {
            camera_ns,
            odom_started_camera_ns: started_ns,
            depth_frame_id: 1,
            position_m,
            orientation: Quat::IDENTITY,
            velocity_m_s: Vec3::ZERO,
            position_sigma_m: Vec3::splat(0.001),
            tilt_sigma_rad: 0.001,
            heading_sigma_rad: 0.001,
            corrected: true,
            matched_points: 100,
            weakest: None,
        }
    }

    fn people(frame: PositionFrame) -> PeopleTracks {
        let mut skeleton_3d = vec![None; 17];
        skeleton_3d[5] = Some(Point { x_m: 2.0, y_m: 0.0, z_m: 1.4 });
        let track = PersonTrack {
            track_id: 4,
            state: TrackState::Confirmed,
            score: 0.9,
            bbox: PixelBox { x_px: 0.0, y_px: 0.0, w_px: 1.0, h_px: 1.0 },
            keypoints: vec![Keypoint { x_px: 0.0, y_px: 0.0, score: 0.9 }; 17],
            position: Some(Point { x_m: 2.0, y_m: 0.0, z_m: 1.0 }),
            velocity: None,
            keypoints_3d: vec![Some(Point { x_m: 9.0, y_m: 9.0, z_m: 9.0 }); 17],
            skeleton_3d,
        };
        PeopleTracks {
            frame_id: 1,
            captured_camera_ns: 1,
            width: 8,
            height: 6,
            position_frame: frame,
            gravity_reaction: None,
            tracks: vec![track],
        }
    }

    fn object(id: u64, kind: ObjectUpdateKind) -> ObjectUpdate {
        ObjectUpdate {
            kind,
            object_id: ObjectId(id),
            merged_into: None,
            label: "chair".into(),
            score: 0.8,
            confidence: 0.75,
            seen_count: 3,
            frame_id: FrameId(1),
            captured: UnixNanos(1),
            map_version: MapVersion(1),
            extent: ObjectExtent {
                centre_m: Vec3::new(1.0, 2.0, 0.45),
                footprint_m: [Vec2::new(0.8, 1.8), Vec2::new(1.2, 1.8), Vec2::new(1.2, 2.2), Vec2::new(0.8, 2.2)],
                long_m: 0.4,
                short_m: 0.4,
                height_m: 0.9,
                top_m: 0.9,
            },
            crop_jpeg: None,
            voxel_m: 0.05,
            voxels: vec![glam::IVec3::new(20, 40, 0), glam::IVec3::new(21, 40, 0)],
        }
    }

    fn close(a: &Value, b: [f64; 3]) -> bool {
        (0..3).all(|i| (a[i].as_f64().unwrap() - b[i]).abs() < 1e-3)
    }

    #[test]
    fn a_page_that_connects_late_gets_everything_after_a_reset() {
        let mut scene = Scene::new(SceneSettings::DEFAULT);
        scene.on_keyframe(&wall_keyframe(1, 2000, [9, 9, 9])).unwrap();
        scene.on_keyframe_pose(&keyframe_pose(1, 0.5, Tracking::Tracking));
        scene.on_object_update(&object(3, ObjectUpdateKind::Appeared));
        assert_eq!(types(&scene.snapshot()), ["reset", "cloud", "keyframe_pose", "object"]);
    }

    #[test]
    fn a_lost_keyframe_is_sent_unplaced_and_a_later_move_places_it() {
        let mut scene = Scene::new(SceneSettings::DEFAULT);
        let lost = json(&scene.on_keyframe_pose(&keyframe_pose(2, 0.0, Tracking::Lost)));
        assert_eq!(lost[0]["placed"], false);
        let moved = KeyframePosesMoved {
            map_version: MapVersion(2),
            keyframes: vec![MovedKeyframe { frame_id: FrameId(2), pose: Pose4 { x_m: 3.0, y_m: 0.0, z_m: 0.0, yaw_rad: 0.1 } }],
        };
        let sent = json(&scene.on_keyframe_poses_moved(&moved));
        assert_eq!((sent[0]["frame_id"].as_u64(), sent[0]["placed"].as_bool()), (Some(2), Some(true)));
        assert_eq!(sent[0]["pose"]["x_m"], 3.0);
    }

    #[test]
    fn the_oldest_cloud_goes_once_there_are_too_many() {
        let mut scene = Scene::new(SceneSettings { max_keyframes: 2, ..SceneSettings::DEFAULT });
        for frame_id in 1..=2 {
            scene.on_keyframe(&wall_keyframe(frame_id, 2000, [0; 3])).unwrap();
            scene.on_keyframe_pose(&keyframe_pose(frame_id, 0.0, Tracking::Tracking));
        }
        let third = scene.on_keyframe(&wall_keyframe(3, 2000, [0; 3])).unwrap();
        assert_eq!(types(&third), ["cloud", "keyframe_dropped"]);
        assert_eq!(json(&third)[0]["frame_id"], 1);
        assert_eq!(scene.keyframe_count(), 2);
        assert_eq!(types(&scene.snapshot()), ["reset", "cloud", "keyframe_pose", "cloud"]);
    }

    #[test]
    fn the_robot_waits_for_a_correction_then_is_drawn_in_the_map_once_per_tick() {
        let mut scene = Scene::new(SceneSettings::DEFAULT);
        scene.on_odom_pose(&odom_pose(0, 10, Vec3::new(1.0, 0.0, 0.0)));
        let waiting = json(&scene.on_tick());
        assert_eq!(waiting.len(), 1);
        assert_eq!((waiting[0]["type"].as_str(), waiting[0]["topic"].as_str()), (Some("notice"), Some("robot")));
        scene.on_map_to_odom(&quarter_turn(5));
        scene.on_odom_pose(&odom_pose(0, 20, Vec3::new(1.0, 0.0, 0.0)));
        let drawn = json(&scene.on_tick());
        assert_eq!(drawn[0]["type"], "notice");
        assert!(drawn[0]["text"].is_null(), "the notice clears: {drawn:?}");
        assert_eq!(drawn[1]["type"], "robot");
        assert!(close(&drawn[1]["position"], [1.0, 1.0, 0.0]), "1 m along odom x is 1 m along map y, plus 1 m x: {drawn:?}");
        assert!(scene.on_tick().is_empty(), "nothing changed since the last tick");
    }

    #[test]
    fn a_correction_from_before_odometry_restarted_is_not_used() {
        let mut scene = Scene::new(SceneSettings::DEFAULT);
        scene.on_map_to_odom(&quarter_turn(5));
        scene.on_odom_pose(&odom_pose(100, 120, Vec3::ZERO));
        let sent = json(&scene.on_tick());
        assert_eq!(sent.len(), 1, "no robot: {sent:?}");
        assert!(sent[0]["text"].as_str().unwrap().contains("started again"), "{sent:?}");
    }

    #[test]
    fn people_are_placed_with_their_fitted_skeleton_in_the_map() {
        let mut scene = Scene::new(SceneSettings::DEFAULT);
        scene.on_odom_pose(&odom_pose(0, 10, Vec3::ZERO));
        scene.on_map_to_odom(&quarter_turn(5));
        scene.on_people_tracks(&people(PositionFrame::Odom));
        let sent = json(&scene.on_tick());
        let people = sent.iter().find(|m| m["type"] == "people").expect("people sent");
        let track = &people["tracks"][0];
        assert_eq!((track["track_id"].as_u64(), track["state"].as_str()), (Some(4), Some("confirmed")));
        assert!(close(&track["torso"], [1.0, 2.0, 1.0]), "{track}");
        assert!(close(&track["joints"][5], [1.0, 2.0, 1.4]), "the skeleton, not the raw keypoints: {track}");
        assert!(track["joints"][0].is_null());
    }

    #[test]
    fn camera_frame_people_are_not_placed_and_say_why() {
        let mut scene = Scene::new(SceneSettings::DEFAULT);
        scene.on_odom_pose(&odom_pose(0, 10, Vec3::ZERO));
        scene.on_map_to_odom(&quarter_turn(5));
        scene.on_people_tracks(&people(PositionFrame::Camera));
        let sent = json(&scene.on_tick());
        let notice = sent.iter().find(|m| m["topic"] == "people").expect("a people notice");
        assert!(notice["text"].as_str().unwrap().contains("camera frame"));
        let people = sent.iter().find(|m| m["type"] == "people").unwrap();
        assert_eq!(people["tracks"].as_array().unwrap().len(), 0);
    }

    fn patches(version: u64, blocks: &[(i32, bool)]) -> SurfacePatches {
        let patch = |x: i32, surface: bool| SurfacePatch {
            block: glam::IVec3::new(x, 0, 0),
            positions_m: if surface { vec![Vec3::new(0.4 * x as f32, 0.1, 0.1), Vec3::new(0.4 * x as f32 + 0.1, 0.1, 0.1), Vec3::new(0.4 * x as f32, 0.2, 0.1)] } else { vec![] },
            triangles: if surface { vec![[0, 1, 2]] } else { vec![] },
        };
        SurfacePatches {
            map_version: MapVersion(version),
            built: UnixNanos(1),
            voxel_m: 0.05,
            block_voxels: 8,
            patches: blocks.iter().map(|&(x, surface)| patch(x, surface)).collect(),
        }
    }

    #[test]
    fn surface_blocks_are_kept_for_late_pages_until_a_patch_empties_them() {
        let mut scene = Scene::new(SceneSettings::DEFAULT);
        assert_eq!(types(&scene.on_surface_patches(&patches(1, &[(0, true), (1, true)])).unwrap()), ["surface"]);
        assert_eq!(scene.surface_block_count(), 2);
        scene.on_surface_patches(&patches(1, &[(1, false)])).unwrap();
        assert_eq!(scene.surface_block_count(), 1);
        assert_eq!(types(&scene.snapshot()), ["reset", "surface"], "one binary surface message");
    }

    #[test]
    fn surface_patches_on_another_grid_are_refused() {
        let mut scene = Scene::new(SceneSettings::DEFAULT);
        scene.on_surface_patches(&patches(1, &[(0, true)])).unwrap();
        let mut finer = patches(2, &[(0, true)]);
        finer.voxel_m = 0.04;
        assert!(matches!(scene.on_surface_patches(&finer), Err(SurfaceError::GridChanged { .. })));
    }

    #[test]
    fn an_object_carries_its_voxels_to_the_page() {
        let mut scene = Scene::new(SceneSettings::DEFAULT);
        let sent = json(&scene.on_object_update(&object(3, ObjectUpdateKind::Appeared)));
        assert_eq!(sent[0]["voxel_m"].as_f64().map(|v| v as f32), Some(0.05));
        assert_eq!(sent[0]["voxels"], serde_json::json!([[20, 40, 0], [21, 40, 0]]));
    }

    #[test]
    fn an_object_that_disappears_or_merges_is_removed() {
        let mut scene = Scene::new(SceneSettings::DEFAULT);
        let appeared = json(&scene.on_object_update(&object(3, ObjectUpdateKind::Appeared)));
        assert_eq!((appeared[0]["label"].as_str(), appeared[0]["bottom_m"].as_f64()), (Some("chair"), Some(0.0)));
        scene.on_object_update(&object(4, ObjectUpdateKind::Appeared));
        assert_eq!(types(&scene.on_object_update(&object(3, ObjectUpdateKind::Disappeared))), ["object_gone"]);
        scene.on_object_update(&object(4, ObjectUpdateKind::Merged));
        assert_eq!(types(&scene.snapshot()), ["reset"]);
    }
}
