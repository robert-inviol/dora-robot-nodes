//! Everything the page draws, kept so a page that connects late gets the whole scene, and turned into the updates
//! that every connected page receives as messages arrive.
//!
//! Odometry and people tracks are in the odom frame; they are carried into the map with the localizer's latest
//! `map_to_odom`, which applies only to the odom frame in use: a correction made before odometry last started
//! belongs to an earlier frame. The robot and the people are sent on each tick at most, whatever their rate.

use std::collections::BTreeMap;

use glam::{Quat, Vec3};
use perception_wire::localization::{FrameId, KeyframePose, KeyframePosesMoved, MapToOdom, Pose4, Tracking};
use perception_wire::objects::{ObjectId, ObjectUpdate, ObjectUpdateKind};
use perception_wire::odometry::OdomPose;
use perception_wire::people::{PeopleTracks, PositionFrame, TrackState};
use perception_wire::scene_layout::SceneLayout;
use perception_wire::space_map::{SurfacePatch, SurfacePatches, Topology};
use perception_wire::Keyframe;

use crate::cloud::{CloudError, CloudSettings};
use crate::clouds::{CloudStore, Placement};
use crate::surface::{encode_patches, BlockGrid, SurfaceError};
use crate::view::{
    mm, mm3, DoorwayView, Layer, NoticeTopic, Outgoing, PersonView, PlaceView, PlaneView, RoomBoxView, RoomView, Update,
};

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
    clouds: CloudStore,
    correction: Option<MapToOdom>,
    odometry: Option<OdomPose>,
    robot_due: bool,
    people: Option<PeopleTracks>,
    people_due: bool,
    /// Every object as last sent, with its voxels.
    objects: BTreeMap<ObjectId, Update>,
    topology: Option<Update>,
    /// The latest layout, drawn once its keyframe is placed.
    layout: Option<SceneLayout>,
    notices: BTreeMap<NoticeTopic, String>,
    surface_grid: Option<BlockGrid>,
    /// The latest surface of every block that has one.
    surfaces: BTreeMap<[i32; 3], SurfacePatch>,
}

impl Scene {
    pub fn new(settings: SceneSettings) -> Scene {
        Scene {
            clouds: CloudStore::new(settings.cloud, settings.max_keyframes),
            correction: None,
            odometry: None,
            robot_due: false,
            people: None,
            people_due: false,
            objects: BTreeMap::new(),
            topology: None,
            layout: None,
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

    /// Everything a newly connected page needs, starting with a reset; the keyframe clouds come only with
    /// [`Scene::clouds_snapshot`].
    pub fn snapshot(&self) -> Vec<Outgoing> {
        let mut out = vec![Outgoing::from(&Update::Reset)];
        for (&topic, text) in &self.notices {
            out.push((&Update::Notice { topic, text: Some(text.clone()) }).into());
        }
        if let Some(grid) = self.surface_grid {
            let encoded = encode_patches(grid, self.surfaces.values()).expect("every kept patch was encoded once already");
            out.push(Outgoing::binary(Layer::Always, encoded));
        }
        out.extend(self.robot().map(|robot| Outgoing::from(&robot)));
        out.extend(self.people_update().map(|people| Outgoing::from(&people)));
        out.extend(self.objects.values().map(Outgoing::from));
        out.extend(self.topology.iter().map(Outgoing::from));
        out.extend(self.layout_update().map(|layout| Outgoing::from(&layout)));
        out
    }

    /// Every keyframe cloud, for a page that starts to show them.
    pub fn clouds_snapshot(&self) -> Vec<Outgoing> {
        self.clouds.snapshot()
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
        Ok(vec![Outgoing::binary(Layer::Always, encoded)])
    }

    pub fn on_keyframe(&mut self, keyframe: &Keyframe) -> Result<Vec<Outgoing>, CloudError> {
        self.clouds.on_keyframe(keyframe)
    }

    pub fn on_keyframe_pose(&mut self, pose: &KeyframePose) -> Vec<Outgoing> {
        self.place_keyframe(pose.frame_id, Placement { pose: pose.pose, placed: pose.tracking != Tracking::Lost })
    }

    pub fn on_keyframe_poses_moved(&mut self, moved: &KeyframePosesMoved) -> Vec<Outgoing> {
        moved
            .keyframes
            .iter()
            .flat_map(|keyframe| self.place_keyframe(keyframe.frame_id, Placement { pose: keyframe.pose, placed: true }))
            .collect()
    }

    fn place_keyframe(&mut self, frame_id: FrameId, placement: Placement) -> Vec<Outgoing> {
        let mut out = self.clouds.place(frame_id, placement);
        if self.layout.as_ref().is_some_and(|layout| layout.frame_id == frame_id.0) {
            out.extend(self.layout_update().map(|layout| Outgoing::from(&layout)));
        }
        out
    }

    /// Keeps the planes and room of the latest keyframe, sent once that keyframe is placed.
    pub fn on_scene_layout(&mut self, layout: SceneLayout) -> Vec<Outgoing> {
        self.layout = Some(layout);
        self.layout_update().map(|layout| Outgoing::from(&layout)).into_iter().collect()
    }

    fn layout_update(&self) -> Option<Update> {
        let layout = self.layout.as_ref()?;
        let placement = self.clouds.placement(FrameId(layout.frame_id))?;
        Some(Update::Layout {
            frame_id: layout.frame_id,
            pose: placement.floor_pose(),
            placed: placement.placed,
            planes: layout
                .planes
                .iter()
                .map(|plane| PlaneView { kind: plane.kind.name(), area_m2: mm(plane.area_m2), outline: plane.outline_m.iter().map(|c| mm3(*c)).collect() })
                .collect(),
            room: layout.room.as_ref().ok().map(|room| RoomBoxView {
                floor_m: mm(room.floor_m),
                ceiling_m: mm(room.ceiling_m),
                footprint: room.footprint_m.iter().map(|c| [mm(c.x), mm(c.y)]).collect(),
                edge_seen: room.edge_seen.clone(),
            }),
            room_problem: layout.room.as_ref().err().cloned(),
        })
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

    /// An object that looks as it was last sent is not sent again, and its voxels go only when they changed.
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
            confidence: (update.confidence * 100.0).round() / 100.0,
            centre: mm3(extent.centre_m),
            footprint: extent.footprint_m.map(|corner| [mm(corner.x), mm(corner.y)]),
            bottom_m: mm(extent.top_m - extent.height_m),
            top_m: mm(extent.top_m),
            voxel_m: update.voxel_m,
            voxels: Some(update.voxels.iter().map(|v| v.to_array()).collect()),
        };
        let sent = self.objects.get(&update.object_id);
        if sent == Some(&object) {
            return Vec::new();
        }
        let voxels_of = |update: &Update| match update {
            Update::Object { voxel_m, voxels, .. } => Some((*voxel_m, voxels.clone())),
            _ => None,
        };
        let out = if sent.and_then(voxels_of) == voxels_of(&object) {
            let mut without = object.clone();
            if let Update::Object { voxels, .. } = &mut without {
                *voxels = None;
            }
            Outgoing::from(&without)
        } else {
            Outgoing::from(&object)
        };
        self.objects.insert(update.object_id, object);
        vec![out]
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
    use crate::view::Payload;

    fn json(outgoing: &[Outgoing]) -> Vec<Value> {
        outgoing
            .iter()
            .filter_map(|o| match &o.payload {
                Payload::Json(text) => Some(serde_json::from_str(text).unwrap()),
                Payload::Binary(_) => None,
            })
            .collect()
    }

    fn types(outgoing: &[Outgoing]) -> Vec<String> {
        outgoing
            .iter()
            .map(|o| match &o.payload {
                Payload::Json(text) => serde_json::from_str::<Value>(text).unwrap()["type"].as_str().unwrap().to_owned(),
                Payload::Binary(bytes) => match bytes[0] {
                    crate::surface::SURFACE_MESSAGE => "surface",
                    crate::clouds::CARVE_MESSAGE => "carve",
                    _ => "cloud",
                }
                .to_owned(),
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
        assert_eq!(types(&scene.snapshot()), ["reset", "object"], "clouds only for pages that show them");
        assert_eq!(types(&scene.clouds_snapshot()), ["cloud", "keyframe_pose"]);
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
        assert_eq!(types(&scene.clouds_snapshot()), ["cloud", "keyframe_pose", "cloud"]);
        assert!(third.iter().all(|o| o.layer == Layer::Clouds));
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
    fn an_object_is_sent_again_only_when_it_changes_and_its_voxels_only_when_they_change() {
        let mut scene = Scene::new(SceneSettings::DEFAULT);
        scene.on_object_update(&object(3, ObjectUpdateKind::Appeared));
        assert!(scene.on_object_update(&object(3, ObjectUpdateKind::Updated)).is_empty(), "nothing changed");
        let mut seen_again = object(3, ObjectUpdateKind::Updated);
        seen_again.confidence = 0.5;
        let sent = json(&scene.on_object_update(&seen_again));
        assert_eq!(sent[0]["confidence"].as_f64(), Some(0.5));
        assert!(sent[0]["voxels"].is_null(), "the voxels are the ones the page has");
        seen_again.voxels.push(glam::IVec3::new(22, 40, 0));
        let sent = json(&scene.on_object_update(&seen_again));
        assert_eq!(sent[0]["voxels"].as_array().map(Vec::len), Some(3));
        assert_eq!(json(&scene.snapshot())[1]["voxels"].as_array().map(Vec::len), Some(3), "a late page gets them all");
    }

    fn layout(frame_id: u64) -> SceneLayout {
        use perception_wire::scene_layout::{LayoutPlane, LayoutRoom};
        SceneLayout {
            frame_id,
            planes: vec![LayoutPlane {
                kind: depth_perception::SurfaceKind::Floor,
                area_m2: 4.0,
                outline_m: vec![Vec3::new(0.0, 1.0, -1.0), Vec3::new(1.0, 1.0, -1.0), Vec3::new(0.0, 2.0, -1.0)],
            }],
            room: Ok(LayoutRoom {
                floor_m: -1.0,
                ceiling_m: 1.5,
                footprint_m: vec![Vec2::new(-1.0, 0.5), Vec2::new(1.0, 0.5), Vec2::new(1.0, 3.0), Vec2::new(-1.0, 3.0)],
                edge_seen: vec![false, true, true, true],
            }),
        }
    }

    #[test]
    fn a_layout_waits_for_its_keyframe_pose_and_follows_it_when_it_moves() {
        let mut scene = Scene::new(SceneSettings::DEFAULT);
        assert!(scene.on_scene_layout(layout(5)).is_empty(), "no pose yet");
        let placed = scene.on_keyframe_pose(&keyframe_pose(5, 2.0, Tracking::Tracking));
        let sent = json(&placed);
        let layout_sent = sent.iter().find(|m| m["type"] == "layout").expect("the layout goes with the pose");
        assert_eq!((layout_sent["placed"].as_bool(), layout_sent["pose"]["x_m"].as_f64()), (Some(true), Some(2.0)));
        assert_eq!(layout_sent["planes"][0]["kind"], "floor");
        assert_eq!(layout_sent["room"]["edge_seen"], serde_json::json!([false, true, true, true]));
        let moved = KeyframePosesMoved {
            map_version: MapVersion(2),
            keyframes: vec![MovedKeyframe { frame_id: FrameId(5), pose: Pose4 { x_m: 2.5, y_m: 0.0, z_m: 0.0, yaw_rad: 0.0 } }],
        };
        let sent = json(&scene.on_keyframe_poses_moved(&moved));
        assert_eq!(sent.iter().find(|m| m["type"] == "layout").unwrap()["pose"]["x_m"].as_f64(), Some(2.5));
        assert_eq!(types(&scene.snapshot()), ["reset", "layout"]);
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
