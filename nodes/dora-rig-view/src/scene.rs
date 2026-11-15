//! Everything the page draws apart from the keyframe clouds, kept so a page that connects late gets the whole
//! scene, and turned into the updates that every connected page receives as messages arrive. The clouds are kept
//! apart, in [`crate::clouds::CloudStore`], so that their heavy work never holds up the robot and the people.
//!
//! Odometry and people tracks are in the odom frame; they are carried into the map with the localizer's latest
//! `map_to_odom`, which applies only to the odom frame it was made in: the robot is drawn with it only while the
//! odometry is in that frame, and people only when their tracks name that frame. The robot goes out with every
//! odometry pose and the people with every people frame.
//!
//! When the dense map's surface is replaced (a snapshot repairs it, or a new run of the dense map starts), every
//! page is sent the whole scene, starting with a reset, as a page that connects is.

use std::collections::BTreeMap;

use bytes::Bytes;
use glam::{Quat, Vec3};
use perception_wire::localization::{FrameId, KeyframePose, PoseGraphUpdate, MapToOdom, Pose4Dof};
use perception_wire::objects::{ObjectId, ObjectUpdate, ObjectUpdateKind};
use perception_wire::odometry::OdomPose;
use perception_wire::people::{PeopleTracks, PersonTrack, PositionFrame, TrackId, TrackState};
use perception_wire::room_layout::RoomLayout;
use perception_wire::space_map::Topology;

use crate::surface::MeshCacheChange;
use crate::surface_stream::{SurfaceApplication, SurfaceStreamStatus};

use crate::clouds::{KeyframeMapPoses, KeyframeMapPose};
use crate::view::{
    mm, mm3, DoorwayView, Layer, NoticeTopic, Outgoing, PersonView, PlaceView, PlaneView, RoomBoxView, RoomView, Update,
};

pub struct Scene {
    correction: Option<MapToOdom>,
    odometry: Option<OdomPose>,
    people: Option<PeopleTracks>,
    /// Each track of the latest people frame as it was last seen with a position, for drawing it faded while a
    /// frame has none.
    last_with_position: BTreeMap<TrackId, PersonTrack>,
    /// Every object as last sent, with its voxels.
    objects: BTreeMap<ObjectId, Update>,
    topology: Option<Update>,
    /// The latest room layout, drawn once its keyframe has a map pose that is drawn.
    layout: Option<RoomLayout>,
    /// Where the localizer put the recent keyframes, for drawing the room layout at its keyframe's pose.
    keyframe_map_poses: KeyframeMapPoses,
    notices: BTreeMap<NoticeTopic, String>,
    /// The whole surface as one message, encoded when it last changed.
    surface: Option<Bytes>,
    /// How the surface messages have come, once one has.
    surface_stream: Option<Update>,
}

impl Scene {
    /// `max_keyframes` bounds the keyframe poses kept for placing a layout, as it bounds the clouds.
    pub fn new(max_keyframes: usize) -> Scene {
        Scene {
            correction: None,
            odometry: None,
            people: None,
            last_with_position: BTreeMap::new(),
            objects: BTreeMap::new(),
            topology: None,
            layout: None,
            keyframe_map_poses: KeyframeMapPoses::new(max_keyframes),
            notices: BTreeMap::new(),
            surface: None,
            surface_stream: None,
        }
    }

    /// Everything a newly connected page needs, starting with a reset; the keyframe clouds come from the cloud
    /// store.
    pub fn snapshot(&self) -> Vec<Outgoing> {
        let mut out = vec![Outgoing::from(&Update::Reset)];
        for (&topic, text) in &self.notices {
            out.push((&Update::Notice { topic, text: Some(text.clone()) }).into());
        }
        out.extend(self.surface_stream.iter().map(Outgoing::from));
        out.extend(self.surface.iter().map(|surface| Outgoing::binary(Layer::Always, surface.clone())));
        out.extend(self.robot().map(|robot| Outgoing::from(&robot)));
        out.extend(self.people_update().map(|people| Outgoing::from(&people)));
        out.extend(self.objects.values().map(Outgoing::from));
        out.extend(self.topology.iter().map(Outgoing::from));
        out.extend(self.layout_update().map(|layout| Outgoing::from(&layout)));
        out
    }

    /// The whole surface as a page that connects now gets it, encoded.
    pub fn whole_surface(&self) -> Option<&Bytes> {
        self.surface.as_ref()
    }

    /// Keeps the whole surface, already encoded, for pages that connect from now on.
    pub fn set_surface(&mut self, surface: Bytes) {
        self.surface = Some(surface);
    }

    /// One surface message, meshed: its mesh blocks go to every page, or, when it replaced the surface held, the
    /// whole scene does.
    pub fn on_surface(&mut self, change: MeshCacheChange, application: SurfaceApplication, status: SurfaceStreamStatus) -> Vec<Outgoing> {
        self.surface = Some(change.snapshot);
        let mut out: Vec<Outgoing> = self.notice(NoticeTopic::Surface, status.untrusted.map(|untrusted| format!("{untrusted}; waiting for the dense map's snapshot"))).into_iter().collect();
        let counts = Update::SurfaceStream { gaps: status.gaps, snapshots: status.snapshots };
        if self.surface_stream.as_ref() != Some(&counts) {
            out.push(Outgoing::from(&counts));
            self.surface_stream = Some(counts);
        }
        match application {
            SurfaceApplication::Changes => {
                out.push(Outgoing::binary(Layer::Always, change.update));
                out
            }
            SurfaceApplication::Replace => self.snapshot(),
        }
    }

    pub fn on_keyframe_pose(&mut self, pose: &KeyframePose) -> Vec<Outgoing> {
        self.set_keyframe_map_pose(pose.frame_id, KeyframeMapPose::of_keyframe_pose(pose))
    }

    pub fn on_pose_graph_update(&mut self, moved: &PoseGraphUpdate) -> Vec<Outgoing> {
        moved.keyframes.iter().flat_map(|keyframe| self.set_keyframe_map_pose(keyframe.frame_id, self.keyframe_map_poses.moved(keyframe))).collect()
    }

    fn set_keyframe_map_pose(&mut self, frame_id: FrameId, map_pose: KeyframeMapPose) -> Vec<Outgoing> {
        self.keyframe_map_poses.insert(frame_id, map_pose);
        if self.layout.as_ref().is_some_and(|layout| layout.frame_id == frame_id.0) {
            return self.layout_update().map(|layout| Outgoing::from(&layout)).into_iter().collect();
        }
        Vec::new()
    }

    /// Keeps the planes and room of the latest keyframe, sent once that keyframe has a map pose.
    pub fn on_room_layout(&mut self, layout: RoomLayout) -> Vec<Outgoing> {
        self.layout = Some(layout);
        self.layout_update().map(|layout| Outgoing::from(&layout)).into_iter().collect()
    }

    fn layout_update(&self) -> Option<Update> {
        let layout = self.layout.as_ref()?;
        let map_pose = self.keyframe_map_poses.get(FrameId(layout.frame_id))?;
        Some(Update::RoomLayout {
            frame_id: layout.frame_id,
            pose: map_pose.floor_pose(),
            drawn: map_pose.standing.drawn(),
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

    /// The robot and the people in the map with the new correction.
    pub fn on_map_to_odom(&mut self, correction: &MapToOdom) -> Vec<Outgoing> {
        self.correction = Some(*correction);
        let mut out = self.robot_updates();
        out.extend(self.people_updates());
        out
    }

    pub fn on_odom_pose(&mut self, pose: &OdomPose) -> Vec<Outgoing> {
        self.odometry = Some(*pose);
        self.robot_updates()
    }

    pub fn on_people_tracks(&mut self, tracks: &PeopleTracks) -> Vec<Outgoing> {
        self.last_with_position.retain(|track_id, _| tracks.tracks.iter().any(|track| track.track_id == *track_id));
        for track in tracks.tracks.iter().filter(|track| track.position.is_some()) {
            self.last_with_position.insert(track.track_id, track.clone());
        }
        self.people = Some(tracks.clone());
        self.people_updates()
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

    /// The robot, and its notice when that changed.
    fn robot_updates(&mut self) -> Vec<Outgoing> {
        let notice = self.robot_notice();
        let mut out: Vec<Outgoing> = self.notice(NoticeTopic::Robot, notice).into_iter().collect();
        out.extend(self.robot().map(|robot| Outgoing::from(&robot)));
        out
    }

    /// The people, and their notice when that changed.
    fn people_updates(&mut self) -> Vec<Outgoing> {
        let notice = self.people_notice();
        let mut out: Vec<Outgoing> = self.notice(NoticeTopic::People, notice).into_iter().collect();
        out.extend(self.people_update().map(|people| Outgoing::from(&people)));
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
        self.correction.as_ref().filter(|c| started.is_none_or(|started| c.odom_started.0 == started))
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

    /// The correction for the odom frame the people tracks name, if the localizer has made one in it.
    fn people_correction(&self, frame: &PeopleTracks) -> Option<&MapToOdom> {
        let PositionFrame::Odom { odom_started } = frame.position_frame else { return None };
        self.correction.as_ref().filter(|correction| correction.odom_started == odom_started)
    }

    fn people_notice(&self) -> Option<String> {
        let frame = self.people.as_ref()?;
        let PositionFrame::Odom { odom_started } = frame.position_frame else {
            return Some("people tracks are in the camera frame; only odom-frame tracks can be transformed into the map frame".into());
        };
        match &self.correction {
            None => Some("people wait for the localizer's first map_to_odom".into()),
            Some(correction) if correction.odom_started != odom_started => Some(format!(
                "people tracks are in the odom frame started at camera time {} ns; map_to_odom is for the one started at {} ns",
                odom_started.0, correction.odom_started.0
            )),
            Some(_) => None,
        }
    }

    fn people_update(&self) -> Option<Update> {
        let frame = self.people.as_ref()?;
        let Some(correction) = self.people_correction(frame) else { return Some(Update::People { tracks: Vec::new() }) };
        let (turn, shift) = turn_and_shift(&correction.correction);
        let to_map = |p: &perception_wire::people::Point| mm3(turn * Vec3::new(p.x_m, p.y_m, p.z_m) + shift);
        let tracks = frame
            .tracks
            .iter()
            .map(|track| {
                // A track seen without a position this frame is drawn as it was last seen with one, faded.
                let (drawn, stale) = match (&track.position, self.last_with_position.get(&track.track_id)) {
                    (None, Some(last)) => (last, true),
                    _ => (track, false),
                };
                PersonView {
                    track_id: track.track_id,
                    state: match track.state {
                        TrackState::Tentative => "tentative",
                        TrackState::Confirmed => "confirmed",
                        TrackState::Lost => "lost",
                    },
                    stale,
                    torso: drawn.position.as_ref().map(to_map),
                    joints: drawn.skeleton_3d.iter().map(|joint| joint.as_ref().map(to_map)).collect(),
                }
            })
            .collect();
        Some(Update::People { tracks })
    }
}

/// `p_map = R_z(yaw) p_odom + (x, y, z)`.
fn turn_and_shift(correction: &Pose4Dof) -> (Quat, Vec3) {
    (Quat::from_rotation_z(correction.yaw_rad), Vec3::new(correction.x_m, correction.y_m, correction.z_m))
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};
    use perception_wire::localization::{CameraNanos, UpdatedKeyframePose, Tracking};
    use perception_wire::navigation::{MapVersion, UnixNanos};
    use perception_wire::objects::ObjectExtent;
    use perception_wire::people::{Keypoint, PersonTrack, PixelBox, Point};
    use serde_json::Value;

    use super::*;
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
                    crate::clouds::SPACE_CARVING_MESSAGE => "space_carving",
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
            pose: Pose4Dof { x_m, y_m: 0.0, z_m: 0.0, yaw_rad: 0.0 },
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
            odom_started: CameraNanos(0),
            correction: Pose4Dof { x_m: 1.0, y_m: 0.0, z_m: 0.0, yaw_rad: std::f32::consts::FRAC_PI_2 },
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
            degenerate: None,
        }
    }

    /// The odom frame the corrections of these tests are for.
    const ODOM: PositionFrame = PositionFrame::Odom { odom_started: CameraNanos(0) };

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
            specific_force: None,
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
        let mut scene = Scene::new(600);
        scene.on_object_update(&object(3, ObjectUpdateKind::Appeared));
        scene.set_surface(Bytes::from_static(&[crate::surface::SURFACE_MESSAGE]));
        assert_eq!(types(&scene.snapshot()), ["reset", "surface", "object"]);
    }

    #[test]
    fn keyframe_poses_for_layouts_are_kept_only_for_the_newest_frames() {
        let mut scene = Scene::new(2);
        scene.on_room_layout(layout(1));
        scene.on_keyframe_pose(&keyframe_pose(1, 0.0, Tracking::Tracking));
        for frame_id in 2..=3 {
            scene.on_keyframe_pose(&keyframe_pose(frame_id, 0.0, Tracking::Tracking));
        }
        assert_eq!(types(&scene.snapshot()), ["reset"], "frame 1's pose is no longer kept");
    }

    #[test]
    fn the_robot_waits_for_a_correction_then_goes_out_with_every_odometry_pose() {
        let mut scene = Scene::new(600);
        let waiting = json(&scene.on_odom_pose(&odom_pose(0, 10, Vec3::new(1.0, 0.0, 0.0))));
        assert_eq!(waiting.len(), 1);
        assert_eq!((waiting[0]["type"].as_str(), waiting[0]["topic"].as_str()), (Some("notice"), Some("robot")));
        let corrected = json(&scene.on_map_to_odom(&quarter_turn(5)));
        assert_eq!(corrected[0]["type"], "notice");
        assert!(corrected[0]["text"].is_null(), "the notice clears: {corrected:?}");
        assert_eq!(corrected[1]["type"], "robot");
        let drawn = json(&scene.on_odom_pose(&odom_pose(0, 20, Vec3::new(1.0, 0.0, 0.0))));
        assert_eq!(drawn.len(), 1, "the notice is unchanged, so only the robot: {drawn:?}");
        assert!(close(&drawn[0]["position"], [1.0, 1.0, 0.0]), "1 m along odom x is 1 m along map y, plus 1 m x: {drawn:?}");
        assert_eq!(json(&scene.on_odom_pose(&odom_pose(0, 53, Vec3::ZERO)))[0]["type"], "robot", "every pose goes out");
    }

    #[test]
    fn a_correction_from_before_odometry_restarted_is_not_used() {
        let mut scene = Scene::new(600);
        scene.on_map_to_odom(&quarter_turn(5));
        let sent = json(&scene.on_odom_pose(&odom_pose(100, 120, Vec3::ZERO)));
        assert_eq!(sent.len(), 1, "no robot: {sent:?}");
        assert!(sent[0]["text"].as_str().unwrap().contains("started again"), "{sent:?}");
    }

    #[test]
    fn people_go_out_with_every_people_frame_with_their_fitted_skeleton_in_the_map() {
        let mut scene = Scene::new(600);
        scene.on_odom_pose(&odom_pose(0, 10, Vec3::ZERO));
        scene.on_map_to_odom(&quarter_turn(5));
        for _ in 0..3 {
            let sent = json(&scene.on_people_tracks(&people(ODOM)));
            let people = sent.iter().find(|m| m["type"] == "people").expect("people sent");
            let track = &people["tracks"][0];
            assert_eq!((track["track_id"].as_u64(), track["state"].as_str()), (Some(4), Some("confirmed")));
            assert_eq!(track["stale"], false);
            assert!(close(&track["torso"], [1.0, 2.0, 1.0]), "{track}");
            assert!(close(&track["joints"][5], [1.0, 2.0, 1.4]), "the skeleton, not the raw keypoints: {track}");
            assert!(track["joints"][0].is_null());
        }
    }

    #[test]
    fn a_track_seen_without_a_position_is_drawn_faded_where_it_was_last_seen() {
        let mut scene = Scene::new(600);
        scene.on_odom_pose(&odom_pose(0, 10, Vec3::ZERO));
        scene.on_map_to_odom(&quarter_turn(5));
        scene.on_people_tracks(&people(ODOM));
        let mut without_depth = people(ODOM);
        without_depth.tracks[0].position = None;
        without_depth.tracks[0].skeleton_3d = vec![None; 17];
        let sent = json(&scene.on_people_tracks(&without_depth));
        let track = &sent.iter().find(|m| m["type"] == "people").unwrap()["tracks"][0];
        assert_eq!(track["stale"], true);
        assert!(close(&track["torso"], [1.0, 2.0, 1.0]), "{track}");
        assert!(close(&track["joints"][5], [1.0, 2.0, 1.4]), "{track}");
        let mut nobody = people(ODOM);
        nobody.tracks.clear();
        scene.on_people_tracks(&nobody);
        let sent = json(&scene.on_people_tracks(&without_depth));
        let track = &sent.iter().find(|m| m["type"] == "people").unwrap()["tracks"][0];
        assert_eq!(track["stale"], false, "a track that left the frame is forgotten");
        assert!(track["torso"].is_null());
    }

    #[test]
    fn camera_frame_people_are_not_transformed_and_say_why() {
        let mut scene = Scene::new(600);
        scene.on_odom_pose(&odom_pose(0, 10, Vec3::ZERO));
        scene.on_map_to_odom(&quarter_turn(5));
        let sent = json(&scene.on_people_tracks(&people(PositionFrame::Camera)));
        let notice = sent.iter().find(|m| m["topic"] == "people").expect("a people notice");
        assert!(notice["text"].as_str().unwrap().contains("camera frame"));
        let people = sent.iter().find(|m| m["type"] == "people").unwrap();
        assert_eq!(people["tracks"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn people_of_a_new_odom_frame_wait_for_a_correction_made_in_it() {
        let mut scene = Scene::new(600);
        scene.on_odom_pose(&odom_pose(0, 10, Vec3::ZERO));
        scene.on_map_to_odom(&quarter_turn(5));
        let restarted = PositionFrame::Odom { odom_started: CameraNanos(100) };
        let sent = json(&scene.on_people_tracks(&people(restarted)));
        let notice = sent.iter().find(|m| m["topic"] == "people").expect("a people notice");
        assert!(notice["text"].as_str().unwrap().contains("started at camera time 100 ns"), "{notice}");
        let drawn = sent.iter().find(|m| m["type"] == "people").unwrap();
        assert_eq!(drawn["tracks"].as_array().unwrap().len(), 0, "not drawn with the old frame's correction");
        let correction = MapToOdom { odom_started: CameraNanos(100), valid_from: CameraNanos(120), ..quarter_turn(5) };
        let sent = json(&scene.on_map_to_odom(&correction));
        let notice = sent.iter().find(|m| m["topic"] == "people").expect("the people notice clears");
        assert!(notice["text"].is_null(), "{notice}");
        let drawn = sent.iter().find(|m| m["type"] == "people").unwrap();
        assert!(close(&drawn["tracks"][0]["torso"], [1.0, 2.0, 1.0]), "{drawn}");
    }

    #[test]
    fn a_surface_that_replaced_the_one_held_sends_every_page_the_whole_scene() {
        use crate::surface::MeshCacheChange;
        use crate::surface_stream::{SurfaceApplication, SurfaceStreamStatus, Untrusted};
        use perception_wire::space_map::MeshSequence;
        let surface = || Bytes::from_static(&[crate::surface::SURFACE_MESSAGE]);
        let change = || MeshCacheChange { update: surface(), snapshot: surface() };
        let mut scene = Scene::new(600);
        scene.on_object_update(&object(3, ObjectUpdateKind::Appeared));
        let trusted = SurfaceStreamStatus { untrusted: None, gaps: 0, snapshots: 0 };
        assert_eq!(types(&scene.on_surface(change(), SurfaceApplication::Changes, trusted)), ["surface_stream", "surface"]);
        let missed = Untrusted::MissedMessages { expected: MeshSequence(2), came: MeshSequence(3) };
        let gap = SurfaceStreamStatus { untrusted: Some(missed), gaps: 1, snapshots: 0 };
        let sent = scene.on_surface(change(), SurfaceApplication::Changes, gap);
        assert_eq!(types(&sent), ["notice", "surface_stream", "surface"]);
        assert!(json(&sent)[0]["text"].as_str().unwrap().contains("missed surface message 2;"), "{:?}", json(&sent));
        let repaired = SurfaceStreamStatus { untrusted: None, gaps: 1, snapshots: 1 };
        let sent = scene.on_surface(change(), SurfaceApplication::Replace, repaired);
        assert_eq!(types(&sent), ["reset", "surface_stream", "surface", "object"], "the reset path, as for a page that connects");
        assert_eq!(json(&sent)[1], serde_json::json!({"type": "surface_stream", "gaps": 1, "snapshots": 1}));
        assert_eq!(sent, scene.snapshot());
    }

    #[test]
    fn an_object_carries_its_voxels_to_the_page() {
        let mut scene = Scene::new(600);
        let sent = json(&scene.on_object_update(&object(3, ObjectUpdateKind::Appeared)));
        assert_eq!(sent[0]["voxel_m"].as_f64().map(|v| v as f32), Some(0.05));
        assert_eq!(sent[0]["voxels"], serde_json::json!([[20, 40, 0], [21, 40, 0]]));
    }

    #[test]
    fn an_object_is_sent_again_only_when_it_changes_and_its_voxels_only_when_they_change() {
        let mut scene = Scene::new(600);
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

    fn layout(frame_id: u64) -> RoomLayout {
        use perception_wire::room_layout::{RoomLayoutPlane, RoomBox};
        RoomLayout {
            frame_id,
            planes: vec![RoomLayoutPlane {
                kind: depth_perception::SurfaceKind::Floor,
                area_m2: 4.0,
                outline_m: vec![Vec3::new(0.0, 1.0, -1.0), Vec3::new(1.0, 1.0, -1.0), Vec3::new(0.0, 2.0, -1.0)],
            }],
            room: Ok(RoomBox {
                floor_m: -1.0,
                ceiling_m: 1.5,
                footprint_m: vec![Vec2::new(-1.0, 0.5), Vec2::new(1.0, 0.5), Vec2::new(1.0, 3.0), Vec2::new(-1.0, 3.0)],
                edge_seen: vec![false, true, true, true],
            }),
        }
    }

    #[test]
    fn a_layout_waits_for_its_keyframe_pose_and_follows_it_when_it_moves() {
        let mut scene = Scene::new(600);
        assert!(scene.on_room_layout(layout(5)).is_empty(), "no pose yet");
        let posed = scene.on_keyframe_pose(&keyframe_pose(5, 2.0, Tracking::Tracking));
        let sent = json(&posed);
        let layout_sent = sent.iter().find(|m| m["type"] == "room_layout").expect("the layout goes with the pose");
        assert_eq!((layout_sent["drawn"].as_bool(), layout_sent["pose"]["x_m"].as_f64()), (Some(true), Some(2.0)));
        assert_eq!(layout_sent["planes"][0]["kind"], "floor");
        assert_eq!(layout_sent["room"]["edge_seen"], serde_json::json!([false, true, true, true]));
        let moved = PoseGraphUpdate {
            map_version: MapVersion(2),
            keyframes: vec![UpdatedKeyframePose { frame_id: FrameId(5), pose: Pose4Dof { x_m: 2.5, y_m: 0.0, z_m: 0.0, yaw_rad: 0.0 } }],
        };
        let sent = json(&scene.on_pose_graph_update(&moved));
        assert_eq!(sent.iter().find(|m| m["type"] == "room_layout").unwrap()["pose"]["x_m"].as_f64(), Some(2.5));
        assert_eq!(types(&scene.snapshot()), ["reset", "room_layout"]);
    }

    #[test]
    fn an_object_that_disappears_or_merges_is_removed() {
        let mut scene = Scene::new(600);
        let appeared = json(&scene.on_object_update(&object(3, ObjectUpdateKind::Appeared)));
        assert_eq!((appeared[0]["label"].as_str(), appeared[0]["bottom_m"].as_f64()), (Some("chair"), Some(0.0)));
        scene.on_object_update(&object(4, ObjectUpdateKind::Appeared));
        assert_eq!(types(&scene.on_object_update(&object(3, ObjectUpdateKind::Disappeared))), ["object_gone"]);
        scene.on_object_update(&object(4, ObjectUpdateKind::Merged));
        assert_eq!(types(&scene.snapshot()), ["reset"]);
    }
}
