//! People tracks in 3D: each tracked person's keypoints placed with depth, their torso as their position, smoothed
//! over time, and held for a while as lost once the pose model's tracker stops reporting them.
//!
//! A person keeps the id this tracker gave them while the pose model's tracker renumbers them: a new pose-tracker id
//! that turns up where a person it stopped reporting was heading takes over that person.

use std::collections::HashMap;
use std::time::Duration;

use glam::{Affine3A, Vec3};
use perception_wire::camera::CameraCalibration;
use perception_wire::people::{
    PeopleTracks, PersonPose, PersonPoses, PersonTrack, Point, PositionFrame, TrackId, TrackState,
    Velocity,
};

use crate::depth::DepthImage;
use crate::one_euro::{OneEuro, OneEuroSettings};
use crate::skeleton::{JointSighting, SkeletonFit, SkeletonSettings};

/// COCO shoulders and hips.
const TORSO_KEYPOINTS: [usize; 4] = [5, 6, 11, 12];

#[derive(Clone, Copy, Debug)]
pub struct TrackSettings {
    /// A keypoint the model is less sure of than this gets no 3D point.
    pub min_keypoint_score: f32,
    /// Torso points further than this from the torso's median depth are taken to have hit something behind.
    pub torso_spread_m: f32,
    /// Any keypoint further than this from the torso's depth is taken to have hit something behind or in front.
    pub limb_spread_m: f32,
    /// Frames a track must be seen in before it is confirmed.
    pub confirm_after_frames: u32,
    /// How long a person the pose tracker has dropped is still reported, as lost, and can be taken over by a new
    /// pose-tracker id.
    pub keep_lost: Duration,
    /// How far from where a dropped person was heading a new pose-tracker id may turn up and still take them over.
    pub rejoin_radius_m: f32,
    pub smoothing: OneEuroSettings,
    pub skeleton: SkeletonSettings,
}

impl Default for TrackSettings {
    fn default() -> TrackSettings {
        TrackSettings {
            min_keypoint_score: 0.5,
            torso_spread_m: 0.3,
            limb_spread_m: 0.8,
            confirm_after_frames: 3,
            keep_lost: Duration::from_secs(2),
            rejoin_radius_m: 0.5,
            smoothing: OneEuroSettings::default(),
            skeleton: SkeletonSettings::default(),
        }
    }
}

/// One tracked person of a pose frame, placed in the colour camera's frame.
pub struct LiftedPerson {
    pose: PersonPose,
    pose_track_id: TrackId,
    centre: Option<Vec3>,
    keypoints_3d: Vec<Option<Vec3>>,
    /// Unit directions from the colour camera through each keypoint the model is sure of.
    rays: Vec<Option<Vec3>>,
}

/// One tracked person of a pose frame, placed in the frame tracks are kept in.
struct PlacedPerson {
    pose: PersonPose,
    pose_track_id: TrackId,
    centre: Option<Vec3>,
    keypoints_3d: Vec<Option<Vec3>>,
    sightings: Vec<Option<JointSighting>>,
}

/// A pose frame's tracked people placed in the colour camera's frame, before tracking.
pub struct LiftedFrame {
    pub frame_id: u64,
    pub captured_camera_ns: i64,
    /// Whether the frame's depth was there to place the people with.
    pub depth_found: bool,
    width: u16,
    height: u16,
    gravity_reaction: Option<[f32; 3]>,
    people: Vec<LiftedPerson>,
}

/// The frame tracks are kept and reported in.
#[derive(Clone, Copy, Debug)]
pub enum Placement {
    /// The colour camera's own frame.
    Camera,
    /// The odom frame, given the transform from the colour camera's frame into it at the pose frame's time.
    Odom(Affine3A),
}

/// The depth frame a pose frame was taken with, and the gravity estimate at the time.
pub struct DepthFrame<'a> {
    pub image: DepthImage<'a>,
    pub gravity_reaction: Option<[f32; 3]>,
}

struct Followed {
    /// The pose tracker's current id for this person.
    pose_track_id: TrackId,
    frames_seen: u32,
    last_seen_ns: i64,
    filter: OneEuro,
    filtered_at_ns: Option<i64>,
    skeleton: SkeletonFit,
    last: PersonTrack,
}

impl Followed {
    /// Where this person would be at `now_ns`, carried on at their last velocity.
    fn heading_to(&self, now_ns: i64) -> Option<Vec3> {
        let position = self.last.position?;
        let position = Vec3::new(position.x_m, position.y_m, position.z_m);
        let gap_s = (now_ns - self.last_seen_ns) as f32 * 1e-9;
        Some(match self.last.velocity {
            Some(v) => position + Vec3::new(v.vx_m_s, v.vy_m_s, v.vz_m_s) * gap_s,
            None => position,
        })
    }
}

pub struct PeopleTracker {
    settings: TrackSettings,
    /// Keyed by this tracker's own person id.
    followed: HashMap<TrackId, Followed>,
    next_id: TrackId,
    /// Time spent fitting skeletons since it was last taken, and how many fits.
    fitting: (std::time::Duration, u32),
}

fn point(v: Vec3) -> Point {
    Point { x_m: v.x, y_m: v.y, z_m: v.z }
}

fn median(mut values: Vec<f32>) -> f32 {
    let middle = values.len() / 2;
    *values.select_nth_unstable_by(middle, f32::total_cmp).1
}

impl PeopleTracker {
    pub fn new(settings: TrackSettings) -> PeopleTracker {
        PeopleTracker { settings, followed: HashMap::new(), next_id: 1, fitting: (std::time::Duration::ZERO, 0) }
    }

    /// Time spent fitting skeletons, and how many fits, since the last call.
    pub fn take_fitting_time(&mut self) -> (std::time::Duration, u32) {
        std::mem::replace(&mut self.fitting, (std::time::Duration::ZERO, 0))
    }

    /// The torso centre and every keypoint in 3D, colour camera coordinates; None where depth cannot say.
    fn lift(&self, person: &PersonPose, depth: &DepthImage, calibration: &CameraCalibration) -> (Option<Vec3>, Vec<Option<Vec3>>) {
        let raw: Vec<Option<Vec3>> = person
            .keypoints
            .iter()
            .map(|k| (k.score >= self.settings.min_keypoint_score).then(|| depth.point_at_color_pixel(calibration, k.x_px, k.y_px)).flatten())
            .collect();
        let torso: Vec<Vec3> = TORSO_KEYPOINTS.iter().filter_map(|&i| raw.get(i).copied().flatten()).collect();
        if torso.is_empty() {
            return (None, vec![None; raw.len()]);
        }
        let torso_z = median(torso.iter().map(|p| p.z).collect());
        let agreeing: Vec<Vec3> = torso.into_iter().filter(|p| (p.z - torso_z).abs() <= self.settings.torso_spread_m).collect();
        let centre = (agreeing.len() >= 2).then(|| agreeing.iter().copied().sum::<Vec3>() / agreeing.len() as f32);
        let keypoints = raw
            .into_iter()
            .map(|p| p.filter(|p| (p.z - torso_z).abs() <= self.settings.limb_spread_m))
            .collect();
        (centre, keypoints)
    }

    /// The person id each pose-tracker id in this frame belongs to: the person it already follows, else a dropped
    /// person it turns up close to where they were heading, else a new person.
    fn assign_people(&mut self, seen: &[(TrackId, Option<Vec3>)], now_ns: i64) -> Vec<TrackId> {
        let following: HashMap<TrackId, TrackId> = self.followed.iter().map(|(&id, f)| (f.pose_track_id, id)).collect();
        let mut assigned: Vec<Option<TrackId>> = seen.iter().map(|(pose_id, _)| following.get(pose_id).copied()).collect();
        let dropped: Vec<(TrackId, Vec3)> = self
            .followed
            .iter()
            .filter(|(&id, f)| f.last_seen_ns < now_ns && !assigned.contains(&Some(id)))
            .filter_map(|(&id, f)| Some((id, f.heading_to(now_ns)?)))
            .collect();
        let mut candidates: Vec<(f32, usize, TrackId)> = seen
            .iter()
            .enumerate()
            .filter(|&(i, _)| assigned[i].is_none())
            .filter_map(|(i, (_, centre))| Some((i, (*centre)?)))
            .flat_map(|(i, centre)| dropped.iter().map(move |&(id, heading)| (centre.distance(heading), i, id)))
            .filter(|&(distance, _, _)| distance <= self.settings.rejoin_radius_m)
            .collect();
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (_, i, id) in candidates {
            if assigned[i].is_none() && !assigned.contains(&Some(id)) {
                assigned[i] = Some(id);
            }
        }
        assigned
            .into_iter()
            .map(|id| {
                id.unwrap_or_else(|| {
                    self.next_id += 1;
                    self.next_id - 1
                })
            })
            .collect()
    }

    /// Forgets everyone, for when positions so far are in a frame that no longer applies. Ids keep counting on.
    pub fn forget_everyone(&mut self) {
        self.followed.clear();
    }

    /// A pose frame's tracked people in the colour camera's frame. `depth` is None when its depth frame never arrived.
    pub fn lift_frame(&self, poses: &PersonPoses, depth: Option<&DepthFrame>, calibration: Option<&CameraCalibration>) -> LiftedFrame {
        let people = poses
            .people
            .iter()
            .filter_map(|person| {
                let pose_track_id = person.track_id?;
                let (centre, keypoints_3d) = match (depth, calibration) {
                    (Some(depth), Some(calibration)) => self.lift(person, &depth.image, calibration),
                    _ => (None, vec![None; person.keypoints.len()]),
                };
                let rays = person
                    .keypoints
                    .iter()
                    .map(|k| {
                        let calibration = calibration.filter(|_| k.score >= self.settings.min_keypoint_score)?;
                        Some(calibration.color.deproject(k.x_px, k.y_px, 1.0).normalize())
                    })
                    .collect();
                Some(LiftedPerson { pose: person.clone(), pose_track_id, centre, keypoints_3d, rays })
            })
            .collect();
        LiftedFrame {
            frame_id: poses.frame_id,
            captured_camera_ns: poses.captured_camera_ns,
            depth_found: depth.is_some(),
            width: poses.width,
            height: poses.height,
            gravity_reaction: depth.and_then(|d| d.gravity_reaction),
            people,
        }
    }

    /// The tracks for one pose frame in the colour camera's frame. `depth` is None when its depth frame never arrived.
    pub fn update(&mut self, poses: &PersonPoses, depth: Option<&DepthFrame>, calibration: Option<&CameraCalibration>) -> PeopleTracks {
        let lifted = self.lift_frame(poses, depth, calibration);
        self.track(lifted, Placement::Camera)
    }

    /// Follows a lifted frame's people, placed as `placement` says. Every frame given here must use the same kind of
    /// placement as the frames before it.
    pub fn track(&mut self, frame: LiftedFrame, placement: Placement) -> PeopleTracks {
        let now_ns = frame.captured_camera_ns;
        let camera_to_frame = match placement {
            Placement::Camera => Affine3A::IDENTITY,
            Placement::Odom(camera_to_odom) => camera_to_odom,
        };
        let place = |p: Vec3| camera_to_frame.transform_point3(p);
        let camera_centre = place(Vec3::ZERO);
        let lifted: Vec<PlacedPerson> = frame
            .people
            .into_iter()
            .map(|p| {
                let keypoints_3d: Vec<Option<Vec3>> = p.keypoints_3d.into_iter().map(|k| k.map(place)).collect();
                let sightings = p
                    .rays
                    .iter()
                    .zip(&keypoints_3d)
                    .map(|(ray, surface)| {
                        let direction = camera_to_frame.transform_vector3((*ray)?);
                        Some(JointSighting { ray_origin: camera_centre, ray_direction: direction, surface: *surface })
                    })
                    .collect();
                PlacedPerson { pose: p.pose, pose_track_id: p.pose_track_id, centre: p.centre.map(place), keypoints_3d, sightings }
            })
            .collect();
        let seen: Vec<(TrackId, Option<Vec3>)> = lifted.iter().map(|p| (p.pose_track_id, p.centre)).collect();
        let person_ids = self.assign_people(&seen, now_ns);
        let mut tracks = Vec::new();
        for (placed, track_id) in lifted.into_iter().zip(person_ids) {
            let PlacedPerson { pose: person, pose_track_id, centre, keypoints_3d, sightings } = placed;
            let person = &person;
            let followed = self.followed.entry(track_id).or_insert_with(|| Followed {
                pose_track_id,
                frames_seen: 0,
                last_seen_ns: now_ns,
                filter: OneEuro::new(self.settings.smoothing),
                filtered_at_ns: None,
                skeleton: SkeletonFit::new(self.settings.skeleton),
                last: PersonTrack {
                    track_id,
                    state: TrackState::Tentative,
                    score: person.score,
                    bbox: person.bbox,
                    keypoints: person.keypoints.clone(),
                    position: None,
                    velocity: None,
                    keypoints_3d: vec![],
                    skeleton_3d: vec![],
                },
            });
            followed.pose_track_id = pose_track_id;
            followed.frames_seen += 1;
            followed.last_seen_ns = now_ns;
            let (position, velocity) = match centre {
                Some(centre) => {
                    let dt_s = followed.filtered_at_ns.map_or(0.0, |then| (now_ns - then) as f32 * 1e-9);
                    followed.filtered_at_ns = Some(now_ns);
                    let (smoothed, rate) = followed.filter.add(centre, dt_s);
                    (Some(point(smoothed)), Some(Velocity { vx_m_s: rate.x, vy_m_s: rate.y, vz_m_s: rate.z }))
                }
                None => (None, None),
            };
            let state = if followed.frames_seen >= self.settings.confirm_after_frames { TrackState::Confirmed } else { TrackState::Tentative };
            let skeleton_3d = match centre {
                Some(centre) => {
                    let started = std::time::Instant::now();
                    let fitted = followed.skeleton.fit(now_ns, &sightings, f64::from(centre.distance(camera_centre)));
                    self.fitting.0 += started.elapsed();
                    self.fitting.1 += 1;
                    fitted.into_iter().map(|p| p.map(point)).collect()
                }
                None => vec![None; person.keypoints.len()],
            };
            let track = PersonTrack {
                track_id,
                state,
                score: person.score,
                bbox: person.bbox,
                keypoints: person.keypoints.clone(),
                position: position.or(followed.last.position),
                velocity,
                keypoints_3d: keypoints_3d.into_iter().map(|p| p.map(point)).collect(),
                skeleton_3d,
            };
            followed.last = track.clone();
            tracks.push(track);
        }
        let keep_lost_ns = self.settings.keep_lost.as_nanos() as i64;
        self.followed.retain(|_, followed| now_ns - followed.last_seen_ns <= keep_lost_ns);
        let mut lost: Vec<PersonTrack> = self
            .followed
            .values()
            .filter(|followed| followed.last_seen_ns < now_ns && followed.last.state == TrackState::Confirmed)
            .map(|followed| PersonTrack { state: TrackState::Lost, velocity: None, ..followed.last.clone() })
            .collect();
        lost.sort_by_key(|track| track.track_id);
        tracks.extend(lost);
        let (position_frame, gravity_reaction) = match placement {
            Placement::Camera => (PositionFrame::Camera, frame.gravity_reaction),
            Placement::Odom(_) => (PositionFrame::Odom, None),
        };
        PeopleTracks {
            frame_id: frame.frame_id,
            captured_camera_ns: now_ns,
            width: frame.width,
            height: frame.height,
            position_frame,
            gravity_reaction,
            tracks,
        }
    }
}

#[cfg(test)]
mod tests {
    use perception_wire::camera::{RigidTransform, StreamIntrinsics};
    use perception_wire::people::{Keypoint, PixelBox};

    use super::*;

    /// A person standing still in a pose's box.
    fn standing_person(track_id: Option<TrackId>, x_px: f32) -> PersonPose {
        let keypoints = (0..17)
            .map(|i| Keypoint { x_px, y_px: 10.0 + i as f32, score: 0.9 })
            .collect();
        PersonPose { track_id, score: 0.875, bbox: PixelBox { x_px: x_px - 5.0, y_px: 5.0, w_px: 10.0, h_px: 30.0 }, keypoints }
    }

    fn calibration() -> CameraCalibration {
        let intrinsics = StreamIntrinsics { width: 64, height: 48, fx: 50.0, fy: 50.0, ppx: 32.0, ppy: 24.0 };
        let identity = RigidTransform { rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], translation_m: [0.0; 3] };
        CameraCalibration { color: intrinsics, depth: intrinsics, depth_to_color: identity, depth_to_imu: identity }
    }

    fn poses(frame_id: u64, people: Vec<PersonPose>) -> PersonPoses {
        PersonPoses { frame_id, captured_camera_ns: frame_id as i64 * 33_333_333, width: 64, height: 48, people }
    }

    fn wall(mm: u16) -> Vec<u16> {
        vec![mm; 64 * 48]
    }

    #[test]
    fn a_person_is_confirmed_after_three_frames_at_the_depth_they_stand() {
        let depth_mm = wall(2500);
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, gravity_reaction: Some([0.0, -9.8, 0.0]) };
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        let states: Vec<TrackState> = (0..3)
            .map(|frame| tracker.update(&poses(frame, vec![standing_person(Some(4), 32.0)]), Some(&depth), Some(&calibration())).tracks[0].state)
            .collect();
        assert_eq!(states, vec![TrackState::Tentative, TrackState::Tentative, TrackState::Confirmed]);
        let tracks = tracker.update(&poses(3, vec![standing_person(Some(4), 32.0)]), Some(&depth), Some(&calibration()));
        let position = tracks.tracks[0].position.unwrap();
        assert!((position.z_m - 2.5).abs() < 1e-4 && position.x_m.abs() < 1e-4, "{position:?}");
        assert_eq!(tracks.gravity_reaction, Some([0.0, -9.8, 0.0]));
        assert_eq!(tracks.position_frame, PositionFrame::Camera);
    }

    #[test]
    fn an_untracked_person_is_left_out() {
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        assert!(tracker.update(&poses(0, vec![standing_person(None, 32.0)]), None, None).tracks.is_empty());
    }

    #[test]
    fn a_keypoint_that_lands_on_the_wall_behind_gets_no_point() {
        // The person is a column at 2 m in front of a wall at 5 m; their right wrist (10) is drawn out over the wall.
        let mut depth_mm = wall(5000);
        for row in 0..48 {
            for column in 28..37 {
                depth_mm[row * 64 + column] = 2000;
            }
        }
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, gravity_reaction: None };
        let mut person = standing_person(Some(1), 32.0);
        person.keypoints[10].x_px = 10.0;
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        let track = &tracker.update(&poses(0, vec![person]), Some(&depth), Some(&calibration())).tracks[0];
        assert!(track.keypoints_3d[10].is_none());
        assert!((track.keypoints_3d[9].unwrap().z_m - 2.0).abs() < 1e-4);
    }

    #[test]
    fn a_confirmed_person_the_tracker_drops_is_lost_for_two_seconds_then_gone() {
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        for frame in 0..3 {
            tracker.update(&poses(frame, vec![standing_person(Some(7), 32.0)]), None, None);
        }
        let tracks = tracker.update(&poses(4, vec![]), None, None);
        assert_eq!(tracks.tracks.iter().map(|t| (t.track_id, t.state)).collect::<Vec<_>>(), vec![(1, TrackState::Lost)]);
        assert!(tracker.update(&poses(70, vec![]), None, None).tracks.is_empty());
    }

    #[test]
    fn a_tentative_person_the_tracker_drops_is_not_reported_lost() {
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        tracker.update(&poses(0, vec![standing_person(Some(7), 32.0)]), None, None);
        assert!(tracker.update(&poses(1, vec![]), None, None).tracks.is_empty());
    }

    /// The person ids and states in one frame's tracks.
    fn ids(tracks: &PeopleTracks) -> Vec<(TrackId, TrackState)> {
        tracks.tracks.iter().map(|t| (t.track_id, t.state)).collect()
    }

    #[test]
    fn a_person_the_pose_tracker_renumbers_keeps_their_id_and_stays_confirmed() {
        let depth_mm = wall(2500);
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, gravity_reaction: None };
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        for frame in 0..3 {
            tracker.update(&poses(frame, vec![standing_person(Some(14), 32.0)]), Some(&depth), Some(&calibration()));
        }
        assert_eq!(ids(&tracker.update(&poses(3, vec![]), Some(&depth), Some(&calibration()))), vec![(1, TrackState::Lost)]);
        let renumbered = tracker.update(&poses(10, vec![standing_person(Some(18), 32.0)]), Some(&depth), Some(&calibration()));
        assert_eq!(ids(&renumbered), vec![(1, TrackState::Confirmed)]);
    }

    #[test]
    fn a_walking_person_is_taken_over_where_they_were_heading() {
        // Walking right at 1 m/s, 2.5 m away: 50 px focal length, so 20 px per metre, 2/3 px per frame.
        let depth_mm = wall(2500);
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, gravity_reaction: None };
        let x_px = |frame: u64| 20.0 + frame as f32 * 20.0 / 30.0;
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        for frame in 0..30 {
            tracker.update(&poses(frame, vec![standing_person(Some(3), x_px(frame))]), Some(&depth), Some(&calibration()));
        }
        // Gone for a second, then back under a new pose-tracker id, 1 m on from where they were last seen.
        let back = tracker.update(&poses(60, vec![standing_person(Some(9), x_px(60))]), Some(&depth), Some(&calibration()));
        assert_eq!(ids(&back), vec![(1, TrackState::Confirmed)]);
    }

    #[test]
    fn someone_who_turns_up_elsewhere_is_a_new_person() {
        let depth_mm = wall(2500);
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, gravity_reaction: None };
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        for frame in 0..3 {
            tracker.update(&poses(frame, vec![standing_person(Some(4), 12.0)]), Some(&depth), Some(&calibration()));
        }
        let tracks = tracker.update(&poses(5, vec![standing_person(Some(5), 52.0)]), Some(&depth), Some(&calibration()));
        assert_eq!(ids(&tracks), vec![(2, TrackState::Tentative), (1, TrackState::Lost)]);
    }

    #[test]
    fn two_renumbered_people_each_keep_their_own_id() {
        let depth_mm = wall(2500);
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, gravity_reaction: None };
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        for frame in 0..3 {
            let people = vec![standing_person(Some(4), 12.0), standing_person(Some(5), 52.0)];
            tracker.update(&poses(frame, people), Some(&depth), Some(&calibration()));
        }
        let people = vec![standing_person(Some(8), 52.0), standing_person(Some(9), 12.0)];
        let tracks = tracker.update(&poses(5, people), Some(&depth), Some(&calibration()));
        assert_eq!(ids(&tracks), vec![(2, TrackState::Confirmed), (1, TrackState::Confirmed)]);
    }

    #[test]
    fn a_person_still_tracked_is_never_taken_over() {
        let depth_mm = wall(2500);
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, gravity_reaction: None };
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        tracker.update(&poses(0, vec![standing_person(Some(4), 32.0)]), Some(&depth), Some(&calibration()));
        let tracks = tracker.update(&poses(1, vec![standing_person(Some(4), 32.0), standing_person(Some(6), 32.0)]), Some(&depth), Some(&calibration()));
        assert_eq!(tracks.tracks.iter().map(|t| t.track_id).collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn tracks_placed_in_odom_carry_the_odom_position_and_no_gravity() {
        let depth_mm = wall(2500);
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, gravity_reaction: Some([0.0, -9.8, 0.0]) };
        let camera_to_odom = Affine3A::from_translation(Vec3::new(1.0, 2.0, 0.5));
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        let lifted = tracker.lift_frame(&poses(0, vec![standing_person(Some(4), 32.0)]), Some(&depth), Some(&calibration()));
        let tracks = tracker.track(lifted, Placement::Odom(camera_to_odom));
        assert_eq!((tracks.position_frame, tracks.gravity_reaction), (PositionFrame::Odom, None));
        // The torso's keypoints sit 5.5 px above the image centre: 0.275 m up at 2.5 m.
        let position = tracks.tracks[0].position.unwrap();
        assert!((position.x_m - 1.0).abs() < 1e-4 && (position.y_m - 1.725).abs() < 1e-4 && (position.z_m - 3.0).abs() < 1e-4, "{position:?}");
        assert!((tracks.tracks[0].keypoints_3d[0].unwrap().z_m - 3.0).abs() < 1e-4);
    }

    #[test]
    fn people_forgotten_after_an_odometry_restart_come_back_under_new_ids() {
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        tracker.update(&poses(0, vec![standing_person(Some(4), 32.0)]), None, None);
        tracker.forget_everyone();
        let tracks = tracker.update(&poses(1, vec![standing_person(Some(4), 32.0)]), None, None);
        assert_eq!(ids(&tracks), vec![(2, TrackState::Tentative)]);
    }
}
