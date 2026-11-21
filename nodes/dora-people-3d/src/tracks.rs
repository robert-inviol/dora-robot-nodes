//! People tracks in 3D: each tracked person's keypoints lifted to 3D with depth, their torso as their position, their
//! joints fitted to a skeleton, both smoothed over time, and held for a while as lost once the pose model's tracker
//! stops reporting them.
//!
//! A person keeps the id this tracker gave them while the pose model's tracker renumbers them: a new pose-tracker id
//! that turns up on the way from where a person it stopped reporting was last seen to where they were heading, nearer
//! to them than to anyone it still reports, takes over that person.

use std::collections::HashMap;
use std::time::Duration;

use glam::{Affine3A, Vec3};
use perception_wire::camera::CameraCalibration;
use perception_wire::localization::CameraNanos;
use perception_wire::people::{
    PeopleTracks, PersonPose, PersonPoses, PersonTrack, Point, PositionFrame, TrackId, TrackState,
    Velocity,
};

use crate::depth::DepthImage;
use crate::joint_smoothing::{JointSmoothing, JointSmoothingSettings};
use crate::one_euro::{OneEuro, OneEuroSettings};
use crate::skeleton::{JointObservation, SkeletonFit, SkeletonSettings};

/// COCO shoulders and hips.
const TORSO_KEYPOINTS: [usize; 4] = [5, 6, 11, 12];

#[derive(Clone, Copy, Debug)]
pub struct TrackSettings {
    /// A keypoint the model is less sure of than this gets no 3D point.
    pub min_keypoint_score: f32,
    /// The depth span of one group of torso readings; see [`torso_in_front`].
    pub torso_spread_m: f32,
    /// Any keypoint further than this from the torso's depth is taken to have hit something behind or in front.
    pub limb_spread_m: f32,
    /// Frames a track must be seen in before it is confirmed.
    pub confirm_after_frames: u32,
    /// How long a person the pose tracker has dropped is still reported, as lost, and can be taken over by a new
    /// pose-tracker id.
    pub keep_lost: Duration,
    /// How far from the way between where a dropped person was last seen and where they were heading a new
    /// pose-tracker id may turn up and still take them over.
    pub rejoin_radius_m: f32,
    /// The torso position's filter.
    pub smoothing: OneEuroSettings,
    pub skeleton: SkeletonSettings,
    /// Each fitted joint's filter.
    pub joint_smoothing: JointSmoothingSettings,
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
            joint_smoothing: JointSmoothingSettings::DEFAULT,
        }
    }
}

/// One tracked person of a pose frame, lifted to 3D in the colour camera's frame.
#[derive(Clone)]
pub struct LiftedPerson {
    pose: PersonPose,
    pose_track_id: TrackId,
    centre: Option<Vec3>,
    keypoints_3d: Vec<Option<Vec3>>,
    /// Unit directions from the colour camera through each keypoint the model is sure of.
    rays: Vec<Option<Vec3>>,
}

/// One tracked person of a pose frame, in the frame tracks are kept in.
struct PersonInTrackFrame {
    pose: PersonPose,
    pose_track_id: TrackId,
    centre: Option<Vec3>,
    keypoints_3d: Vec<Option<Vec3>>,
    observations: Vec<Option<JointObservation>>,
}

/// A pose frame's tracked people lifted to 3D in the colour camera's frame, before tracking.
#[derive(Clone)]
pub struct LiftedFrame {
    pub frame_id: u64,
    pub captured_camera_ns: i64,
    /// Whether the frame's depth was there to lift the people with.
    pub depth_found: bool,
    width: u16,
    height: u16,
    specific_force: Option<[f32; 3]>,
    people: Vec<LiftedPerson>,
}

/// The frame tracks are kept and reported in.
#[derive(Clone, Copy, Debug)]
pub enum TrackFrame {
    /// The colour camera's own frame.
    Camera,
    /// The odom frame named by `odom_started`, given the transform from the colour camera's frame into it at the
    /// pose frame's time.
    Odom { camera_to_odom: Affine3A, odom_started: CameraNanos },
}

/// The depth frame a pose frame was taken with, and the gravity estimate at the time.
pub struct DepthFrame<'a> {
    pub image: DepthImage<'a>,
    pub specific_force: Option<[f32; 3]>,
}

/// Where a person was when they were last seen with a position.
struct LastPosition {
    seen_ns: i64,
    position: Vec3,
    velocity: Vec3,
    skeleton_3d: Vec<Option<Point>>,
}

struct Followed {
    /// The pose tracker's current id for this person.
    pose_track_id: TrackId,
    frames_seen: u32,
    last_seen_ns: i64,
    filter: OneEuro,
    filtered_at_ns: Option<i64>,
    skeleton: SkeletonFit,
    joint_smoothing: JointSmoothing,
    last: PersonTrack,
    last_position: Option<LastPosition>,
}

/// The way a dropped person may have gone: from where they were last seen to where their last velocity carries them
/// by now. The velocity of someone standing still is noise, so a person who turns up again where they stood is on
/// it as surely as one who walked on.
#[derive(Clone, Copy, Debug)]
struct WayAhead {
    from: Vec3,
    to: Vec3,
}

impl WayAhead {
    fn distance(&self, point: Vec3) -> f32 {
        let along = self.to - self.from;
        let t = if along.length_squared() > 0.0 { ((point - self.from).dot(along) / along.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
        point.distance(self.from + along * t)
    }
}

impl Followed {
    /// The way this person may have gone by `now_ns`, carried on at their last velocity.
    fn way_ahead(&self, now_ns: i64) -> Option<WayAhead> {
        let last = self.last_position.as_ref()?;
        let gap_s = (now_ns - last.seen_ns) as f32 * 1e-9;
        Some(WayAhead { from: last.position, to: last.position + last.velocity * gap_s })
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

/// The torso readings that are the person, by the in-front rule: a person stands in front of whatever is behind
/// them. The readings are grouped by depth, nearest first, each group holding the readings within `spread_m` beyond
/// its nearest. The group holding a strict majority of the readings is the person; with no strict majority the
/// nearest group of two or more is. None when the chosen group has fewer than two readings, for then depth cannot
/// say where the torso is.
fn torso_in_front(readings: &[Vec3], spread_m: f32) -> Option<Vec<Vec3>> {
    let mut by_depth = readings.to_vec();
    by_depth.sort_by(|a, b| a.z.total_cmp(&b.z));
    let mut groups: Vec<Vec<Vec3>> = Vec::new();
    for reading in by_depth {
        match groups.last_mut() {
            Some(group) if reading.z - group[0].z <= spread_m => group.push(reading),
            _ => groups.push(vec![reading]),
        }
    }
    let majority = groups.iter().find(|group| group.len() * 2 > readings.len());
    let chosen = majority.or_else(|| groups.iter().find(|group| group.len() >= 2))?;
    (chosen.len() >= 2).then(|| chosen.clone())
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
        let Some(person) = torso_in_front(&torso, self.settings.torso_spread_m) else {
            return (None, vec![None; raw.len()]);
        };
        let centre = person.iter().copied().sum::<Vec3>() / person.len() as f32;
        let keypoints = raw
            .into_iter()
            .map(|p| p.filter(|p| (p.z - centre.z).abs() <= self.settings.limb_spread_m))
            .collect();
        (Some(centre), keypoints)
    }

    /// The person id each pose-tracker id in this frame belongs to: the person it already follows, else a dropped
    /// person it turns up close to the way they were going, else a new person. A dropped person is one whose
    /// pose-tracker id is not in this frame. A newcomer takes over a dropped person only when it is nearer to where
    /// they were going than to anyone the pose tracker still reports, since otherwise it may be one of them seen
    /// twice, or someone beside them.
    fn assign_people(&mut self, seen: &[(TrackId, Option<Vec3>)], now_ns: i64) -> Vec<TrackId> {
        let following: HashMap<TrackId, TrackId> = self.followed.iter().map(|(&id, f)| (f.pose_track_id, id)).collect();
        let mut assigned: Vec<Option<TrackId>> = seen.iter().map(|(pose_id, _)| following.get(pose_id).copied()).collect();
        let dropped: Vec<(TrackId, WayAhead)> = self
            .followed
            .iter()
            .filter(|(&id, f)| !assigned.contains(&Some(id)) && !seen.iter().any(|(pose_id, _)| *pose_id == f.pose_track_id))
            .filter_map(|(&id, f)| Some((id, f.way_ahead(now_ns)?)))
            .collect();
        let still_reported: Vec<Vec3> =
            seen.iter().zip(&assigned).filter(|(_, person)| person.is_some()).filter_map(|((_, centre), _)| *centre).collect();
        let nearest_reported = |centre: Vec3| still_reported.iter().map(|other| centre.distance(*other)).fold(f32::INFINITY, f32::min);
        let mut candidates: Vec<(f32, usize, TrackId)> = seen
            .iter()
            .enumerate()
            .filter(|&(i, _)| assigned[i].is_none())
            .filter_map(|(i, (_, centre))| Some((i, (*centre)?)))
            .flat_map(|(i, centre)| dropped.iter().map(move |&(id, way)| (centre, i, id, way.distance(centre))))
            .filter(|&(centre, _, _, distance)| distance <= self.settings.rejoin_radius_m && distance < nearest_reported(centre))
            .map(|(_, i, id, distance)| (distance, i, id))
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
            specific_force: depth.and_then(|d| d.specific_force),
            people,
        }
    }

    /// The tracks for one pose frame in the colour camera's frame. `depth` is None when its depth frame never arrived.
    pub fn update(&mut self, poses: &PersonPoses, depth: Option<&DepthFrame>, calibration: Option<&CameraCalibration>) -> PeopleTracks {
        let lifted = self.lift_frame(poses, depth, calibration);
        self.track(lifted, TrackFrame::Camera)
    }

    /// Follows a lifted frame's people, transformed into `track_frame`. Every frame given here must use the same kind of
    /// track frame as the frames before it.
    pub fn track(&mut self, frame: LiftedFrame, track_frame: TrackFrame) -> PeopleTracks {
        let now_ns = frame.captured_camera_ns;
        let camera_to_frame = match track_frame {
            TrackFrame::Camera => Affine3A::IDENTITY,
            TrackFrame::Odom { camera_to_odom, .. } => camera_to_odom,
        };
        let to_track_frame = |p: Vec3| camera_to_frame.transform_point3(p);
        let camera_centre = to_track_frame(Vec3::ZERO);
        let lifted: Vec<PersonInTrackFrame> = frame
            .people
            .into_iter()
            .map(|p| {
                let keypoints_3d: Vec<Option<Vec3>> = p.keypoints_3d.into_iter().map(|k| k.map(to_track_frame)).collect();
                let observations = p
                    .rays
                    .iter()
                    .zip(&keypoints_3d)
                    .map(|(ray, surface)| {
                        let direction = camera_to_frame.transform_vector3((*ray)?);
                        Some(JointObservation { ray_origin: camera_centre, ray_direction: direction, surface: *surface })
                    })
                    .collect();
                PersonInTrackFrame { pose: p.pose, pose_track_id: p.pose_track_id, centre: p.centre.map(to_track_frame), keypoints_3d, observations }
            })
            .collect();
        let seen: Vec<(TrackId, Option<Vec3>)> = lifted.iter().map(|p| (p.pose_track_id, p.centre)).collect();
        let person_ids = self.assign_people(&seen, now_ns);
        let mut tracks = Vec::new();
        for (in_frame, track_id) in lifted.into_iter().zip(person_ids) {
            let PersonInTrackFrame { pose: person, pose_track_id, centre, keypoints_3d, observations } = in_frame;
            let person = &person;
            let followed = self.followed.entry(track_id).or_insert_with(|| Followed {
                pose_track_id,
                frames_seen: 0,
                last_seen_ns: now_ns,
                filter: OneEuro::new(self.settings.smoothing),
                filtered_at_ns: None,
                skeleton: SkeletonFit::new(self.settings.skeleton),
                joint_smoothing: JointSmoothing::new(self.settings.joint_smoothing),
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
                last_position: None,
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
                    let fitted = followed.skeleton.fit(now_ns, &observations, f64::from(centre.distance(camera_centre)));
                    self.fitting.0 += started.elapsed();
                    self.fitting.1 += 1;
                    followed.joint_smoothing.smooth(now_ns, &fitted).into_iter().map(|p| p.map(point)).collect()
                }
                None => vec![None; person.keypoints.len()],
            };
            let track = PersonTrack {
                track_id,
                state,
                score: person.score,
                bbox: person.bbox,
                keypoints: person.keypoints.clone(),
                position,
                velocity,
                keypoints_3d: keypoints_3d.into_iter().map(|p| p.map(point)).collect(),
                skeleton_3d,
            };
            if let (Some(centre), Some(rate)) = (position, velocity) {
                followed.last_position = Some(LastPosition {
                    seen_ns: now_ns,
                    position: Vec3::new(centre.x_m, centre.y_m, centre.z_m),
                    velocity: Vec3::new(rate.vx_m_s, rate.vy_m_s, rate.vz_m_s),
                    skeleton_3d: track.skeleton_3d.clone(),
                });
            }
            followed.last = track.clone();
            tracks.push(track);
        }
        let keep_lost_ns = self.settings.keep_lost.as_nanos() as i64;
        self.followed.retain(|_, followed| now_ns - followed.last_seen_ns <= keep_lost_ns);
        let mut lost: Vec<PersonTrack> = self
            .followed
            .values()
            .filter(|followed| followed.last_seen_ns < now_ns && followed.last.state == TrackState::Confirmed)
            .map(|followed| PersonTrack {
                state: TrackState::Lost,
                position: followed.last_position.as_ref().map(|last| point(last.position)),
                velocity: None,
                skeleton_3d: followed.last_position.as_ref().map_or_else(|| vec![None; followed.last.keypoints.len()], |last| last.skeleton_3d.clone()),
                ..followed.last.clone()
            })
            .collect();
        lost.sort_by_key(|track| track.track_id);
        tracks.extend(lost);
        let (position_frame, specific_force) = match track_frame {
            TrackFrame::Camera => (PositionFrame::Camera, frame.specific_force),
            TrackFrame::Odom { odom_started, .. } => (PositionFrame::Odom { odom_started }, None),
        };
        PeopleTracks {
            frame_id: frame.frame_id,
            captured_camera_ns: now_ns,
            width: frame.width,
            height: frame.height,
            position_frame,
            specific_force,
            tracks,
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3A;
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
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, specific_force: Some([0.0, -9.8, 0.0]) };
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        let states: Vec<TrackState> = (0..3)
            .map(|frame| tracker.update(&poses(frame, vec![standing_person(Some(4), 32.0)]), Some(&depth), Some(&calibration())).tracks[0].state)
            .collect();
        assert_eq!(states, vec![TrackState::Tentative, TrackState::Tentative, TrackState::Confirmed]);
        let tracks = tracker.update(&poses(3, vec![standing_person(Some(4), 32.0)]), Some(&depth), Some(&calibration()));
        let position = tracks.tracks[0].position.unwrap();
        assert!((position.z_m - 2.5).abs() < 1e-4 && position.x_m.abs() < 1e-4, "{position:?}");
        assert_eq!(tracks.specific_force, Some([0.0, -9.8, 0.0]));
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
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, specific_force: None };
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
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, specific_force: None };
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
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, specific_force: None };
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
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, specific_force: None };
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
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, specific_force: None };
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
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, specific_force: None };
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        tracker.update(&poses(0, vec![standing_person(Some(4), 32.0)]), Some(&depth), Some(&calibration()));
        let tracks = tracker.update(&poses(1, vec![standing_person(Some(4), 32.0), standing_person(Some(6), 32.0)]), Some(&depth), Some(&calibration()));
        assert_eq!(tracks.tracks.iter().map(|t| t.track_id).collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn tracks_in_odom_carry_the_odom_position_and_no_gravity() {
        let depth_mm = wall(2500);
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, specific_force: Some([0.0, -9.8, 0.0]) };
        let camera_to_odom = Affine3A::from_translation(Vec3::new(1.0, 2.0, 0.5));
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        let lifted = tracker.lift_frame(&poses(0, vec![standing_person(Some(4), 32.0)]), Some(&depth), Some(&calibration()));
        let tracks = tracker.track(lifted, TrackFrame::Odom { camera_to_odom, odom_started: CameraNanos(250) });
        assert_eq!((tracks.position_frame, tracks.specific_force), (PositionFrame::Odom { odom_started: CameraNanos(250) }, None));
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

    /// A person 2 m away in front of a wall at 5 m: a column of the image 9 px wide round x = 32 px.
    fn person_before_a_wall() -> Vec<u16> {
        let mut depth_mm = wall(5000);
        for row in 0..48 {
            for column in 28..37 {
                depth_mm[row * 64 + column] = 2000;
            }
        }
        depth_mm
    }

    #[test]
    fn a_torso_half_on_the_wall_behind_is_placed_on_the_person_in_front() {
        let depth_mm = person_before_a_wall();
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, specific_force: None };
        let mut person = standing_person(Some(1), 32.0);
        // The right shoulder and right hip are drawn out over the wall.
        person.keypoints[6].x_px = 10.0;
        person.keypoints[12].x_px = 10.0;
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        let track = &tracker.update(&poses(0, vec![person]), Some(&depth), Some(&calibration())).tracks[0];
        let position = track.position.expect("the two torso readings on the person place them");
        assert!((position.z_m - 2.0).abs() < 1e-4, "{position:?}");
        assert!((track.keypoints_3d[9].unwrap().z_m - 2.0).abs() < 1e-4, "the person's own keypoints are kept");
        assert!(track.keypoints_3d[6].is_none(), "the reading on the wall is not the person");
    }

    #[test]
    fn a_torso_with_one_reading_on_the_person_and_one_on_the_wall_has_no_position() {
        let depth_mm = person_before_a_wall();
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, specific_force: None };
        let mut person = standing_person(Some(1), 32.0);
        person.keypoints[6].x_px = 10.0;
        person.keypoints[11].score = 0.1;
        person.keypoints[12].score = 0.1;
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        let track = &tracker.update(&poses(0, vec![person]), Some(&depth), Some(&calibration())).tracks[0];
        assert!(track.position.is_none());
        assert!(track.keypoints_3d.iter().all(Option::is_none), "no keypoint is trusted without a torso depth: {:?}", track.keypoints_3d);
    }

    #[test]
    fn the_in_front_rule_takes_a_majority_then_the_nearest_group_of_two() {
        let at = |z: f32| Vec3::new(0.0, 0.0, z);
        let depths = |readings: &[f32]| torso_in_front(&readings.iter().map(|&z| at(z)).collect::<Vec<_>>(), 0.3).map(|group| group.iter().map(|p| p.z).collect::<Vec<_>>());
        assert_eq!(depths(&[5.0, 2.0, 2.1, 5.1]), Some(vec![2.0, 2.1]), "two and two: the nearer pair");
        assert_eq!(depths(&[2.0, 5.0, 5.1]), Some(vec![5.0, 5.1]), "a strict majority wins");
        assert_eq!(depths(&[1.5, 2.0, 2.1, 5.0]), Some(vec![2.0, 2.1]), "a lone nearer reading is not a group");
        assert_eq!(depths(&[2.0, 5.0]), None);
        assert_eq!(depths(&[2.0]), None);
    }

    #[test]
    fn a_person_seen_without_depth_has_no_position_that_frame_and_is_lost_where_last_positioned() {
        let depth_mm = wall(2500);
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, specific_force: None };
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        for frame in 0..3 {
            tracker.update(&poses(frame, vec![standing_person(Some(4), 32.0)]), Some(&depth), Some(&calibration()));
        }
        let without_depth = tracker.update(&poses(3, vec![standing_person(Some(4), 32.0)]), None, Some(&calibration()));
        assert_eq!(ids(&without_depth), vec![(1, TrackState::Confirmed)]);
        assert!(without_depth.tracks[0].position.is_none(), "no position is reported for a frame without depth");
        let lost = tracker.update(&poses(4, vec![]), None, Some(&calibration()));
        assert_eq!(ids(&lost), vec![(1, TrackState::Lost)]);
        let position = lost.tracks[0].position.expect("lost where last positioned");
        assert!((position.z_m - 2.5).abs() < 1e-3, "{position:?}");
        assert!(lost.tracks[0].skeleton_3d.iter().any(Option::is_some), "with the skeleton last fitted");
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

    /// The D455's colour camera at 1280x720.
    const FOCAL_PX: f32 = 640.0;
    const CENTRE_PX: (f32, f32) = (640.0, 360.0);

    /// What the camera sees of a person, and how noisy it is.
    struct Sighting {
        /// Each keypoint's pixel is off by this much each way.
        pixel_sigma_px: f32,
        /// Each depth reading is off by this much along the ray.
        depth_sigma_m: f32,
    }

    /// The joint centres of a 1.7 m body standing facing the camera with its hips' midpoint `at`, camera
    /// coordinates (x right, y down, z forward); the face at the head.
    fn body_at(at: Vec3) -> [Vec3; 17] {
        let mut joints = [at + Vec3::new(0.0, -0.62, 0.0); 17];
        let offsets = [
            (5, 0.183, -0.48), (6, -0.183, -0.48), (7, 0.20, -0.164), (8, -0.20, -0.164), (9, 0.21, 0.084), (10, -0.21, 0.084),
            (11, 0.10, 0.0), (12, -0.10, 0.0), (13, 0.10, 0.417), (14, -0.10, 0.417), (15, 0.10, 0.835), (16, -0.10, 0.835),
        ];
        for (k, x, y) in offsets {
            joints[k] = at + Vec3::new(x, y, 0.0);
        }
        joints
    }

    /// One pose frame of a person, as `lift_frame` would make it from noisy keypoints and depth readings.
    fn sighted(frame_id: u64, pose_track_id: TrackId, body: &[Vec3; 17], sighting: &Sighting, noise: &mut Noise) -> LiftedFrame {
        let mut keypoints = Vec::new();
        let mut keypoints_3d = Vec::new();
        let mut rays = Vec::new();
        for joint in body {
            let x_px = FOCAL_PX * joint.x / joint.z + CENTRE_PX.0 + noise.normal(sighting.pixel_sigma_px);
            let y_px = FOCAL_PX * joint.y / joint.z + CENTRE_PX.1 + noise.normal(sighting.pixel_sigma_px);
            let ray = Vec3::new((x_px - CENTRE_PX.0) / FOCAL_PX, (y_px - CENTRE_PX.1) / FOCAL_PX, 1.0);
            // The reading is on the skin facing the camera, 5 cm in front of the joint's centre.
            let skin_z = joint.z - 0.05 + noise.normal(sighting.depth_sigma_m);
            keypoints.push(Keypoint { x_px, y_px, score: 0.9 });
            keypoints_3d.push(Some(ray * skin_z));
            rays.push(Some(ray.normalize()));
        }
        let torso: Vec<Vec3> = TORSO_KEYPOINTS.iter().map(|&k| keypoints_3d[k].unwrap()).collect();
        let centre = torso.iter().copied().sum::<Vec3>() / torso.len() as f32;
        let pose = PersonPose { track_id: Some(pose_track_id), score: 0.9, bbox: PixelBox { x_px: 0.0, y_px: 0.0, w_px: 1.0, h_px: 1.0 }, keypoints };
        LiftedFrame {
            frame_id,
            captured_camera_ns: frame_id as i64 * 33_333_333,
            depth_found: true,
            width: 1280,
            height: 720,
            specific_force: None,
            people: vec![LiftedPerson { pose, pose_track_id, centre: Some(centre), keypoints_3d, rays }],
        }
    }

    /// A still camera 1 m above the odom origin looking along odom y.
    fn still_camera() -> TrackFrame {
        let camera_to_odom = Affine3A::from_cols(Vec3A::X, -Vec3A::Z, Vec3A::Y, Vec3A::new(0.0, 0.0, 1.0));
        TrackFrame::Odom { camera_to_odom, odom_started: CameraNanos(0) }
    }

    /// A tracker whose joints are the fit's own, unsmoothed: a cutoff far above the frame rate passes every sample.
    fn unsmoothed() -> TrackSettings {
        let pass_through = OneEuroSettings { min_cutoff_hz: 1e9, ..JointSmoothingSettings::DEFAULT.filter };
        TrackSettings { joint_smoothing: JointSmoothingSettings { filter: pass_through, ..JointSmoothingSettings::DEFAULT }, ..TrackSettings::default() }
    }

    fn joint_in_odom(track_frame: TrackFrame, camera_point: Vec3) -> Vec3 {
        let TrackFrame::Odom { camera_to_odom, .. } = track_frame else { unreachable!("tests track in odom") };
        camera_to_odom.transform_point3(camera_point)
    }

    fn fitted(track: &PersonTrack, joint: usize) -> Vec3 {
        let p = track.skeleton_3d[joint].expect("the body joints are fitted");
        Vec3::new(p.x_m, p.y_m, p.z_m)
    }

    /// The body joints, shoulders to ankles.
    const BODY_JOINTS: std::ops::Range<usize> = 5..17;

    /// Root mean square distance of each body joint from its own mean over the frames, averaged over the joints.
    fn joint_jitter_m(frames: &[PersonTrack]) -> f32 {
        let per_joint: Vec<f32> = BODY_JOINTS
            .map(|joint| {
                let mean = frames.iter().map(|t| fitted(t, joint)).sum::<Vec3>() / frames.len() as f32;
                (frames.iter().map(|t| fitted(t, joint).distance_squared(mean)).sum::<f32>() / frames.len() as f32).sqrt()
            })
            .collect();
        per_joint.iter().sum::<f32>() / per_joint.len() as f32
    }


    #[test]
    fn a_still_persons_joints_hold_still_at_three_and_a_half_metres() {
        // Pixels off by 1.5 px and depth by 3 cm each frame: the D455 at 3.5 m after the 5 x 5 median.
        let sighting = Sighting { pixel_sigma_px: 1.5, depth_sigma_m: 0.03 };
        let body = body_at(Vec3::new(0.3, 0.2, 3.5));
        let (mut smoothed, mut raw) = (PeopleTracker::new(TrackSettings::default()), PeopleTracker::new(unsmoothed()));
        let (mut smoothed_tracks, mut raw_tracks) = (Vec::new(), Vec::new());
        let mut noise = Noise(7);
        for frame in 0..300 {
            let seen = sighted(frame, 4, &body, &sighting, &mut noise);
            let again = seen.clone();
            let (a, b) = (smoothed.track(seen, still_camera()), raw.track(again, still_camera()));
            // The first second settles the bone lengths and the filters.
            if frame >= 30 {
                smoothed_tracks.push(a.tracks[0].clone());
                raw_tracks.push(b.tracks[0].clone());
            }
        }
        let (before, after) = (joint_jitter_m(&raw_tracks), joint_jitter_m(&smoothed_tracks));
        assert!(before > 0.012, "the fit alone jitters by {before} m");
        assert!(after < before / 3.0, "smoothing cuts the jitter from {before} m to {after} m");
        let hip = joint_in_odom(still_camera(), body[11]);
        let mean_hip = smoothed_tracks.iter().map(|t| fitted(t, 11)).sum::<Vec3>() / smoothed_tracks.len() as f32;
        assert!(mean_hip.distance(hip) < 0.05, "the smoothed hip stays where the hip is: {mean_hip} for {hip}");
    }

    #[test]
    fn a_walking_persons_joints_follow_with_a_lag_of_a_few_centimetres() {
        // Walking across the view at 1.2 m/s, 3 m away, with the same noise as above.
        let sighting = Sighting { pixel_sigma_px: 1.5, depth_sigma_m: 0.03 };
        let walk_m_s = 1.2;
        let body = |frame: u64| body_at(Vec3::new(-1.0 + walk_m_s * frame as f32 / 30.0, 0.2, 3.0));
        let (mut smoothed, mut raw) = (PeopleTracker::new(TrackSettings::default()), PeopleTracker::new(unsmoothed()));
        let mut noise = Noise(11);
        let (mut smoothed_behind, mut raw_behind) = (Vec::new(), Vec::new());
        for frame in 0..60 {
            let seen = sighted(frame, 4, &body(frame), &sighting, &mut noise);
            let again = seen.clone();
            let (a, b) = (smoothed.track(seen, still_camera()), raw.track(again, still_camera()));
            if frame >= 30 {
                // Camera x is odom x, the way the person walks.
                for joint in BODY_JOINTS {
                    let truth = joint_in_odom(still_camera(), body(frame)[joint]);
                    smoothed_behind.push(truth.x - fitted(&a.tracks[0], joint).x);
                    raw_behind.push(truth.x - fitted(&b.tracks[0], joint).x);
                }
            }
        }
        let mean = |values: &[f32]| values.iter().sum::<f32>() / values.len() as f32;
        let lag_m = mean(&smoothed_behind) - mean(&raw_behind);
        assert!(lag_m < 0.05, "the smoothed joints trail the fit's by {lag_m} m, {} ms at 1.2 m/s", lag_m / walk_m_s * 1e3);
    }

    #[test]
    fn a_still_person_the_pose_tracker_drops_for_almost_two_seconds_keeps_their_id() {
        // Sitting 4.5 m away, depth off by 6 cm a reading: the velocity of the noisy torso is noise of up to a few
        // tenths of a metre a second, which carried on over a 1.8 s gap points well away from where they sit.
        let sighting = Sighting { pixel_sigma_px: 2.0, depth_sigma_m: 0.06 };
        let body = body_at(Vec3::new(-0.8, 0.0, 4.5));
        for seed in 1..=20 {
            let mut tracker = PeopleTracker::new(TrackSettings::default());
            let mut noise = Noise(seed);
            for frame in 0..90 {
                tracker.track(sighted(frame, 9, &body, &sighting, &mut noise), still_camera());
            }
            let back = tracker.track(sighted(144, 11, &body, &sighting, &mut noise), still_camera());
            assert_eq!(ids(&back), vec![(1, TrackState::Confirmed)], "seed {seed}");
        }
    }

    #[test]
    fn a_newcomer_beside_someone_still_tracked_does_not_take_over_a_person_missing_for_a_frame() {
        // A at 32 px and B at 38 px, 0.3 m apart at 2.5 m. A is missed for one frame while a newcomer turns up 0.1 m
        // from B and 0.2 m from A: B seen twice, or someone beside B, not A.
        let depth_mm = wall(2500);
        let depth = DepthFrame { image: DepthImage { width: 64, height: 48, depth_mm: &depth_mm }, specific_force: None };
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        for frame in 0..3 {
            let people = vec![standing_person(Some(4), 32.0), standing_person(Some(5), 38.0)];
            tracker.update(&poses(frame, people), Some(&depth), Some(&calibration()));
        }
        let people = vec![standing_person(Some(5), 38.0), standing_person(Some(7), 36.0)];
        let missed = tracker.update(&poses(3, people), Some(&depth), Some(&calibration()));
        assert_eq!(ids(&missed), vec![(2, TrackState::Confirmed), (3, TrackState::Tentative), (1, TrackState::Lost)]);
        let people = vec![standing_person(Some(4), 32.0), standing_person(Some(5), 38.0)];
        let back = tracker.update(&poses(4, people), Some(&depth), Some(&calibration()));
        assert_eq!(ids(&back), vec![(1, TrackState::Confirmed), (2, TrackState::Confirmed)]);
    }
}
