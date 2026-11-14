//! People tracks in 3D: each tracked person's keypoints placed with depth, their torso as their position, smoothed
//! over time, and held for a while as lost once the pose model's tracker stops reporting them.

use std::collections::HashMap;
use std::time::Duration;

use glam::Vec3;
use perception_wire::camera::CameraCalibration;
use perception_wire::people::{
    PeopleTracks, PersonPose, PersonPoses, PersonTrack, Point, PositionFrame, TrackId, TrackState,
    Velocity,
};

use crate::depth::DepthImage;
use crate::one_euro::{OneEuro, OneEuroSettings};

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
    /// How long a track the tracker has dropped is still reported, as lost.
    pub keep_lost: Duration,
    pub smoothing: OneEuroSettings,
}

impl Default for TrackSettings {
    fn default() -> TrackSettings {
        TrackSettings {
            min_keypoint_score: 0.5,
            torso_spread_m: 0.3,
            limb_spread_m: 0.8,
            confirm_after_frames: 3,
            keep_lost: Duration::from_secs(1),
            smoothing: OneEuroSettings::default(),
        }
    }
}

/// The depth frame a pose frame was taken with, and the gravity estimate at the time.
pub struct DepthFrame<'a> {
    pub image: DepthImage<'a>,
    pub gravity_reaction: Option<[f32; 3]>,
}

struct Followed {
    frames_seen: u32,
    last_seen_ns: i64,
    filter: OneEuro,
    filtered_at_ns: Option<i64>,
    last: PersonTrack,
}

pub struct PeopleTracker {
    settings: TrackSettings,
    followed: HashMap<TrackId, Followed>,
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
        PeopleTracker { settings, followed: HashMap::new() }
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

    /// The tracks for one pose frame. `depth` is None when its depth frame never arrived.
    pub fn update(&mut self, poses: &PersonPoses, depth: Option<&DepthFrame>, calibration: Option<&CameraCalibration>) -> PeopleTracks {
        let now_ns = poses.captured_camera_ns;
        let mut tracks = Vec::new();
        for person in &poses.people {
            let Some(track_id) = person.track_id else { continue };
            let (centre, keypoints_3d) = match (depth, calibration) {
                (Some(depth), Some(calibration)) => self.lift(person, &depth.image, calibration),
                _ => (None, vec![None; person.keypoints.len()]),
            };
            let followed = self.followed.entry(track_id).or_insert_with(|| Followed {
                frames_seen: 0,
                last_seen_ns: now_ns,
                filter: OneEuro::new(self.settings.smoothing),
                filtered_at_ns: None,
                last: PersonTrack {
                    track_id,
                    state: TrackState::Tentative,
                    score: person.score,
                    bbox: person.bbox,
                    keypoints: person.keypoints.clone(),
                    position: None,
                    velocity: None,
                    keypoints_3d: vec![],
                },
            });
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
            let track = PersonTrack {
                track_id,
                state,
                score: person.score,
                bbox: person.bbox,
                keypoints: person.keypoints.clone(),
                position: position.or(followed.last.position),
                velocity,
                keypoints_3d: keypoints_3d.into_iter().map(|p| p.map(point)).collect(),
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
        PeopleTracks {
            frame_id: poses.frame_id,
            captured_camera_ns: now_ns,
            width: poses.width,
            height: poses.height,
            position_frame: PositionFrame::Camera,
            gravity_reaction: depth.and_then(|d| d.gravity_reaction),
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
    fn a_confirmed_person_the_tracker_drops_is_lost_for_a_second_then_gone() {
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        for frame in 0..3 {
            tracker.update(&poses(frame, vec![standing_person(Some(7), 32.0)]), None, None);
        }
        let tracks = tracker.update(&poses(4, vec![]), None, None);
        assert_eq!(tracks.tracks.iter().map(|t| (t.track_id, t.state)).collect::<Vec<_>>(), vec![(7, TrackState::Lost)]);
        assert!(tracker.update(&poses(40, vec![]), None, None).tracks.is_empty());
    }

    #[test]
    fn a_tentative_person_the_tracker_drops_is_not_reported_lost() {
        let mut tracker = PeopleTracker::new(TrackSettings::default());
        tracker.update(&poses(0, vec![standing_person(Some(7), 32.0)]), None, None);
        assert!(tracker.update(&poses(1, vec![]), None, None).tracks.is_empty());
    }
}
