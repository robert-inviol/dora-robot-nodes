//! Transforming the people tracker's confirmed tracks into the map frame.
//!
//! Tracks in the odometry frame are carried into the map with the localizer's latest map to odom correction, when
//! that correction was made in the odom frame the tracks name: after odometry starts again, the last correction
//! belongs to the old odom frame and would put people in the wrong place. Tracks in the camera frame cannot be
//! transformed: the node does not know where the camera was. A track's joints are its
//! `skeleton_3d`, the joints fitted to the person's skeleton; the raw `keypoints_3d` are not recorded.

use perception_wire::localization::{CameraNanos, MapToOdom};
use perception_wire::navigation::{TrackId, UnixNanos};
use perception_wire::people::{self, PeopleTracks, PositionFrame, TrackState};
use world_agent::world::Joint;
use world_store::{FloorVelocity, JointInMap, PersonObservation, Position3};

/// Why a people tracks frame was not transformed into the map frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum NotTransformed {
    #[error("people tracks are in the camera frame; only odometry-frame tracks can be transformed into the map frame")]
    CameraFrame,
    #[error("people tracks are in the odometry frame and the localizer has sent no map to odom correction yet")]
    NoCorrection,
    #[error(
        "people tracks are in the odom frame started at camera time {tracks:?} and the localizer's latest map to odom \
         correction is for the one started at {correction:?}"
    )]
    CorrectionFromAnotherOdomFrame { tracks: CameraNanos, correction: CameraNanos },
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("people track {track} has {joints} skeleton joints, not {}", Joint::ALL.len())]
pub struct WrongJointCount {
    pub track: u64,
    pub joints: usize,
}

/// The confirmed tracks of one frame transformed into the map frame, each with every joint its skeleton fit found. A confirmed
/// track with no torso position is left out.
pub fn transform_people_to_map(
    frame: &PeopleTracks,
    correction: Option<&MapToOdom>,
    seen: UnixNanos,
) -> Result<Result<Vec<PersonObservation>, NotTransformed>, WrongJointCount> {
    let correction = match (frame.position_frame, correction) {
        (PositionFrame::Camera, _) => return Ok(Err(NotTransformed::CameraFrame)),
        (PositionFrame::Odom { .. }, None) => return Ok(Err(NotTransformed::NoCorrection)),
        (PositionFrame::Odom { odom_started }, Some(correction)) if correction.odom_started != odom_started => {
            return Ok(Err(NotTransformed::CorrectionFromAnotherOdomFrame { tracks: odom_started, correction: correction.odom_started }));
        }
        (PositionFrame::Odom { .. }, Some(correction)) => correction,
    };
    let (sin, cos) = correction.correction.yaw_rad.sin_cos();
    let c = correction.correction;
    let to_map = |p: &people::Point| Position3 {
        x_m: cos * p.x_m - sin * p.y_m + c.x_m,
        y_m: sin * p.x_m + cos * p.y_m + c.y_m,
        z_m: p.z_m + c.z_m,
    };
    let mut observations = Vec::new();
    for track in frame.tracks.iter().filter(|t| t.state == TrackState::Confirmed) {
        if track.skeleton_3d.len() != Joint::ALL.len() {
            return Err(WrongJointCount { track: track.track_id, joints: track.skeleton_3d.len() });
        }
        let Some(torso) = &track.position else { continue };
        let joints = Joint::ALL
            .into_iter()
            .zip(&track.skeleton_3d)
            .filter_map(|(joint, point)| point.as_ref().map(|p| JointInMap { joint, position: to_map(p) }))
            .collect();
        let velocity = track.velocity.map(|v| FloorVelocity { vx_m_s: cos * v.vx_m_s - sin * v.vy_m_s, vy_m_s: sin * v.vx_m_s + cos * v.vy_m_s });
        observations.push(PersonObservation {
            track: TrackId(track.track_id),
            seen,
            map_version: correction.map_version,
            torso: to_map(torso),
            velocity,
            joints,
        });
    }
    Ok(Ok(observations))
}

#[cfg(test)]
mod tests {
    use perception_wire::localization::Pose4Dof;
    use perception_wire::navigation::MapVersion;
    use perception_wire::people::{Keypoint, PersonTrack, PixelBox, Velocity};

    use super::*;

    fn point(x_m: f32, y_m: f32, z_m: f32) -> people::Point {
        people::Point { x_m, y_m, z_m }
    }

    fn track(id: u64, state: TrackState, position: Option<people::Point>) -> PersonTrack {
        let mut skeleton_3d = vec![None; Joint::ALL.len()];
        skeleton_3d[5] = Some(point(1.0, 0.2, 1.4));
        skeleton_3d[6] = Some(point(1.0, -0.2, 1.4));
        PersonTrack {
            track_id: id,
            state,
            score: 0.9,
            bbox: PixelBox { x_px: 0.0, y_px: 0.0, w_px: 10.0, h_px: 20.0 },
            keypoints: vec![Keypoint { x_px: 0.0, y_px: 0.0, score: 0.9 }; Joint::ALL.len()],
            position,
            velocity: Some(Velocity { vx_m_s: 0.5, vy_m_s: 0.0, vz_m_s: 0.0 }),
            keypoints_3d: vec![Some(point(9.0, 9.0, 9.0)); Joint::ALL.len()],
            skeleton_3d,
        }
    }

    fn frame(position_frame: PositionFrame, tracks: Vec<PersonTrack>) -> PeopleTracks {
        PeopleTracks { frame_id: 1, captured_camera_ns: 5, width: 1280, height: 720, position_frame, specific_force: None, tracks }
    }

    /// The odom frame the test correction is for.
    const ODOM: PositionFrame = PositionFrame::Odom { odom_started: CameraNanos(0) };

    /// Odometry turned a quarter turn anticlockwise and shifted against the map.
    fn correction() -> MapToOdom {
        MapToOdom {
            map_version: MapVersion(4),
            valid_from: CameraNanos(0),
            odom_started: CameraNanos(0),
            correction: Pose4Dof { x_m: 2.0, y_m: 1.0, z_m: -0.5, yaw_rad: std::f32::consts::FRAC_PI_2 },
            sigma_xy_m: 0.02,
            sigma_z_m: 0.01,
            sigma_yaw_rad: 0.01,
        }
    }

    fn close(a: Position3, b: Position3) -> bool {
        (a.x_m - b.x_m).abs() < 1e-5 && (a.y_m - b.y_m).abs() < 1e-5 && (a.z_m - b.z_m).abs() < 1e-5
    }

    #[test]
    fn confirmed_tracks_in_odometry_are_carried_into_the_map_with_their_joints() {
        let tracks = vec![
            track(3, TrackState::Confirmed, Some(point(1.0, 0.0, 1.2))),
            track(4, TrackState::Tentative, Some(point(0.0, 0.0, 1.2))),
            track(5, TrackState::Lost, Some(point(0.0, 0.0, 1.2))),
            track(6, TrackState::Confirmed, None),
        ];
        let in_map = transform_people_to_map(&frame(ODOM, tracks), Some(&correction()), UnixNanos(9)).unwrap().unwrap();
        assert_eq!(in_map.len(), 1, "only the confirmed track with a torso position");
        let person = &in_map[0];
        assert_eq!((person.track, person.seen, person.map_version), (TrackId(3), UnixNanos(9), MapVersion(4)));
        assert!(close(person.torso, Position3 { x_m: 2.0, y_m: 2.0, z_m: 0.7 }), "{:?}", person.torso);
        let velocity = person.velocity.unwrap();
        assert!(velocity.vx_m_s.abs() < 1e-6 && (velocity.vy_m_s - 0.5).abs() < 1e-6, "{velocity:?}");
        assert_eq!(person.joints.iter().map(|j| j.joint).collect::<Vec<_>>(), [Joint::LeftShoulder, Joint::RightShoulder]);
        assert!(close(person.joints[0].position, Position3 { x_m: 1.8, y_m: 2.0, z_m: 0.9 }), "{:?}", person.joints[0]);
    }

    #[test]
    fn camera_frame_tracks_and_odometry_without_a_correction_are_not_transformed() {
        let tracks = || vec![track(3, TrackState::Confirmed, Some(point(1.0, 0.0, 1.2)))];
        assert_eq!(transform_people_to_map(&frame(PositionFrame::Camera, tracks()), Some(&correction()), UnixNanos(9)), Ok(Err(NotTransformed::CameraFrame)));
        assert_eq!(transform_people_to_map(&frame(ODOM, tracks()), None, UnixNanos(9)), Ok(Err(NotTransformed::NoCorrection)));
    }

    #[test]
    fn after_an_odometry_restart_tracks_wait_for_a_correction_in_their_own_odom_frame() {
        let tracks = || vec![track(3, TrackState::Confirmed, Some(point(1.0, 0.0, 1.2)))];
        let restarted = PositionFrame::Odom { odom_started: CameraNanos(500) };
        assert_eq!(
            transform_people_to_map(&frame(restarted, tracks()), Some(&correction()), UnixNanos(9)),
            Ok(Err(NotTransformed::CorrectionFromAnotherOdomFrame { tracks: CameraNanos(500), correction: CameraNanos(0) })),
            "the old odom frame's correction does not place them"
        );
        let in_the_new_frame = MapToOdom { odom_started: CameraNanos(500), valid_from: CameraNanos(600), ..correction() };
        let in_map = transform_people_to_map(&frame(restarted, tracks()), Some(&in_the_new_frame), UnixNanos(9)).unwrap().unwrap();
        assert!(close(in_map[0].torso, Position3 { x_m: 2.0, y_m: 2.0, z_m: 0.7 }), "{:?}", in_map[0].torso);
    }

    #[test]
    fn a_track_without_every_joint_slot_is_refused() {
        let mut short = track(3, TrackState::Confirmed, Some(point(1.0, 0.0, 1.2)));
        short.skeleton_3d.pop();
        let refused = transform_people_to_map(&frame(ODOM, vec![short]), Some(&correction()), UnixNanos(9));
        assert_eq!(refused, Err(WrongJointCount { track: 3, joints: 16 }));
    }
}
