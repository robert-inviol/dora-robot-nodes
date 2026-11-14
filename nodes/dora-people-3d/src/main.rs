//! dora node: joins each frame's `person_poses` with that frame's raw `depth` and the camera's `calibration`, and
//! publishes `people_tracks` for every pose frame, even an empty one. With `POSITION_FRAME=odom` it also places each
//! frame with `odom_pose`, so a frame goes out once the odometry has reached its time.

use std::collections::{BTreeSet, VecDeque};
use std::time::Instant;

use dora_node_api::arrow::array::{Array, AsArray};
use dora_node_api::arrow::buffer::ScalarBuffer;
use dora_node_api::arrow::datatypes::UInt16Type;
use dora_node_api::{DoraNode, Event, MetadataParameters, Parameter};
use dora_people_3d::depth::DepthImage;
use dora_people_3d::odom::{CameraPlacement, OdomHistory};
use dora_people_3d::tracks::{DepthFrame, LiftedFrame, PeopleTracker, Placement, TrackSettings};
use eyre::{bail, eyre, Context, Result};
use perception_wire::camera::decode_camera_calibration;
use perception_wire::camera::CameraCalibration;
use perception_wire::odometry::decode_odom_pose;
use perception_wire::people::{decode_person_poses, encode_people_tracks, PeopleTracks};

const POSES_INPUT: &str = "person_poses";
const DEPTH_INPUT: &str = "depth";
const CALIBRATION_INPUT: &str = "calibration";
const ODOM_INPUT: &str = "odom_pose";
const TRACKS_OUTPUT: &str = "people_tracks";
/// Depth frames kept while their pose frames are still on the Hailo; the pose model runs a few frames behind.
const DEPTH_FRAMES_KEPT: usize = 15;
/// Odometry poses kept for placing pose frames that arrive late, in camera time.
const ODOM_KEPT_NS: i64 = 2_000_000_000;
/// Pose frames held for the odometry to reach their time; past this many the oldest goes unsent.
const FRAMES_AWAITING_ODOMETRY: usize = 60;
/// Pose frames between the progress lines this node logs.
const LOG_EVERY_FRAMES: u32 = 30;

struct HeldDepth {
    frame_id: u64,
    width: usize,
    height: usize,
    depth_mm: ScalarBuffer<u16>,
    gravity_reaction: Option<[f32; 3]>,
}

fn integer(metadata: &MetadataParameters, name: &str) -> Result<i64> {
    match metadata.get(name) {
        Some(Parameter::Integer(value)) => Ok(*value),
        other => bail!("depth metadata {name} is {other:?}, not an integer"),
    }
}

fn hold_depth(metadata: &MetadataParameters, data: &dyn Array) -> Result<HeldDepth> {
    let (width, height) = (integer(metadata, "width")? as usize, integer(metadata, "height")? as usize);
    let depth_mm = data.as_primitive_opt::<UInt16Type>().ok_or_else(|| eyre!("depth is {:?}, not u16", data.data_type()))?.values().clone();
    if depth_mm.len() != width * height {
        bail!("depth has {} values for {width}x{height}", depth_mm.len());
    }
    let gravity_reaction = match metadata.get("gravity_reaction") {
        Some(Parameter::ListFloat(values)) if values.len() == 3 => Some([values[0] as f32, values[1] as f32, values[2] as f32]),
        None => None,
        other => bail!("depth metadata gravity_reaction is {other:?}, not three floats"),
    };
    Ok(HeldDepth { frame_id: integer(metadata, "frame_id")? as u64, width, height, depth_mm, gravity_reaction })
}

/// The frame `people_tracks` positions are in, from `POSITION_FRAME`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PositionFrameSetting {
    Camera,
    Odom,
}

fn position_frame_setting() -> Result<PositionFrameSetting> {
    match std::env::var("POSITION_FRAME").as_deref() {
        Ok("camera") => Ok(PositionFrameSetting::Camera),
        Ok("odom") => Ok(PositionFrameSetting::Odom),
        other => bail!("POSITION_FRAME is {other:?}; set it to camera or odom"),
    }
}

/// Counts for the progress line.
#[derive(Default)]
struct Progress {
    frames: u32,
    with_depth: u32,
    placed: u32,
    sent: u32,
    never_placed: u32,
    gave_up_waiting: u32,
    odometry_restarts: u32,
    track_ids: BTreeSet<u64>,
    last: Option<String>,
}

/// Pose frames waiting for the odometry, and the odom frame the tracker's positions are in.
struct OdomPlacing {
    history: OdomHistory,
    waiting: VecDeque<LiftedFrame>,
    odom_started_camera_ns: Option<i64>,
}

fn send(node: &mut DoraNode, tracks: &PeopleTracks, depth_found: bool, progress: &mut Progress) -> Result<()> {
    let mut parameters = MetadataParameters::default();
    parameters.insert("depth_found".to_owned(), Parameter::Bool(depth_found));
    node.send_output(TRACKS_OUTPUT.to_owned().into(), parameters, encode_people_tracks(tracks))?;
    progress.sent += 1;
    progress.placed += tracks.tracks.iter().filter(|t| t.position.is_some()).count() as u32;
    progress.track_ids.extend(tracks.tracks.iter().map(|t| t.track_id));
    progress.last = tracks.tracks.first().map(|t| format!(", track {} {:?} at {:?}", t.track_id, t.state, t.position));
    Ok(())
}

/// Tracks and sends every waiting frame the odometry has reached, in order.
fn place_waiting(
    node: &mut DoraNode,
    tracker: &mut PeopleTracker,
    placing: &mut OdomPlacing,
    calibration: Option<&CameraCalibration>,
    progress: &mut Progress,
) -> Result<()> {
    let Some(calibration) = calibration else { return Ok(()) };
    while let Some(frame) = placing.waiting.front() {
        match placing.history.camera_at(frame.captured_camera_ns, calibration) {
            CameraPlacement::NotYet => break,
            CameraPlacement::Never => {
                placing.waiting.pop_front();
                progress.never_placed += 1;
            }
            CameraPlacement::Placed { camera_to_odom, odom_started_camera_ns } => {
                let frame = placing.waiting.pop_front().expect("a waiting frame");
                if placing.odom_started_camera_ns.is_some_and(|started| started != odom_started_camera_ns) {
                    tracker.forget_everyone();
                }
                placing.odom_started_camera_ns = Some(odom_started_camera_ns);
                let depth_found = frame.depth_found;
                let tracks = tracker.track(frame, Placement::Odom(camera_to_odom));
                send(node, &tracks, depth_found, progress)?;
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let position_frame = position_frame_setting()?;
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let mut tracker = PeopleTracker::new(TrackSettings::default());
    let mut calibration = None;
    let mut depths: VecDeque<HeldDepth> = VecDeque::with_capacity(DEPTH_FRAMES_KEPT + 1);
    let mut placing = OdomPlacing { history: OdomHistory::new(ODOM_KEPT_NS), waiting: VecDeque::new(), odom_started_camera_ns: None };
    let mut progress = Progress::default();
    let mut logged_at = Instant::now();
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } => match id.as_str() {
                CALIBRATION_INPUT => calibration = Some(decode_camera_calibration(data.as_ref()).wrap_err("reading the calibration")?),
                DEPTH_INPUT => {
                    depths.push_back(hold_depth(&metadata.parameters, data.as_ref()).wrap_err("reading a depth frame")?);
                    if depths.len() > DEPTH_FRAMES_KEPT {
                        depths.pop_front();
                    }
                }
                ODOM_INPUT => {
                    if position_frame != PositionFrameSetting::Odom {
                        bail!("odom_pose arrived but POSITION_FRAME is camera");
                    }
                    let pose = decode_odom_pose(data.as_ref()).wrap_err("reading an odom pose")?;
                    progress.odometry_restarts += placing.history.push(pose) as u32;
                    place_waiting(&mut node, &mut tracker, &mut placing, calibration.as_ref(), &mut progress)?;
                }
                POSES_INPUT => {
                    let poses = decode_person_poses(data.as_ref())?;
                    let held = depths.iter().find(|d| d.frame_id == poses.frame_id);
                    let depth = held.map(|d| DepthFrame {
                        image: DepthImage { width: d.width, height: d.height, depth_mm: &d.depth_mm },
                        gravity_reaction: d.gravity_reaction,
                    });
                    let lifted = tracker.lift_frame(&poses, depth.as_ref(), calibration.as_ref());
                    progress.frames += 1;
                    progress.with_depth += held.is_some() as u32;
                    match position_frame {
                        PositionFrameSetting::Camera => {
                            let depth_found = lifted.depth_found;
                            let tracks = tracker.track(lifted, Placement::Camera);
                            send(&mut node, &tracks, depth_found, &mut progress)?;
                        }
                        PositionFrameSetting::Odom => {
                            placing.waiting.push_back(lifted);
                            if placing.waiting.len() > FRAMES_AWAITING_ODOMETRY {
                                placing.waiting.pop_front();
                                progress.gave_up_waiting += 1;
                            }
                            place_waiting(&mut node, &mut tracker, &mut placing, calibration.as_ref(), &mut progress)?;
                        }
                    }
                    if progress.frames == LOG_EVERY_FRAMES {
                        let p = std::mem::take(&mut progress);
                        let (fitting, fits) = tracker.take_fitting_time();
                        eprintln!(
                            "[people-3d] {:.1} pose frames/s, {}/{} with depth, {} sent ({} before the odometry's poses, {} gave up \
                             waiting, {} odometry restarts), {} people placed, {} skeleton fits at {:.2} ms each, track ids {:?}{}",
                            p.frames as f64 / logged_at.elapsed().as_secs_f64(),
                            p.with_depth,
                            p.frames,
                            p.sent,
                            p.never_placed,
                            p.gave_up_waiting,
                            p.odometry_restarts,
                            p.placed,
                            fits,
                            fitting.as_secs_f64() * 1e3 / f64::from(fits.max(1)),
                            p.track_ids,
                            p.last.unwrap_or_default()
                        );
                        logged_at = Instant::now();
                    }
                }
                other => bail!("unexpected input {other}"),
            },
            Event::Stop(_) => break,
            // The daemon has gone or the connection to it broke; nothing more will arrive.
            Event::Error(error) => bail!("dora reported an error: {error}"),
            _ => {}
        }
    }
    Ok(())
}
