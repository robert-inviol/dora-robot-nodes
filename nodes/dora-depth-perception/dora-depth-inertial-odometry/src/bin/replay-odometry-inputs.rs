//! Runs the odometry node's handling over a recording of its inputs, offline, in the order they arrived, and says
//! what became of every depth frame: how many were corrected, every loss with the second before it, the pose once a
//! second of camera time with how far it moved, and how long the frames took to track.
//!
//! `replay-odometry-inputs <recording directory>`; the recording is made by `dora-odometry-input-recorder`.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use dora_depth_inertial_odometry::recording::{RecordedMessage, RecordingReader};
use dora_depth_inertial_odometry::{DepthOutcome, FrameOutcome, ImuOutcome, NodeSettings, OdometryNode, StartProgress};
use eyre::{eyre, Result};
use glam::Vec3;
use perception_wire::odometry::OdomPose;

/// Poses between progress lines, as the node logs them.
const LOG_EVERY_POSES: u32 = 300;
/// Camera time between pose lines.
const DRIFT_EVERY_NS: i64 = 1_000_000_000;

#[derive(Default)]
struct Tally {
    depth_frames: u32,
    poses: u32,
    corrected: u32,
    losses: u32,
    since_progress: u32,
    /// Wall-clock time of each call that tracked a frame.
    tracking_times: Vec<Duration>,
    /// The odom frame being followed and the pose printed last in it.
    odom_frame: Option<OdomFrameDrift>,
}

/// Where the poses of one odom frame have gone since it started, for a recording of a camera at rest.
struct OdomFrameDrift {
    started_camera_ns: i64,
    first_position_m: Vec3,
    last_printed: Option<(i64, Vec3)>,
}

impl Tally {
    /// Prints the pose once a second of camera time: its position, how far it moved since the last line, and its
    /// sigma, so that a recording of a camera at rest shows the odometry's drift per axis.
    fn track_drift(&mut self, pose: &OdomPose) {
        let frame = match &mut self.odom_frame {
            Some(frame) if frame.started_camera_ns == pose.odom_started_camera_ns => frame,
            other => other.insert(OdomFrameDrift { started_camera_ns: pose.odom_started_camera_ns, first_position_m: pose.position_m, last_printed: None }),
        };
        if frame.last_printed.is_some_and(|(camera_ns, _)| pose.camera_ns - camera_ns < DRIFT_EVERY_NS) {
            return;
        }
        let millimetres = |metres: Vec3| metres * 1e3;
        let since_start = millimetres(pose.position_m - frame.first_position_m);
        let since_last = frame.last_printed.map_or(Vec3::ZERO, |(_, position)| millimetres(pose.position_m - position));
        let seconds = (pose.camera_ns - frame.started_camera_ns) as f64 * 1e-9;
        println!(
            "pose {seconds:7.1} s into its odom frame: moved since the frame's first pose [{:+.1}, {:+.1}, {:+.1}] mm, in the last second [{:+.1}, {:+.1}, {:+.1}] mm, sigma [{:.1}, {:.1}, {:.1}] mm",
            since_start.x, since_start.y, since_start.z, since_last.x, since_last.y, since_last.z,
            pose.position_sigma_m.x * 1e3, pose.position_sigma_m.y * 1e3, pose.position_sigma_m.z * 1e3,
        );
        frame.last_printed = Some((pose.camera_ns, pose.position_m));
    }

    fn frame(&mut self, frame: FrameOutcome, took: Duration, odometry: &mut OdometryNode) {
        if !matches!(frame, FrameOutcome::BeforeStart) {
            self.tracking_times.push(took);
        }
        match frame {
            FrameOutcome::BeforeStart => {}
            FrameOutcome::Posed(pose) => {
                self.track_drift(&pose);
                self.poses += 1;
                self.corrected += u32::from(pose.corrected);
                self.since_progress += 1;
                if self.since_progress == LOG_EVERY_POSES {
                    let frames = odometry.take_progress();
                    println!("poses through depth frame {}: {frames}", pose.depth_frame_id);
                    self.since_progress = 0;
                }
            }
            FrameOutcome::Lost { uncorrected_for, largest_position_sigma_m, last_second } => {
                self.losses += 1;
                println!("lost after {uncorrected_for:?} uncorrected, position sigma {largest_position_sigma_m:.4} m; the second before: {last_second}");
            }
        }
    }
}

fn main() -> Result<()> {
    let directory = PathBuf::from(std::env::args().nth(1).ok_or_else(|| eyre!("give the recording directory"))?);
    let mut recording = RecordingReader::open(&directory)?;
    let mut odometry = OdometryNode::new(NodeSettings::d455());
    let mut tally = Tally::default();
    while let Some(message) = recording.next_message()? {
        match message {
            RecordedMessage::Calibration(calibration) => {
                odometry.on_calibration(&calibration)?;
            }
            RecordedMessage::PersonPoses(poses) => odometry.on_person_poses(&poses)?,
            RecordedMessage::Imu(imu) => {
                let began = Instant::now();
                let outcome = odometry.on_imu(&imu)?;
                let took = began.elapsed();
                match outcome {
                ImuOutcome::BeforeCalibration => {}
                ImuOutcome::Fed { start, held_frame } => {
                    match start {
                        StartProgress::Started => println!("odometry started"),
                        StartProgress::NotStationary(moving) => println!("odometry has not started: {moving}"),
                        StartProgress::StaticInitialization { .. } | StartProgress::Tracking => {}
                    }
                    if let Some(frame) = held_frame {
                        tally.frame(frame, took, &mut odometry);
                    }
                }
                }
            }
            RecordedMessage::Depth(frame) => {
                tally.depth_frames += 1;
                let began = Instant::now();
                let outcome = odometry.on_depth(frame.as_frame())?;
                let took = began.elapsed();
                match outcome {
                    DepthOutcome::BeforeCalibration | DepthOutcome::HeldForImu { replaced: None } => {}
                    DepthOutcome::HeldForImu { replaced: Some(replaced) } => println!("depth frame {} dropped: a newer frame arrived before its imu message", replaced.0),
                    DepthOutcome::Tracked(frame) => tally.frame(frame, took, &mut odometry),
                }
            }
        }
    }
    let Tally { depth_frames, poses, corrected, losses, mut tracking_times, .. } = tally;
    tracking_times.sort_unstable();
    let at = |share: f64| tracking_times.get(((tracking_times.len() as f64 - 1.0) * share).round() as usize).map_or(0.0, |time| time.as_secs_f64() * 1e3);
    println!("remaining: {}", odometry.take_progress());
    println!(
        "{depth_frames} depth frames, {poses} poses, {corrected} corrected, {losses} losses; tracking median {:.2} ms, 95th percentile {:.2} ms, worst {:.2} ms",
        at(0.5),
        at(0.95),
        at(1.0)
    );
    Ok(())
}
