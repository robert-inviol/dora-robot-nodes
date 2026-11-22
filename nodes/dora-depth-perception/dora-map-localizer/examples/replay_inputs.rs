//! Runs the node's handling of its inputs over a recording of them, in the order they arrived, and reports each
//! keyframe as the node logs it.
//!
//! Usage: cargo run --release -p dora-map-localizer --example replay_inputs -- <recording directory> <settings.toml> <keyframes.csv>
//! `<keyframes.csv>` receives one row per localized keyframe. Settings without an `[odometry]` section replay the
//! recording as a dataflow that wires no odometry: its odometry poses and calibrations are left out.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use dora_map_localizer::config::NodeConfig;
use dora_map_localizer::recording::{RecordedMessage, RecordingReader};
use dora_map_localizer::{Handled, LocalizerNode};
use eyre::Result;

const HEADER: &str = "frame_id,tracking,selection,fitness,inlier_rmse_m,x_m,y_m,z_m,yaw_deg,tracked_against,loop_closed,odometry_step_m,no_odometry_step\n";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: replay_inputs <recording directory> <settings.toml> <keyframes.csv>";
    let (recording, settings, csv_path) = (args.get(1).expect(usage), args.get(2).expect(usage), args.get(3).expect(usage));
    let mut reader = RecordingReader::open(Path::new(recording))?;
    let settings = NodeConfig::read(Path::new(settings))?.settings();
    let odometry_wired = settings.odometry.is_some();
    let mut node = LocalizerNode::new(settings);
    let (mut csv, mut states, mut loops, mut moved) = (HEADER.to_owned(), BTreeMap::<&'static str, usize>::new(), 0usize, 0usize);
    while let Some(message) = reader.next_message()? {
        let handled: Handled = match message {
            RecordedMessage::Keyframe(keyframe) => node.on_keyframe(keyframe),
            RecordedMessage::Imu(imu) => node.on_imu(&imu)?,
            RecordedMessage::OdomPose(_) | RecordedMessage::Calibration(_) if !odometry_wired => continue,
            RecordedMessage::OdomPose(pose) => node.on_odom_pose(&pose)?,
            RecordedMessage::Calibration(calibration) => {
                node.on_calibration(&calibration)?;
                continue;
            }
        };
        moved += handled.moved.is_some() as usize;
        let Some(estimate) = handled.estimate else { continue };
        let work = handled.keyframe_work.expect("a keyframe estimate comes with the node's work on it");
        *states.entry(estimate.tracking.wire_name()).or_default() += 1;
        loops += estimate.loop_closed as usize;
        let number = |value: Option<f32>| value.map_or(String::new(), |value| format!("{value:.4}"));
        writeln!(
            csv,
            "{},{},{},{},{},{:.4},{:.4},{:.4},{:.2},{},{},{},{}",
            estimate.frame_id.0,
            estimate.tracking.wire_name(),
            format!("{:?}", estimate.selection).split([' ', '{']).next().unwrap_or_default(),
            number(estimate.registration.map(|r| r.fitness)),
            number(estimate.registration.map(|r| r.inlier_rmse_m)),
            estimate.pose.translation.x,
            estimate.pose.translation.y,
            estimate.pose.translation.z,
            estimate.pose.yaw_rad.to_degrees(),
            estimate.submap.len(),
            estimate.loop_closed,
            number(work.odometry_step.as_ref().ok().map(|step| step.translation.length())),
            work.odometry_step.as_ref().err().map_or(String::new(), |problem| problem.to_string().replace(',', ";")),
        )?;
    }
    std::fs::write(csv_path, csv)?;
    println!("keyframes by tracking: {states:?}; loop closures: {loops}; optimizations that moved keyframes: {moved}; rows in {csv_path}");
    Ok(())
}
