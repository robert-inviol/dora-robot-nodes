//! Runs the node's handling of a recording, as `replay_inputs` does, with each keyframe's specific force replaced by
//! another estimate of up, so its floor frame comes from that estimate instead of the one the camera recorded:
//!
//! - `odometry`: the odometry's orientation at the keyframe's camera time, up in camera axes scaled to 9.81 m/s^2. A
//!   keyframe odometry does not cover keeps its own specific force and is counted.
//! - `gyro:<time constant s>`: the camera's own gravity filter (`rgbd_keyframes::GravityFilter`) over the recorded IMU
//!   samples with the given time constant, in time order, advanced to each keyframe's camera time once both streams
//!   have passed it.
//!
//! Usage: cargo run --release -p dora-map-localizer --example replay_with_gravity -- <recording directory>
//! <settings.toml> <keyframes.csv> <odometry|gyro:SECONDS>

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

use dora_map_localizer::config::NodeConfig;
use dora_map_localizer::recording::{RecordedMessage, RecordingReader};
use dora_map_localizer::{Handled, LocalizerNode};
use eyre::{bail, eyre, Result};
use glam::Vec3;
use map_localizer::map_to_odom::CameraExtrinsics;
use perception_wire::odometry::OdomPose;
use perception_wire::Imu;
use rgbd_keyframes::GravityFilter;

const HEADER: &str = "frame_id,tracking,selection,fitness,inlier_rmse_m,x_m,y_m,z_m,yaw_deg,tracked_against,loop_closed,odometry_step_m,no_odometry_step\n";
const GRAVITY_M_S2: f32 = 9.81;

enum GravitySource {
    Odometry,
    Gyro { time_constant_s: f32 },
}

fn gravity_source(argument: &str) -> Result<GravitySource> {
    if argument == "odometry" {
        return Ok(GravitySource::Odometry);
    }
    match argument.strip_prefix("gyro:") {
        Some(seconds) => Ok(GravitySource::Gyro { time_constant_s: seconds.parse()? }),
        None => bail!("the gravity source is odometry or gyro:<time constant s>, not {argument}"),
    }
}

/// Up in the colour camera's axes at each keyframe, from the odometry's orientation at the keyframe's camera time.
fn odometry_up(odom: &mut [OdomPose], imus: &[Imu], extrinsics: CameraExtrinsics) -> HashMap<u64, Vec3> {
    odom.sort_by_key(|p| p.camera_ns);
    let mut up = HashMap::new();
    for imu in imus {
        let time = imu.keyframe_camera_ns;
        let after = odom.partition_point(|p| p.camera_ns < time);
        let (Some(later), Some(earlier)) = (odom.get(after), after.checked_sub(1).and_then(|i| odom.get(i))) else { continue };
        if earlier.odom_started_camera_ns != later.odom_started_camera_ns {
            continue;
        }
        let share = (time - earlier.camera_ns) as f32 / (later.camera_ns - earlier.camera_ns) as f32;
        let orientation = earlier.orientation.slerp(later.orientation, share).normalize();
        up.insert(imu.frame_id, (orientation * extrinsics.camera_to_imu).inverse() * Vec3::Z);
    }
    up
}

/// The specific force at each keyframe from the camera's gravity filter run over the recorded IMU samples.
fn gravity_filter_specific_force(imus: &[Imu], time_constant_s: f32) -> Result<HashMap<u64, Vec3>> {
    #[derive(Clone, Copy)]
    enum Sample {
        Accel(Vec3),
        Gyro(Vec3),
    }
    let camera_time = |ns: i64| u64::try_from(ns).map(Duration::from_nanos).map_err(|_| eyre!("camera time {ns} ns is negative"));
    let mut samples: Vec<(i64, Sample)> = imus
        .iter()
        .flat_map(|imu| {
            let accel = imu.accel_camera_ns.iter().zip(&imu.accel_m_s2).map(|(&t, &a)| (t, Sample::Accel(a)));
            let gyro = imu.gyro_camera_ns.iter().zip(&imu.gyro_rad_s).map(|(&t, &g)| (t, Sample::Gyro(g)));
            accel.chain(gyro).collect::<Vec<_>>()
        })
        .collect();
    samples.sort_by_key(|(t, _)| *t);
    let mut keyframes: Vec<(i64, u64)> = imus.iter().map(|imu| (imu.keyframe_camera_ns, imu.frame_id)).collect();
    keyframes.sort();

    let mut filter = GravityFilter::new(Duration::from_secs_f32(time_constant_s));
    let (mut next, mut found) = (0, HashMap::new());
    let (mut latest_accel_ns, mut latest_gyro_ns) = (i64::MIN, i64::MIN);
    for (keyframe_ns, frame_id) in keyframes {
        // In time order, each sample applied once both streams reach it, as the camera's frames apply them; then on
        // until both streams pass the keyframe's time, as they have when the camera sends a frame.
        while let Some(&(t, sample)) = samples.get(next).filter(|(t, _)| *t <= keyframe_ns || latest_accel_ns <= keyframe_ns || latest_gyro_ns <= keyframe_ns) {
            match sample {
                Sample::Accel(reading) => {
                    filter.add_accel(camera_time(t)?, reading)?;
                    latest_accel_ns = t;
                }
                Sample::Gyro(reading) => {
                    filter.add_gyro(camera_time(t)?, reading)?;
                    latest_gyro_ns = t;
                }
            }
            filter.advance_to(camera_time(t.min(keyframe_ns))?);
            next += 1;
        }
        filter.advance_to(camera_time(keyframe_ns)?);
        if let Some(specific_force) = filter.specific_force() {
            found.insert(frame_id, specific_force);
        }
    }
    Ok(found)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: replay_with_gravity <recording directory> <settings.toml> <keyframes.csv> <odometry|gyro:SECONDS>";
    let (recording, settings, csv_path) = (args.get(1).expect(usage), args.get(2).expect(usage), args.get(3).expect(usage));
    let source = gravity_source(args.get(4).expect(usage))?;

    let mut reader = RecordingReader::open(Path::new(recording))?;
    let (mut odom, mut imus, mut extrinsics) = (Vec::<OdomPose>::new(), Vec::<Imu>::new(), None);
    while let Some(message) = reader.next_message()? {
        match message {
            RecordedMessage::OdomPose(pose) => odom.push(pose),
            RecordedMessage::Imu(imu) => imus.push(imu),
            RecordedMessage::Calibration(c) => extrinsics = extrinsics.or(Some(CameraExtrinsics::from_calibration(&c)?)),
            RecordedMessage::Keyframe(_) => {}
        }
    }
    let replacement = match source {
        GravitySource::Odometry => odometry_up(&mut odom, &imus, extrinsics.ok_or_else(|| eyre!("no calibration in the recording"))?)
            .into_iter()
            .map(|(frame_id, up)| (frame_id, up * GRAVITY_M_S2))
            .collect(),
        GravitySource::Gyro { time_constant_s } => gravity_filter_specific_force(&imus, time_constant_s)?,
    };
    drop(imus);

    let mut reader = RecordingReader::open(Path::new(recording))?;
    let settings = NodeConfig::read(Path::new(settings))?.settings();
    let odometry_wired = settings.odometry.is_some();
    let mut node = LocalizerNode::new(settings);
    let (mut csv, mut states, mut uncovered) = (HEADER.to_owned(), BTreeMap::<&'static str, usize>::new(), 0usize);
    while let Some(message) = reader.next_message()? {
        let handled: Handled = match message {
            RecordedMessage::Keyframe(mut keyframe) => {
                match replacement.get(&keyframe.frame_id) {
                    Some(&specific_force) => keyframe.specific_force = specific_force,
                    None => uncovered += 1,
                }
                node.on_keyframe(keyframe)
            }
            RecordedMessage::Imu(imu) => node.on_imu(&imu)?,
            RecordedMessage::OdomPose(_) | RecordedMessage::Calibration(_) if !odometry_wired => continue,
            RecordedMessage::OdomPose(pose) => node.on_odom_pose(&pose)?,
            RecordedMessage::Calibration(calibration) => {
                node.on_calibration(&calibration)?;
                continue;
            }
        };
        let Some(estimate) = handled.estimate else { continue };
        let work = handled.keyframe_work.expect("a keyframe estimate comes with the node's work on it");
        *states.entry(estimate.tracking.wire_name()).or_default() += 1;
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
    println!("keyframes by tracking: {states:?}; keyframes without a replacement up (own specific force kept): {uncovered}; rows in {csv_path}");
    Ok(())
}
