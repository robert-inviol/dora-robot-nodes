//! Localizes every keyframe of a recording, as the node does, and reports how tracking went.
//!
//! Usage: cargo run --release -p map-localizer --example localize_recording -- <keyframes.arrow> <trajectory.csv> <refuse|search> [<imu.arrow>]
//! The third argument is what to do with a keyframe whose turn the gyro cannot give (`HeadingWithoutGyro`);
//! without an imu recording no keyframe has a gyro turn, so it must be `search`. The trajectory lists each
//! keyframe's final map pose.

use std::fmt::Write as _;
use std::time::Instant;

use map_localizer::cloud::{registration_cloud, CloudSettings};
use map_localizer::gyro::{GyroHeading, GyroSettings, KeyframeTiming};
use eyre::bail;
use map_localizer::{HeadingWithoutGyro, Localizer, LocalizerInput, LocalizerSettings};
use perception_wire::localization::{CameraNanos, FrameId, Tracking};
use perception_wire::navigation::UnixNanos;
use room_map::recording::{read_imu_for, read_keyframes};

fn main() -> eyre::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: localize_recording <keyframes.arrow> <trajectory.csv> <refuse|search> [<imu.arrow>]";
    let (recording, trajectory_path) = (args.get(1).expect(usage), args.get(2).expect(usage));
    let without_gyro = match args.get(3).map(String::as_str) {
        Some("refuse") => HeadingWithoutGyro::Refuse,
        Some("search") => HeadingWithoutGyro::Search,
        _ => bail!(usage),
    };
    let keyframes = read_keyframes(recording)?;
    let imu = args.get(4).map(|path| read_imu_for(path, &keyframes)).transpose()?;
    if imu.is_none() && without_gyro == HeadingWithoutGyro::Refuse {
        bail!("without an imu recording every keyframe's heading has to be searched for: pass search");
    }
    let mut gyro = GyroHeading::new(GyroSettings::default());
    let mut localizer = Localizer::new(LocalizerSettings::default());
    let mut previous = None;
    let (mut times_ms, mut states, mut loops, mut gyro_gaps) = (Vec::new(), Vec::new(), 0, 0);
    for (index, keyframe) in keyframes.iter().enumerate() {
        let started = Instant::now();
        let heading = match &imu {
            Some(imu) => {
                gyro.add(&imu[index]);
                let timing = KeyframeTiming { camera_time: CameraNanos(imu[index].keyframe_camera_ns), gravity_reaction: keyframe.gravity_reaction };
                let heading = gyro.keyframe_heading(previous, timing, without_gyro);
                previous = Some(timing);
                if let Some(problem) = &heading.gyro_problem {
                    eprintln!("keyframe {}: {problem}; handled as {without_gyro:?}", keyframe.frame_id);
                }
                gyro_gaps += heading.gyro_problem.is_some() as usize;
                heading.hint
            }
            None => without_gyro.hint(),
        };
        let cloud = registration_cloud(keyframe, &CloudSettings::default());
        let estimate = localizer.localize(LocalizerInput {
            frame_id: FrameId(keyframe.frame_id),
            captured: UnixNanos(keyframe.captured_unix_ns),
            cloud,
            heading,
        });
        times_ms.push(started.elapsed().as_secs_f64() * 1e3);
        states.push(estimate.tracking);
        loops += estimate.loop_closed as usize;
    }
    let count = |state: Tracking| states.iter().filter(|s| **s == state).count();
    let mut sorted = times_ms.clone();
    sorted.sort_by(f64::total_cmp);
    println!(
        "{} keyframes: {} tracking, {} relocalised, {} lost; {} in the map; {loops} loop closures; map version {}; {gyro_gaps} gyro gaps",
        keyframes.len(),
        count(Tracking::Tracking),
        count(Tracking::Relocalised),
        count(Tracking::Lost),
        localizer.graph().len(),
        localizer.map_version().0
    );
    println!(
        "per keyframe: mean {:.1} ms, median {:.1} ms, 95th percentile {:.1} ms, worst {:.1} ms",
        times_ms.iter().sum::<f64>() / times_ms.len() as f64,
        sorted[sorted.len() / 2],
        sorted[sorted.len() * 95 / 100],
        sorted[sorted.len() - 1]
    );
    let mut csv = String::from("frame_id,x_m,y_m,z_m,yaw_deg\n");
    for (frame, pose) in localizer.keyframe_poses() {
        writeln!(csv, "{},{:.4},{:.4},{:.4},{:.2}", frame.0, pose.translation.x, pose.translation.y, pose.translation.z, pose.yaw_rad.to_degrees())?;
    }
    std::fs::write(trajectory_path, csv)?;
    Ok(())
}
