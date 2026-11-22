//! Per-keyframe diagnostics over a span of a localizer input recording: the depth the keyframe holds, how far its
//! floor frame (from the smoothed accelerometer) leans from the gravity odometry gives, the camera's motion, and
//! pairwise registration of each keyframe against the one before it, with either floor frame. Optionally writes
//! the colour JPEG and a depth picture (PGM) of chosen keyframes.
//!
//! Usage: cargo run --release -p dora-map-localizer --example diagnose_keyframes -- <recording directory> <first frame>
//! <last frame> <out.csv> [<image directory> <frame>...]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use depth_perception::{find_planes, sample_point_cloud, DepthImage, FloorFrame, PlaneSettings, RangeLimits, SurfaceKind};
use dora_map_localizer::recording::{RecordedMessage, RecordingReader};
use eyre::{eyre, Result};
use glam::{Quat, Vec3};
use map_localizer::map_to_odom::CameraExtrinsics;
use perception_wire::odometry::OdomPose;
use perception_wire::{Imu, Keyframe};
use room_map::grid::voxel_downsample;
use room_map::pose::wrap_angle;
use room_map::registration::{register, RegistrationSettings, Target};
use room_map::Pose;

const STRIDE_PX: usize = 4;
const VOXEL_M: f32 = 0.05;

/// The odometry's IMU pose at a camera time, interpolated between the poses either side.
fn odom_at(poses: &[OdomPose], time: i64) -> Option<(Vec3, Quat, &OdomPose)> {
    let after = poses.partition_point(|p| p.camera_ns < time);
    let later = poses.get(after)?;
    let earlier = poses.get(after.checked_sub(1)?)?;
    if earlier.odom_started_camera_ns != later.odom_started_camera_ns {
        return None;
    }
    let share = (time - earlier.camera_ns) as f32 / (later.camera_ns - earlier.camera_ns) as f32;
    Some((earlier.position_m.lerp(later.position_m, share), earlier.orientation.slerp(later.orientation, share).normalize(), later))
}

fn cloud(keyframe: &Keyframe, up_camera: Vec3) -> Vec<Vec3> {
    let depth_m = keyframe.depth_m();
    let depth = DepthImage::new(keyframe.width, keyframe.height, &depth_m);
    let floor = FloorFrame::from_specific_force(up_camera);
    voxel_downsample(&sample_point_cloud(&depth, &keyframe.intrinsics, &floor, RangeLimits::D455, STRIDE_PX).points, VOXEL_M)
}

/// The lean of the camera's optical axis above the horizon and its roll, in degrees, given "up" in camera axes.
fn pitch_roll_deg(up: Vec3) -> (f32, f32) {
    let up = up.normalize();
    let pitch = up.z.asin().to_degrees();
    let roll = up.x.atan2(-up.y).to_degrees();
    (pitch, roll)
}

fn percentile(values: &mut [f32], share: f32) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) as f32 * share) as usize]
}


/// The tilt of the largest floor or level surface in a floor-frame cloud, from a least-squares fit z = a x + b y + c
/// to its points: the slope along x and along y in degrees and the plane's area. NaN when there is none.
fn level_tilt_deg(points: &[Vec3]) -> (f32, f32, f32) {
    let fit = find_planes(points, &PlaneSettings::default());
    let Some((index, plane)) = fit
        .planes
        .iter()
        .enumerate()
        .filter(|(_, p)| matches!(p.kind, SurfaceKind::Floor | SurfaceKind::LevelSurface))
        .max_by(|a, b| a.1.area_m2.total_cmp(&b.1.area_m2))
    else {
        return (f32::NAN, f32::NAN, f32::NAN);
    };
    let members: Vec<Vec3> = points.iter().zip(&fit.point_planes).filter(|(_, p)| **p == Some(index as u8)).map(|(p, _)| *p).collect();
    let (mut a, mut b) = ([[0.0f64; 3]; 3], [0.0f64; 3]);
    for p in &members {
        let row = [p.x as f64, p.y as f64, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += row[i] * row[j];
            }
            b[i] += row[i] * p.z as f64;
        }
    }
    let m = glam::DMat3::from_cols_array_2d(&a).transpose();
    if m.determinant().abs() < 1e-9 {
        return (f32::NAN, f32::NAN, plane.area_m2);
    }
    let solution = m.inverse() * glam::DVec3::from_array(b);
    ((solution.x.atan().to_degrees()) as f32, (solution.y.atan().to_degrees()) as f32, plane.area_m2)
}

/// The keyframe before: its frame id, camera time, cloud with each floor frame, and odometry floor pose.
type PreviousKeyframe = (u64, i64, Vec<Vec3>, Vec<Vec3>, Option<Pose>);

struct PairResult {
    fitness: f32,
    rmse: f32,
    step: Pose,
}

fn pairwise(source: &[Vec3], target: &Target, seed: Pose, settings: &RegistrationSettings) -> PairResult {
    let found = register(source, target, seed, settings);
    PairResult { fitness: found.fitness, rmse: found.inlier_rmse_m, step: found.pose }
}

fn write_depth_pgm(keyframe: &Keyframe, path: &Path) -> Result<()> {
    let mut bytes = format!("P6\n{} {}\n255\n", keyframe.width, keyframe.height).into_bytes();
    for &mm in keyframe.depth_mm.iter() {
        let rgb = if mm == 0 {
            [0, 0, 0]
        } else if (mm as f32) < 400.0 {
            [255, 0, 255]
        } else if (mm as f32) > 8000.0 {
            [0, 0, 255]
        } else {
            let v = 255.0 * (1.0 - ((mm as f32 - 400.0) / 6000.0).clamp(0.0, 1.0));
            [v as u8, v as u8, v as u8]
        };
        bytes.extend_from_slice(&rgb);
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: diagnose_keyframes <recording directory> <first frame> <last frame> <out.csv> [<image directory> <frame>...]";
    let recording = args.get(1).ok_or_else(|| eyre!(usage))?;
    let first: u64 = args.get(2).ok_or_else(|| eyre!(usage))?.parse()?;
    let last: u64 = args.get(3).ok_or_else(|| eyre!(usage))?.parse()?;
    let csv_path = args.get(4).ok_or_else(|| eyre!(usage))?;
    let image_directory = args.get(5);
    let image_frames: Vec<u64> = args.iter().skip(6).map(|a| a.parse()).collect::<Result<_, _>>()?;

    let mut reader = RecordingReader::open(Path::new(recording))?;
    let mut keyframes: BTreeMap<u64, Keyframe> = BTreeMap::new();
    let mut imus: BTreeMap<u64, Imu> = BTreeMap::new();
    let mut odom: Vec<OdomPose> = Vec::new();
    let mut extrinsics = None;
    while let Some(message) = reader.next_message()? {
        match message {
            RecordedMessage::Keyframe(k) if (first..=last).contains(&k.frame_id) || image_frames.contains(&k.frame_id) => {
                keyframes.insert(k.frame_id, k);
            }
            RecordedMessage::Keyframe(_) => {}
            RecordedMessage::Imu(imu) => {
                imus.insert(imu.frame_id, imu);
            }
            RecordedMessage::OdomPose(pose) => odom.push(pose),
            RecordedMessage::Calibration(c) => extrinsics = extrinsics.or(Some(CameraExtrinsics::from_calibration(&c)?)),
        }
    }
    let extrinsics = extrinsics.ok_or_else(|| eyre!("no calibration in the recording"))?;
    odom.sort_by_key(|p| p.camera_ns);
    let (axis, angle) = extrinsics.camera_to_imu.to_axis_angle();
    println!("colour camera to IMU: turned {:.2} degrees about {axis:?}; up in camera axes when the IMU is level: {:?}", angle.to_degrees(), extrinsics.camera_to_imu.inverse() * Vec3::NEG_Y);

    if let Some(directory) = image_directory {
        std::fs::create_dir_all(directory)?;
        for frame in &image_frames {
            let keyframe = keyframes.get(frame).ok_or_else(|| eyre!("no keyframe {frame}"))?;
            std::fs::write(Path::new(directory).join(format!("{frame}-colour.jpg")), &keyframe.color_jpeg)?;
            write_depth_pgm(keyframe, &Path::new(directory).join(format!("{frame}-depth.ppm")))?;
        }
    }

    let settings = RegistrationSettings::default();
    let mut csv = String::from(
        "frame_id,dt_s,valid_share,near_share_lt_0_6m,median_depth_m,p90_depth_m,beyond_4m_share,cloud_points,\
         sf_norm_m_s2,accel_min,accel_max,gyro_max_deg_s,sf_pitch_deg,sf_roll_deg,odom_pitch_deg,odom_roll_deg,tilt_error_deg,\
         odom_corrected,odom_matched,odom_sigma_z_m,odom_step_m,odom_step_dz_m,odom_step_yaw_deg,\
         floor_slope_x_sf_deg,floor_slope_y_sf_deg,floor_area_m2,floor_slope_x_odomup_deg,floor_slope_y_odomup_deg,ceiling_sf,pair_sf_fitness,pair_sf_rmse,pair_sf_dxy_from_odom_m,pair_sf_dz_from_odom_m,pair_sf_dyaw_from_odom_deg,\
         pair_odomup_fitness,pair_odomup_dxy_from_odom_m,pair_odomup_dz_from_odom_m,pair_odomup_dyaw_from_odom_deg,pair_zero_seed_sf_fitness\n",
    );
    let mut previous: Option<PreviousKeyframe> = None;
    for (&frame, keyframe) in keyframes.range(first..=last) {
        let Some(imu) = imus.get(&frame) else { continue };
        let time = imu.keyframe_camera_ns;
        let total = keyframe.depth_mm.len() as f32;
        let mut valid: Vec<f32> = keyframe.depth_mm.iter().filter(|&&mm| mm > 0).map(|&mm| mm as f32 * 0.001).collect();
        let valid_share = valid.len() as f32 / total;
        let near_share = valid.iter().filter(|&&d| d < 0.6).count() as f32 / total;
        let beyond_share = valid.iter().filter(|&&d| d > 4.0).count() as f32 / total;
        let median = percentile(&mut valid, 0.5);
        let p90 = percentile(&mut valid, 0.9);
        let accel_norms: Vec<f32> = imu.accel_m_s2.iter().map(|a| a.length()).collect();
        let (accel_min, accel_max) = accel_norms.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &a| (lo.min(a), hi.max(a)));
        let gyro_max = imu
            .gyro_camera_ns
            .iter()
            .zip(&imu.gyro_rad_s)
            .filter(|(t, _)| time - **t < 100_000_000 && **t <= time)
            .map(|(_, g)| g.length().to_degrees())
            .fold(0.0, f32::max);
        let sf_up = keyframe.specific_force;
        let (sf_pitch, sf_roll) = pitch_roll_deg(sf_up);
        let odom_here = odom_at(&odom, time);
        let odom_up = odom_here.map(|(_, q, _)| (q * extrinsics.camera_to_imu).inverse() * Vec3::Z);
        let odom_floor = odom_here.and_then(|(p, q, _)| extrinsics.floor_pose(p, q));
        let (odom_pitch, odom_roll) = odom_up.map_or((f32::NAN, f32::NAN), pitch_roll_deg);
        let tilt_error = odom_up.map_or(f32::NAN, |up| up.normalize().angle_between(sf_up.normalize()).to_degrees());

        let cloud_sf = cloud(keyframe, sf_up);
        let cloud_odom = odom_up.map_or_else(Vec::new, |up| cloud(keyframe, up));
        let ceiling = {
            let own = Target::new(cloud_sf.clone(), &settings);
            register(&cloud_sf, &own, Pose::IDENTITY, &RegistrationSettings { correspondence_schedule_m: vec![0.08], iterations_per_stage: 1, ..settings.clone() }).fitness
        };
        let mut row = String::new();
        let (corrected, matched, sigma_z) = odom_here.map_or((false, 0, f32::NAN), |(_, _, p)| (p.corrected, p.matched_points, p.position_sigma_m.z));
        let dt = previous.as_ref().map_or(f32::NAN, |(_, t, ..)| (time - t) as f32 * 1e-9);
        let odom_step = match (&previous, odom_floor) {
            (Some((.., Some(before))), Some(now)) => Some(now.relative_to(before)),
            _ => None,
        };
        write!(
            row,
            "{frame},{dt:.3},{valid_share:.3},{near_share:.3},{median:.2},{p90:.2},{beyond_share:.3},{},{:.2},{accel_min:.2},{accel_max:.2},{gyro_max:.1},\
             {sf_pitch:.1},{sf_roll:.1},{odom_pitch:.1},{odom_roll:.1},{tilt_error:.2},{corrected},{matched},{sigma_z:.4},",
            cloud_sf.len(),
            sf_up.length(),
        )?;
        match odom_step {
            Some(step) => write!(row, "{:.3},{:.3},{:.2},", step.translation.length(), step.translation.z, step.yaw_rad.to_degrees())?,
            None => row.push_str(",,,"),
        }
        let (sx, sy, area) = level_tilt_deg(&cloud_sf);
        let (ox, oy, _) = if cloud_odom.is_empty() { (f32::NAN, f32::NAN, f32::NAN) } else { level_tilt_deg(&cloud_odom) };
        write!(row, "{sx:.2},{sy:.2},{area:.1},{ox:.2},{oy:.2},{ceiling:.3},")?;
        match (&previous, odom_step) {
            (Some((_, _, before_sf, before_odom, _)), Some(step)) => {
                let diff = |found: &Pose| {
                    let d = found.relative_to(&step);
                    (Vec3::new(d.translation.x, d.translation.y, 0.0).length(), found.translation.z - step.translation.z, wrap_angle(found.yaw_rad - step.yaw_rad).to_degrees())
                };
                let target_sf = Target::new(before_sf.clone(), &settings);
                let sf = pairwise(&cloud_sf, &target_sf, step, &settings);
                let (dxy, dz, dyaw) = diff(&sf.step);
                write!(row, "{:.3},{:.4},{dxy:.3},{dz:.3},{dyaw:.2},", sf.fitness, sf.rmse)?;
                if !before_odom.is_empty() && !cloud_odom.is_empty() {
                    let target_odom = Target::new(before_odom.clone(), &settings);
                    let od = pairwise(&cloud_odom, &target_odom, step, &settings);
                    let (dxy, dz, dyaw) = diff(&od.step);
                    write!(row, "{:.3},{dxy:.3},{dz:.3},{dyaw:.2},", od.fitness)?;
                } else {
                    row.push_str(",,,,");
                }
                let zero_seed = Pose { yaw_rad: step.yaw_rad, translation: Vec3::ZERO };
                write!(row, "{:.3}", pairwise(&cloud_sf, &target_sf, zero_seed, &settings).fitness)?;
            }
            _ => row.push_str(",,,,,,,,,,"),
        }
        csv.push_str(&row);
        csv.push('\n');
        previous = Some((frame, time, cloud_sf, cloud_odom, odom_floor));
    }
    std::fs::write(csv_path, csv)?;
    println!("rows in {csv_path}");
    Ok(())
}
