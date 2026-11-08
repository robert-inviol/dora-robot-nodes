//! Plays an Arrow IPC recording of keyframes (one per batch) on each tick, restamped as new frames. With an
//! IMU recording beside it, each keyframe is preceded by the imu message that leads up to it.

use std::collections::HashMap;
use std::fs::File;
use std::time::{SystemTime, UNIX_EPOCH};

use dora_node_api::arrow::array::StructArray;
use dora_node_api::arrow::ipc::reader::FileReader;
use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters};
use eyre::{Context, Result};
use perception_wire::{decode_imu, decode_keyframe, encode_imu, encode_keyframe, Imu, Keyframe, MessageContext};

const RECORDING_VARIABLE: &str = "KEYFRAME_RECORDING";
/// Optional: an IMU recording made beside the keyframe recording.
const IMU_RECORDING_VARIABLE: &str = "IMU_RECORDING";
/// Set to "true" to start the recording again after its last keyframe.
const LOOP_VARIABLE: &str = "KEYFRAME_REPLAY_LOOP";
const TICK_INPUT: &str = "tick";
const KEYFRAME_OUTPUT: &str = "keyframe";
const IMU_OUTPUT: &str = "imu";

fn read_batches<T>(path: &str, name: &str, decode: fn(&StructArray) -> Result<T>) -> Result<Vec<T>> {
    let file = File::open(path).wrap_err_with(|| format!("opening the {name} recording {path}"))?;
    let reader = FileReader::try_new(file, None).wrap_err_with(|| format!("reading {path} as Arrow IPC"))?;
    let mut messages = Vec::new();
    for batch in reader {
        let row = StructArray::from(batch.wrap_err_with(|| format!("reading a batch of {path}"))?);
        messages.push(decode(&row).for_message(name)?);
    }
    eyre::ensure!(!messages.is_empty(), "the {name} recording {path} is empty");
    Ok(messages)
}

fn now_unix_ns() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).expect("clock after 1970").as_nanos() as i64
}

/// The imu message for each keyframe, keyed by the keyframe's recorded id.
fn imu_by_keyframe(path: &str, keyframes: &[Keyframe]) -> Result<HashMap<u64, Imu>> {
    let imus: HashMap<u64, Imu> = read_batches(path, IMU_OUTPUT, |row| decode_imu(row))?.into_iter().map(|imu| (imu.frame_id, imu)).collect();
    if let Some(missing) = keyframes.iter().find(|k| !imus.contains_key(&k.frame_id)) {
        eyre::bail!("the IMU recording {path} has no samples for keyframe {}", missing.frame_id);
    }
    Ok(imus)
}

/// How far camera times move on each pass of a looped recording, so they keep increasing: the recording's
/// span plus one typical gap between keyframes.
fn loop_shift_ns(imus: &HashMap<u64, Imu>, keyframes: &[Keyframe]) -> i64 {
    let times: Vec<i64> = keyframes.iter().map(|k| imus[&k.frame_id].keyframe_camera_ns).collect();
    let first_sample = imus[&keyframes[0].frame_id].accel_camera_ns.iter().chain(&imus[&keyframes[0].frame_id].gyro_camera_ns).min().copied();
    let start = first_sample.unwrap_or(times[0]).min(times[0]);
    let gap = if times.len() > 1 { (times[times.len() - 1] - times[0]) / (times.len() as i64 - 1) } else { 0 };
    times[times.len() - 1] - start + gap
}

fn shifted(imu: &Imu, frame_id: u64, shift_ns: i64) -> Imu {
    Imu {
        frame_id,
        keyframe_camera_ns: imu.keyframe_camera_ns + shift_ns,
        accel_camera_ns: imu.accel_camera_ns.iter().map(|t| t + shift_ns).collect(),
        accel_m_s2: imu.accel_m_s2.clone(),
        gyro_camera_ns: imu.gyro_camera_ns.iter().map(|t| t + shift_ns).collect(),
        gyro_rad_s: imu.gyro_rad_s.clone(),
    }
}

fn main() -> Result<()> {
    let path = std::env::var(RECORDING_VARIABLE).wrap_err_with(|| format!("{RECORDING_VARIABLE} is not set"))?;
    let looping = std::env::var(LOOP_VARIABLE).is_ok_and(|v| v == "true");
    let mut keyframes = read_batches(&path, KEYFRAME_OUTPUT, |row| decode_keyframe(row))?;
    let imus = match std::env::var(IMU_RECORDING_VARIABLE) {
        Ok(imu_path) => Some(imu_by_keyframe(&imu_path, &keyframes)?),
        Err(_) => None,
    };
    let shift_per_pass_ns = imus.as_ref().map_or(0, |imus| loop_shift_ns(imus, &keyframes));
    let recorded_ids: Vec<u64> = keyframes.iter().map(|k| k.frame_id).collect();
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let keyframe_output = DataId::from(KEYFRAME_OUTPUT.to_owned());
    let imu_output = DataId::from(IMU_OUTPUT.to_owned());
    let mut sent: u64 = 0;

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, .. } if id.as_str() == TICK_INPUT => {
                if !looping && sent as usize == keyframes.len() {
                    continue;
                }
                let index = sent as usize % keyframes.len();
                if let Some(imus) = &imus {
                    let pass = sent / keyframes.len() as u64;
                    let imu = shifted(&imus[&recorded_ids[index]], sent, pass as i64 * shift_per_pass_ns);
                    node.send_output(imu_output.clone(), MetadataParameters::default(), encode_imu(&imu))?;
                }
                let keyframe = &mut keyframes[index];
                keyframe.frame_id = sent;
                keyframe.captured_unix_ns = now_unix_ns();
                node.send_output(keyframe_output.clone(), MetadataParameters::default(), encode_keyframe(keyframe))?;
                sent += 1;
            }
            Event::Stop(_) => break,
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use depth_perception::CameraIntrinsics;
    use glam::Vec3;

    use super::*;

    fn keyframe(frame_id: u64) -> Keyframe {
        Keyframe {
            frame_id,
            captured_unix_ns: 0,
            width: 1,
            height: 1,
            intrinsics: CameraIntrinsics { fx: 1.0, fy: 1.0, ppx: 0.0, ppy: 0.0 },
            gravity_reaction: Vec3::NEG_Y,
            depth_mm: vec![0u16].into(),
            color_jpeg: Vec::new(),
        }
    }

    fn imu(frame_id: u64, keyframe_camera_ns: i64, sample_ns: i64) -> Imu {
        Imu {
            frame_id,
            keyframe_camera_ns,
            accel_camera_ns: vec![sample_ns],
            accel_m_s2: vec![Vec3::NEG_Y * 9.8],
            gyro_camera_ns: vec![sample_ns],
            gyro_rad_s: vec![Vec3::ZERO],
        }
    }

    #[test]
    fn a_looped_pass_moves_camera_times_past_the_end_of_the_previous_pass() {
        let keyframes = [keyframe(7), keyframe(8), keyframe(9)];
        let imus: HashMap<u64, Imu> = [imu(7, 1_000, 900), imu(8, 1_500, 1_400), imu(9, 2_000, 1_900)].into_iter().map(|i| (i.frame_id, i)).collect();
        let shift = loop_shift_ns(&imus, &keyframes);
        assert_eq!(shift, 2_000 - 900 + 500);
        let second_pass_first = shifted(&imus[&7], 3, shift);
        assert!(second_pass_first.accel_camera_ns[0] > imus[&9].keyframe_camera_ns);
        assert_eq!((second_pass_first.frame_id, second_pass_first.accel_m_s2.clone()), (3, imus[&7].accel_m_s2.clone()));
    }
}
