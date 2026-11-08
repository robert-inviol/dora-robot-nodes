//! Reading recordings: Arrow IPC files with one keyframe, or one imu, message per batch.

use std::fs::File;

use dora_node_api::arrow::array::StructArray;
use dora_node_api::arrow::ipc::reader::FileReader;
use eyre::{Context, Result};
use perception_wire::{decode_imu, decode_keyframe, Imu, Keyframe, MessageContext};

fn read_messages<T>(path: &str, name: &str, decode: impl Fn(&StructArray) -> Result<T>) -> Result<Vec<T>> {
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

pub fn read_keyframes(path: &str) -> Result<Vec<Keyframe>> {
    read_messages(path, "keyframe", |row| decode_keyframe(row))
}

/// For each keyframe, in order, the imu message that leads up to it, matched by frame id.
pub fn read_imu_for(path: &str, keyframes: &[Keyframe]) -> Result<Vec<Imu>> {
    let mut by_frame: std::collections::HashMap<u64, Imu> =
        read_messages(path, "imu", |row| decode_imu(row))?.into_iter().map(|imu| (imu.frame_id, imu)).collect();
    keyframes
        .iter()
        .map(|k| by_frame.remove(&k.frame_id).ok_or_else(|| eyre::eyre!("the imu recording {path} has no message for keyframe {}", k.frame_id)))
        .collect()
}

/// The colour image as packed RGB rows.
pub fn decode_color(keyframe: &Keyframe) -> Result<Vec<u8>> {
    let mut decoder = jpeg_decoder::Decoder::new(&keyframe.color_jpeg[..]);
    let pixels = decoder.decode().wrap_err_with(|| format!("decoding the colour of keyframe {}", keyframe.frame_id))?;
    let info = decoder.info().ok_or_else(|| eyre::eyre!("keyframe {} colour has no image info", keyframe.frame_id))?;
    eyre::ensure!(
        (info.width as usize, info.height as usize) == (keyframe.width, keyframe.height) && info.pixel_format == jpeg_decoder::PixelFormat::RGB24,
        "keyframe {} colour is {}x{} {:?}, depth {}x{}",
        keyframe.frame_id, info.width, info.height, info.pixel_format, keyframe.width, keyframe.height
    );
    Ok(pixels)
}
