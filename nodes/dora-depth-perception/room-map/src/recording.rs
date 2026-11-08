//! Reading a keyframe recording: an Arrow IPC file with one keyframe message per batch.

use std::fs::File;

use dora_node_api::arrow::array::StructArray;
use dora_node_api::arrow::ipc::reader::FileReader;
use eyre::{Context, Result};
use perception_wire::{decode_keyframe, Keyframe, MessageContext};

pub fn read_keyframes(path: &str) -> Result<Vec<Keyframe>> {
    let file = File::open(path).wrap_err_with(|| format!("opening the keyframe recording {path}"))?;
    let reader = FileReader::try_new(file, None).wrap_err_with(|| format!("reading {path} as Arrow IPC"))?;
    let mut keyframes = Vec::new();
    for batch in reader {
        let row = StructArray::from(batch.wrap_err_with(|| format!("reading a batch of {path}"))?);
        keyframes.push(decode_keyframe(&row).for_message("keyframe")?);
    }
    eyre::ensure!(!keyframes.is_empty(), "the keyframe recording {path} is empty");
    Ok(keyframes)
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
