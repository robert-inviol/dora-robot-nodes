//! dora node: selects keyframes from dora-realsense's frames. It follows the camera's `imu` and `depth`, selects a frame
//! at least once a second and sooner when the camera turns, asks the camera for that frame's colour, and sends an
//! `imu` message with every sample since the previous keyframe, then the `keyframe`: depth aligned to colour, the
//! colour as JPEG, and the gravity estimate.

use std::fs::File;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use dora_keyframe_selector::selection::{KeyframeAssembler, SelectedKeyframe};
use dora_node_api::arrow::array::{Array, AsArray, RecordBatch, StructArray, UInt64Array};
use dora_node_api::arrow::datatypes::{Schema, UInt16Type, UInt8Type};
use dora_node_api::arrow::ipc::writer::FileWriter;
use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters, Parameter};
use eyre::{bail, eyre, Context, Result};
use glam::Vec3;
use perception_wire::camera::{decode_camera_calibration, CameraCalibration};
use perception_wire::{decode_imu, encode_imu, encode_keyframe};
use rgbd_keyframes::KeyframeSettings;
use shared_memory_outputs::SharedMemoryOutputs;

const IMU_INPUT: &str = "imu";
const DEPTH_INPUT: &str = "depth";
const COLOR_INPUT: &str = "color";
const CALIBRATION_INPUT: &str = "calibration";
const REQUEST_COLOR_OUTPUT: &str = "request_color";
const IMU_OUTPUT: &str = "imu";
const KEYFRAME_OUTPUT: &str = "keyframe";
const RECORD_VARIABLE: &str = "KEYFRAME_RECORD_PATH";
const IMU_RECORD_VARIABLE: &str = "IMU_RECORD_PATH";
const JPEG_QUALITY: u8 = 85;
/// The most shared-memory samples of one output its receivers may hold unreleased: half a minute of keyframes, more
/// than its receivers and dora's region cache keep at once.
const UNRELEASED_SAMPLES_LIMIT: usize = 30;

fn integer(metadata: &MetadataParameters, input: &str, name: &str) -> Result<i64> {
    match metadata.get(name) {
        Some(Parameter::Integer(value)) => Ok(*value),
        other => bail!("{input} metadata {name} is {other:?}, not an integer"),
    }
}

fn specific_force(metadata: &MetadataParameters) -> Result<Option<Vec3>> {
    match metadata.get("specific_force") {
        Some(Parameter::ListFloat(values)) if values.len() == 3 => Ok(Some(Vec3::new(values[0] as f32, values[1] as f32, values[2] as f32))),
        None => Ok(None),
        other => bail!("depth metadata specific_force is {other:?}, not three floats"),
    }
}

fn now_unix_ns() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as i64)
}

/// Appends messages of one type to an Arrow IPC file, created when the first one arrives; `dora-keyframe-replay`
/// plays it.
struct Recorder {
    path: String,
    writer: Option<FileWriter<File>>,
}

impl Recorder {
    fn from_env(variable: &str) -> Option<Recorder> {
        std::env::var(variable).ok().map(|path| Recorder { path, writer: None })
    }

    fn write(&mut self, message: &StructArray) -> Result<()> {
        let batch = RecordBatch::from(message.clone());
        if self.writer.is_none() {
            let file = File::create(&self.path).wrap_err_with(|| format!("creating the recording {}", self.path))?;
            self.writer = Some(FileWriter::try_new(file, &Schema::new(batch.schema().fields().clone()))?);
        }
        self.writer.as_mut().expect("just created").write(&batch)?;
        Ok(())
    }

    fn finish(self) -> Result<()> {
        if let Some(mut writer) = self.writer {
            writer.finish().wrap_err_with(|| format!("finishing the recording {}", self.path))?;
        }
        Ok(())
    }
}

fn encode_jpeg(rgb: &[u8], width: u16, height: u16) -> Result<Vec<u8>> {
    let mut jpeg = Vec::with_capacity(256 * 1024);
    jpeg_encoder::Encoder::new(&mut jpeg, JPEG_QUALITY)
        .encode(rgb, width, height, jpeg_encoder::ColorType::Rgb)
        .wrap_err("encoding the colour image")?;
    Ok(jpeg)
}

/// Sends a selected frame's imu message, then its keyframe, recording each when asked to.
fn send_selected(
    node: &mut DoraNode,
    outputs: &mut SharedMemoryOutputs,
    selected: SelectedKeyframe,
    imu_recorder: &mut Option<Recorder>,
    keyframe_recorder: &mut Option<Recorder>,
) -> Result<()> {
    let output = |name: &str| DataId::from(name.to_owned());
    // The imu message goes first, so a consumer has the samples before the keyframe they lead up to.
    let imu_message = encode_imu(&selected.imu);
    if let Some(recorder) = imu_recorder.as_mut() {
        recorder.write(&imu_message)?;
    }
    let mut imu_parameters = MetadataParameters::default();
    imu_parameters.insert("accel_samples".to_owned(), Parameter::Integer(selected.imu.accel_camera_ns.len() as i64));
    imu_parameters.insert("gyro_samples".to_owned(), Parameter::Integer(selected.imu.gyro_camera_ns.len() as i64));
    outputs.send_array(node, &output(IMU_OUTPUT), imu_parameters, imu_message)?;
    let keyframe_message = encode_keyframe(&selected.keyframe);
    if let Some(recorder) = keyframe_recorder.as_mut() {
        recorder.write(&keyframe_message)?;
    }
    outputs.send_array(node, &output(KEYFRAME_OUTPUT), MetadataParameters::default(), keyframe_message)?;
    Ok(())
}

fn main() -> Result<()> {
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let mut outputs = SharedMemoryOutputs::new(&node, UNRELEASED_SAMPLES_LIMIT)?;
    let mut assembler = KeyframeAssembler::new(KeyframeSettings::default());
    let mut calibration: Option<CameraCalibration> = None;
    let mut keyframe_recorder = Recorder::from_env(RECORD_VARIABLE);
    let mut imu_recorder = Recorder::from_env(IMU_RECORD_VARIABLE);
    let output = |name: &str| DataId::from(name.to_owned());
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } => match id.as_str() {
                CALIBRATION_INPUT => calibration = Some(decode_camera_calibration(data.as_ref()).wrap_err("reading the calibration")?),
                IMU_INPUT => {
                    assembler.add_imu(&decode_imu(data.as_ref()).wrap_err("reading an imu message")?);
                    if let Some(calibration) = calibration.as_ref() {
                        for selected in assembler.take_ready(calibration) {
                            send_selected(&mut node, &mut outputs, selected, &mut imu_recorder, &mut keyframe_recorder)?;
                        }
                    }
                }
                DEPTH_INPUT => {
                    let parameters = &metadata.parameters;
                    let frame_id = integer(parameters, DEPTH_INPUT, "frame_id")? as u64;
                    let camera_ns = integer(parameters, DEPTH_INPUT, "camera_ns")?;
                    let depth_mm = data.as_primitive_opt::<UInt16Type>().ok_or_else(|| eyre!("depth is {:?}, not u16", data.data_type()))?;
                    if calibration.is_some()
                        && assembler.add_depth(frame_id, camera_ns, now_unix_ns(), specific_force(parameters)?, depth_mm.values())
                    {
                        outputs.send_array(&mut node, &output(REQUEST_COLOR_OUTPUT), MetadataParameters::default(), UInt64Array::from(vec![frame_id]))?;
                    }
                }
                COLOR_INPUT => {
                    let parameters = &metadata.parameters;
                    let frame_id = integer(parameters, COLOR_INPUT, "frame_id")? as u64;
                    let (width, height) = (integer(parameters, COLOR_INPUT, "width")?, integer(parameters, COLOR_INPUT, "height")?);
                    let calibration = calibration.as_ref().ok_or_else(|| eyre!("colour arrived before the calibration"))?;
                    if !assembler.is_waiting_for(frame_id) {
                        bail!("colour for frame {frame_id}, which was not asked for");
                    }
                    let rgb = data.as_primitive_opt::<UInt8Type>().ok_or_else(|| eyre!("colour is {:?}, not u8", data.data_type()))?;
                    let started = Instant::now();
                    let jpeg = encode_jpeg(rgb.values(), width as u16, height as u16)?;
                    eprintln!("[keyframes] keyframe {frame_id}: jpeg {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
                    assembler.add_color(frame_id, jpeg);
                    for selected in assembler.take_ready(calibration) {
                        send_selected(&mut node, &mut outputs, selected, &mut imu_recorder, &mut keyframe_recorder)?;
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
    if let Some(recorder) = keyframe_recorder {
        recorder.finish()?;
    }
    if let Some(recorder) = imu_recorder {
        recorder.finish()?;
    }
    Ok(())
}
