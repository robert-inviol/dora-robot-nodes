//! A recording of the node's inputs as they arrived: a directory with one Arrow IPC stream file per input. Each
//! batch is one wire message, with an `arrival` column that numbers every message of every input in the order they
//! arrived, so a replay feeds them to the node in that order.
//!
//! A stream file is readable up to its last whole batch, so a recording cut short by a crash still replays.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dora_node_api::arrow::array::{Array, ArrayRef, AsArray, RecordBatch, StructArray, UInt64Array};
use dora_node_api::arrow::datatypes::{DataType, Field, Fields, Schema, UInt64Type};
use dora_node_api::arrow::ipc::reader::StreamReader;
use dora_node_api::arrow::ipc::writer::StreamWriter;
use eyre::{bail, ensure, eyre, Result, WrapErr};
use perception_wire::camera::{decode_camera_calibration, CameraCalibration};
use perception_wire::odometry::{decode_odom_pose, OdomPose};
use perception_wire::{decode_imu, decode_keyframe, Imu, Keyframe};

/// The node's inputs, each in its own file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordedInput {
    Keyframe,
    Imu,
    OdomPose,
    Calibration,
}

impl RecordedInput {
    pub const ALL: [RecordedInput; 4] = [RecordedInput::Keyframe, RecordedInput::Imu, RecordedInput::OdomPose, RecordedInput::Calibration];

    /// The node input this is, which names its file.
    pub fn input_name(self) -> &'static str {
        match self {
            RecordedInput::Keyframe => "keyframe",
            RecordedInput::Imu => "imu",
            RecordedInput::OdomPose => "odom_pose",
            RecordedInput::Calibration => "calibration",
        }
    }

    pub fn named(input_name: &str) -> Option<RecordedInput> {
        RecordedInput::ALL.into_iter().find(|input| input.input_name() == input_name)
    }

    fn file(self, directory: &Path) -> PathBuf {
        directory.join(format!("{}.arrows", self.input_name()))
    }
}

const ARRIVAL: &str = "arrival";

/// Writes a recording into a new directory.
pub struct RecordingWriter {
    directory: PathBuf,
    streams: [Option<StreamWriter<BufWriter<File>>>; 4],
    next_arrival: u64,
}

impl RecordingWriter {
    /// Creates the directory, which must not exist yet, so a recording is never written over.
    pub fn create(directory: &Path) -> Result<RecordingWriter> {
        ensure!(!directory.exists(), "the recording directory {} exists already", directory.display());
        fs::create_dir_all(directory).wrap_err_with(|| format!("creating the recording directory {}", directory.display()))?;
        Ok(RecordingWriter { directory: directory.to_owned(), streams: [None, None, None, None], next_arrival: 0 })
    }

    /// Writes one message: a one-row struct as it came over the wire.
    pub fn write(&mut self, input: RecordedInput, message: &StructArray) -> Result<()> {
        ensure!(message.len() == 1, "a {} message has {} rows, not one", input.input_name(), message.len());
        let mut fields = vec![Arc::new(Field::new(ARRIVAL, DataType::UInt64, false))];
        fields.extend(message.fields().iter().cloned());
        let mut columns: Vec<ArrayRef> = vec![Arc::new(UInt64Array::from(vec![self.next_arrival]))];
        columns.extend(message.columns().iter().cloned());
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).wrap_err_with(|| format!("a {} message as a batch", input.input_name()))?;
        let stream = match &mut self.streams[input as usize] {
            Some(stream) => stream,
            empty => {
                let path = input.file(&self.directory);
                let file = File::create(&path).wrap_err_with(|| format!("creating {}", path.display()))?;
                empty.insert(StreamWriter::try_new(BufWriter::new(file), &batch.schema()).wrap_err_with(|| format!("starting {}", path.display()))?)
            }
        };
        stream.write(&batch).wrap_err_with(|| format!("writing a {} message", input.input_name()))?;
        self.next_arrival += 1;
        Ok(())
    }

    /// Messages written so far, of every input.
    pub fn messages(&self) -> u64 {
        self.next_arrival
    }

    pub fn finish(self) -> Result<()> {
        for (input, stream) in RecordedInput::ALL.into_iter().zip(self.streams) {
            let Some(mut stream) = stream else { continue };
            stream.finish().wrap_err_with(|| format!("ending the {} stream", input.input_name()))?;
            stream.into_inner().wrap_err_with(|| format!("flushing the {} stream", input.input_name()))?;
        }
        Ok(())
    }
}

/// One recorded message.
pub enum RecordedMessage {
    Keyframe(Keyframe),
    Imu(Imu),
    OdomPose(OdomPose),
    Calibration(CameraCalibration),
}

fn decode(input: RecordedInput, batch: &RecordBatch) -> Result<(u64, RecordedMessage)> {
    let schema = batch.schema();
    ensure!(schema.field(0).name() == ARRIVAL, "a {} batch does not start with its arrival", input.input_name());
    let arrivals = batch.column(0).as_primitive_opt::<UInt64Type>().ok_or_else(|| eyre!("arrival is {:?}", batch.column(0).data_type()))?;
    ensure!(arrivals.len() == 1, "a recorded {} batch has {} rows, not one", input.input_name(), arrivals.len());
    let fields: Fields = schema.fields().iter().skip(1).cloned().collect();
    let message = StructArray::try_new(fields, batch.columns()[1..].to_vec(), None).wrap_err_with(|| format!("a recorded {} message", input.input_name()))?;
    let decoded = match input {
        RecordedInput::Keyframe => RecordedMessage::Keyframe(decode_keyframe(&message)?),
        RecordedInput::Imu => RecordedMessage::Imu(decode_imu(&message)?),
        RecordedInput::OdomPose => RecordedMessage::OdomPose(decode_odom_pose(&message)?),
        RecordedInput::Calibration => RecordedMessage::Calibration(decode_camera_calibration(&message)?),
    };
    Ok((arrivals.value(0), decoded))
}

struct InputStream {
    input: RecordedInput,
    reader: StreamReader<BufReader<File>>,
    next: Option<(u64, RecordedMessage)>,
}

impl InputStream {
    fn advance(&mut self) -> Result<()> {
        self.next = match self.reader.next() {
            None => None,
            Some(batch) => Some(decode(self.input, &batch.wrap_err_with(|| format!("reading the {} stream", self.input.input_name()))?)?),
        };
        Ok(())
    }
}

/// Reads a recording's messages back in the order they arrived.
pub struct RecordingReader {
    streams: Vec<InputStream>,
    last_arrival: Option<u64>,
}

impl RecordingReader {
    /// Opens every input's file in the directory; keyframes must be there, an input the dataflow did not wire may
    /// not.
    pub fn open(directory: &Path) -> Result<RecordingReader> {
        let mut streams = Vec::new();
        for input in RecordedInput::ALL {
            let path = input.file(directory);
            if !path.exists() {
                if input == RecordedInput::Keyframe {
                    bail!("the recording {} has no {}", directory.display(), path.display());
                }
                continue;
            }
            let file = File::open(&path).wrap_err_with(|| format!("opening {}", path.display()))?;
            let reader = StreamReader::try_new(BufReader::new(file), None).wrap_err_with(|| format!("reading {} as an Arrow IPC stream", path.display()))?;
            let mut stream = InputStream { input, reader, next: None };
            stream.advance()?;
            streams.push(stream);
        }
        Ok(RecordingReader { streams, last_arrival: None })
    }

    /// The next message to arrive, or None at the end of every input's file.
    pub fn next_message(&mut self) -> Result<Option<RecordedMessage>> {
        let Some(stream) = self.streams.iter_mut().filter(|stream| stream.next.is_some()).min_by_key(|stream| stream.next.as_ref().map(|(arrival, _)| *arrival)) else { return Ok(None) };
        let (arrival, message) = stream.next.take().expect("the stream was picked for its next message");
        stream.advance()?;
        ensure!(self.last_arrival.is_none_or(|last| arrival > last), "message {arrival} of the recording arrived after {:?}", self.last_arrival);
        self.last_arrival = Some(arrival);
        Ok(Some(message))
    }
}

#[cfg(test)]
mod tests {
    use depth_perception::CameraIntrinsics;
    use glam::Vec3;
    use perception_wire::{encode_imu, encode_keyframe};

    use super::*;

    #[test]
    fn a_recording_reads_back_every_message_in_the_order_it_was_written() {
        let directory = std::env::temp_dir().join(format!("localizer-recording-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        let imu = Imu { frame_id: 7, keyframe_camera_ns: 100, accel_camera_ns: vec![90], accel_m_s2: vec![Vec3::Z], gyro_camera_ns: vec![95, 99], gyro_rad_s: vec![Vec3::X, Vec3::Y] };
        let keyframe = Keyframe {
            frame_id: 7,
            captured_unix_ns: 1,
            width: 2,
            height: 1,
            intrinsics: CameraIntrinsics { fx: 1.0, fy: 1.0, ppx: 1.0, ppy: 0.5 },
            specific_force: Vec3::new(0.0, -9.8, 0.0),
            depth_mm: vec![1000u16, 2000].into(),
            color_jpeg: vec![1, 2, 3],
        };

        let mut writer = RecordingWriter::create(&directory).unwrap();
        writer.write(RecordedInput::Imu, &encode_imu(&imu)).unwrap();
        writer.write(RecordedInput::Keyframe, &encode_keyframe(&keyframe)).unwrap();
        writer.write(RecordedInput::Imu, &encode_imu(&Imu { frame_id: 8, ..imu.clone() })).unwrap();
        assert_eq!(writer.messages(), 3);
        writer.finish().unwrap();
        assert!(RecordingWriter::create(&directory).is_err(), "a recording is not written over");

        let mut reader = RecordingReader::open(&directory).unwrap();
        let mut read = Vec::new();
        while let Some(message) = reader.next_message().unwrap() {
            read.push(match message {
                RecordedMessage::Imu(read) => format!("imu {} with {} gyro samples", read.frame_id, read.gyro_rad_s.len()),
                RecordedMessage::Keyframe(read) => format!("keyframe {} with depth {:?} and colour {:?}", read.frame_id, &read.depth_mm[..], read.color_jpeg),
                RecordedMessage::OdomPose(_) | RecordedMessage::Calibration(_) => "another input".to_owned(),
            });
        }
        fs::remove_dir_all(&directory).unwrap();
        assert_eq!(read, ["imu 7 with 2 gyro samples", "keyframe 7 with depth [1000, 2000] and colour [1, 2, 3]", "imu 8 with 2 gyro samples"]);
    }

    #[test]
    fn a_recording_without_keyframes_does_not_open() {
        let directory = std::env::temp_dir().join(format!("localizer-recording-empty-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        RecordingWriter::create(&directory).unwrap().finish().unwrap();
        let opened = RecordingReader::open(&directory).is_ok();
        fs::remove_dir_all(&directory).unwrap();
        assert!(!opened);
    }
}
