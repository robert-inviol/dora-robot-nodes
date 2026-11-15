//! A recording of the odometry node's inputs as they arrived: a directory with one Arrow IPC stream file per input.
//! Each batch is one message, with an `arrival` column that numbers every message of every input in the order they
//! arrived, so a replay feeds them in that order. Depth frames are kept whole, as the camera sent them; imu messages,
//! calibrations and person poses are kept as their wire messages.
//!
//! A stream file is readable up to its last whole batch, so a recording cut short by a crash still replays.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dora_node_api::arrow::array::{Array, ArrayRef, AsArray, Int64Array, ListArray, RecordBatch, StructArray, UInt16Array, UInt32Array, UInt64Array};
use dora_node_api::arrow::buffer::OffsetBuffer;
use dora_node_api::arrow::datatypes::{DataType, Field, Fields, Int64Type, Schema, UInt16Type, UInt32Type, UInt64Type};
use dora_node_api::arrow::ipc::reader::StreamReader;
use dora_node_api::arrow::ipc::writer::StreamWriter;
use eyre::{bail, ensure, eyre, Result, WrapErr};
use perception_wire::camera::{decode_camera_calibration, CameraCalibration};
use perception_wire::localization::CameraNanos;
use perception_wire::people::{decode_person_poses, PersonPoses};
use perception_wire::{decode_imu, Imu};

use crate::DepthFrame;
use depth_inertial_odometry::tracker::DepthFrameId;

/// The inputs a recording keeps, each in its own file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordedInput {
    Depth,
    Imu,
    Calibration,
    PersonPoses,
}

impl RecordedInput {
    pub const ALL: [RecordedInput; 4] = [RecordedInput::Depth, RecordedInput::Imu, RecordedInput::Calibration, RecordedInput::PersonPoses];

    /// The node input this is, which names its file.
    pub fn input_name(self) -> &'static str {
        match self {
            RecordedInput::Depth => "depth",
            RecordedInput::Imu => "imu",
            RecordedInput::Calibration => "calibration",
            RecordedInput::PersonPoses => "person_poses",
        }
    }

    fn file(self, directory: &Path) -> PathBuf {
        directory.join(format!("{}.arrows", self.input_name()))
    }

    fn index(self) -> usize {
        self as usize
    }
}

const ARRIVAL: &str = "arrival";

fn arrival_field() -> Field {
    Field::new(ARRIVAL, DataType::UInt64, false)
}

fn depth_schema() -> Schema {
    Schema::new(vec![
        arrival_field(),
        Field::new("frame_id", DataType::UInt64, false),
        Field::new("camera_ns", DataType::Int64, false),
        Field::new("width", DataType::UInt32, false),
        Field::new("height", DataType::UInt32, false),
        Field::new("depth_mm", DataType::List(Arc::new(Field::new("item", DataType::UInt16, false))), false),
    ])
}

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

    fn write(&mut self, input: RecordedInput, batch: &RecordBatch) -> Result<()> {
        let stream = match &mut self.streams[input.index()] {
            Some(stream) => stream,
            empty => {
                let path = input.file(&self.directory);
                let file = File::create(&path).wrap_err_with(|| format!("creating {}", path.display()))?;
                empty.insert(StreamWriter::try_new(BufWriter::new(file), &batch.schema()).wrap_err_with(|| format!("starting {}", path.display()))?)
            }
        };
        stream.write(batch).wrap_err_with(|| format!("writing a {} message", input.input_name()))?;
        self.next_arrival += 1;
        Ok(())
    }

    pub fn write_depth(&mut self, frame: &DepthFrame) -> Result<()> {
        let width = u32::try_from(frame.width).wrap_err("a depth frame's width")?;
        let height = u32::try_from(frame.height).wrap_err("a depth frame's height")?;
        let item = Arc::new(Field::new("item", DataType::UInt16, false));
        let depth_mm = ListArray::new(item, OffsetBuffer::from_lengths([frame.depth_mm.len()]), Arc::new(UInt16Array::from(frame.depth_mm.to_vec())), None);
        let columns: Vec<ArrayRef> = vec![
            Arc::new(UInt64Array::from(vec![self.next_arrival])),
            Arc::new(UInt64Array::from(vec![frame.frame_id.0])),
            Arc::new(Int64Array::from(vec![frame.time.0])),
            Arc::new(UInt32Array::from(vec![width])),
            Arc::new(UInt32Array::from(vec![height])),
            Arc::new(depth_mm),
        ];
        let batch = RecordBatch::try_new(Arc::new(depth_schema()), columns).wrap_err("a depth frame as a batch")?;
        self.write(RecordedInput::Depth, &batch)
    }

    /// Writes an imu message, a calibration or person poses: a one-row struct as it came over the wire.
    pub fn write_message(&mut self, input: RecordedInput, message: &StructArray) -> Result<()> {
        ensure!(input != RecordedInput::Depth, "depth frames are written with write_depth");
        ensure!(message.len() == 1, "a {} message has {} rows, not one", input.input_name(), message.len());
        let mut fields = vec![Arc::new(arrival_field())];
        fields.extend(message.fields().iter().cloned());
        let mut columns: Vec<ArrayRef> = vec![Arc::new(UInt64Array::from(vec![self.next_arrival]))];
        columns.extend(message.columns().iter().cloned());
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).wrap_err_with(|| format!("a {} message as a batch", input.input_name()))?;
        self.write(input, &batch)
    }

    /// Messages written so far, of every input.
    pub fn messages(&self) -> u64 {
        self.next_arrival
    }

    /// Ends every stream and hands what is buffered to the operating system, which writes it to disk in its own
    /// time: waiting for gigabytes to reach the disk would hold up the node's inputs.
    pub fn finish(self) -> Result<()> {
        for (input, stream) in RecordedInput::ALL.into_iter().zip(self.streams) {
            let Some(mut stream) = stream else { continue };
            stream.finish().wrap_err_with(|| format!("ending the {} stream", input.input_name()))?;
            stream.into_inner().wrap_err_with(|| format!("flushing the {} stream", input.input_name()))?;
        }
        Ok(())
    }
}

/// A depth frame read back from a recording.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordedDepthFrame {
    pub frame_id: DepthFrameId,
    pub time: CameraNanos,
    pub width: usize,
    pub height: usize,
    pub depth_mm: Vec<u16>,
}

impl RecordedDepthFrame {
    pub fn as_frame(&self) -> DepthFrame<'_> {
        DepthFrame { frame_id: self.frame_id, time: self.time, width: self.width, height: self.height, depth_mm: &self.depth_mm }
    }
}

/// One recorded message.
#[derive(Clone, Debug, PartialEq)]
pub enum RecordedMessage {
    Depth(RecordedDepthFrame),
    Imu(Imu),
    Calibration(CameraCalibration),
    PersonPoses(PersonPoses),
}

fn single_value<T: dora_node_api::arrow::datatypes::ArrowPrimitiveType>(batch: &RecordBatch, column: usize) -> Result<T::Native> {
    let values = batch.column(column).as_primitive_opt::<T>().ok_or_else(|| eyre!("column {} is {:?}", batch.schema().field(column).name(), batch.column(column).data_type()))?;
    ensure!(values.len() == 1, "a recorded batch has {} rows, not one", values.len());
    Ok(values.value(0))
}

fn decode(input: RecordedInput, batch: &RecordBatch) -> Result<(u64, RecordedMessage)> {
    ensure!(batch.schema().field(0).name() == ARRIVAL, "a {} batch does not start with its arrival", input.input_name());
    let arrival = single_value::<UInt64Type>(batch, 0)?;
    if input == RecordedInput::Depth {
        let lists = batch.column(5).as_list_opt::<i32>().ok_or_else(|| eyre!("depth_mm is {:?}", batch.column(5).data_type()))?;
        let depth_mm = lists.value(0).as_primitive_opt::<UInt16Type>().ok_or_else(|| eyre!("depth_mm holds {:?}", lists.value_type()))?.values().to_vec();
        let frame = RecordedDepthFrame {
            frame_id: DepthFrameId(single_value::<UInt64Type>(batch, 1)?),
            time: CameraNanos(single_value::<Int64Type>(batch, 2)?),
            width: single_value::<UInt32Type>(batch, 3)? as usize,
            height: single_value::<UInt32Type>(batch, 4)? as usize,
            depth_mm,
        };
        return Ok((arrival, RecordedMessage::Depth(frame)));
    }
    let schema = batch.schema();
    let fields: Fields = schema.fields().iter().skip(1).cloned().collect();
    let message = StructArray::try_new(fields, batch.columns()[1..].to_vec(), None).wrap_err_with(|| format!("a recorded {} message", input.input_name()))?;
    let decoded = match input {
        RecordedInput::Imu => RecordedMessage::Imu(decode_imu(&message)?),
        RecordedInput::Calibration => RecordedMessage::Calibration(decode_camera_calibration(&message)?),
        RecordedInput::PersonPoses => RecordedMessage::PersonPoses(decode_person_poses(&message)?),
        RecordedInput::Depth => unreachable!("depth is decoded above"),
    };
    Ok((arrival, decoded))
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
    /// Opens every input's file in the directory; depth, imu and calibration must be there, person poses may not.
    pub fn open(directory: &Path) -> Result<RecordingReader> {
        let mut streams = Vec::new();
        for input in RecordedInput::ALL {
            let path = input.file(directory);
            if !path.exists() {
                if input == RecordedInput::PersonPoses {
                    continue;
                }
                bail!("the recording {} has no {}", directory.display(), path.display());
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
    use glam::Vec3;
    use perception_wire::camera::{encode_camera_calibration, RigidTransform, StreamIntrinsics};
    use perception_wire::encode_imu;
    use perception_wire::people::{encode_person_poses, Keypoint, PersonPose, PixelBox};

    use super::*;

    #[test]
    fn a_recording_reads_back_every_message_in_the_order_it_was_written() {
        let directory = std::env::temp_dir().join(format!("odometry-recording-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        let intrinsics = StreamIntrinsics { width: 4, height: 2, fx: 3.0, fy: 3.0, ppx: 2.0, ppy: 1.0 };
        let identity = RigidTransform { rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], translation_m: [0.01, 0.0, 0.0] };
        let calibration = CameraCalibration { color: intrinsics, depth: intrinsics, depth_to_color: identity, depth_to_imu: identity };
        let imu = Imu { frame_id: 7, keyframe_camera_ns: 100, accel_camera_ns: vec![90], accel_m_s2: vec![Vec3::Z], gyro_camera_ns: vec![95, 99], gyro_rad_s: vec![Vec3::X, Vec3::Y] };
        let poses = PersonPoses { frame_id: 6, captured_camera_ns: 80, width: 4, height: 2, people: vec![PersonPose { track_id: Some(3), score: 0.5, bbox: PixelBox { x_px: 1.0, y_px: 0.0, w_px: 2.0, h_px: 1.0 }, keypoints: vec![Keypoint { x_px: 1.5, y_px: 0.5, score: 0.25 }; 17] }] };
        let depth = RecordedDepthFrame { frame_id: DepthFrameId(7), time: CameraNanos(100), width: 4, height: 2, depth_mm: vec![0, 1, 2, 3, 4, 5, 6, 65535] };

        let mut writer = RecordingWriter::create(&directory).unwrap();
        writer.write_message(RecordedInput::Calibration, &encode_camera_calibration(&calibration)).unwrap();
        writer.write_depth(&depth.as_frame()).unwrap();
        writer.write_message(RecordedInput::PersonPoses, &encode_person_poses(&poses)).unwrap();
        writer.write_message(RecordedInput::Imu, &encode_imu(&imu)).unwrap();
        writer.write_depth(&RecordedDepthFrame { frame_id: DepthFrameId(8), ..depth.clone() }.as_frame()).unwrap();
        writer.finish().unwrap();
        assert!(RecordingWriter::create(&directory).is_err(), "a recording is not written over");

        let mut reader = RecordingReader::open(&directory).unwrap();
        let mut read = Vec::new();
        while let Some(message) = reader.next_message().unwrap() {
            read.push(message);
        }
        fs::remove_dir_all(&directory).unwrap();
        assert_eq!(
            read,
            vec![
                RecordedMessage::Calibration(calibration),
                RecordedMessage::Depth(depth.clone()),
                RecordedMessage::PersonPoses(poses),
                RecordedMessage::Imu(imu),
                RecordedMessage::Depth(RecordedDepthFrame { frame_id: DepthFrameId(8), ..depth }),
            ]
        );
    }
}
