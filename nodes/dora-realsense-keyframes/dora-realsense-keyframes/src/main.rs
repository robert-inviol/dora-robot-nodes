//! A RealSense D400 with an IMU as a dora node: depth aligned to colour, sent as a `keyframe` whenever the
//! picker calls for one, each preceded by an `imu` message holding every accelerometer and gyro sample since the
//! previous keyframe. Video and motion run on pipelines and threads of their own: a pipeline that also carries
//! video hands over only the newest motion sample per video frame, while a motion-only one delivers every sample.
//! Neither thread waits on dora's events.

use std::collections::HashSet;
use std::fs::File;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use depth_perception::CameraIntrinsics;
use dora_node_api::arrow::datatypes::Schema;
use dora_node_api::arrow::ipc::writer::FileWriter;
use dora_node_api::arrow::record_batch::RecordBatch;
use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters, Parameter};
use eyre::{eyre, Context, Result};
use glam::Vec3;
use perception_wire::{encode_imu, encode_keyframe, Imu, Keyframe};
use realsense_rust::config::Config;
use realsense_rust::context::Context as RealsenseContext;
use realsense_rust::frame::{AccelFrame, ColorFrame, DepthFrame, FrameEx, GyroFrame};
use realsense_rust::kind::{Rs2Format, Rs2Option, Rs2StreamKind, Rs2TimestampDomain};
use realsense_rust::pipeline::{ActivePipeline, InactivePipeline};
use realsense_rust::processing_blocks::align::Align;
use rgbd_keyframes::{GravityEstimate, KeyframePicker, KeyframeSettings};

const TICK_INPUT: &str = "tick";
const KEYFRAME_OUTPUT: &str = "keyframe";
const IMU_OUTPUT: &str = "imu";
/// Optional: append every keyframe to this Arrow IPC file, the format dora-keyframe-replay plays.
const RECORD_VARIABLE: &str = "KEYFRAME_RECORD_PATH";
/// Optional: append every imu message to this Arrow IPC file, which dora-keyframe-replay plays beside the keyframes.
const IMU_RECORD_VARIABLE: &str = "IMU_RECORD_PATH";
const COLOR_SIZE: (usize, usize) = (1280, 720);
const DEPTH_SIZE: (usize, usize) = (848, 480);
const VIDEO_FPS: usize = 30;
/// The D455's highest IMU rates.
const ACCEL_HZ: usize = 200;
const GYRO_HZ: usize = 400;
const JPEG_QUALITY: u8 = 90;
const GRAVITY_TIME_CONSTANT: Duration = Duration::from_millis(500);
const FRAME_WAIT: Duration = Duration::from_secs(2);

struct Captured {
    imu: Imu,
    keyframe: Keyframe,
    timings_ms: Vec<(&'static str, f64)>,
}

fn now_unix_ns() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).expect("clock after 1970").as_nanos() as i64
}

fn camera_time(frame: &impl FrameEx) -> Duration {
    Duration::from_secs_f64(frame.timestamp() / 1e3)
}

fn camera_ns(frame: &impl FrameEx) -> i64 {
    (frame.timestamp() * 1e6).round() as i64
}

/// The raw pixels of a video frame, which librealsense packs row by row with no padding for these formats.
fn pixels<T: Copy>(data: &std::os::raw::c_void, size: usize) -> Vec<T> {
    let pointer = data as *const std::os::raw::c_void as *const T;
    unsafe { std::slice::from_raw_parts(pointer, size / std::mem::size_of::<T>()) }.to_vec()
}

/// Motion samples the motion thread has delivered and the video thread has not yet taken.
#[derive(Default)]
struct MotionSamples {
    accel: Vec<(i64, Vec3)>,
    gyro: Vec<(i64, Vec3)>,
}

/// The motion-only pipeline, handed to the motion thread.
struct MotionPipeline(ActivePipeline);

// SAFETY: librealsense pipelines are thread-safe C objects; the wrapper only lacks Send because it holds raw
// pointers. The pipeline is used by one thread at a time and is joined before its context is dropped.
unsafe impl Send for MotionPipeline {}

fn start_motion(context: &RealsenseContext) -> Result<MotionPipeline> {
    let mut config = Config::new();
    config.enable_stream(Rs2StreamKind::Accel, None, 0, 0, Rs2Format::MotionXyz32F, ACCEL_HZ)?;
    config.enable_stream(Rs2StreamKind::Gyro, None, 0, 0, Rs2Format::MotionXyz32F, GYRO_HZ)?;
    let pipeline = InactivePipeline::try_from(context)
        .and_then(|pipeline| pipeline.start(Some(config)))
        .map_err(|e| eyre!("starting the motion streams: {e:#}"))?;
    use_camera_clock(&pipeline)?;
    Ok(MotionPipeline(pipeline))
}

/// Every keyframe and IMU sample is stamped on the camera's own clock, which all its sensors share. librealsense's
/// host-synced global time drifts apart between the motion and video sensors, so it is turned off on the sensors a
/// pipeline streams from, which are the only ones whose setting it reads.
fn use_camera_clock(pipeline: &ActivePipeline) -> Result<()> {
    for mut sensor in pipeline.profile().device().sensors() {
        if sensor.supports_option(Rs2Option::GlobalTimeEnabled) {
            sensor.set_option(Rs2Option::GlobalTimeEnabled, 0.0).map_err(|e| eyre!("turning off global time: {e}"))?;
        }
    }
    Ok(())
}

fn on_camera_clock(frame: &impl FrameEx) -> bool {
    frame.timestamp_domain() == Rs2TimestampDomain::HardwareClock
}

/// Whether a motion frame is one not seen before: the pipeline pairs each new sample of one stream with the
/// latest of the other, so most framesets repeat a sample.
fn is_new(frame: &impl FrameEx, last_frame_number: &mut Option<u64>) -> bool {
    let number = frame.frame_number();
    if !on_camera_clock(frame) || last_frame_number.is_some_and(|last| number <= last) {
        return false;
    }
    *last_frame_number = Some(number);
    true
}

fn stream_motion(mut motion: MotionPipeline, samples: &Mutex<MotionSamples>, stopping: &AtomicBool) -> Result<()> {
    let (mut last_accel, mut last_gyro) = (None, None);
    while !stopping.load(Ordering::Relaxed) {
        let frames = motion.0.wait(Some(FRAME_WAIT)).map_err(|e| eyre!("waiting for motion samples: {e}"))?;
        let mut shared = samples.lock().expect("motion samples lock");
        for accel in frames.frames_of_type::<AccelFrame>() {
            if is_new(&accel, &mut last_accel) {
                shared.accel.push((camera_ns(&accel), Vec3::from_array(*accel.acceleration())));
            }
        }
        for gyro in frames.frames_of_type::<GyroFrame>() {
            if is_new(&gyro, &mut last_gyro) {
                shared.gyro.push((camera_ns(&gyro), Vec3::from_array(*gyro.rotational_velocity())));
            }
        }
    }
    Ok(())
}

fn capture(sender: mpsc::SyncSender<Captured>, stopping: Arc<AtomicBool>) -> Result<()> {
    let context = RealsenseContext::new().wrap_err("creating the librealsense context")?;
    eyre::ensure!(!context.query_devices(HashSet::new()).is_empty(), "no RealSense camera found");
    let motion_samples = Mutex::new(MotionSamples::default());
    let motion_pipeline = start_motion(&context)?;
    std::thread::scope(|scope| {
        let motion = scope.spawn(|| stream_motion(motion_pipeline, &motion_samples, &stopping));
        let video = stream_video(&context, &sender, &motion_samples, &stopping, || motion.is_finished());
        stopping.store(true, Ordering::Relaxed);
        let motion = motion.join().map_err(|_| eyre!("the motion thread panicked"))?;
        video.and(motion)
    })
}

fn stream_video(
    context: &RealsenseContext,
    sender: &mpsc::SyncSender<Captured>,
    motion_samples: &Mutex<MotionSamples>,
    stopping: &AtomicBool,
    motion_finished: impl Fn() -> bool,
) -> Result<()> {
    let mut config = Config::new();
    config.enable_stream(Rs2StreamKind::Color, None, COLOR_SIZE.0, COLOR_SIZE.1, Rs2Format::Rgb8, VIDEO_FPS)?;
    config.enable_stream(Rs2StreamKind::Depth, None, DEPTH_SIZE.0, DEPTH_SIZE.1, Rs2Format::Z16, VIDEO_FPS)?;
    let mut pipeline = InactivePipeline::try_from(context)
        .and_then(|pipeline| pipeline.start(Some(config)))
        .map_err(|e| eyre!("starting the colour and depth streams: {e:#}"))?;
    use_camera_clock(&pipeline)?;
    let mut align = Align::new(Rs2StreamKind::Color, 1)?;
    let mut picker = KeyframePicker::new(KeyframeSettings::default());
    let mut gravity = GravityEstimate::new(GRAVITY_TIME_CONSTANT);
    let (mut last_accel_ns, mut last_gyro_ns): (Option<i64>, Option<i64>) = (None, None);
    let since = |last: Option<i64>, now: i64| Duration::from_nanos(last.map_or(0, |l| (now - l).max(0) as u64));
    let mut frame_id = 0u64;
    let mut pending = Imu::default();
    // The camera can still hold motion samples from an earlier session; only those after the first depth frame count.
    let mut session_start_ns: Option<i64> = None;
    eprintln!(
        "[realsense] streaming {}x{} colour, {}x{} depth, accel at {ACCEL_HZ} Hz and gyro at {GYRO_HZ} Hz",
        COLOR_SIZE.0, COLOR_SIZE.1, DEPTH_SIZE.0, DEPTH_SIZE.1
    );

    while !stopping.load(Ordering::Relaxed) {
        if motion_finished() {
            break;
        }
        let frames = pipeline.wait(Some(FRAME_WAIT)).map_err(|e| eyre!("waiting for camera frames: {e}"))?;
        let delivered = std::mem::take(&mut *motion_samples.lock().expect("motion samples lock"));
        if session_start_ns.is_none() {
            session_start_ns = frames.frames_of_type::<DepthFrame>().first().filter(|d| on_camera_clock(*d)).map(camera_ns);
        }
        let Some(start_ns) = session_start_ns else { continue };
        for (t, sample) in delivered.accel.into_iter().filter(|(t, _)| *t >= start_ns) {
            gravity.add(sample, since(last_accel_ns, t));
            last_accel_ns = Some(t);
            pending.accel_camera_ns.push(t);
            pending.accel_m_s2.push(sample);
        }
        for (t, sample) in delivered.gyro.into_iter().filter(|(t, _)| *t >= start_ns) {
            picker.add_rotation(sample, since(last_gyro_ns, t));
            last_gyro_ns = Some(t);
            pending.gyro_camera_ns.push(t);
            pending.gyro_rad_s.push(sample);
        }
        let Some(depth_frame) = frames.frames_of_type::<DepthFrame>().into_iter().next().filter(on_camera_clock) else {
            continue;
        };
        let Some(gravity_reaction) = gravity.reaction() else { continue };
        if !picker.take(camera_time(&depth_frame)) {
            continue;
        }
        let captured_unix_ns = now_unix_ns();
        let keyframe_camera_ns = camera_ns(&depth_frame);
        let started = Instant::now();
        align.queue(frames).map_err(|e| eyre!("queueing frames for alignment: {e}"))?;
        let aligned = align.wait(FRAME_WAIT).map_err(|e| eyre!("aligning depth to colour: {e}"))?;
        let depth = aligned.frames_of_type::<DepthFrame>().into_iter().next().ok_or_else(|| eyre!("aligned set has no depth"))?;
        let color = aligned.frames_of_type::<ColorFrame>().into_iter().next().ok_or_else(|| eyre!("aligned set has no colour"))?;
        let align_ms = started.elapsed().as_secs_f64() * 1e3;
        let (width, height) = (color.width(), color.height());
        let depth_units_mm = depth.depth_units().map_err(|e| eyre!("reading depth units: {e}"))? * 1e3;
        let mut depth_mm: Vec<u16> = pixels(unsafe { depth.get_data() }, depth.get_data_size());
        if (depth_units_mm - 1.0).abs() > 1e-6 {
            depth_mm.iter_mut().for_each(|d| *d = (*d as f32 * depth_units_mm).round().min(u16::MAX as f32) as u16);
        }
        let rgb: Vec<u8> = pixels(unsafe { color.get_data() }, color.get_data_size());
        let started = Instant::now();
        let mut color_jpeg = Vec::with_capacity(256 * 1024);
        jpeg_encoder::Encoder::new(&mut color_jpeg, JPEG_QUALITY)
            .encode(&rgb, width as u16, height as u16, jpeg_encoder::ColorType::Rgb)
            .wrap_err("encoding the colour image")?;
        let jpeg_ms = started.elapsed().as_secs_f64() * 1e3;
        let k = color.stream_profile().intrinsics().map_err(|e| eyre!("reading colour intrinsics: {e}"))?;
        let keyframe = Keyframe {
            frame_id,
            captured_unix_ns,
            width,
            height,
            intrinsics: CameraIntrinsics { fx: k.fx(), fy: k.fy(), ppx: k.ppx(), ppy: k.ppy() },
            gravity_reaction,
            depth_mm: depth_mm.into(),
            color_jpeg,
        };
        let imu = Imu { frame_id, keyframe_camera_ns, ..std::mem::take(&mut pending) };
        frame_id += 1;
        let timings_ms = vec![("align_ms", align_ms), ("jpeg_ms", jpeg_ms)];
        match sender.try_send(Captured { imu, keyframe, timings_ms }) {
            Ok(()) => {}
            // The keyframe is lost but its samples are not: they lead the next imu message.
            Err(TrySendError::Full(dropped)) => {
                eprintln!("[realsense] dora is behind; dropped keyframe {}", dropped.keyframe.frame_id);
                let mut kept = dropped.imu;
                kept.accel_camera_ns.append(&mut pending.accel_camera_ns);
                kept.accel_m_s2.append(&mut pending.accel_m_s2);
                kept.gyro_camera_ns.append(&mut pending.gyro_camera_ns);
                kept.gyro_rad_s.append(&mut pending.gyro_rad_s);
                pending = kept;
            }
            Err(TrySendError::Disconnected(_)) => break,
        }
    }
    Ok(())
}

/// Appends messages of one type to an Arrow IPC file, created when the first one arrives.
struct Recorder {
    path: String,
    writer: Option<FileWriter<File>>,
}

impl Recorder {
    fn from_env(variable: &str) -> Option<Recorder> {
        std::env::var(variable).ok().map(|path| Recorder { path, writer: None })
    }

    fn write(&mut self, message: &dora_node_api::arrow::array::StructArray) -> Result<()> {
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

fn main() -> Result<()> {
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let keyframe_output = DataId::from(KEYFRAME_OUTPUT.to_owned());
    let imu_output = DataId::from(IMU_OUTPUT.to_owned());
    let mut keyframe_recorder = Recorder::from_env(RECORD_VARIABLE);
    let mut imu_recorder = Recorder::from_env(IMU_RECORD_VARIABLE);
    let (sender, receiver) = mpsc::sync_channel::<Captured>(2);
    let stopping = Arc::new(AtomicBool::new(false));
    let camera = {
        let stopping = stopping.clone();
        std::thread::spawn(move || capture(sender, stopping))
    };

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, .. } if id.as_str() == TICK_INPUT => {
                for captured in receiver.try_iter() {
                    // The imu message goes first, so a consumer has the samples before the keyframe they lead up to.
                    let imu_message = encode_imu(&captured.imu);
                    if let Some(recorder) = imu_recorder.as_mut() {
                        recorder.write(&imu_message)?;
                    }
                    let mut imu_parameters = MetadataParameters::default();
                    imu_parameters.insert("accel_samples".to_owned(), Parameter::Integer(captured.imu.accel_camera_ns.len() as i64));
                    imu_parameters.insert("gyro_samples".to_owned(), Parameter::Integer(captured.imu.gyro_camera_ns.len() as i64));
                    node.send_output(imu_output.clone(), imu_parameters, imu_message)?;

                    let keyframe_message = encode_keyframe(&captured.keyframe);
                    if let Some(recorder) = keyframe_recorder.as_mut() {
                        recorder.write(&keyframe_message)?;
                    }
                    let mut parameters = MetadataParameters::default();
                    for (stage, ms) in captured.timings_ms {
                        parameters.insert(stage.to_owned(), Parameter::Float(ms));
                    }
                    node.send_output(keyframe_output.clone(), parameters, keyframe_message)?;
                }
                if camera.is_finished() {
                    break;
                }
            }
            Event::Stop(_) => break,
            _ => {}
        }
    }
    stopping.store(true, Ordering::Relaxed);
    if let Some(recorder) = keyframe_recorder {
        recorder.finish()?;
    }
    if let Some(recorder) = imu_recorder {
        recorder.finish()?;
    }
    camera.join().map_err(|_| eyre!("the camera thread panicked"))?
}
