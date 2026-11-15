//! A RealSense D400 with an IMU as a dora node that owns the camera and publishes every frame: the detector's
//! letterboxed `model_input`, raw `depth`, the `imu` samples leading up to the frame, and periodically the
//! `calibration` that relates them; full-size `color` when a dataflow lists it. Every stamp is on the camera's own
//! clock. Framesets go out once the accelerometer and gyro have both delivered a sample, and every frameset that goes
//! out has its colour and depth captured within `COLOR_DEPTH_GAP_LIMIT` (`rgbd_frames::frameset`); one further apart
//! stops the node. Outputs a dataflow does not list cost nothing. Video runs on a pipeline, which pairs colour with depth;
//! motion comes straight from the same device's motion sensor on a thread of its own, since a pipeline that also
//! carries video hands over only the newest motion sample per video frame. The video thread sends straight to dora,
//! so a frame waits on no timer.

mod motion_sensor;

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use dora_node_api::arrow::array::{Array, AsArray, UInt16Array};
use dora_node_api::arrow::datatypes::UInt64Type;
use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters, Parameter};
use eyre::{bail, eyre, Context, Result};
use glam::Vec3;
use perception_wire::camera::{encode_camera_calibration, CameraCalibration, RigidTransform, StreamIntrinsics};
use perception_wire::{encode_imu, Imu};
use realsense_rust::base::{Rs2Extrinsics, Rs2Intrinsics};
use realsense_rust::config::Config;
use realsense_rust::context::Context as RealsenseContext;
use realsense_rust::frame::{ColorFrame, DepthFrame, FrameEx};
use realsense_rust::kind::{Rs2Format, Rs2Option, Rs2StreamKind, Rs2TimestampDomain};
use realsense_rust::pipeline::{ActivePipeline, InactivePipeline};
use motion_sensor::{MotionSample, MotionStream, MotionStreamRate};
use realsense_rust::stream_profile::StreamProfile;
use rgbd_frames::frameset::{frameset_disposition, FramesetDisposition};
use rgbd_frames::imu_calibration::ImuCalibration;
use rgbd_frames::letterbox::Letterbox;
use rgbd_keyframes::GravityEstimate;
use shared_memory_outputs::SharedMemoryOutputs;

const MODEL_INPUT_OUTPUT: &str = "model_input";
const DEPTH_OUTPUT: &str = "depth";
const COLOR_OUTPUT: &str = "color";
const IMU_OUTPUT: &str = "imu";
const CALIBRATION_OUTPUT: &str = "calibration";
/// Required input: any timer. It carries nothing; it keeps the node's events open so a stop can reach it.
const TICK_INPUT: &str = "tick";
/// Optional input: when the dataflow gives it, `color` is sent only for the frames it names, instead of every frame.
const REQUEST_COLOR_INPUT: &str = "request_color";
/// Colour frames kept for `request_color` to name; a request for an older one is an error.
const COLOR_FRAMES_KEPT: usize = 15;
/// Required: this camera's `calibration.json` from librealsense's `rs-imu-calibration.py`, applied to every sample.
const IMU_CALIBRATION_VARIABLE: &str = "IMU_CALIBRATION_PATH";
/// Optional: the side of the square `model_input`, default 640.
const MODEL_INPUT_SIZE_VARIABLE: &str = "MODEL_INPUT_SIZE";
const COLOR_SIZE: (usize, usize) = (1280, 720);
const DEPTH_SIZE: (usize, usize) = (848, 480);
const VIDEO_FPS: usize = 30;
/// The D455's highest IMU rates.
const ACCEL_HZ: usize = 200;
const GYRO_HZ: usize = 400;
/// Motion samples the motion sensor's queue keeps for the motion thread: one second of both streams.
const MOTION_QUEUE_CAPACITY: usize = ACCEL_HZ + GYRO_HZ;
/// `calibration` goes out with the first frame and then once a second, so a node that starts late still gets it.
const CALIBRATION_EVERY_FRAMES: u64 = VIDEO_FPS as u64;
const GRAVITY_TIME_CONSTANT: Duration = Duration::from_millis(500);
const FRAME_WAIT: Duration = Duration::from_secs(2);
/// The most shared-memory samples of one output its receivers may hold unreleased: four seconds of frames, several
/// times what the dataflow's receivers and dora's region cache keep at once.
const UNRELEASED_SAMPLES_LIMIT: usize = 4 * VIDEO_FPS;
const EVENT_WAIT: Duration = Duration::from_millis(200);

fn camera_ns(frame: &impl FrameEx) -> i64 {
    (frame.timestamp() * 1e6).round() as i64
}

/// The pixels of a video frame, which librealsense packs row by row with no padding for these formats.
fn pixels<T: Copy>(data: &std::os::raw::c_void, size: usize) -> &[T] {
    let pointer = data as *const std::os::raw::c_void as *const T;
    unsafe { std::slice::from_raw_parts(pointer, size / std::mem::size_of::<T>()) }
}

/// Motion samples the motion thread has delivered and the video thread has not yet taken.
#[derive(Default)]
struct MotionSamples {
    accel: Vec<(i64, Vec3)>,
    gyro: Vec<(i64, Vec3)>,
}

fn start_pipeline(context: &RealsenseContext, config: Config, what: &str) -> Result<ActivePipeline> {
    let pipeline = InactivePipeline::try_from(context)
        .and_then(|pipeline| pipeline.start(Some(config)))
        .map_err(|e| eyre!("starting the {what} streams: {e:#}"))?;
    use_camera_clock(&pipeline)?;
    Ok(pipeline)
}

/// librealsense's host-synced global time drifts apart between the motion and video sensors, so it is turned off on
/// every sensor of the pipeline's device, the motion sensor included.
fn use_camera_clock(pipeline: &ActivePipeline) -> Result<()> {
    for mut sensor in pipeline.profile().device().sensors() {
        if sensor.supports_option(Rs2Option::GlobalTimeEnabled) {
            sensor.set_option(Rs2Option::GlobalTimeEnabled, 0.0).map_err(|e| eyre!("turning off global time: {e}"))?;
        }
    }
    Ok(())
}

/// The frame numbers one stream has delivered. A pipeline pairs each new frame of one stream with the latest of the
/// others, so framesets repeat frames, and this tells a new frame from a repeated one.
struct FrameSequence {
    stream: &'static str,
    last_number: Option<u64>,
}

impl FrameSequence {
    fn new(stream: &'static str) -> FrameSequence {
        FrameSequence { stream, last_number: None }
    }

    /// Whether `frame` is one this stream has not delivered before. Every stamp has to be on the camera's own clock,
    /// so a frame stamped on any other is an error.
    fn is_new(&mut self, frame: &impl FrameEx) -> Result<bool> {
        let number = frame.frame_number();
        let domain = frame.timestamp_domain();
        if domain != Rs2TimestampDomain::HardwareClock {
            bail!("{} frame {number} is stamped on the {domain:?} clock, not the camera's hardware clock", self.stream);
        }
        if self.last_number.is_some_and(|last| number <= last) {
            return Ok(false);
        }
        self.last_number = Some(number);
        Ok(true)
    }
}

fn stream_motion(
    mut motion: MotionStream,
    calibration: &ImuCalibration,
    samples: &Mutex<MotionSamples>,
    stopping: &AtomicBool,
) -> Result<()> {
    let (mut accel_frames, mut gyro_frames) = (FrameSequence::new("accel"), FrameSequence::new("gyro"));
    while !stopping.load(Ordering::Relaxed) {
        match motion.next_sample(FRAME_WAIT)? {
            MotionSample::Accel(accel) => {
                if accel_frames.is_new(&accel)? {
                    let sample = calibration.accel.apply(Vec3::from_array(*accel.acceleration()));
                    samples.lock().expect("motion samples lock").accel.push((camera_ns(&accel), sample));
                }
            }
            MotionSample::Gyro(gyro) => {
                if gyro_frames.is_new(&gyro)? {
                    let sample = calibration.gyro.apply(Vec3::from_array(*gyro.rotational_velocity()));
                    samples.lock().expect("motion samples lock").gyro.push((camera_ns(&gyro), sample));
                }
            }
        }
    }
    Ok(())
}

fn stream_profile(pipeline: &ActivePipeline, kind: Rs2StreamKind) -> Result<&StreamProfile> {
    pipeline.profile().streams().iter().find(|s| s.kind() == kind).ok_or_else(|| eyre!("no {kind:?} stream is running"))
}

fn intrinsics(profile: &StreamProfile) -> Result<StreamIntrinsics> {
    let k: Rs2Intrinsics = profile.intrinsics().map_err(|e| eyre!("reading {:?} intrinsics: {e}", profile.kind()))?;
    Ok(StreamIntrinsics { width: k.width() as u16, height: k.height() as u16, fx: k.fx(), fy: k.fy(), ppx: k.ppx(), ppy: k.ppy() })
}

fn transform(from: &StreamProfile, to: &StreamProfile) -> Result<RigidTransform> {
    let e: Rs2Extrinsics =
        from.extrinsics(to).map_err(|e| eyre!("reading the {:?} to {:?} extrinsics: {e}", from.kind(), to.kind()))?;
    Ok(RigidTransform { rotation: e.rotation(), translation_m: e.translation() })
}

/// Intrinsics come from the running streams. Transforms are per stream, not per mode, and librealsense only relates
/// profiles of different pipelines through the device's own profile lists, so the IMU transform is read from those.
/// The depth sensor's unit in millimetres, read once: nothing in the node changes it.
fn read_depth_units_mm(video: &ActivePipeline) -> Result<f32> {
    let sensors = video.profile().device().sensors();
    let depth_sensor = sensors
        .iter()
        .find(|sensor| sensor.supports_option(Rs2Option::DepthUnits))
        .ok_or_else(|| eyre!("the camera has no sensor with depth units"))?;
    let units_m = depth_sensor.get_option(Rs2Option::DepthUnits).ok_or_else(|| eyre!("reading depth units"))?;
    Ok(units_m * 1e3)
}

fn read_calibration(video: &ActivePipeline) -> Result<CameraCalibration> {
    let (color, depth) = (stream_profile(video, Rs2StreamKind::Color)?, stream_profile(video, Rs2StreamKind::Depth)?);
    let device_profiles: Vec<StreamProfile> =
        video.profile().device().sensors().iter().flat_map(|sensor| sensor.stream_profiles()).collect();
    let device_profile = |kind: Rs2StreamKind| {
        device_profiles.iter().find(|p| p.kind() == kind).ok_or_else(|| eyre!("the camera has no {kind:?} stream"))
    };
    Ok(CameraCalibration {
        color: intrinsics(color)?,
        depth: intrinsics(depth)?,
        depth_to_color: transform(depth, color)?,
        depth_to_imu: transform(device_profile(Rs2StreamKind::Depth)?, device_profile(Rs2StreamKind::Accel)?)?,
    })
}

/// What the dataflow wants from this node, read once at start.
struct Wanted {
    model_input: Option<Letterbox>,
    depth: bool,
    color: bool,
    color_on_request: bool,
}

impl Wanted {
    fn from_node(node: &DoraNode) -> Result<Wanted> {
        let config = node.node_config();
        let listed = |name: &str| config.outputs.contains(&DataId::from(name.to_owned()));
        let side = match std::env::var(MODEL_INPUT_SIZE_VARIABLE) {
            Ok(text) => text.parse().wrap_err_with(|| format!("{MODEL_INPUT_SIZE_VARIABLE} is not a whole number: {text}"))?,
            Err(_) => 640,
        };
        let model_input = listed(MODEL_INPUT_OUTPUT)
            .then(|| Letterbox::new(COLOR_SIZE.0, COLOR_SIZE.1, side))
            .transpose()
            .wrap_err("sizing the model input")?;
        Ok(Wanted {
            model_input,
            depth: listed(DEPTH_OUTPUT),
            color: listed(COLOR_OUTPUT),
            color_on_request: config.inputs.contains_key(&DataId::from(REQUEST_COLOR_INPUT.to_owned())),
        })
    }
}

fn image_parameters(encoding: &str, width: usize, height: usize, frame_id: u64, captured_camera_ns: i64) -> MetadataParameters {
    let mut parameters = MetadataParameters::default();
    parameters.insert("encoding".to_owned(), Parameter::String(encoding.to_owned()));
    parameters.insert("width".to_owned(), Parameter::Integer(width as i64));
    parameters.insert("height".to_owned(), Parameter::Integer(height as i64));
    parameters.insert("frame_id".to_owned(), Parameter::Integer(frame_id as i64));
    parameters.insert("camera_ns".to_owned(), Parameter::Integer(captured_camera_ns));
    parameters
}

fn capture(
    mut node: DoraNode,
    imu_calibration: ImuCalibration,
    stopping: Arc<AtomicBool>,
    color_requests: Arc<Mutex<Vec<u64>>>,
) -> Result<()> {
    let wanted = Wanted::from_node(&node)?;
    let context = RealsenseContext::new().wrap_err("creating the librealsense context")?;
    eyre::ensure!(!context.query_devices(HashSet::new()).is_empty(), "no RealSense camera found");
    let mut video_config = Config::new();
    video_config.enable_stream(Rs2StreamKind::Color, None, COLOR_SIZE.0, COLOR_SIZE.1, Rs2Format::Rgb8, VIDEO_FPS)?;
    video_config.enable_stream(Rs2StreamKind::Depth, None, DEPTH_SIZE.0, DEPTH_SIZE.1, Rs2Format::Z16, VIDEO_FPS)?;
    let mut video = start_pipeline(&context, video_config, "colour and depth")?;
    let calibration = read_calibration(&video)?;
    let depth_units_mm = read_depth_units_mm(&video)?;
    // The motion sensor is the one of the device the video pipeline opened, which the first frameset names.
    let first_frames = video.wait(Some(FRAME_WAIT)).map_err(|e| eyre!("waiting for the first camera frames: {e}"))?;
    let motion_rates = [MotionStreamRate::accel(ACCEL_HZ), MotionStreamRate::gyro(GYRO_HZ)];
    let motion = MotionStream::start(first_frames, &motion_rates, MOTION_QUEUE_CAPACITY)?;
    let motion_samples = Mutex::new(MotionSamples::default());
    std::thread::scope(|scope| {
        let motion = scope.spawn(|| stream_motion(motion, &imu_calibration, &motion_samples, &stopping));
        let camera = CameraStreams { calibration: &calibration, depth_units_mm, motion_samples: &motion_samples };
        let sent = stream_video(&mut node, video, &wanted, camera, &stopping, &color_requests, || motion.is_finished());
        stopping.store(true, Ordering::Relaxed);
        let motion = motion.join().map_err(|_| eyre!("the motion thread panicked"))?;
        sent.and(motion)
    })
}

/// What the video thread reads besides its frames: the calibration and depth units read at start, and the motion
/// samples the motion thread delivers.
struct CameraStreams<'a> {
    calibration: &'a CameraCalibration,
    depth_units_mm: f32,
    motion_samples: &'a Mutex<MotionSamples>,
}

fn stream_video(
    node: &mut DoraNode,
    mut pipeline: ActivePipeline,
    wanted: &Wanted,
    camera: CameraStreams,
    stopping: &AtomicBool,
    color_requests: &Mutex<Vec<u64>>,
    motion_finished: impl Fn() -> bool,
) -> Result<()> {
    let output = |name: &str| DataId::from(name.to_owned());
    let mut gravity = GravityEstimate::new(GRAVITY_TIME_CONSTANT);
    let mut last_accel_ns: Option<i64> = None;
    let (mut depth_frames, mut color_frames) = (FrameSequence::new("depth"), FrameSequence::new("colour"));
    let mut outputs = SharedMemoryOutputs::new(node, UNRELEASED_SAMPLES_LIMIT)?;
    let mut frame_id = 0u64;
    let mut pending = Imu::default();
    // The camera can still hold motion samples from an earlier session; only those after the first depth frame count.
    let mut session_start_ns: Option<i64> = None;
    // The last colour frames, oldest first, for `request_color`: frame id, capture time and pixels.
    let mut kept_colors: VecDeque<(u64, i64, Vec<u8>)> = VecDeque::with_capacity(COLOR_FRAMES_KEPT);
    let mut waiting_requests: Vec<u64> = Vec::new();
    eprintln!(
        "[realsense] streaming {}x{} colour, {}x{} depth at {VIDEO_FPS} fps, accel at {ACCEL_HZ} Hz and gyro at {GYRO_HZ} Hz",
        COLOR_SIZE.0, COLOR_SIZE.1, DEPTH_SIZE.0, DEPTH_SIZE.1
    );

    while !stopping.load(Ordering::Relaxed) {
        if motion_finished() {
            break;
        }
        let frames = pipeline.wait(Some(FRAME_WAIT)).map_err(|e| eyre!("waiting for camera frames: {e}"))?;
        // A frameset can repeat the previous depth frame beside a new colour one; each depth frame is sent once.
        let Some(depth) = frames.frames_of_type::<DepthFrame>().into_iter().next() else { continue };
        if !depth_frames.is_new(&depth)? {
            continue;
        }
        let Some(color) = frames.frames_of_type::<ColorFrame>().into_iter().next() else { continue };
        // It can also repeat the previous colour frame beside a new depth one. That colour already went out with the
        // frame before, so this depth frame is not sent.
        if !color_frames.is_new(&color)? {
            eprintln!(
                "[realsense] depth frame {} came with colour frame {} again; that depth frame is not sent",
                depth.frame_number(),
                color.frame_number()
            );
            continue;
        }
        let depth_camera_ns = camera_ns(&depth);
        let color_camera_ns = camera_ns(&color);
        let mut motion_samples = camera.motion_samples.lock().expect("motion samples lock");
        let motion_delivering =
            session_start_ns.is_some() || (!motion_samples.accel.is_empty() && !motion_samples.gyro.is_empty());
        let disposition = frameset_disposition(motion_delivering, color_camera_ns, depth_camera_ns).wrap_err_with(|| {
            format!("colour frame {} and depth frame {}", color.frame_number(), depth.frame_number())
        })?;
        if disposition == FramesetDisposition::AwaitingMotion {
            continue;
        }
        let delivered = std::mem::take(&mut *motion_samples);
        drop(motion_samples);
        let start_ns = *session_start_ns.get_or_insert(depth_camera_ns);
        for (t, sample) in delivered.accel.into_iter().filter(|(t, _)| *t >= start_ns) {
            gravity.add(sample, Duration::from_nanos(last_accel_ns.map_or(0, |last| (t - last).max(0) as u64)));
            last_accel_ns = Some(t);
            pending.accel_camera_ns.push(t);
            pending.accel_m_s2.push(sample);
        }
        for (t, sample) in delivered.gyro.into_iter().filter(|(t, _)| *t >= start_ns) {
            pending.gyro_camera_ns.push(t);
            pending.gyro_rad_s.push(sample);
        }

        // The imu message goes first, so a consumer has the samples before the frame they lead up to.
        let imu = Imu { frame_id, keyframe_camera_ns: depth_camera_ns, ..std::mem::take(&mut pending) };
        outputs.send_array(node, &output(IMU_OUTPUT), MetadataParameters::default(), encode_imu(&imu))?;
        if frame_id % CALIBRATION_EVERY_FRAMES == 0 {
            outputs.send_array(node, &output(CALIBRATION_OUTPUT), MetadataParameters::default(), encode_camera_calibration(camera.calibration))?;
        }

        let rgb: &[u8] = pixels(unsafe { color.get_data() }, color.get_data_size());
        if let Some(letterbox) = &wanted.model_input {
            let started = Instant::now();
            let mut parameters = image_parameters("rgb8", letterbox.side, letterbox.side, frame_id, color_camera_ns);
            parameters.insert("source_width".to_owned(), Parameter::Integer(letterbox.source_width as i64));
            parameters.insert("source_height".to_owned(), Parameter::Integer(letterbox.source_height as i64));
            parameters.insert("shrink".to_owned(), Parameter::Integer(letterbox.factor as i64));
            parameters.insert("pad_x".to_owned(), Parameter::Integer(letterbox.pad_x as i64));
            parameters.insert("pad_y".to_owned(), Parameter::Integer(letterbox.pad_y as i64));
            let mut letterbox_ms = 0.0;
            outputs.send_raw(node, &output(MODEL_INPUT_OUTPUT), parameters, letterbox.output_len(), |out| {
                letterbox.fill(rgb, out);
                letterbox_ms = started.elapsed().as_secs_f64() * 1e3;
            })?;
            if frame_id % CALIBRATION_EVERY_FRAMES == 0 {
                eprintln!("[realsense] frame {frame_id}: letterbox {letterbox_ms:.2} ms");
            }
        }
        if wanted.depth {
            let units_mm = camera.depth_units_mm;
            let raw: &[u16] = pixels(unsafe { depth.get_data() }, depth.get_data_size());
            let depth_mm: UInt16Array = if (units_mm - 1.0).abs() > 1e-6 {
                raw.iter().map(|&d| (d as f32 * units_mm).round().min(u16::MAX as f32) as u16).collect()
            } else {
                UInt16Array::from(raw.to_vec())
            };
            let mut parameters = image_parameters("mono16", depth.width(), depth.height(), frame_id, depth_camera_ns);
            if let Some(specific_force) = gravity.specific_force() {
                parameters.insert("specific_force".to_owned(), Parameter::ListFloat(specific_force.to_array().map(f64::from).to_vec()));
            }
            outputs.send_array(node, &output(DEPTH_OUTPUT), parameters, depth_mm)?;
        }
        if wanted.color && !wanted.color_on_request {
            let parameters = image_parameters("rgb8", color.width(), color.height(), frame_id, color_camera_ns);
            outputs.send_raw(node, &output(COLOR_OUTPUT), parameters, rgb.len(), |out| out.copy_from_slice(rgb))?;
        }
        if wanted.color && wanted.color_on_request {
            let mut pixels = if kept_colors.len() == COLOR_FRAMES_KEPT { kept_colors.pop_front().expect("full").2 } else { Vec::new() };
            pixels.clear();
            pixels.extend_from_slice(rgb);
            kept_colors.push_back((frame_id, color_camera_ns, pixels));
            waiting_requests.append(&mut color_requests.lock().expect("colour requests lock"));
            let mut still_waiting = Vec::new();
            for requested in waiting_requests.drain(..) {
                if requested > frame_id {
                    still_waiting.push(requested);
                    continue;
                }
                let Some((_, requested_ns, pixels)) = kept_colors.iter().find(|(id, _, _)| *id == requested) else {
                    bail!("colour for frame {requested} was asked for after it was dropped; the camera keeps the last {COLOR_FRAMES_KEPT}");
                };
                let parameters = image_parameters("rgb8", color.width(), color.height(), requested, *requested_ns);
                outputs.send_raw(node, &output(COLOR_OUTPUT), parameters, pixels.len(), |out| out.copy_from_slice(pixels))?;
            }
            waiting_requests = still_waiting;
        }
        frame_id += 1;
    }
    Ok(())
}

fn load_imu_calibration() -> Result<ImuCalibration> {
    let path = std::env::var(IMU_CALIBRATION_VARIABLE).wrap_err_with(|| format!("{IMU_CALIBRATION_VARIABLE} is not set"))?;
    let text = std::fs::read_to_string(&path).wrap_err_with(|| format!("reading the IMU calibration {path}"))?;
    ImuCalibration::from_librealsense_json(&text).wrap_err_with(|| format!("parsing the IMU calibration {path}"))
}

fn main() -> Result<()> {
    let imu_calibration = load_imu_calibration()?;
    let (node, mut events) = DoraNode::init_from_env()?;
    // dora closes the events of a node without inputs at once, and a stop could then never reach it.
    eyre::ensure!(
        node.node_config().inputs.contains_key(&DataId::from(TICK_INPUT.to_owned())),
        "give the camera a `{TICK_INPUT}` input (any dora timer) so the dataflow can stop it"
    );
    let stopping = Arc::new(AtomicBool::new(false));
    let color_requests = Arc::new(Mutex::new(Vec::new()));
    let camera = {
        let (stopping, color_requests) = (stopping.clone(), color_requests.clone());
        std::thread::spawn(move || capture(node, imu_calibration, stopping, color_requests))
    };
    while !camera.is_finished() {
        match events.recv_timeout(EVENT_WAIT) {
            Some(Event::Input { id, data, .. }) if id.as_str() == REQUEST_COLOR_INPUT => {
                let frames = data.as_primitive_opt::<UInt64Type>().ok_or_else(|| eyre!("request_color is {:?}, not u64 frame ids", data.data_type()))?;
                color_requests.lock().expect("colour requests lock").extend(frames.values().iter().copied());
            }
            Some(Event::Stop(_)) | None => break,
            _ => {}
        }
    }
    stopping.store(true, Ordering::Relaxed);
    camera.join().map_err(|_| eyre!("the camera thread panicked"))?
}
