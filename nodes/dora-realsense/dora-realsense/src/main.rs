//! A RealSense D400 with an IMU as a dora node that owns the camera and publishes every frame: the detector's
//! letterboxed `model_input`, raw `depth`, the `imu` samples leading up to the frame, and periodically the
//! `calibration` that relates them; full-size `color` when a dataflow lists it. Every stamp is on the camera's own
//! clock. Outputs a dataflow does not list cost nothing. Video and motion run on pipelines and threads of their own:
//! a pipeline that also carries video hands over only the newest motion sample per video frame, while a motion-only
//! one delivers every sample. The video thread sends straight to dora, so a frame waits on no timer.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use dora_node_api::arrow::array::UInt16Array;
use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters, Parameter};
use eyre::{eyre, Context, Result};
use glam::Vec3;
use perception_wire::camera::{encode_camera_calibration, CameraCalibration, RigidTransform, StreamIntrinsics};
use perception_wire::{encode_imu, Imu};
use realsense_rust::base::{Rs2Extrinsics, Rs2Intrinsics};
use realsense_rust::config::Config;
use realsense_rust::context::Context as RealsenseContext;
use realsense_rust::frame::{AccelFrame, ColorFrame, DepthFrame, FrameEx, GyroFrame};
use realsense_rust::kind::{Rs2Format, Rs2Option, Rs2StreamKind, Rs2TimestampDomain};
use realsense_rust::pipeline::{ActivePipeline, InactivePipeline};
use realsense_rust::stream_profile::StreamProfile;
use rgbd_frames::imu_calibration::ImuCalibration;
use rgbd_frames::letterbox::Letterbox;
use rgbd_keyframes::GravityEstimate;

const MODEL_INPUT_OUTPUT: &str = "model_input";
const DEPTH_OUTPUT: &str = "depth";
const COLOR_OUTPUT: &str = "color";
const IMU_OUTPUT: &str = "imu";
const CALIBRATION_OUTPUT: &str = "calibration";
/// Required input: any timer. It carries nothing; it keeps the node's events open so a stop can reach it.
const TICK_INPUT: &str = "tick";
/// Optional input: when the dataflow gives it, `color` is sent once for each message on it instead of every frame.
const REQUEST_COLOR_INPUT: &str = "request_color";
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
/// `calibration` goes out with the first frame and then once a second, so a node that starts late still gets it.
const CALIBRATION_EVERY_FRAMES: u64 = VIDEO_FPS as u64;
const GRAVITY_TIME_CONSTANT: Duration = Duration::from_millis(500);
const FRAME_WAIT: Duration = Duration::from_secs(2);
const EVENT_WAIT: Duration = Duration::from_millis(200);

fn camera_ns(frame: &impl FrameEx) -> i64 {
    (frame.timestamp() * 1e6).round() as i64
}

fn on_camera_clock(frame: &impl FrameEx) -> bool {
    frame.timestamp_domain() == Rs2TimestampDomain::HardwareClock
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

/// The motion-only pipeline, handed to the motion thread.
struct MotionPipeline(ActivePipeline);

// SAFETY: librealsense pipelines are thread-safe C objects; the wrapper only lacks Send because it holds raw
// pointers. The pipeline is used by one thread at a time and is joined before its context is dropped.
unsafe impl Send for MotionPipeline {}

fn start_pipeline(context: &RealsenseContext, config: Config, what: &str) -> Result<ActivePipeline> {
    let pipeline = InactivePipeline::try_from(context)
        .and_then(|pipeline| pipeline.start(Some(config)))
        .map_err(|e| eyre!("starting the {what} streams: {e:#}"))?;
    use_camera_clock(&pipeline)?;
    Ok(pipeline)
}

/// librealsense's host-synced global time drifts apart between the motion and video sensors, so it is turned off on
/// the sensors a pipeline streams from, which are the only ones whose setting it reads.
fn use_camera_clock(pipeline: &ActivePipeline) -> Result<()> {
    for mut sensor in pipeline.profile().device().sensors() {
        if sensor.supports_option(Rs2Option::GlobalTimeEnabled) {
            sensor.set_option(Rs2Option::GlobalTimeEnabled, 0.0).map_err(|e| eyre!("turning off global time: {e}"))?;
        }
    }
    Ok(())
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

fn stream_motion(
    mut motion: MotionPipeline,
    calibration: &ImuCalibration,
    samples: &Mutex<MotionSamples>,
    stopping: &AtomicBool,
) -> Result<()> {
    let (mut last_accel, mut last_gyro) = (None, None);
    while !stopping.load(Ordering::Relaxed) {
        let frames = motion.0.wait(Some(FRAME_WAIT)).map_err(|e| eyre!("waiting for motion samples: {e}"))?;
        let mut shared = samples.lock().expect("motion samples lock");
        for accel in frames.frames_of_type::<AccelFrame>() {
            if is_new(&accel, &mut last_accel) {
                shared.accel.push((camera_ns(&accel), calibration.accel.apply(Vec3::from_array(*accel.acceleration()))));
            }
        }
        for gyro in frames.frames_of_type::<GyroFrame>() {
            if is_new(&gyro, &mut last_gyro) {
                shared.gyro.push((camera_ns(&gyro), calibration.gyro.apply(Vec3::from_array(*gyro.rotational_velocity()))));
            }
        }
    }
    Ok(())
}

fn stream_profile<'a>(pipeline: &'a ActivePipeline, kind: Rs2StreamKind) -> Result<&'a StreamProfile> {
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

fn image_parameters(encoding: &str, width: usize, height: usize, frame_id: u64, frame_camera_ns: i64) -> MetadataParameters {
    let mut parameters = MetadataParameters::default();
    parameters.insert("encoding".to_owned(), Parameter::String(encoding.to_owned()));
    parameters.insert("width".to_owned(), Parameter::Integer(width as i64));
    parameters.insert("height".to_owned(), Parameter::Integer(height as i64));
    parameters.insert("frame_id".to_owned(), Parameter::Integer(frame_id as i64));
    parameters.insert("camera_ns".to_owned(), Parameter::Integer(frame_camera_ns));
    parameters
}

fn capture(
    mut node: DoraNode,
    imu_calibration: ImuCalibration,
    stopping: Arc<AtomicBool>,
    color_requested: Arc<AtomicBool>,
) -> Result<()> {
    let wanted = Wanted::from_node(&node)?;
    let context = RealsenseContext::new().wrap_err("creating the librealsense context")?;
    eyre::ensure!(!context.query_devices(HashSet::new()).is_empty(), "no RealSense camera found");
    let mut motion_config = Config::new();
    motion_config.enable_stream(Rs2StreamKind::Accel, None, 0, 0, Rs2Format::MotionXyz32F, ACCEL_HZ)?;
    motion_config.enable_stream(Rs2StreamKind::Gyro, None, 0, 0, Rs2Format::MotionXyz32F, GYRO_HZ)?;
    let motion = start_pipeline(&context, motion_config, "motion")?;
    let mut video_config = Config::new();
    video_config.enable_stream(Rs2StreamKind::Color, None, COLOR_SIZE.0, COLOR_SIZE.1, Rs2Format::Rgb8, VIDEO_FPS)?;
    video_config.enable_stream(Rs2StreamKind::Depth, None, DEPTH_SIZE.0, DEPTH_SIZE.1, Rs2Format::Z16, VIDEO_FPS)?;
    let video = start_pipeline(&context, video_config, "colour and depth")?;
    let calibration = read_calibration(&video)?;
    let motion_samples = Mutex::new(MotionSamples::default());
    let motion = MotionPipeline(motion);
    std::thread::scope(|scope| {
        let motion = scope.spawn(|| stream_motion(motion, &imu_calibration, &motion_samples, &stopping));
        let sent = stream_video(&mut node, video, &wanted, &calibration, &motion_samples, &stopping, &color_requested, || {
            motion.is_finished()
        });
        stopping.store(true, Ordering::Relaxed);
        let motion = motion.join().map_err(|_| eyre!("the motion thread panicked"))?;
        sent.and(motion)
    })
}

#[allow(clippy::too_many_arguments)]
fn stream_video(
    node: &mut DoraNode,
    mut pipeline: ActivePipeline,
    wanted: &Wanted,
    calibration: &CameraCalibration,
    motion_samples: &Mutex<MotionSamples>,
    stopping: &AtomicBool,
    color_requested: &AtomicBool,
    motion_finished: impl Fn() -> bool,
) -> Result<()> {
    let output = |name: &str| DataId::from(name.to_owned());
    let mut gravity = GravityEstimate::new(GRAVITY_TIME_CONSTANT);
    let mut last_accel_ns: Option<i64> = None;
    let mut last_depth: Option<u64> = None;
    let mut frame_id = 0u64;
    let mut pending = Imu::default();
    // The camera can still hold motion samples from an earlier session; only those after the first depth frame count.
    let mut session_start_ns: Option<i64> = None;
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
        let Some(depth) = frames.frames_of_type::<DepthFrame>().into_iter().next().filter(|d| is_new(d, &mut last_depth)) else {
            continue;
        };
        let Some(color) = frames.frames_of_type::<ColorFrame>().into_iter().next() else { continue };
        let frame_camera_ns = camera_ns(&depth);
        let start_ns = *session_start_ns.get_or_insert(frame_camera_ns);
        let delivered = std::mem::take(&mut *motion_samples.lock().expect("motion samples lock"));
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
        let imu = Imu { frame_id, keyframe_camera_ns: frame_camera_ns, ..std::mem::take(&mut pending) };
        node.send_output(output(IMU_OUTPUT), MetadataParameters::default(), encode_imu(&imu))?;
        if frame_id % CALIBRATION_EVERY_FRAMES == 0 {
            node.send_output(output(CALIBRATION_OUTPUT), MetadataParameters::default(), encode_camera_calibration(calibration))?;
        }

        let rgb: &[u8] = pixels(unsafe { color.get_data() }, color.get_data_size());
        if let Some(letterbox) = &wanted.model_input {
            let started = Instant::now();
            let mut parameters = image_parameters("rgb8", letterbox.side, letterbox.side, frame_id, frame_camera_ns);
            parameters.insert("source_width".to_owned(), Parameter::Integer(letterbox.source_width as i64));
            parameters.insert("source_height".to_owned(), Parameter::Integer(letterbox.source_height as i64));
            parameters.insert("shrink".to_owned(), Parameter::Integer(letterbox.factor as i64));
            parameters.insert("pad_x".to_owned(), Parameter::Integer(letterbox.pad_x as i64));
            parameters.insert("pad_y".to_owned(), Parameter::Integer(letterbox.pad_y as i64));
            let mut letterbox_ms = 0.0;
            node.send_output_raw(output(MODEL_INPUT_OUTPUT), parameters, letterbox.output_len(), |out| {
                letterbox.fill(rgb, out);
                letterbox_ms = started.elapsed().as_secs_f64() * 1e3;
            })?;
            if frame_id % CALIBRATION_EVERY_FRAMES == 0 {
                eprintln!("[realsense] frame {frame_id}: letterbox {letterbox_ms:.2} ms");
            }
        }
        if wanted.depth {
            let units_mm = depth.depth_units().map_err(|e| eyre!("reading depth units: {e}"))? * 1e3;
            let raw: &[u16] = pixels(unsafe { depth.get_data() }, depth.get_data_size());
            let depth_mm: UInt16Array = if (units_mm - 1.0).abs() > 1e-6 {
                raw.iter().map(|&d| (d as f32 * units_mm).round().min(u16::MAX as f32) as u16).collect()
            } else {
                UInt16Array::from(raw.to_vec())
            };
            let mut parameters = image_parameters("mono16", depth.width(), depth.height(), frame_id, frame_camera_ns);
            if let Some(reaction) = gravity.reaction() {
                parameters.insert("gravity_reaction".to_owned(), Parameter::ListFloat(reaction.to_array().map(f64::from).to_vec()));
            }
            node.send_output(output(DEPTH_OUTPUT), parameters, depth_mm)?;
        }
        if wanted.color && (!wanted.color_on_request || color_requested.swap(false, Ordering::Relaxed)) {
            let parameters = image_parameters("rgb8", color.width(), color.height(), frame_id, frame_camera_ns);
            node.send_output_raw(output(COLOR_OUTPUT), parameters, rgb.len(), |out| out.copy_from_slice(rgb))?;
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
    let color_requested = Arc::new(AtomicBool::new(false));
    let camera = {
        let (stopping, color_requested) = (stopping.clone(), color_requested.clone());
        std::thread::spawn(move || capture(node, imu_calibration, stopping, color_requested))
    };
    while !camera.is_finished() {
        match events.recv_timeout(EVENT_WAIT) {
            Some(Event::Input { id, .. }) if id.as_str() == REQUEST_COLOR_INPUT => color_requested.store(true, Ordering::Relaxed),
            Some(Event::Stop(_)) | None => break,
            _ => {}
        }
    }
    stopping.store(true, Ordering::Relaxed);
    camera.join().map_err(|_| eyre!("the camera thread panicked"))?
}
