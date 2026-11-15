//! The accelerometer and gyro of the device a video pipeline opened, streamed straight from its motion sensor into a
//! frame queue, so every sample arrives and the video pipeline's device is the only one librealsense holds.
//!
//! A second pipeline would resolve a second device object for the same camera. librealsense polls the clock of every
//! device object with a sensor open, through that object's depth sensor; the second object's depth sensor is not
//! streaming, so each poll tries to open and claim the USB interface the first object already holds. The RSUSB
//! backend of librealsense 2.58 leaks the file descriptor of every such failed claim, ten a second.

use std::ffi::CStr;
use std::ptr::NonNull;
use std::time::Duration;

use eyre::{bail, eyre, Result};
use realsense_rust::frame::{AccelFrame, CompositeFrame, GyroFrame};
use realsense_sys as sys;

/// One sample from the motion sensor.
pub enum MotionSample {
    Accel(AccelFrame),
    Gyro(GyroFrame),
}

/// One motion stream to open: its kind and rate.
#[derive(Debug, Clone, Copy)]
pub struct MotionStreamRate {
    stream: sys::rs2_stream,
    hz: usize,
}

impl MotionStreamRate {
    pub fn accel(hz: usize) -> MotionStreamRate {
        MotionStreamRate { stream: sys::rs2_stream_RS2_STREAM_ACCEL, hz }
    }

    pub fn gyro(hz: usize) -> MotionStreamRate {
        MotionStreamRate { stream: sys::rs2_stream_RS2_STREAM_GYRO, hz }
    }
}

/// Calls a librealsense function with an error slot of its own and turns an error it reports into one saying what
/// was being done.
///
/// # Safety
/// `function` is a librealsense call that is safe with the arguments it captures.
unsafe fn call<T>(doing: &str, function: impl FnOnce(*mut *mut sys::rs2_error) -> T) -> Result<T> {
    let mut error = std::ptr::null_mut();
    let value = function(&mut error);
    if error.is_null() {
        return Ok(value);
    }
    let message = CStr::from_ptr(sys::rs2_get_error_message(error)).to_string_lossy().into_owned();
    sys::rs2_free_error(error);
    bail!("{doing}: {message}")
}

fn non_null<T>(pointer: *mut T, what: &str) -> Result<NonNull<T>> {
    NonNull::new(pointer).ok_or_else(|| eyre!("librealsense returned no {what}"))
}

/// The device that delivered `frames`, which is the device its pipeline opened.
fn device_of(frames: CompositeFrame) -> Result<NonNull<sys::rs2_device>> {
    unsafe {
        let frameset = frames.get_owned_raw();
        let frame = call("taking a frame out of the first frameset", |e| sys::rs2_extract_frame(frameset.as_ptr(), 0, e));
        sys::rs2_release_frame(frameset.as_ptr());
        let frame = non_null(frame?, "frame in the first frameset")?;
        let sensor = call("finding the sensor of the first frame", |e| sys::rs2_get_frame_sensor(frame.as_ptr(), e));
        sys::rs2_release_frame(frame.as_ptr());
        let sensor = non_null(sensor?, "sensor of the first frame")?;
        let device = call("finding the device of the first frame", |e| sys::rs2_create_device_from_sensor(sensor.as_ptr(), e));
        sys::rs2_delete_sensor(sensor.as_ptr());
        non_null(device?, "device of the first frame")
    }
}

/// Whether `sensor` is a motion sensor.
unsafe fn is_motion_sensor(sensor: NonNull<sys::rs2_sensor>) -> Result<bool> {
    let extendable = call("asking whether a sensor is the motion sensor", |e| {
        sys::rs2_is_sensor_extendable_to(sensor.as_ptr(), sys::rs2_extension_RS2_EXTENSION_MOTION_SENSOR, e)
    })?;
    Ok(extendable != 0)
}

/// The device's motion sensor, among the sensors in `sensors`.
unsafe fn find_motion_sensor(sensors: NonNull<sys::rs2_sensor_list>) -> Result<NonNull<sys::rs2_sensor>> {
    let count = call("counting the camera's sensors", |e| sys::rs2_get_sensors_count(sensors.as_ptr(), e))?;
    for index in 0..count {
        let sensor = non_null(call("reading the camera's sensors", |e| sys::rs2_create_sensor(sensors.as_ptr(), index, e))?, "sensor")?;
        match is_motion_sensor(sensor) {
            Ok(true) => return Ok(sensor),
            Ok(false) => sys::rs2_delete_sensor(sensor.as_ptr()),
            Err(error) => {
                sys::rs2_delete_sensor(sensor.as_ptr());
                return Err(error);
            }
        }
    }
    bail!("the camera has no motion sensor")
}

unsafe fn motion_sensor_of(device: NonNull<sys::rs2_device>) -> Result<NonNull<sys::rs2_sensor>> {
    let sensors = non_null(call("listing the camera's sensors", |e| sys::rs2_query_sensors(device.as_ptr(), e))?, "sensor list")?;
    let found = find_motion_sensor(sensors);
    sys::rs2_delete_sensor_list(sensors.as_ptr());
    found
}

/// Whether `profile` is the stream `rate` names, as calibrated XYZ floats.
unsafe fn is_profile_for(profile: *const sys::rs2_stream_profile, rate: MotionStreamRate) -> Result<bool> {
    let (mut stream, mut format, mut index, mut unique_id, mut framerate) = (0, 0, 0, 0, 0);
    call("reading a stream profile", |e| {
        sys::rs2_get_stream_profile_data(profile, &mut stream, &mut format, &mut index, &mut unique_id, &mut framerate, e)
    })?;
    Ok(stream == rate.stream && format == sys::rs2_format_RS2_FORMAT_MOTION_XYZ32F && framerate as usize == rate.hz)
}

/// Opens the streams `rates` names, among the profiles in `list`, on `sensor`.
unsafe fn open_profiles(
    sensor: NonNull<sys::rs2_sensor>,
    list: NonNull<sys::rs2_stream_profile_list>,
    rates: &[MotionStreamRate],
) -> Result<()> {
    let count = call("counting the motion sensor's streams", |e| sys::rs2_get_stream_profiles_count(list.as_ptr(), e))?;
    let mut chosen: Vec<*const sys::rs2_stream_profile> = Vec::with_capacity(rates.len());
    for &rate in rates {
        let mut profile = None;
        for index in 0..count {
            let candidate = call("reading the motion sensor's streams", |e| sys::rs2_get_stream_profile(list.as_ptr(), index, e))?;
            if is_profile_for(candidate, rate)? {
                profile = Some(candidate);
                break;
            }
        }
        chosen.push(profile.ok_or_else(|| eyre!("the motion sensor has no stream {} at {} Hz", rate.stream, rate.hz))?);
    }
    call("opening the motion streams", |e| sys::rs2_open_multiple(sensor.as_ptr(), chosen.as_mut_ptr(), chosen.len() as i32, e))
}

/// The motion sensor of the video pipeline's device, streaming into a frame queue until dropped.
pub struct MotionStream {
    device: NonNull<sys::rs2_device>,
    sensor: NonNull<sys::rs2_sensor>,
    queue: NonNull<sys::rs2_frame_queue>,
    opened: bool,
    started: bool,
}

// SAFETY: librealsense sensors and frame queues are thread-safe C objects; the wrapper only lacks Send because it
// holds raw pointers, and it is used by one thread at a time.
unsafe impl Send for MotionStream {}

impl MotionStream {
    /// Starts the streams `rates` names on the motion sensor of the device that delivered `frames`. The queue keeps
    /// up to `queue_capacity` samples that `next_sample` has not yet taken.
    pub fn start(frames: CompositeFrame, rates: &[MotionStreamRate], queue_capacity: usize) -> Result<MotionStream> {
        let device = device_of(frames)?;
        unsafe {
            let sensor = motion_sensor_of(device).inspect_err(|_| sys::rs2_delete_device(device.as_ptr()))?;
            let queue = call("creating the motion frame queue", |e| sys::rs2_create_frame_queue(queue_capacity as i32, e))
                .and_then(|queue| non_null(queue, "motion frame queue"))
                .inspect_err(|_| {
                    sys::rs2_delete_sensor(sensor.as_ptr());
                    sys::rs2_delete_device(device.as_ptr());
                })?;
            let mut motion = MotionStream { device, sensor, queue, opened: false, started: false };
            let list = non_null(call("listing the motion sensor's streams", |e| sys::rs2_get_stream_profiles(sensor.as_ptr(), e))?, "motion stream list")?;
            let opened = open_profiles(sensor, list, rates);
            sys::rs2_delete_stream_profiles_list(list.as_ptr());
            opened?;
            motion.opened = true;
            call("starting the motion streams", |e| sys::rs2_start_queue(sensor.as_ptr(), queue.as_ptr(), e))?;
            motion.started = true;
            Ok(motion)
        }
    }

    /// The next sample, waiting at most `timeout` for it.
    pub fn next_sample(&mut self, timeout: Duration) -> Result<MotionSample> {
        unsafe {
            let wait_ms = timeout.as_millis() as u32;
            let frame = call("waiting for motion samples", |e| sys::rs2_wait_for_frame(self.queue.as_ptr(), wait_ms, e))?;
            let frame = non_null(frame, "motion sample")?;
            let stream = call("reading a motion sample's stream", |e| sys::rs2_get_frame_stream_profile(frame.as_ptr(), e))
                .and_then(|profile| {
                    let (mut stream, mut format, mut index, mut unique_id, mut framerate) = (0, 0, 0, 0, 0);
                    call("reading a motion sample's stream", |e| {
                        sys::rs2_get_stream_profile_data(profile, &mut stream, &mut format, &mut index, &mut unique_id, &mut framerate, e)
                    })?;
                    Ok(stream)
                })
                .inspect_err(|_| sys::rs2_release_frame(frame.as_ptr()))?;
            match stream {
                sys::rs2_stream_RS2_STREAM_ACCEL => AccelFrame::try_from(frame).map(MotionSample::Accel),
                sys::rs2_stream_RS2_STREAM_GYRO => GyroFrame::try_from(frame).map(MotionSample::Gyro),
                other => {
                    sys::rs2_release_frame(frame.as_ptr());
                    bail!("the motion sensor delivered a frame of stream {other}")
                }
            }
            .map_err(|e| eyre!("reading a motion sample: {e:#}"))
        }
    }
}

impl Drop for MotionStream {
    fn drop(&mut self) {
        unsafe {
            if self.started {
                if let Err(e) = call("stopping the motion streams", |e| sys::rs2_stop(self.sensor.as_ptr(), e)) {
                    eprintln!("[realsense] {e:#}");
                }
            }
            if self.opened {
                if let Err(e) = call("closing the motion streams", |e| sys::rs2_close(self.sensor.as_ptr(), e)) {
                    eprintln!("[realsense] {e:#}");
                }
            }
            sys::rs2_delete_frame_queue(self.queue.as_ptr());
            sys::rs2_delete_sensor(self.sensor.as_ptr());
            sys::rs2_delete_device(self.device.as_ptr());
        }
    }
}
