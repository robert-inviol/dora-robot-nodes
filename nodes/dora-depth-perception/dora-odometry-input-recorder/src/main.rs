//! dora node that records the odometry node's inputs (`depth`, `imu`, `calibration` and `person_poses`, wired as
//! the odometry node wires them) into a recording directory, for `replay-odometry-inputs` to run the odometry over
//! offline.
//!
//! `RECORDINGS_DIRECTORY` names where recordings go: each is a new directory in it named by the Unix time in seconds
//! when the node started. `RECORD_SECONDS` is how much camera time to
//! record, from the first depth frame; after it the files are closed and later inputs are dropped, until the
//! dataflow stops.

use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dora_depth_inertial_odometry::recording::{RecordedInput, RecordingWriter};
use dora_depth_inertial_odometry::DepthFrame;
use dora_node_api::arrow::array::{Array, StructArray};
use dora_node_api::{DoraNode, Event};
use eyre::{bail, eyre, Result, WrapErr};

const DIRECTORY_VARIABLE: &str = "RECORDINGS_DIRECTORY";
const SECONDS_VARIABLE: &str = "RECORD_SECONDS";
/// How often the node says how the recording goes, in wall-clock time.
const PROGRESS_EVERY: Duration = Duration::from_secs(10);

fn message(data: &dyn Array, input: RecordedInput) -> Result<&StructArray> {
    data.as_any().downcast_ref::<StructArray>().ok_or_else(|| eyre!("a {} message is {:?}, not a struct", input.input_name(), data.data_type()))
}

/// What has been recorded, for the log.
#[derive(Default)]
struct Recorded {
    depth_frames: u64,
    /// Depth frames the camera numbered that never arrived.
    depth_frames_missed: u64,
    last_frame_id: Option<u64>,
    bytes: u64,
}

fn main() -> Result<()> {
    let recordings = PathBuf::from(std::env::var(DIRECTORY_VARIABLE).wrap_err_with(|| format!("{DIRECTORY_VARIABLE} names where recordings go"))?);
    let started_unix_s = SystemTime::now().duration_since(UNIX_EPOCH).wrap_err("the clock is before 1970")?.as_secs();
    let directory = recordings.join(started_unix_s.to_string());
    let seconds: u64 = std::env::var(SECONDS_VARIABLE).wrap_err_with(|| format!("{SECONDS_VARIABLE} is how many seconds to record"))?.parse().wrap_err_with(|| format!("{SECONDS_VARIABLE} is a whole number of seconds"))?;
    let record_for = Duration::from_secs(seconds);
    let mut writer = Some(RecordingWriter::create(&directory)?);
    eprintln!("[odometry-recorder] recording {seconds} s of camera time into {}", directory.display());
    let (_node, mut events) = DoraNode::init_from_env()?;
    let (mut recorded, mut first_depth_ns, mut logged_at) = (Recorded::default(), None::<i64>, Instant::now());
    while let Some(event) = events.recv() {
        let Event::Input { id, metadata, data } = event else {
            if let Event::Stop(_) = event {
                break;
            }
            continue;
        };
        let Some(recording) = writer.as_mut() else { continue };
        match id.as_str() {
            "depth" => {
                let frame = DepthFrame::from_input(&metadata.parameters, data.as_ref()).wrap_err("reading a depth frame")?;
                if let Some(last) = recorded.last_frame_id.filter(|&last| frame.frame_id.0 <= last) {
                    bail!("depth frame {} came after frame {last}", frame.frame_id.0);
                }
                let first = *first_depth_ns.get_or_insert(frame.time.0);
                let since_first = u64::try_from(frame.time.0 - first).map_err(|_| eyre!("depth frame {} was taken before the first one recorded", frame.frame_id.0))?;
                if Duration::from_nanos(since_first) > record_for {
                    writer.take().expect("the recording is open").finish()?;
                    eprintln!(
                        "[odometry-recorder] done: {} depth frames ({} missed), {:.0} MB in {}; later inputs are dropped",
                        recorded.depth_frames,
                        recorded.depth_frames_missed,
                        recorded.bytes as f64 / 1e6,
                        directory.display()
                    );
                    continue;
                }
                recording.write_depth(&frame)?;
                recorded.depth_frames += 1;
                recorded.depth_frames_missed += recorded.last_frame_id.map_or(0, |last| frame.frame_id.0 - last - 1);
                recorded.last_frame_id = Some(frame.frame_id.0);
                recorded.bytes += 2 * frame.depth_mm.len() as u64;
            }
            "imu" => recording.write_message(RecordedInput::Imu, message(data.as_ref(), RecordedInput::Imu)?)?,
            "calibration" => recording.write_message(RecordedInput::Calibration, message(data.as_ref(), RecordedInput::Calibration)?)?,
            "person_poses" => recording.write_message(RecordedInput::PersonPoses, message(data.as_ref(), RecordedInput::PersonPoses)?)?,
            other => bail!("unknown input {other}"),
        }
        if logged_at.elapsed() >= PROGRESS_EVERY {
            eprintln!("[odometry-recorder] {} depth frames ({} missed), {:.0} MB of depth, {} messages", recorded.depth_frames, recorded.depth_frames_missed, recorded.bytes as f64 / 1e6, recording.messages());
            logged_at = Instant::now();
        }
    }
    if let Some(recording) = writer {
        recording.finish()?;
        eprintln!("[odometry-recorder] stopped: {} depth frames ({} missed) in {}", recorded.depth_frames, recorded.depth_frames_missed, directory.display());
    }
    Ok(())
}
