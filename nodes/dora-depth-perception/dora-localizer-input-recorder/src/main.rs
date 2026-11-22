//! dora node that records the map localizer node's inputs (`keyframe`, `imu`, `odom_pose` and `calibration`, wired
//! as the localizer wires them, but with queues that drop nothing) into a recording directory, for the localizer's
//! `replay_inputs` example to run the localizer over offline.
//!
//! `RECORDINGS_DIRECTORY` names where recordings go: each is a new directory in it named by the Unix time in seconds
//! when the node started. It records until the dataflow stops.

use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dora_map_localizer::recording::{RecordedInput, RecordingWriter};
use dora_node_api::arrow::array::StructArray;
use dora_node_api::{DoraNode, Event};
use eyre::{eyre, Result, WrapErr};

const DIRECTORY_VARIABLE: &str = "RECORDINGS_DIRECTORY";
/// How often the node says how the recording goes, in wall-clock time.
const PROGRESS_EVERY: Duration = Duration::from_secs(10);

fn main() -> Result<()> {
    let recordings = PathBuf::from(std::env::var(DIRECTORY_VARIABLE).wrap_err_with(|| format!("{DIRECTORY_VARIABLE} names where recordings go"))?);
    let started_unix_s = SystemTime::now().duration_since(UNIX_EPOCH).wrap_err("the clock is before 1970")?.as_secs();
    let directory = recordings.join(started_unix_s.to_string());
    let mut recording = RecordingWriter::create(&directory)?;
    eprintln!("[localizer-recorder] recording into {}", directory.display());
    let (_node, mut events) = DoraNode::init_from_env()?;
    let (mut keyframes, mut logged_at) = (0u64, Instant::now());
    while let Some(event) = events.recv() {
        let Event::Input { id, data, .. } = event else {
            if let Event::Stop(_) = event {
                break;
            }
            continue;
        };
        let input = RecordedInput::named(id.as_str()).ok_or_else(|| eyre!("unknown input {id}"))?;
        let message = data.as_any().downcast_ref::<StructArray>().ok_or_else(|| eyre!("a {id} message is {:?}, not a struct", data.data_type()))?;
        recording.write(input, message)?;
        keyframes += (input == RecordedInput::Keyframe) as u64;
        if logged_at.elapsed() >= PROGRESS_EVERY {
            eprintln!("[localizer-recorder] {keyframes} keyframes, {} messages", recording.messages());
            logged_at = Instant::now();
        }
    }
    eprintln!("[localizer-recorder] stopped: {keyframes} keyframes, {} messages in {}", recording.messages(), directory.display());
    recording.finish()
}
