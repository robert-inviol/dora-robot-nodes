//! What crossed the bridge on each key, logged every `LOG_TRAFFIC_EVERY`: messages, payload bytes after zstd, the
//! time spent encoding or decoding them, and, for messages stamped with a capture time on the camera's clock, how
//! long after that stamp they reached the bridge on this machine's clock. The two clocks are not the same, so that
//! spread is reported as its minimum in nanoseconds (the clock offset plus the fastest message's delay) and the
//! median and maximum above that minimum.

use std::collections::BTreeMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dora_node_api::arrow::array::{Array, AsArray};
use dora_node_api::arrow::datatypes::Int64Type;

/// How often each key's traffic is logged.
pub const LOG_TRAFFIC_EVERY: Duration = Duration::from_secs(10);

/// The fields that hold a message's capture time on the camera's clock, by message: `camera_ns` (odometry poses,
/// posed depth frames), `captured_camera_ns` (people tracks) and `keyframe_camera_ns` (imu messages).
const CAPTURE_TIME_FIELDS: [&str; 3] = ["camera_ns", "captured_camera_ns", "keyframe_camera_ns"];

/// The capture time on the camera's clock of a single-row struct message, when its message type carries one.
pub fn capture_camera_ns(message: &dyn Array) -> Option<i64> {
    let row = message.as_struct_opt()?;
    CAPTURE_TIME_FIELDS.iter().find_map(|field| {
        let column = row.column_by_name(field)?.as_primitive_opt::<Int64Type>()?;
        (!column.is_empty() && column.is_valid(0)).then(|| column.value(0))
    })
}

/// Now on this machine's clock, in nanoseconds since the Unix epoch.
pub fn unix_now_ns() -> i64 {
    let since_epoch = SystemTime::now().duration_since(UNIX_EPOCH).expect("the clock is after 1970");
    i64::try_from(since_epoch.as_nanos()).expect("nanoseconds since 1970 fit an i64 until 2262")
}

/// Which way a key's messages cross this bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Direction {
    Published,
    Received,
}

impl Direction {
    fn verb(self) -> &'static str {
        match self {
            Direction::Published => "published",
            Direction::Received => "received",
        }
    }

    fn work(self) -> &'static str {
        match self {
            Direction::Published => "encode",
            Direction::Received => "decode",
        }
    }
}

#[derive(Default)]
struct KeyTraffic {
    messages: u64,
    payload_bytes: u64,
    work: Duration,
    longest_work: Duration,
    arrival_after_capture_ns: Vec<i64>,
}

/// Each key's traffic since the last log.
pub struct Traffic {
    keys: BTreeMap<(Direction, String), KeyTraffic>,
    logged_at: Instant,
}

impl Traffic {
    pub fn new() -> Traffic {
        Traffic { keys: BTreeMap::new(), logged_at: Instant::now() }
    }

    /// One message on `key`: its payload size, the time encoding or decoding it took, and, when it carries a capture
    /// time, when it reached the bridge.
    pub fn record(&mut self, direction: Direction, key: &str, payload_bytes: usize, work: Duration, arrival_unix_ns: i64, capture_camera_ns: Option<i64>) {
        let traffic = self.keys.entry((direction, key.to_owned())).or_default();
        traffic.messages += 1;
        traffic.payload_bytes += payload_bytes as u64;
        traffic.work += work;
        traffic.longest_work = traffic.longest_work.max(work);
        if let Some(captured) = capture_camera_ns {
            traffic.arrival_after_capture_ns.push(arrival_unix_ns - captured);
        }
    }

    /// Logs every key's traffic and starts counting again, once `LOG_TRAFFIC_EVERY` has passed.
    pub fn log_when_due(&mut self) {
        let span = self.logged_at.elapsed();
        if span < LOG_TRAFFIC_EVERY {
            return;
        }
        for ((direction, key), traffic) in &mut self.keys {
            eprintln!("[zenoh-bridge] {}", traffic.summary(*direction, key, span));
            *traffic = KeyTraffic::default();
        }
        self.logged_at = Instant::now();
    }
}

impl KeyTraffic {
    fn summary(&mut self, direction: Direction, key: &str, span: Duration) -> String {
        let seconds = span.as_secs_f64();
        let mean_work_ms = if self.messages == 0 { 0.0 } else { self.work.as_secs_f64() * 1e3 / self.messages as f64 };
        let mut line = format!(
            "{key} {} {} messages in {seconds:.1} s ({:.1}/s), {} payload bytes ({:.1} kB/s, {:.0} B each), {} {mean_work_ms:.2} ms mean, {:.2} ms max",
            direction.verb(),
            self.messages,
            self.messages as f64 / seconds,
            self.payload_bytes,
            self.payload_bytes as f64 / seconds / 1e3,
            if self.messages == 0 { 0.0 } else { self.payload_bytes as f64 / self.messages as f64 },
            direction.work(),
            self.longest_work.as_secs_f64() * 1e3,
        );
        if !self.arrival_after_capture_ns.is_empty() {
            let spread = &mut self.arrival_after_capture_ns;
            spread.sort_unstable();
            let least = spread[0];
            let above_ms = |ns: i64| (ns - least) as f64 / 1e6;
            line += &format!(
                "; arrival - capture: min {least} ns, median +{:.2} ms, max +{:.2} ms",
                above_ms(spread[spread.len() / 2]),
                above_ms(spread[spread.len() - 1])
            );
        }
        line
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dora_node_api::arrow::array::{ArrayRef, Int64Array, StructArray, UInt64Array};
    use dora_node_api::arrow::datatypes::Field;

    use super::*;

    fn message(fields: &[(&str, ArrayRef)]) -> StructArray {
        StructArray::from(
            fields
                .iter()
                .map(|(name, column)| (Arc::new(Field::new(*name, column.data_type().clone(), true)), column.clone()))
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn the_capture_time_is_read_from_whichever_field_the_message_type_has() {
        let odom_pose = message(&[("camera_ns", Arc::new(Int64Array::from(vec![42])))]);
        let people = message(&[("captured_camera_ns", Arc::new(Int64Array::from(vec![43])))]);
        let imu = message(&[("keyframe_camera_ns", Arc::new(Int64Array::from(vec![44])))]);
        let calibration = message(&[("frame_id", Arc::new(UInt64Array::from(vec![1])))]);
        assert_eq!(capture_camera_ns(&odom_pose), Some(42));
        assert_eq!(capture_camera_ns(&people), Some(43));
        assert_eq!(capture_camera_ns(&imu), Some(44));
        assert_eq!(capture_camera_ns(&calibration), None);
        assert_eq!(capture_camera_ns(&Int64Array::from(vec![1])), None);
    }

    #[test]
    fn a_summary_gives_the_spread_above_the_fastest_message() {
        let mut traffic = KeyTraffic::default();
        for (bytes, after_ns) in [(100, 5_000_000), (300, 7_000_000), (200, 1_000_000)] {
            traffic.messages += 1;
            traffic.payload_bytes += bytes;
            traffic.arrival_after_capture_ns.push(after_ns);
        }
        let line = traffic.summary(Direction::Received, "rig/odom_pose", Duration::from_secs(10));
        assert!(line.starts_with("rig/odom_pose received 3 messages in 10.0 s (0.3/s), 600 payload bytes"), "{line}");
        assert!(line.ends_with("min 1000000 ns, median +4.00 ms, max +6.00 ms"), "{line}");
    }
}
