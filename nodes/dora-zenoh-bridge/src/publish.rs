//! Where each published input goes and how: every key gets its own declared publisher. A key that must not lose a
//! message blocks the bridge until Zenoh takes it. A key named in ZENOH_LATEST_ONLY is sent from a thread of its
//! own, and a message still waiting there when a newer one comes is replaced by it; each replaced message is counted
//! and logged.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use eyre::{bail, eyre, Result};
use zenoh::qos::CongestionControl;
use zenoh::Wait;

/// How a key's messages are delivered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delivery {
    /// Every message, the bridge waiting while Zenoh's queue is full.
    Every,
    /// Only the newest message matters: one still waiting to go out is replaced by a newer one.
    LatestOnly,
}

/// The key each published input goes to and how.
pub fn publish_plan(publish: &[(String, String)], latest_only: &[String]) -> Result<HashMap<String, (String, Delivery)>> {
    let keys: HashSet<&str> = publish.iter().map(|(_, key)| key.as_str()).collect();
    if let Some(unknown) = latest_only.iter().find(|key| !keys.contains(key.as_str())) {
        bail!("ZENOH_LATEST_ONLY names {unknown}, which ZENOH_PUBLISH does not publish");
    }
    let mut plan = HashMap::new();
    for (input, key) in publish {
        let delivery = if latest_only.contains(key) { Delivery::LatestOnly } else { Delivery::Every };
        if plan.insert(input.clone(), (key.clone(), delivery)).is_some() {
            bail!("ZENOH_PUBLISH names input {input} twice");
        }
    }
    Ok(plan)
}

/// The newest message waiting to go out on a key, and how many were replaced before going out.
#[derive(Default)]
struct Slot {
    waiting: Option<Vec<u8>>,
    replaced: u64,
    closed: bool,
    failed: Option<String>,
}

/// A message slot one thread fills and another empties.
#[derive(Default)]
pub struct LatestSlot {
    slot: Mutex<Slot>,
    ready: Condvar,
}

impl LatestSlot {
    /// Puts a message in the slot; returns true when it replaced one still waiting.
    pub fn offer(&self, payload: Vec<u8>) -> bool {
        let mut slot = self.slot.lock().expect("the slot lock is never poisoned: its holders do not panic");
        let replaced = slot.waiting.replace(payload).is_some();
        slot.replaced += u64::from(replaced);
        self.ready.notify_one();
        replaced
    }

    /// The next message, waiting for one; None once the slot is closed.
    pub fn take(&self) -> Option<Vec<u8>> {
        let mut slot = self.slot.lock().expect("the slot lock is never poisoned: its holders do not panic");
        loop {
            if let Some(payload) = slot.waiting.take() {
                return Some(payload);
            }
            if slot.closed {
                return None;
            }
            slot = self.ready.wait(slot).expect("the slot lock is never poisoned: its holders do not panic");
        }
    }

    pub fn close(&self) {
        self.slot.lock().expect("the slot lock is never poisoned: its holders do not panic").closed = true;
        self.ready.notify_one();
    }

    /// How many messages were replaced since the last call.
    pub fn take_replaced(&self) -> u64 {
        std::mem::take(&mut self.slot.lock().expect("the slot lock is never poisoned: its holders do not panic").replaced)
    }

    fn fail(&self, error: String) {
        self.slot.lock().expect("the slot lock is never poisoned: its holders do not panic").failed = Some(error);
    }

    fn failure(&self) -> Option<String> {
        self.slot.lock().expect("the slot lock is never poisoned: its holders do not panic").failed.clone()
    }
}

/// How often the replaced messages of a latest-only key are logged, when there were any.
const LOG_REPLACED_EVERY: Duration = Duration::from_secs(10);

pub enum KeyPublisher {
    Every { key: String, publisher: zenoh::pubsub::Publisher<'static> },
    LatestOnly { key: String, slot: Arc<LatestSlot>, sender: Option<JoinHandle<()>>, logged_at: Instant },
}

impl KeyPublisher {
    pub fn declare(session: &zenoh::Session, key: &str, delivery: Delivery) -> Result<KeyPublisher> {
        let publisher = session
            .declare_publisher(key.to_owned())
            .congestion_control(CongestionControl::Block)
            .wait()
            .map_err(|e| eyre!("declaring the publisher of {key}: {e}"))?;
        Ok(match delivery {
            Delivery::Every => KeyPublisher::Every { key: key.to_owned(), publisher },
            Delivery::LatestOnly => {
                let slot = Arc::new(LatestSlot::default());
                let sending = slot.clone();
                let sender = std::thread::spawn(move || {
                    while let Some(payload) = sending.take() {
                        if let Err(error) = publisher.put(payload).wait() {
                            sending.fail(error.to_string());
                            return;
                        }
                    }
                });
                KeyPublisher::LatestOnly { key: key.to_owned(), slot, sender: Some(sender), logged_at: Instant::now() }
            }
        })
    }

    pub fn publish(&mut self, payload: Vec<u8>) -> Result<()> {
        match self {
            KeyPublisher::Every { key, publisher } => publisher.put(payload).wait().map_err(|e| eyre!("publishing to {key}: {e}")),
            KeyPublisher::LatestOnly { key, slot, logged_at, .. } => {
                if let Some(error) = slot.failure() {
                    bail!("publishing to {key}: {error}");
                }
                slot.offer(payload);
                if logged_at.elapsed() >= LOG_REPLACED_EVERY {
                    let replaced = slot.take_replaced();
                    if replaced > 0 {
                        eprintln!("[zenoh-bridge] {key}: {replaced} messages replaced by newer ones before going out in {:.0} s", logged_at.elapsed().as_secs_f64());
                    }
                    *logged_at = Instant::now();
                }
                Ok(())
            }
        }
    }

    /// Stops the key's sending thread once what is waiting has gone out.
    pub fn close(&mut self) -> Result<()> {
        if let KeyPublisher::LatestOnly { key, slot, sender, .. } = self {
            slot.close();
            if let Some(sender) = sender.take() {
                sender.join().map_err(|_| eyre!("the sending thread of {key} panicked"))?;
            }
            if let Some(error) = slot.failure() {
                bail!("publishing to {key}: {error}");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(entries: &[(&str, &str)]) -> Vec<(String, String)> {
        entries.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
    }

    #[test]
    fn keys_not_named_latest_only_deliver_every_message() {
        let plan = publish_plan(&pairs(&[("keyframe", "rig/keyframe"), ("posed_depth", "rig/posed_depth")]), &["rig/posed_depth".to_owned()]).unwrap();
        assert_eq!(plan["keyframe"], ("rig/keyframe".to_owned(), Delivery::Every));
        assert_eq!(plan["posed_depth"], ("rig/posed_depth".to_owned(), Delivery::LatestOnly));
    }

    #[test]
    fn a_latest_only_key_that_is_not_published_is_refused() {
        let error = publish_plan(&pairs(&[("keyframe", "rig/keyframe")]), &["rig/posed_dept".to_owned()]).unwrap_err();
        assert!(error.to_string().contains("rig/posed_dept"), "{error}");
    }

    #[test]
    fn a_newer_message_replaces_one_still_waiting_and_is_counted() {
        let slot = LatestSlot::default();
        assert!(!slot.offer(vec![1]));
        assert!(slot.offer(vec![2]));
        assert_eq!(slot.take(), Some(vec![2]));
        assert_eq!(slot.take_replaced(), 1);
        assert_eq!(slot.take_replaced(), 0);
        slot.close();
        assert_eq!(slot.take(), None);
    }
}
