//! The store's clock, and when each live input last arrived.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use perception_wire::navigation::UnixNanos;

use crate::types::InputSource;

/// A clock tests can set.
#[derive(Clone, Debug)]
pub struct ManualClock(Arc<AtomicI64>);

impl ManualClock {
    pub fn starting_at(time: UnixNanos) -> Self {
        Self(Arc::new(AtomicI64::new(time.0)))
    }

    pub fn set(&self, time: UnixNanos) {
        self.0.store(time.0, Ordering::SeqCst);
    }

    pub fn advance(&self, span: Duration) {
        let nanos = i64::try_from(span.as_nanos()).expect("a span shorter than 292 years");
        self.0.fetch_add(nanos, Ordering::SeqCst);
    }
}

/// Where the store reads the time: the system clock, or a manual one in tests.
#[derive(Clone, Debug)]
pub enum Clock {
    System,
    Manual(ManualClock),
}

impl Clock {
    pub fn now(&self) -> UnixNanos {
        match self {
            Clock::System => world_model::clock::now(),
            Clock::Manual(clock) => UnixNanos(clock.0.load(Ordering::SeqCst)),
        }
    }
}

/// When each input last arrived at the store.
#[derive(Debug, Default)]
pub struct Arrivals(Mutex<BTreeMap<InputSource, UnixNanos>>);

impl Arrivals {
    pub fn note(&self, source: InputSource, at: UnixNanos) {
        self.0.lock().expect("the arrivals lock is never poisoned").insert(source, at);
    }

    pub fn last(&self, source: InputSource) -> Option<UnixNanos> {
        self.0.lock().expect("the arrivals lock is never poisoned").get(&source).copied()
    }
}
