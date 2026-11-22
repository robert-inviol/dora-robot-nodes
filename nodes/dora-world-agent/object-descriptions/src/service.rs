//! Describing and embedding views in the background. Each view an object update carries is queued; a newer view of a
//! waiting object takes the waiting one's place, and a full queue refuses the view with an error. A few workers take
//! one object each: they read every label the object was reported under, skip a person, then embed the view and,
//! in parallel, describe it and embed the description, keeping each result in the world store as it arrives. A
//! failed call is tried again after each configured pause, then reported as failed; nothing is stored for it. An
//! object is never in two workers at once, so its results are kept in the order of its views.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use perception_wire::navigation::ObjectId;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use world_agent::config::ObjectDescriptionsConfig;
use world_model::{Size, WorldError};
use world_store::{
    DescribedView, EmbeddingSource, FrameId, IngestError, JpegImage, ModelName, ObjectChange, ObjectEmbedding, ObjectUpdate,
    WorldStore, WorldWriter,
};

use crate::describer::Describer;
use crate::embedder::{Embedder, EmbeddingInput};
use crate::facts::{ObjectFacts, PersonLabels};

/// Where descriptions and embeddings are kept, and the labels an object was reported under are read from.
pub trait DescriptionStore: Send + Sync + 'static {
    /// Every label the object was reported under, the first reported first.
    fn object_labels(&self, object: ObjectId) -> impl Future<Output = Result<Vec<String>, WorldError>> + Send;

    fn set_description(&self, object: ObjectId, described: DescribedView) -> Result<(), IngestError>;

    fn set_embedding(&self, object: ObjectId, embedding: ObjectEmbedding) -> Result<(), IngestError>;
}

/// The world store's readers and writer as a `DescriptionStore`.
#[derive(Clone)]
pub struct StoreAndWriter {
    pub store: WorldStore,
    pub writer: WorldWriter,
}

impl DescriptionStore for StoreAndWriter {
    async fn object_labels(&self, object: ObjectId) -> Result<Vec<String>, WorldError> {
        self.store.object_labels(object).await
    }

    fn set_description(&self, object: ObjectId, described: DescribedView) -> Result<(), IngestError> {
        self.writer.set_description(object, described)
    }

    fn set_embedding(&self, object: ObjectId, embedding: ObjectEmbedding) -> Result<(), IngestError> {
        self.writer.set_embedding(object, embedding)
    }
}

/// What the service does, from the settings file's `[object_descriptions]`.
#[derive(Clone, Debug)]
pub struct DescriptionSettings {
    pub retry_after: Vec<Duration>,
    pub queue_capacity: usize,
    pub concurrent_objects: usize,
    pub person_labels: PersonLabels,
}

impl DescriptionSettings {
    pub fn of(config: &ObjectDescriptionsConfig) -> Self {
        Self {
            retry_after: config.retry_after.clone(),
            queue_capacity: config.queue_capacity.get(),
            concurrent_objects: config.concurrent_objects.get(),
            person_labels: PersonLabels::new(&config.person_labels),
        }
    }
}

/// What became of an offered update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Offered {
    Queued,
    /// It took the place of a view of the same object that was still waiting.
    ReplacedWaitingView,
    /// It carries no view, or does not show the object standing (it disappeared or merged).
    NoView,
    /// It is labelled as a person.
    Person,
    /// The queue was full; the view is not described.
    QueueFull,
}

/// One step of describing a view; each is tried, retried and reported on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Step {
    Describe,
    Embed(EmbeddingSource),
}

impl fmt::Display for Step {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Step::Describe => f.write_str("description"),
            Step::Embed(source) => write!(f, "{source} embedding"),
        }
    }
}

/// How many of each step succeeded, were tried again, or failed for good, and what was not described.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DescriptionStats {
    pub succeeded: HashMap<Step, u64>,
    pub retried: HashMap<Step, u64>,
    pub failed: HashMap<Step, u64>,
    pub people_skipped: u64,
    pub refused_queue_full: u64,
    /// Results the store would not take, and label lookups that failed.
    pub store_errors: u64,
}

impl DescriptionStats {
    pub fn succeeded(&self, step: Step) -> u64 {
        self.succeeded.get(&step).copied().unwrap_or(0)
    }

    pub fn retried(&self, step: Step) -> u64 {
        self.retried.get(&step).copied().unwrap_or(0)
    }

    pub fn failed(&self, step: Step) -> u64 {
        self.failed.get(&step).copied().unwrap_or(0)
    }
}

/// One view waiting to be described.
#[derive(Clone, Debug)]
struct Job {
    object: ObjectId,
    view: JpegImage,
    view_frame: FrameId,
    size: Size,
    label: String,
}

#[derive(Default)]
struct Queue {
    /// Objects with a view waiting, oldest first.
    waiting: VecDeque<ObjectId>,
    jobs: HashMap<ObjectId, Job>,
    in_flight: HashSet<ObjectId>,
}

impl Queue {
    /// The oldest waiting view of an object no worker holds.
    fn take(&mut self) -> Option<Job> {
        let at = self.waiting.iter().position(|object| !self.in_flight.contains(object))?;
        let object = self.waiting.remove(at).expect("the position is in the queue");
        self.in_flight.insert(object);
        Some(self.jobs.remove(&object).expect("every waiting object has a job"))
    }

    fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.in_flight.is_empty()
    }
}

struct Inner<D, E, S> {
    describer: D,
    embedder: Arc<E>,
    store: Arc<S>,
    settings: DescriptionSettings,
    queue: Mutex<Queue>,
    /// Woken whenever the queue changes.
    changed: Notify,
    stats: Mutex<DescriptionStats>,
}

/// The background describer of every object's views.
pub struct ObjectDescriptions<D, E, S> {
    inner: Arc<Inner<D, E, S>>,
    workers: TaskTracker,
    cancel: CancellationToken,
}

impl<D: Describer, E: Embedder, S: DescriptionStore> ObjectDescriptions<D, E, S> {
    /// Starts the workers.
    pub fn start(describer: D, embedder: Arc<E>, store: Arc<S>, settings: DescriptionSettings) -> Self {
        let workers_wanted = settings.concurrent_objects;
        let inner = Arc::new(Inner {
            describer,
            embedder,
            store,
            settings,
            queue: Mutex::default(),
            changed: Notify::new(),
            stats: Mutex::default(),
        });
        let (workers, cancel) = (TaskTracker::new(), CancellationToken::new());
        for _ in 0..workers_wanted {
            let (inner, cancel) = (inner.clone(), cancel.clone());
            workers.spawn(async move { inner.work(cancel).await });
        }
        workers.close();
        Self { inner, workers, cancel }
    }

    /// Queues the update's view, if it carries one of an object standing in the map.
    pub fn offer(&self, update: &ObjectUpdate) -> Offered {
        let Some(view) = &update.crop else { return Offered::NoView };
        if matches!(update.change, ObjectChange::Disappeared | ObjectChange::MergedInto(_)) {
            return Offered::NoView;
        }
        let object = update.object_id;
        if self.inner.settings.person_labels.any_in([update.label.as_str()]) {
            self.inner.stats.lock().expect("the stats lock is never poisoned").people_skipped += 1;
            tracing::info!(%object, label = update.label, "a person is not described");
            return Offered::Person;
        }
        let job = Job { object, view: view.clone(), view_frame: update.frame_id, size: update.size, label: update.label.clone() };
        let offered = {
            let mut queue = self.inner.queue.lock().expect("the queue lock is never poisoned");
            if queue.jobs.contains_key(&object) {
                queue.jobs.insert(object, job);
                Offered::ReplacedWaitingView
            } else if queue.waiting.len() >= self.inner.settings.queue_capacity {
                Offered::QueueFull
            } else {
                queue.waiting.push_back(object);
                queue.jobs.insert(object, job);
                Offered::Queued
            }
        };
        match offered {
            Offered::QueueFull => {
                self.inner.stats.lock().expect("the stats lock is never poisoned").refused_queue_full += 1;
                tracing::error!(%object, capacity = self.inner.settings.queue_capacity, "the description queue is full; this view is not described");
            }
            _ => self.inner.changed.notify_waiters(),
        }
        offered
    }

    pub fn stats(&self) -> DescriptionStats {
        self.inner.stats.lock().expect("the stats lock is never poisoned").clone()
    }

    /// Waits until no view is waiting or being described.
    pub async fn settle(&self) {
        loop {
            let changed = self.inner.changed.notified();
            if self.inner.queue.lock().expect("the queue lock is never poisoned").is_idle() {
                return;
            }
            changed.await;
        }
    }

    /// Stops the workers; views in progress or waiting are not described.
    pub async fn shutdown(&self) {
        self.cancel.cancel();
        self.workers.wait().await;
    }
}

impl<D: Describer, E: Embedder, S: DescriptionStore> Inner<D, E, S> {
    async fn work(&self, cancel: CancellationToken) {
        loop {
            let changed = self.changed.notified();
            let job = self.queue.lock().expect("the queue lock is never poisoned").take();
            match job {
                Some(job) => {
                    let object = job.object;
                    tokio::select! {
                        _ = self.describe(job) => {}
                        _ = cancel.cancelled() => return,
                    }
                    self.queue.lock().expect("the queue lock is never poisoned").in_flight.remove(&object);
                    self.changed.notify_waiters();
                }
                None => tokio::select! {
                    _ = changed => {}
                    _ = cancel.cancelled() => return,
                },
            }
        }
    }

    fn count(&self, update: impl FnOnce(&mut DescriptionStats)) {
        update(&mut self.stats.lock().expect("the stats lock is never poisoned"));
    }

    /// The call's result, after as many tries as the retry pauses allow; None, reported, when every try failed.
    async fn retrying<T, Er: fmt::Display, F: Future<Output = Result<T, Er>>>(&self, object: ObjectId, step: Step, call: impl Fn() -> F) -> Option<T> {
        let mut pauses = self.settings.retry_after.iter();
        let mut attempt = 1;
        loop {
            match call().await {
                Ok(value) => {
                    self.count(|s| *s.succeeded.entry(step).or_default() += 1);
                    return Some(value);
                }
                Err(error) => match pauses.next() {
                    Some(pause) => {
                        tracing::warn!(%object, %step, attempt, %error, "failed; trying again in {pause:?}");
                        self.count(|s| *s.retried.entry(step).or_default() += 1);
                        tokio::time::sleep(*pause).await;
                        attempt += 1;
                    }
                    None => {
                        tracing::error!(%object, %step, attempts = attempt, %error, "failed for good; nothing is stored");
                        self.count(|s| *s.failed.entry(step).or_default() += 1);
                        return None;
                    }
                },
            }
        }
    }

    fn kept(&self, object: ObjectId, step: Step, result: Result<(), IngestError>) {
        if let Err(error) = result {
            tracing::error!(%object, %step, %error, "the world store did not take the result");
            self.count(|s| s.store_errors += 1);
        }
    }

    async fn describe(&self, job: Job) {
        let object = job.object;
        let mut labels = match self.store.object_labels(object).await {
            Ok(labels) => labels,
            Err(error) => {
                tracing::error!(%object, %error, "the object's labels could not be read; this view is not described");
                self.count(|s| s.store_errors += 1);
                return;
            }
        };
        if !labels.contains(&job.label) {
            labels.push(job.label.clone());
        }
        if self.settings.person_labels.any_in(labels.iter().map(String::as_str)) {
            tracing::info!(%object, ?labels, "a person is not described");
            self.count(|s| s.people_skipped += 1);
            return;
        }
        let facts = ObjectFacts { size: job.size, labels };
        tracing::info!(%object, frame = job.view_frame.0, facts = facts.line(), "describing a view");
        let embedding = |source: EmbeddingSource, vector| ObjectEmbedding {
            source,
            model: self.embedder.model().clone(),
            vector,
            view_frame: job.view_frame,
        };
        let view = async {
            let step = Step::Embed(EmbeddingSource::View);
            if let Some(vector) = self.retrying(object, step, || self.embedder.embed(EmbeddingInput::View(&job.view))).await {
                self.kept(object, step, self.store.set_embedding(object, embedding(EmbeddingSource::View, vector)));
            }
        };
        let description = async {
            let Some(caption) = self.retrying(object, Step::Describe, || self.describer.describe(object, &job.view, &facts)).await else {
                return;
            };
            tracing::info!(%object, name = caption.name, description = caption.description, "described");
            let model = ModelName(self.describer.model().0.clone());
            let described = DescribedView { caption: caption.clone(), view_frame: job.view_frame, model };
            self.kept(object, Step::Describe, self.store.set_description(object, described));
            let step = Step::Embed(EmbeddingSource::Description);
            if let Some(vector) = self.retrying(object, step, || self.embedder.embed(EmbeddingInput::Description(&caption))).await {
                self.kept(object, step, self.store.set_embedding(object, embedding(EmbeddingSource::Description, vector)));
            }
        };
        tokio::join!(view, description);
    }
}
