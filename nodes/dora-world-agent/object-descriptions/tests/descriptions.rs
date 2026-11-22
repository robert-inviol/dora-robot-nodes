//! The background describer against a real world store, with fakes for Claude and Gemini.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use object_descriptions::describer::{DescribeError, Describer};
use object_descriptions::embedder::{EmbedError, Embedder, EmbeddingInput, QueriesOf};
use object_descriptions::facts::{ObjectFacts, PersonLabels};
use object_descriptions::service::{DescriptionSettings, ObjectDescriptions, Offered, Step, StoreAndWriter};
use perception_wire::navigation::{MapPose, MapVersion, ObjectId, UnixNanos};
use world_agent::config::ModelName;
use world_agent::search::QueryEmbedder;
use world_model::{Embedding, EmbeddingModel, FindQuery, FoundObject, ObjectCaption, Point, Size, WorldModel};
use world_store::{
    EmbeddingSource, Footprint, FrameId, JpegImage, ObjectChange, ObjectUpdate, Position3, RobotPoseSample, StoreConfig, WorldStore,
};

const MUG: ObjectId = ObjectId(51);
const CUP: ObjectId = ObjectId(52);
const NOW: UnixNanos = UnixNanos(1_794_610_000_000_000_000);

/// An 8x8 JPEG whose grey level is its tag, which tells the fakes which view it is.
fn view(tag: u8) -> JpegImage {
    let mut jpeg = Vec::new();
    jpeg_encoder::Encoder::new(&mut jpeg, 90).encode(&[tag * 60; 8 * 8 * 3], 8, 8, jpeg_encoder::ColorType::Rgb).unwrap();
    JpegImage(jpeg)
}

fn tag_of(jpeg: &JpegImage) -> u8 {
    [RED_MUG_VIEW, SHINY_MUG_VIEW, PLAIN_CUP_VIEW].into_iter().find(|&tag| &view(tag) == jpeg).expect("one of the test's views")
}

const RED_MUG_VIEW: u8 = 1;
const SHINY_MUG_VIEW: u8 = 2;
const PLAIN_CUP_VIEW: u8 = 3;

fn caption(name: &str, colour: &str) -> ObjectCaption {
    ObjectCaption {
        name: name.to_owned(),
        category: "kitchenware".to_owned(),
        description: format!("A {colour} {name}."),
        colour: colour.to_owned(),
        material: "ceramic".to_owned(),
    }
}

/// Describes each view by its tag; can be held shut so views pile up behind it.
struct FakeDescriber {
    model: ModelName,
    calls: Mutex<Vec<(ObjectId, u8, String)>>,
    gate: tokio::sync::Semaphore,
}

impl FakeDescriber {
    fn open() -> Self {
        Self { model: ModelName("fake-haiku".to_owned()), calls: Mutex::default(), gate: tokio::sync::Semaphore::new(Self::OPEN) }
    }

    const OPEN: usize = 1_000;

    fn shut() -> Self {
        Self { gate: tokio::sync::Semaphore::new(0), ..Self::open() }
    }

    fn calls(&self) -> Vec<(ObjectId, u8, String)> {
        self.calls.lock().unwrap().clone()
    }
}

impl Describer for FakeDescriber {
    fn model(&self) -> &ModelName {
        &self.model
    }

    async fn describe(&self, object: ObjectId, view: &JpegImage, facts: &ObjectFacts) -> Result<ObjectCaption, DescribeError> {
        let tag = tag_of(view);
        self.calls.lock().unwrap().push((object, tag, facts.line()));
        self.gate.acquire().await.expect("the gate stays open").forget();
        Ok(match tag {
            RED_MUG_VIEW => caption("mug", "red"),
            SHINY_MUG_VIEW => caption("mug", "shiny red"),
            PLAIN_CUP_VIEW => caption("cup", "white"),
            other => panic!("no view {other}"),
        })
    }
}

/// Three dimensions: x is red, y is a cup, z is shine. Views, descriptions and queries map onto them by name.
struct FakeEmbedder {
    model: EmbeddingModel,
    calls: Mutex<HashMap<&'static str, usize>>,
    view_failures_left: AtomicUsize,
}

impl FakeEmbedder {
    fn new() -> Self {
        Self { model: EmbeddingModel("fake-embedding".to_owned()), calls: Mutex::default(), view_failures_left: AtomicUsize::new(0) }
    }

    fn failing_views(times: usize) -> Self {
        Self { view_failures_left: AtomicUsize::new(times), ..Self::new() }
    }

    fn calls(&self, kind: &str) -> usize {
        self.calls.lock().unwrap().get(kind).copied().unwrap_or(0)
    }
}

fn vector(values: [f32; 3]) -> Embedding {
    Embedding::new(values.to_vec()).unwrap()
}

impl Embedder for FakeEmbedder {
    fn model(&self) -> &EmbeddingModel {
        &self.model
    }

    async fn embed(&self, input: EmbeddingInput<'_>) -> Result<Embedding, EmbedError> {
        *self.calls.lock().unwrap().entry(input.kind()).or_default() += 1;
        Ok(match input {
            EmbeddingInput::View(jpeg) => {
                let failing = self.view_failures_left.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| left.checked_sub(1)).is_ok();
                if failing {
                    return Err("the embedding API answered 503".into());
                }
                match tag_of(jpeg) {
                    RED_MUG_VIEW => vector([0.9, 0.1, 0.0]),
                    SHINY_MUG_VIEW => vector([0.7, 0.1, 0.7]),
                    _ => vector([0.0, 1.0, 0.0]),
                }
            }
            EmbeddingInput::Description(caption) => match caption.colour.as_str() {
                "red" => vector([1.0, 0.2, 0.0]),
                "shiny red" => vector([0.6, 0.2, 0.8]),
                _ => vector([0.0, 1.0, 0.1]),
            },
            EmbeddingInput::Query(text) => match text {
                "red cup" => vector([1.0, 0.0, 0.0]),
                "something shiny" => vector([0.0, 0.0, 1.0]),
                other => panic!("no query {other:?}"),
            },
        })
    }
}

struct Room {
    store: WorldStore,
    handles: Arc<StoreAndWriter>,
    _dir: tempfile::TempDir,
}

fn room() -> Room {
    let dir = tempfile::tempdir().unwrap();
    let mut config = StoreConfig::at(dir.path().join("world.sqlite"));
    config.batch_interval = Duration::from_millis(10);
    let (store, writer) = WorldStore::open(config).unwrap();
    let pose = MapPose { x_m: 0.0, y_m: 0.0, yaw_rad: 0.0 };
    writer.record_robot_pose(RobotPoseSample { captured: NOW, map_version: MapVersion(1), pose }).unwrap();
    Room { store: store.clone(), handles: Arc::new(StoreAndWriter { store, writer }), _dir: dir }
}

fn update(object: ObjectId, label: &str, change: ObjectChange, crop: Option<JpegImage>, frame: u64) -> ObjectUpdate {
    let at = Point { x_m: 1.0 + object.0 as f32 / 100.0, y_m: 1.0 };
    let half = 0.05;
    let corner = |dx: f32, dy: f32| Point { x_m: at.x_m + dx, y_m: at.y_m + dy };
    ObjectUpdate {
        change,
        object_id: object,
        label: label.to_owned(),
        score: 0.9,
        confidence: 1.0,
        seen_count: 3,
        frame_id: FrameId(frame),
        captured: UnixNanos(NOW.0 + frame as i64),
        map_version: MapVersion(1),
        centre: Position3 { x_m: at.x_m, y_m: at.y_m, z_m: 0.8 },
        footprint: Footprint([corner(-half, -half), corner(half, -half), corner(half, half), corner(-half, half)]),
        size: Size { long_m: 0.12, short_m: 0.08, height_m: 0.1 },
        top_m: 0.85,
        crop,
    }
}

fn settings(retry_after_ms: &[u64], queue_capacity: usize) -> DescriptionSettings {
    DescriptionSettings {
        retry_after: retry_after_ms.iter().map(|ms| Duration::from_millis(*ms)).collect(),
        queue_capacity,
        concurrent_objects: 2,
        person_labels: PersonLabels::new(&["person".to_owned(), "man".to_owned()]),
    }
}

type Service = ObjectDescriptions<Arc<FakeDescriber>, FakeEmbedder, StoreAndWriter>;

/// Hands the update to the store first and then to the service, as the node does.
fn arrive(room: &Room, service: &Service, update: ObjectUpdate) -> Offered {
    room.handles.writer.apply_object_update(update.clone()).unwrap();
    service.offer(&update)
}

async fn settled(room: &Room, service: &Service) {
    tokio::time::timeout(Duration::from_secs(10), service.settle()).await.expect("the service settles");
    room.handles.writer.flush().unwrap();
}

async fn find(room: &Room, embedder: &Arc<FakeEmbedder>, text: &str) -> Vec<FoundObject> {
    let meaning = QueriesOf(embedder.clone()).embed_query(text).await.unwrap();
    room.store.find(FindQuery { text: Some(text.to_owned()), meaning: Some(meaning), ..FindQuery::default() }).await.unwrap()
}

#[tokio::test]
async fn an_appeared_object_is_described_and_embedded_once() {
    let room = room();
    let (describer, embedder) = (Arc::new(FakeDescriber::open()), Arc::new(FakeEmbedder::new()));
    let service = ObjectDescriptions::start(describer.clone(), embedder.clone(), room.handles.clone(), settings(&[], 8));

    assert_eq!(arrive(&room, &service, update(MUG, "cup", ObjectChange::Appeared, Some(view(RED_MUG_VIEW)), 1)), Offered::Queued);
    assert_eq!(arrive(&room, &service, update(MUG, "mug", ObjectChange::Updated, None, 2)), Offered::NoView);
    settled(&room, &service).await;

    assert_eq!(describer.calls(), [(MUG, RED_MUG_VIEW, "About 12 x 8 x 10 cm (long x short x high). Segmented as: cup.".to_owned())]);
    assert_eq!((embedder.calls("view"), embedder.calls("description")), (1, 1));
    let described = room.store.describe_object(MUG).await.unwrap();
    assert_eq!(described.caption, Some(caption("mug", "red")));
    let found = find(&room, &embedder, "red cup").await;
    assert_eq!(found[0].id, MUG);
    assert!(found[0].matched.description.is_some() && found[0].matched.view.is_some(), "{found:?}");
    let stats = service.stats();
    assert_eq!((stats.succeeded(Step::Describe), stats.succeeded(Step::Embed(EmbeddingSource::View))), (1, 1));
}

#[tokio::test]
async fn find_ranks_the_described_red_mug_above_a_white_cup_that_only_shares_the_word() {
    let room = room();
    let (describer, embedder) = (Arc::new(FakeDescriber::open()), Arc::new(FakeEmbedder::new()));
    let service = ObjectDescriptions::start(describer, embedder.clone(), room.handles.clone(), settings(&[], 8));
    arrive(&room, &service, update(MUG, "mug", ObjectChange::Appeared, Some(view(RED_MUG_VIEW)), 1));
    arrive(&room, &service, update(CUP, "cup", ObjectChange::Appeared, Some(view(PLAIN_CUP_VIEW)), 1));
    settled(&room, &service).await;

    let found = find(&room, &embedder, "red cup").await;
    let ids: Vec<ObjectId> = found.iter().map(|f| f.id).collect();
    assert_eq!(ids, [MUG, CUP], "{found:#?}");
    assert!(found[0].score > 0.9, "{found:#?}");
    assert!(found[1].matched.words.get() > 0.0 && found[1].score < 0.1, "the cup only shares a word: {found:#?}");
}

#[tokio::test]
async fn a_clearly_better_view_replaces_the_description_and_both_vectors() {
    let room = room();
    let (describer, embedder) = (Arc::new(FakeDescriber::open()), Arc::new(FakeEmbedder::new()));
    let service = ObjectDescriptions::start(describer.clone(), embedder.clone(), room.handles.clone(), settings(&[], 8));
    arrive(&room, &service, update(MUG, "mug", ObjectChange::Appeared, Some(view(RED_MUG_VIEW)), 1));
    settled(&room, &service).await;
    let before = find(&room, &embedder, "something shiny").await;

    arrive(&room, &service, update(MUG, "mug", ObjectChange::Updated, Some(view(SHINY_MUG_VIEW)), 2));
    settled(&room, &service).await;

    assert_eq!(describer.calls().len(), 2);
    assert_eq!(room.store.describe_object(MUG).await.unwrap().caption, Some(caption("mug", "shiny red")));
    let after = find(&room, &embedder, "something shiny").await;
    assert!(after[0].matched.description.unwrap().get() > before[0].matched.description.unwrap().get() + 0.5, "{before:?} {after:?}");
    assert!(after[0].matched.view.unwrap().get() > before[0].matched.view.unwrap().get() + 0.5, "{before:?} {after:?}");
}

#[tokio::test]
async fn a_person_is_never_described_or_embedded() {
    let room = room();
    let (describer, embedder) = (Arc::new(FakeDescriber::open()), Arc::new(FakeEmbedder::new()));
    let service = ObjectDescriptions::start(describer.clone(), embedder.clone(), room.handles.clone(), settings(&[], 8));

    assert_eq!(arrive(&room, &service, update(MUG, " Person", ObjectChange::Appeared, Some(view(RED_MUG_VIEW)), 1)), Offered::Person);
    room.handles.writer.apply_object_update(update(CUP, "man", ObjectChange::Appeared, None, 1)).unwrap();
    room.handles.writer.flush().unwrap();
    assert_eq!(arrive(&room, &service, update(CUP, "coat", ObjectChange::Updated, Some(view(PLAIN_CUP_VIEW)), 2)), Offered::Queued);
    settled(&room, &service).await;

    assert!(describer.calls().is_empty(), "{:?}", describer.calls());
    assert_eq!((embedder.calls("view"), embedder.calls("description")), (0, 0));
    assert_eq!(service.stats().people_skipped, 2, "one by its label, one by a label it was reported under before");
    assert_eq!(room.store.describe_object(CUP).await.unwrap().caption, None);
}

#[tokio::test]
async fn a_failed_view_embedding_is_retried_then_reported_and_nothing_is_stored_for_it() {
    let room = room();
    let (describer, embedder) = (Arc::new(FakeDescriber::open()), Arc::new(FakeEmbedder::failing_views(usize::MAX)));
    let service = ObjectDescriptions::start(describer, embedder.clone(), room.handles.clone(), settings(&[1, 1], 8));
    arrive(&room, &service, update(MUG, "mug", ObjectChange::Appeared, Some(view(RED_MUG_VIEW)), 1));
    settled(&room, &service).await;

    let stats = service.stats();
    let view = Step::Embed(EmbeddingSource::View);
    assert_eq!((stats.retried(view), stats.failed(view), stats.succeeded(view)), (2, 1, 0));
    assert_eq!(embedder.calls("view"), 3);
    let found = find(&room, &embedder, "red cup").await;
    assert!(found[0].matched.view.is_none() && found[0].matched.description.is_some(), "the description is kept on its own: {found:?}");
    let rows = room.store.query_text("SELECT count(*) FROM object_embedding WHERE source = 'view'".to_owned(), std::num::NonZeroUsize::MIN).await.unwrap();
    assert_eq!(rows.rows[0][0], world_model::SqlValue::Integer(0));
}

#[tokio::test]
async fn a_view_embedding_that_fails_once_is_stored_on_the_next_try() {
    let room = room();
    let (describer, embedder) = (Arc::new(FakeDescriber::open()), Arc::new(FakeEmbedder::failing_views(1)));
    let service = ObjectDescriptions::start(describer, embedder.clone(), room.handles.clone(), settings(&[1, 1], 8));
    arrive(&room, &service, update(MUG, "mug", ObjectChange::Appeared, Some(view(RED_MUG_VIEW)), 1));
    settled(&room, &service).await;

    let view = Step::Embed(EmbeddingSource::View);
    let stats = service.stats();
    assert_eq!((stats.retried(view), stats.failed(view), stats.succeeded(view)), (1, 0, 1));
    assert!(find(&room, &embedder, "red cup").await[0].matched.view.is_some());
}

#[tokio::test]
async fn a_newer_view_takes_the_place_of_a_waiting_one_and_a_full_queue_refuses_more() {
    let room = room();
    let (describer, embedder) = (Arc::new(FakeDescriber::shut()), Arc::new(FakeEmbedder::new()));
    let mut one_at_a_time = settings(&[], 1);
    one_at_a_time.concurrent_objects = 1;
    let service = ObjectDescriptions::start(describer.clone(), embedder, room.handles.clone(), one_at_a_time);
    let third = ObjectId(53);

    arrive(&room, &service, update(MUG, "mug", ObjectChange::Appeared, Some(view(RED_MUG_VIEW)), 1));
    tokio::time::timeout(Duration::from_secs(5), async {
        while describer.calls().is_empty() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the mug is taken");
    assert_eq!(arrive(&room, &service, update(CUP, "cup", ObjectChange::Appeared, Some(view(RED_MUG_VIEW)), 1)), Offered::Queued);
    assert_eq!(arrive(&room, &service, update(CUP, "cup", ObjectChange::Updated, Some(view(PLAIN_CUP_VIEW)), 2)), Offered::ReplacedWaitingView);
    assert_eq!(arrive(&room, &service, update(third, "plate", ObjectChange::Appeared, Some(view(PLAIN_CUP_VIEW)), 1)), Offered::QueueFull);
    assert_eq!(service.stats().refused_queue_full, 1);

    describer.gate.add_permits(FakeDescriber::OPEN);
    settled(&room, &service).await;
    let tags: Vec<(ObjectId, u8)> = describer.calls().into_iter().map(|(object, tag, _)| (object, tag)).collect();
    assert_eq!(tags, [(MUG, RED_MUG_VIEW), (CUP, PLAIN_CUP_VIEW)], "the cup is described from its newest view only");
}

#[tokio::test]
async fn a_view_of_an_object_that_disappeared_is_not_offered() {
    let room = room();
    let service = ObjectDescriptions::start(Arc::new(FakeDescriber::open()), Arc::new(FakeEmbedder::new()), room.handles.clone(), settings(&[], 8));
    let gone = update(MUG, "mug", ObjectChange::Disappeared, Some(view(RED_MUG_VIEW)), 1);
    assert_eq!(service.offer(&gone), Offered::NoView);
    service.shutdown().await;
}
