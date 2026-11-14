//! The room namer with `fake-claude` as its model.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{claude, watchdog, WorkDir};
use perception_wire::navigation::RoomId;
use tokio::sync::mpsc;
use world_agent::claude_code::ClaudeCodeClient;
use world_agent::config::{CompositionChange, Effort, ModelName, SystemPrompt};
use world_agent::identity::AgentRegistry;
use world_agent::process::{Launcher, ProcessProfile, ToolAccess};
use world_agent::room_names::{RoomContents, RoomNameSink, RoomNameSinkError, RoomNamer};
use world_agent::world::RoomName;

struct ChannelRoomNames(mpsc::UnboundedSender<(RoomId, RoomName)>);

impl RoomNameSink for ChannelRoomNames {
    fn name_room(&self, room: RoomId, name: RoomName) -> Result<(), RoomNameSinkError> {
        self.0.send((room, name)).map_err(|_| RoomNameSinkError("the test stopped listening".to_owned()))
    }
}

struct Harness {
    namer: Arc<RoomNamer<ClaudeCodeClient, ChannelRoomNames>>,
    names: mpsc::UnboundedReceiver<(RoomId, RoomName)>,
    _work_dir: WorkDir,
}

impl Harness {
    fn start(reply: &str) -> Self {
        let work_dir = WorkDir::new();
        std::fs::create_dir_all(&work_dir.0).unwrap();
        let profile = ProcessProfile {
            model: ModelName("namer".to_owned()),
            effort: Effort::Low,
            system_prompt: SystemPrompt(format!("You name rooms. [fake:reply={reply}]")),
            tools: ToolAccess::None,
        };
        let registry = Arc::new(AgentRegistry::default());
        let launcher = Launcher::new(claude(&work_dir.0), profile, registry.clone(), watchdog().startup);
        let client = Arc::new(ClaudeCodeClient::start(launcher, registry, watchdog()));
        let (sender, names) = mpsc::unbounded_channel();
        let namer = Arc::new(RoomNamer::new(client, Arc::new(ChannelRoomNames(sender)), CompositionChange::new(0.5).unwrap()));
        Self { namer, names, _work_dir: work_dir }
    }

    async fn changed(&mut self, rooms: Vec<RoomContents>) -> (Vec<RoomId>, Vec<(RoomId, RoomName)>) {
        let started = self.namer.rooms_changed(rooms);
        tokio::time::timeout(Duration::from_secs(20), self.namer.settle()).await.expect("naming settles");
        (started, std::iter::from_fn(|| self.names.try_recv().ok()).collect())
    }
}

fn room(id: u64, name: Option<&str>, labels: &[&str]) -> RoomContents {
    RoomContents {
        room: RoomId(id),
        name: name.map(|n| RoomName(n.to_owned())),
        labels: labels.iter().map(|l| (*l).to_owned()).collect(),
    }
}

fn living_room() -> RoomName {
    RoomName("living_room".to_owned())
}

#[tokio::test]
async fn unnamed_rooms_are_named_and_named_ones_kept_until_their_objects_change_a_lot() {
    let mut harness = Harness::start("living room");
    let (started, names) = harness
        .changed(vec![
            room(1, None, &["sofa", "armchair", "tv"]),
            room(2, Some("kitchen"), &["table", "chair", "chair", "fridge"]),
            room(3, None, &[]),
        ])
        .await;
    assert_eq!(started, [RoomId(1)]);
    assert_eq!(names, [(RoomId(1), living_room())]);

    let (started, _) = harness
        .changed(vec![
            room(1, Some("living_room"), &["sofa", "armchair", "tv", "lamp"]),
            room(2, Some("kitchen"), &["table", "chair", "chair", "fridge", "kettle"]),
        ])
        .await;
    assert_eq!(started, [], "small changes keep the names");

    let (started, names) = harness
        .changed(vec![
            room(1, Some("living_room"), &["sofa", "armchair", "tv"]),
            room(2, Some("kitchen"), &["sofa", "tv", "armchair", "lamp"]),
        ])
        .await;
    assert_eq!(started, [RoomId(2)]);
    assert_eq!(names, [(RoomId(2), living_room())]);
}

#[tokio::test]
async fn a_reply_that_is_not_a_name_names_nothing_and_is_tried_again() {
    let mut harness = Harness::start("I think this is a kitchen.");
    let (started, names) = harness.changed(vec![room(1, None, &["fridge"])]).await;
    assert_eq!((started, names), (vec![RoomId(1)], vec![]));
    let (started, _) = harness.changed(vec![room(1, None, &["fridge"])]).await;
    assert_eq!(started, [RoomId(1)]);
}
