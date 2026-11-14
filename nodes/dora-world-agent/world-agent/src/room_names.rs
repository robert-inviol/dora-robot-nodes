//! Names rooms with a small model call from the objects seen in them. When the room list changes, a room with no
//! name, or one whose objects have changed by more than the configured share since it was named, is named again.
//! Each name goes to a `RoomNameSink`; the world store keeps it.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write;
use std::sync::{Arc, Mutex};

use perception_wire::navigation::RoomId;
use tokio_util::task::TaskTracker;

use crate::config::CompositionChange;
use crate::model::{Duty, ModelClient, Prompt};
use crate::world::RoomName;

/// Where room names go. `name_room` may block until the name is kept; the namer calls it on a blocking thread.
pub trait RoomNameSink: Send + Sync + 'static {
    fn name_room(&self, room: RoomId, name: RoomName) -> Result<(), RoomNameSinkError>;
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("the room name was not kept: {0}")]
pub struct RoomNameSinkError(pub String);

/// One room as the namer sees it: its name if it has one, and the label of every object seen in it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoomContents {
    pub room: RoomId,
    pub name: Option<RoomName>,
    pub labels: Vec<String>,
}

/// How many objects of each label a room holds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Composition(BTreeMap<String, u32>);

impl Composition {
    pub fn of(labels: &[String]) -> Self {
        let mut counts = BTreeMap::new();
        for label in labels {
            *counts.entry(label.clone()).or_insert(0) += 1;
        }
        Self(counts)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The share of objects not in common, from 0 (the same) to 1 (nothing in common).
    pub fn change_from(&self, earlier: &Composition) -> f32 {
        let labels: HashSet<&String> = self.0.keys().chain(earlier.0.keys()).collect();
        let (mut common, mut either) = (0_u32, 0_u32);
        for label in labels {
            let (now, then) = (self.0.get(label).copied().unwrap_or(0), earlier.0.get(label).copied().unwrap_or(0));
            common += now.min(then);
            either += now.max(then);
        }
        match either {
            0 => 0.0,
            _ => 1.0 - common as f32 / either as f32,
        }
    }

    fn describe(&self) -> String {
        let mut text = String::new();
        for (index, (label, count)) in self.0.iter().enumerate() {
            let separator = if index == 0 { "" } else { ", " };
            write!(text, "{separator}{count} {label}").expect("writing to a string");
        }
        text
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0:?} is not a room name: expected one to three lowercase words")]
pub struct NotARoomName(pub String);

/// A model's reply as a room name: one to three lowercase ASCII words, joined with underscores.
pub fn parse_room_name(reply: &str) -> Result<RoomName, NotARoomName> {
    let words: Vec<&str> = reply.split_whitespace().collect();
    let well_formed = (1..=3).contains(&words.len()) && words.iter().all(|w| w.bytes().all(|b| b.is_ascii_lowercase()));
    match well_formed {
        true => Ok(RoomName(words.join("_"))),
        false => Err(NotARoomName(reply.to_owned())),
    }
}

pub fn naming_prompt(composition: &Composition, other_names: &[RoomName]) -> Prompt {
    let others: Vec<&str> = other_names.iter().map(|n| n.0.as_str()).collect();
    let others = if others.is_empty() { "none".to_owned() } else { others.join(", ") };
    Prompt(format!("Objects in this room: {}.\nOther rooms: {others}.", composition.describe()))
}

#[derive(Default)]
struct NamerState {
    /// What each room held when it was named, or when it was first seen with a name.
    named_with: HashMap<RoomId, Composition>,
    in_flight: HashSet<RoomId>,
}

pub struct RoomNamer<C, S> {
    client: Arc<C>,
    sink: Arc<S>,
    rename_after: CompositionChange,
    state: Mutex<NamerState>,
    tasks: TaskTracker,
}

impl<C: ModelClient, S: RoomNameSink> RoomNamer<C, S> {
    pub fn new(client: Arc<C>, sink: Arc<S>, rename_after: CompositionChange) -> Self {
        Self { client, sink, rename_after, state: Mutex::default(), tasks: TaskTracker::new() }
    }

    /// Starts naming every room that needs a name and returns which. A room with no objects waits for some.
    pub fn rooms_changed(self: &Arc<Self>, rooms: Vec<RoomContents>) -> Vec<RoomId> {
        let mut state = self.state.lock().expect("namer lock");
        let mut started = Vec::new();
        for contents in &rooms {
            let composition = Composition::of(&contents.labels);
            if composition.is_empty() || state.in_flight.contains(&contents.room) {
                continue;
            }
            let needs_name = match (&contents.name, state.named_with.get(&contents.room)) {
                (None, _) => true,
                (Some(_), None) => {
                    state.named_with.insert(contents.room, composition.clone());
                    false
                }
                (Some(_), Some(earlier)) => composition.change_from(earlier) > self.rename_after.get(),
            };
            if !needs_name {
                continue;
            }
            let others: Vec<RoomName> =
                rooms.iter().filter(|r| r.room != contents.room).filter_map(|r| r.name.clone()).collect();
            state.in_flight.insert(contents.room);
            started.push(contents.room);
            let namer = self.clone();
            let room = contents.room;
            self.tasks.spawn(async move { namer.name(room, composition, others).await });
        }
        started
    }

    async fn name(&self, room: RoomId, composition: Composition, others: Vec<RoomName>) {
        let reply = self.client.ask(Duty::NameRoom { room }, naming_prompt(&composition, &others)).await;
        let named = match reply {
            Ok(reply) => match parse_room_name(&reply.text) {
                Ok(name) => {
                    tracing::info!(%room, %name, "room named");
                    let sink = self.sink.clone();
                    match tokio::task::spawn_blocking(move || sink.name_room(room, name)).await {
                        Ok(kept) => kept.map_err(|error| error.to_string()),
                        Err(error) => Err(format!("the room name sink panicked: {error}")),
                    }
                }
                Err(error) => Err(error.to_string()),
            },
            Err(error) => Err(error.to_string()),
        };
        let mut state = self.state.lock().expect("namer lock");
        state.in_flight.remove(&room);
        match named {
            Ok(()) => {
                state.named_with.insert(room, composition);
            }
            Err(error) => tracing::error!(%room, error, "the room was not named; it is tried again on the next change"),
        }
    }

    /// Waits for every naming in flight.
    pub async fn settle(&self) {
        self.tasks.close();
        self.tasks.wait().await;
        self.tasks.reopen();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| (*n).to_owned()).collect()
    }

    #[test]
    fn change_is_the_share_of_objects_not_in_common() {
        let before = Composition::of(&labels(&["sofa", "armchair", "tv", "lamp"]));
        assert_eq!(before.change_from(&before), 0.0);
        let after = Composition::of(&labels(&["sofa", "armchair", "tv", "lamp", "lamp"]));
        assert!((after.change_from(&before) - 0.2).abs() < 1e-6);
        let other = Composition::of(&labels(&["bed", "wardrobe"]));
        assert_eq!(other.change_from(&before), 1.0);
        assert_eq!(Composition::default().change_from(&Composition::default()), 0.0);
    }

    #[test]
    fn a_reply_of_one_to_three_lowercase_words_is_a_name() {
        assert_eq!(parse_room_name(" living room\n"), Ok(RoomName("living_room".to_owned())));
        assert_eq!(parse_room_name("kitchen"), Ok(RoomName("kitchen".to_owned())));
        for bad in ["", "The kitchen.", "a very large open hall", "Kitchen", "kitchen!"] {
            assert!(parse_room_name(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_prompt_counts_the_objects_and_names_the_other_rooms() {
        let prompt = naming_prompt(&Composition::of(&labels(&["chair", "table", "chair"])), &[RoomName("hallway".to_owned())]);
        assert_eq!(prompt.0, "Objects in this room: 2 chair, 1 table.\nOther rooms: hallway.");
        assert_eq!(naming_prompt(&Composition::of(&labels(&["bed"])), &[]).0, "Objects in this room: 1 bed.\nOther rooms: none.");
    }
}
