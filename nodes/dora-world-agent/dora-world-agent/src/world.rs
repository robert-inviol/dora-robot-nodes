//! The world the node answers from: the SQLite world store, fed from the node's inputs.

use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use dora_node_api::arrow::array::Array;
use eyre::{eyre, Result, WrapErr};
use perception_wire::localization::{decode_keyframe_pose, decode_map_to_odom, MapToOdom, Tracking};
use perception_wire::navigation::{decode_goal_status, GoalTarget, MapPose, ObjectId, RoomId, UnixNanos};
use perception_wire::objects::decode_object_update;
use perception_wire::people::decode_people_tracks;
use perception_wire::space_map::{decode_occupancy_grid, decode_topology};
use world_agent::config::WorldStoreSettings;
use world_agent::room_names::{RoomContents, RoomNameSink, RoomNameSinkError};
use world_agent::world::{
    Change, Clearance, EntityRef, FindQuery, FoundObject, Landmark, ObjectDescription, PngImage, ReadOnlySql, RoomName,
    Route, Observation, SqlRows, Timestamp, ViewRequest, WorldError, WorldModel, WorldSnapshot,
};
use world_store::{
    BackupConfig, OccupancyGrid, IssuedGoal, ObjectUpdate, OpenError, RobotPoseSample, StoreConfig, Topology, WorldWriter,
};

use object_descriptions::service::DescriptionStore;
use world_store::{DescribedView, IngestError, ObjectEmbedding};

use crate::people::{transform_people_to_map, NotTransformed};

/// The world store's inputs, by their dora input ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreInput {
    Topology,
    OccupancyGrid,
    ObjectUpdate,
    KeyframePose,
    GoalStatus,
    MapToOdom,
    PeopleTracks,
}

impl StoreInput {
    pub const ALL: [StoreInput; 7] = [
        StoreInput::Topology,
        StoreInput::OccupancyGrid,
        StoreInput::ObjectUpdate,
        StoreInput::KeyframePose,
        StoreInput::GoalStatus,
        StoreInput::MapToOdom,
        StoreInput::PeopleTracks,
    ];

    pub fn input_id(self) -> &'static str {
        match self {
            StoreInput::Topology => "topology",
            StoreInput::OccupancyGrid => "occupancy_grid",
            StoreInput::ObjectUpdate => "object_update",
            StoreInput::KeyframePose => "keyframe_pose",
            StoreInput::GoalStatus => "goal_status",
            StoreInput::MapToOdom => "map_to_odom",
            StoreInput::PeopleTracks => "people_tracks",
        }
    }

    pub fn from_input_id(id: &str) -> Option<StoreInput> {
        StoreInput::ALL.into_iter().find(|input| input.input_id() == id)
    }
}

/// What an ingested message changed that the node has to act on.
#[derive(Clone, Debug, PartialEq)]
pub enum Ingested {
    Recorded,
    /// The rooms changed; the room namer has to see them.
    RoomsChanged,
    /// An object update was recorded; its view, if it carries one, goes to be described.
    ObjectUpdated(Box<ObjectUpdate>),
}

/// The world model the node answers from, which also keeps room names and object descriptions, takes the store's
/// inputs and records the goals the node sends.
pub trait WorldStore: WorldModel + RoomNameSink + DescriptionStore {
    /// Decodes one input message and hands it to the store.
    fn ingest(&self, input: StoreInput, data: &dyn Array) -> Result<Ingested>;

    fn record_goal(&self, goal: IssuedGoal) -> Result<()>;

    /// Every room with its name, if it has been given one, and the labels of the objects in it, once everything
    /// ingested so far is in.
    fn rooms(&self) -> impl std::future::Future<Output = Result<Vec<RoomContents>>> + Send;

    /// Commits what is pending and stops taking input.
    fn close(&self) -> Result<()>;
}

/// The SQLite world store and its writer, as one world.
#[derive(Clone)]
pub struct SqliteWorld {
    store: world_store::WorldStore,
    writer: WorldWriter,
    /// The localizer's tracking state at the last keyframe pose, to log when it changes.
    tracking: Arc<Mutex<Option<Tracking>>>,
    /// The localizer's latest map to odom correction, which places odometry-frame people tracks.
    correction: Arc<Mutex<Option<MapToOdom>>>,
    /// Why the last people tracks frame was not transformed, to log when it changes.
    not_transformed: Arc<Mutex<Option<NotTransformed>>>,
}

impl SqliteWorld {
    pub fn open(settings: &WorldStoreSettings) -> Result<Self, OpenError> {
        let mut config = StoreConfig::at(&settings.path);
        config.backup = settings.backup.as_ref().map(|backup| BackupConfig { path: backup.path.clone(), every: backup.every });
        Self::open_with(config)
    }

    pub fn open_with(config: StoreConfig) -> Result<Self, OpenError> {
        let (store, writer) = world_store::WorldStore::open(config)?;
        Ok(Self { store, writer, tracking: Arc::default(), correction: Arc::default(), not_transformed: Arc::default() })
    }

    /// Runs SQL on the store's read-only connection, for checking what was written.
    pub async fn query_text(&self, sql: &str, limit: NonZeroUsize) -> Result<SqlRows, WorldError> {
        self.store.query_text(sql.to_owned(), limit).await
    }

    /// A keyframe pose gives the robot pose only while the localizer is registered against the map. A lost or
    /// restarted pose is dead reckoning, and the store never revises a robot pose when pose graph updates move a
    /// restarted keyframe, so the robot pose goes stale in the digest instead.
    fn record_keyframe_pose(&self, data: &dyn Array) -> Result<()> {
        let keyframe = decode_keyframe_pose(data).wrap_err("decoding a keyframe pose")?;
        let previous = self.tracking.lock().expect("the tracking lock is never poisoned").replace(keyframe.tracking);
        if previous != Some(keyframe.tracking) {
            tracing::warn!(tracking = keyframe.tracking.wire_name(), frame = keyframe.frame_id.0, "the localizer's tracking changed");
        }
        match keyframe.tracking {
            Tracking::Tracking | Tracking::Relocalized => {}
            Tracking::Restarted | Tracking::Lost => return Ok(()),
        }
        let pose = MapPose { x_m: keyframe.pose.x_m, y_m: keyframe.pose.y_m, yaw_rad: keyframe.pose.yaw_rad };
        let sample = RobotPoseSample { captured: keyframe.captured, map_version: keyframe.map_version, pose };
        Ok(self.writer.record_robot_pose(sample)?)
    }

    /// Transforms a people tracks frame's confirmed tracks into the map frame and records them. A frame that cannot be transformed is
    /// logged at warn when the reason first appears, and records nothing, so people tracking goes stale in the
    /// digest.
    fn record_people(&self, data: &dyn Array) -> Result<()> {
        let frame = decode_people_tracks(data).wrap_err("decoding people tracks")?;
        let correction = *self.correction.lock().expect("the correction lock is never poisoned");
        let in_map = transform_people_to_map(&frame, correction.as_ref(), world_agent::clock::now())?;
        let reason = in_map.as_ref().err().copied();
        let previous = std::mem::replace(&mut *self.not_transformed.lock().expect("the not_transformed lock is never poisoned"), reason);
        match (previous, reason) {
            (_, Some(reason)) if previous != Some(reason) => tracing::warn!(%reason, frame = frame.frame_id, "people tracks not transformed into the map frame"),
            (Some(_), None) => tracing::info!(frame = frame.frame_id, "people tracks are transformed into the map frame again"),
            _ => {}
        }
        match in_map {
            Ok(observations) => Ok(self.writer.record_people(observations)?),
            Err(_) => Ok(()),
        }
    }
}

impl WorldStore for SqliteWorld {
    fn ingest(&self, input: StoreInput, data: &dyn Array) -> Result<Ingested> {
        match input {
            StoreInput::Topology => {
                let topology = decode_topology(data).wrap_err("decoding a topology")?;
                self.writer.set_topology(Topology::from(topology))?;
                return Ok(Ingested::RoomsChanged);
            }
            StoreInput::OccupancyGrid => {
                let occupancy_grid = decode_occupancy_grid(data).wrap_err("decoding a occupancy grid")?;
                self.writer.set_occupancy_grid(OccupancyGrid::try_from(occupancy_grid).wrap_err("a occupancy grid")?)?;
            }
            StoreInput::ObjectUpdate => {
                let update = ObjectUpdate::try_from(decode_object_update(data).wrap_err("decoding an object update")?).wrap_err("an object update")?;
                self.writer.apply_object_update(update.clone())?;
                return Ok(Ingested::ObjectUpdated(Box::new(update)));
            }
            StoreInput::KeyframePose => self.record_keyframe_pose(data)?,
            StoreInput::MapToOdom => {
                let correction = decode_map_to_odom(data).wrap_err("decoding a map to odom correction")?;
                *self.correction.lock().expect("the correction lock is never poisoned") = Some(correction);
            }
            StoreInput::PeopleTracks => self.record_people(data)?,
            StoreInput::GoalStatus => {
                let status = decode_goal_status(data).wrap_err("decoding a goal status")?;
                self.writer.record_goal_status(status)?;
            }
        }
        Ok(Ingested::Recorded)
    }

    fn record_goal(&self, goal: IssuedGoal) -> Result<()> {
        Ok(self.writer.record_goal(goal)?)
    }

    async fn rooms(&self) -> Result<Vec<RoomContents>> {
        let writer = self.writer.clone();
        tokio::task::spawn_blocking(move || writer.flush()).await.wrap_err("waiting for the world store's writer")??;
        let rooms = self.store.room_labels().await?;
        Ok(rooms.into_iter().map(|r| RoomContents { room: r.room, name: r.name, labels: r.labels }).collect())
    }

    fn close(&self) -> Result<()> {
        Ok(self.writer.close()?)
    }
}

impl DescriptionStore for SqliteWorld {
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

impl RoomNameSink for SqliteWorld {
    fn name_room(&self, room: RoomId, name: RoomName) -> Result<(), RoomNameSinkError> {
        self.writer.name_room(room, name).map_err(|error| RoomNameSinkError(error.to_string()))
    }
}

impl WorldModel for SqliteWorld {
    async fn snapshot(&self) -> Result<WorldSnapshot, WorldError> {
        self.store.snapshot().await
    }

    async fn stamp(&self) -> Result<Timestamp, WorldError> {
        self.store.stamp().await
    }

    async fn find(&self, query: FindQuery) -> Result<Vec<FoundObject>, WorldError> {
        self.store.find(query).await
    }

    async fn describe_object(&self, id: ObjectId) -> Result<ObjectDescription, WorldError> {
        self.store.describe_object(id).await
    }

    async fn clearance_between(&self, a: Landmark, b: Landmark) -> Result<Clearance, WorldError> {
        self.store.clearance_between(a, b).await
    }

    async fn route(&self, from: Landmark, to: Landmark) -> Result<Route, WorldError> {
        self.store.route(from, to).await
    }

    async fn last_seen(&self, subject: EntityRef) -> Result<Observation, WorldError> {
        self.store.last_seen(subject).await
    }

    async fn changes_since(&self, since: UnixNanos, room: Option<RoomName>) -> Result<Vec<Change>, WorldError> {
        self.store.changes_since(since, room).await
    }

    async fn render_view(&self, request: ViewRequest) -> Result<PngImage, WorldError> {
        self.store.render_view(request).await
    }

    async fn query_sql(&self, sql: ReadOnlySql, limit: NonZeroUsize) -> Result<SqlRows, WorldError> {
        self.store.query_sql(sql, limit).await
    }

    async fn check_target(&self, target: GoalTarget) -> Result<(), WorldError> {
        self.store.check_target(target).await
    }
}

/// The input a dora input id names, or an error for an input the node does not take.
pub fn store_input(id: &str) -> Result<StoreInput> {
    StoreInput::from_input_id(id).ok_or_else(|| eyre!("unexpected input {id}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_inputs_are_named_by_their_input_ids() {
        for input in StoreInput::ALL {
            assert_eq!(StoreInput::from_input_id(input.input_id()), Some(input));
        }
        assert_eq!(StoreInput::from_input_id("spatial_query"), None);
        assert!(store_input("keep_out").is_err());
    }
}
