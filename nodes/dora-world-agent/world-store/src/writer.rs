//! The single writer: one connection on its own thread, taking ingest commands over a channel and committing them
//! in batches. Each command runs in a savepoint, so a rejected command leaves no trace; a database error stops the
//! writer, and every later ingest call reports it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use perception_wire::navigation::{GoalId, GoalStatus, GoalTarget, LookTarget, MapVersion, ObjectId, UnixNanos};
use rusqlite::{named_params, params, Connection, OptionalExtension};
use world_model::{Point, RoomName};

use crate::geometry::bounds;
use crate::inputs::{Arrivals, Clock};
use crate::map::{parse_unnamed_room, MapState, UNMAPPED_ROOM};
use crate::relations::{relation, Solid, NEXT_TO_M};
use crate::types::{
    Caption, FreeSpace, InputSource, IssuedGoal, JpegImage, ObjectChange, ObjectUpdate, RobotPoseSample, RoomId, Topology,
};

/// How many crops of each object the store keeps, best first.
const VIEWS_KEPT: i64 = 3;

/// Why the writer refused a command; nothing of it was written.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Rejection {
    #[error("no object {0} stands in the map")]
    UnknownObject(ObjectId),
    #[error("object {0} already stands in the map")]
    AlreadyPresent(ObjectId),
    #[error("object {0} cannot merge into itself")]
    MergedIntoItself(ObjectId),
    #[error("the crop of {0} is not a JPEG")]
    NotAJpeg(ObjectId),
    #[error("no goal {0} was recorded")]
    UnknownGoal(GoalId),
    #[error("goal {0} is already recorded")]
    GoalExists(GoalId),
    #[error("no {0} in the current topology")]
    UnknownRoom(RoomId),
    #[error("a room name cannot be empty")]
    EmptyName,
    #[error("{0} is reserved for unnamed rooms and the unmapped area")]
    ReservedName(RoomName),
    #[error("{name} already names {by}")]
    NameTaken { name: RoomName, by: RoomId },
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum IngestError {
    #[error("the world store's writer has stopped: {0}")]
    WriterStopped(String),
    #[error("the world store refused the update: {0}")]
    Rejected(#[from] Rejection),
}

/// How much the writer has done since the store opened.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WriterStats {
    pub commands: u64,
    pub batches: u64,
    pub rejected: u64,
}

enum Command {
    Object(Box<ObjectUpdate>),
    Caption(ObjectId, Caption, UnixNanos),
    Topology(Box<Topology>),
    FreeSpace(Box<FreeSpace>),
    RobotPose(RobotPoseSample),
    Goal(Box<IssuedGoal>),
    GoalStatus(GoalStatus),
    NameRoom(RoomId, RoomName, Sender<Result<(), IngestError>>),
    Flush(Sender<()>),
    Shutdown,
}

#[derive(Default)]
struct Counters {
    commands: AtomicU64,
    batches: AtomicU64,
    rejected: AtomicU64,
}

struct Shared {
    failure: Mutex<Option<String>>,
    counters: Counters,
    arrivals: Arc<Arrivals>,
    clock: Clock,
    thread: Mutex<Option<JoinHandle<()>>>,
}

/// The handle every ingest call goes through. Clones share the one writer thread.
#[derive(Clone)]
pub struct WorldWriter {
    commands: Sender<Command>,
    shared: Arc<Shared>,
}

impl WorldWriter {
    pub(crate) fn start(
        connection: Connection,
        map: Arc<RwLock<MapState>>,
        arrivals: Arc<Arrivals>,
        clock: Clock,
        batch_interval: Duration,
    ) -> std::io::Result<Self> {
        let (commands, receiver) = mpsc::channel();
        let shared = Arc::new(Shared {
            failure: Mutex::new(None),
            counters: Counters::default(),
            arrivals,
            clock,
            thread: Mutex::new(None),
        });
        let thread_shared = shared.clone();
        let thread = std::thread::Builder::new()
            .name("world-store-writer".to_owned())
            .spawn(move || run(connection, receiver, map, thread_shared, batch_interval))?;
        *shared.thread.lock().expect("the writer thread lock is never poisoned") = Some(thread);
        Ok(Self { commands, shared })
    }

    fn stopped(&self) -> IngestError {
        let failure = self.shared.failure.lock().expect("the failure lock is never poisoned").clone();
        IngestError::WriterStopped(failure.unwrap_or_else(|| "it was closed".to_owned()))
    }

    fn send(&self, command: Command) -> Result<(), IngestError> {
        self.commands.send(command).map_err(|_| self.stopped())
    }

    fn arrived(&self, source: InputSource) {
        self.shared.arrivals.note(source, self.shared.clock.now());
    }

    pub fn apply_object_update(&self, update: ObjectUpdate) -> Result<(), IngestError> {
        self.arrived(InputSource::Objects);
        self.send(Command::Object(Box::new(update)))
    }

    pub fn set_caption(&self, object: ObjectId, caption: Caption) -> Result<(), IngestError> {
        self.send(Command::Caption(object, caption, self.shared.clock.now()))
    }

    pub fn set_topology(&self, topology: Topology) -> Result<(), IngestError> {
        self.arrived(InputSource::Topology);
        self.send(Command::Topology(Box::new(topology)))
    }

    pub fn set_free_space(&self, free_space: FreeSpace) -> Result<(), IngestError> {
        self.arrived(InputSource::FreeSpace);
        self.send(Command::FreeSpace(Box::new(free_space)))
    }

    pub fn record_robot_pose(&self, sample: RobotPoseSample) -> Result<(), IngestError> {
        self.arrived(InputSource::RobotPose);
        self.send(Command::RobotPose(sample))
    }

    pub fn record_goal(&self, goal: IssuedGoal) -> Result<(), IngestError> {
        self.send(Command::Goal(Box::new(goal)))
    }

    pub fn record_goal_status(&self, status: GoalStatus) -> Result<(), IngestError> {
        self.send(Command::GoalStatus(status))
    }

    /// Names a room of the current topology, and returns once the name is committed. Blocks for up to one batch.
    pub fn name_room(&self, room: RoomId, name: RoomName) -> Result<(), IngestError> {
        let (reply, answer) = mpsc::channel();
        self.send(Command::NameRoom(room, name, reply))?;
        answer.recv().map_err(|_| self.stopped())?
    }

    /// Returns once everything sent before it is committed.
    pub fn flush(&self) -> Result<(), IngestError> {
        let (reply, done) = mpsc::channel();
        self.send(Command::Flush(reply))?;
        done.recv().map_err(|_| self.stopped())
    }

    pub fn stats(&self) -> WriterStats {
        let counters = &self.shared.counters;
        WriterStats {
            commands: counters.commands.load(Ordering::SeqCst),
            batches: counters.batches.load(Ordering::SeqCst),
            rejected: counters.rejected.load(Ordering::SeqCst),
        }
    }

    /// Commits what is pending, stops the writer thread and waits for it. Later ingest calls fail.
    pub fn close(&self) -> Result<(), IngestError> {
        let _ = self.commands.send(Command::Shutdown);
        let thread = self.shared.thread.lock().expect("the writer thread lock is never poisoned").take();
        if let Some(thread) = thread {
            thread.join().map_err(|_| IngestError::WriterStopped("the writer thread panicked".to_owned()))?;
        }
        match self.shared.failure.lock().expect("the failure lock is never poisoned").clone() {
            Some(failure) => Err(IngestError::WriterStopped(failure)),
            None => Ok(()),
        }
    }
}

enum BatchEnd {
    Continue,
    Shutdown,
}

enum Reply {
    Flushed(Sender<()>),
    Named(Sender<Result<(), IngestError>>, Result<(), Rejection>),
}

fn run(mut connection: Connection, commands: Receiver<Command>, map: Arc<RwLock<MapState>>, shared: Arc<Shared>, batch_interval: Duration) {
    while let Ok(first) = commands.recv() {
        match write_batch(&mut connection, first, &commands, &map, &shared, batch_interval) {
            Ok(BatchEnd::Continue) => {}
            Ok(BatchEnd::Shutdown) => return,
            Err(error) => {
                tracing::error!(%error, "the world store's writer failed and has stopped");
                *shared.failure.lock().expect("the failure lock is never poisoned") = Some(error.to_string());
                return;
            }
        }
    }
}

fn write_batch(
    connection: &mut Connection,
    first: Command,
    commands: &Receiver<Command>,
    map: &RwLock<MapState>,
    shared: &Shared,
    batch_interval: Duration,
) -> rusqlite::Result<BatchEnd> {
    let deadline = Instant::now() + batch_interval;
    let mut transaction = connection.transaction()?;
    let mut replies = Vec::new();
    let mut end = BatchEnd::Continue;
    let mut next = Some(first);
    let now = shared.clock.now();
    while let Some(command) = next.take() {
        let reply_now = match command {
            Command::Shutdown => {
                end = BatchEnd::Shutdown;
                break;
            }
            Command::Flush(reply) => {
                replies.push(Reply::Flushed(reply));
                true
            }
            Command::NameRoom(room, name, reply) => {
                let savepoint = transaction.savepoint()?;
                let outcome = name_room(&savepoint, map, room, &name, now)?;
                if outcome.is_ok() {
                    savepoint.commit()?;
                }
                replies.push(Reply::Named(reply, outcome));
                true
            }
            command => {
                let savepoint = transaction.savepoint()?;
                match apply(&savepoint, map, command, now)? {
                    Ok(()) => savepoint.commit()?,
                    Err(rejection) => {
                        shared.counters.rejected.fetch_add(1, Ordering::SeqCst);
                        tracing::error!(%rejection, "the world store refused an update");
                    }
                }
                false
            }
        };
        shared.counters.commands.fetch_add(1, Ordering::SeqCst);
        let remaining = deadline.saturating_duration_since(Instant::now());
        if reply_now || remaining.is_zero() {
            break;
        }
        match commands.recv_timeout(remaining) {
            Ok(command) => next = Some(command),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => end = BatchEnd::Shutdown,
        }
    }
    transaction.commit()?;
    shared.counters.batches.fetch_add(1, Ordering::SeqCst);
    for reply in replies {
        // A caller that stopped waiting has nothing to be told.
        match reply {
            Reply::Flushed(sender) => drop(sender.send(())),
            Reply::Named(sender, outcome) => drop(sender.send(outcome.map_err(IngestError::from))),
        }
    }
    Ok(end)
}

type Applied = rusqlite::Result<Result<(), Rejection>>;

fn apply(connection: &Connection, map: &RwLock<MapState>, command: Command, now: UnixNanos) -> Applied {
    match command {
        Command::Object(update) => apply_object(connection, map, &update, now),
        Command::Caption(object, caption, at) => set_caption(connection, object, &caption, at),
        Command::Topology(topology) => apply_topology(connection, map, *topology, now),
        Command::FreeSpace(free_space) => apply_free_space(connection, map, *free_space, now),
        Command::RobotPose(sample) => record_robot_pose(connection, map, sample, now),
        Command::Goal(goal) => record_goal(connection, &goal),
        Command::GoalStatus(status) => record_goal_status(connection, status),
        Command::NameRoom(..) | Command::Flush(_) | Command::Shutdown => unreachable!("write_batch handles these itself"),
    }
}

fn note_version(connection: &Connection, map: &RwLock<MapState>, version: MapVersion, now: UnixNanos) -> rusqlite::Result<()> {
    connection.execute("INSERT OR IGNORE INTO map_version VALUES (?1, ?2)", params![version.0 as i64, now.0])?;
    map.write().expect("the map lock is never poisoned").note_version(version);
    Ok(())
}

struct CurrentRow {
    history_id: i64,
    room: Option<i64>,
    last_seen: i64,
    label: String,
}

fn current_row(connection: &Connection, object: ObjectId) -> rusqlite::Result<Option<CurrentRow>> {
    connection
        .query_row(
            "SELECT history_id, room_id, last_seen_unix_ns, label FROM object WHERE object_id = ?1 AND valid_to_unix_ns IS NULL",
            [object.0 as i64],
            |row| Ok(CurrentRow { history_id: row.get(0)?, room: row.get(1)?, last_seen: row.get(2)?, label: row.get(3)? }),
        )
        .optional()
}

fn is_jpeg(image: &JpegImage) -> bool {
    image.0.starts_with(&[0xFF, 0xD8, 0xFF])
}

fn apply_object(connection: &Connection, map: &RwLock<MapState>, update: &ObjectUpdate, now: UnixNanos) -> Applied {
    let id = update.object_id;
    if update.crop.as_ref().is_some_and(|crop| !is_jpeg(crop)) {
        return Ok(Err(Rejection::NotAJpeg(id)));
    }
    let current = current_row(connection, id)?;
    let room = map.read().expect("the map lock is never poisoned").room_at(update.centre.on_floor()).map(|r| r.0 as i64);
    let captured = update.captured.0;
    match (update.change, &current) {
        (ObjectChange::Appeared, Some(_)) => return Ok(Err(Rejection::AlreadyPresent(id))),
        (ObjectChange::Appeared, None) => {}
        (_, None) => return Ok(Err(Rejection::UnknownObject(id))),
        (_, Some(_)) => {}
    }
    if let ObjectChange::MergedInto(into) = update.change {
        if into == id {
            return Ok(Err(Rejection::MergedIntoItself(id)));
        }
        if current_row(connection, into)?.is_none() {
            return Ok(Err(Rejection::UnknownObject(into)));
        }
    }
    note_version(connection, map, update.map_version, now)?;
    let event = |kind: &str, earliest: i64, room: Option<i64>, from_room: Option<i64>, other: Option<ObjectId>| {
        connection.execute(
            "INSERT INTO event (kind, detected_unix_ns, earliest_unix_ns, object_id, other_object_id, label, room_id, from_room_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![kind, captured, earliest, id.0 as i64, other.map(|o| o.0 as i64), update.label, room, from_room],
        )
    };
    match (update.change, current) {
        (ObjectChange::Appeared, _) => {
            insert_row(connection, update, room)?;
            event("appeared", captured, room, None, None)?;
        }
        (ObjectChange::Updated, Some(current)) => {
            update_row(connection, current.history_id, update, room)?;
        }
        (ObjectChange::Moved, Some(current)) => {
            close_row(connection, current.history_id, captured)?;
            insert_row(connection, update, room)?;
            event("moved", current.last_seen, room, current.room, None)?;
        }
        (ObjectChange::Disappeared, Some(current)) => {
            close_row(connection, current.history_id, captured)?;
            forget_relations(connection, id)?;
            connection.execute(
                "INSERT INTO event (kind, detected_unix_ns, earliest_unix_ns, object_id, label, room_id) VALUES ('disappeared', ?1, ?2, ?3, ?4, ?5)",
                params![captured, current.last_seen, id.0 as i64, current.label, current.room],
            )?;
            return Ok(Ok(()));
        }
        (ObjectChange::MergedInto(into), Some(current)) => {
            close_row(connection, current.history_id, captured)?;
            forget_relations(connection, id)?;
            event("merged", current.last_seen, current.room, None, Some(into))?;
            return Ok(Ok(()));
        }
        (_, None) => unreachable!("checked above"),
    }
    if let Some(crop) = &update.crop {
        connection.execute(
            "INSERT INTO object_view (object_id, frame_id, captured_unix_ns, score, jpeg) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id.0 as i64, update.frame_id.0 as i64, captured, update.score, crop.0],
        )?;
        connection.execute(
            "DELETE FROM object_view WHERE object_id = ?1 AND view_id NOT IN
             (SELECT view_id FROM object_view WHERE object_id = ?1 ORDER BY score DESC, view_id DESC LIMIT ?2)",
            params![id.0 as i64, VIEWS_KEPT],
        )?;
    }
    relate(connection, id, update.captured)?;
    Ok(Ok(()))
}

fn insert_row(connection: &Connection, update: &ObjectUpdate, room: Option<i64>) -> rusqlite::Result<()> {
    let [c0, c1, c2, c3] = update.footprint.0;
    connection.execute(
        "INSERT INTO object (object_id, label, room_id, centre_x_m, centre_y_m, centre_z_m,
             corner0_x_m, corner0_y_m, corner1_x_m, corner1_y_m, corner2_x_m, corner2_y_m, corner3_x_m, corner3_y_m,
             long_m, short_m, height_m, top_m, score, confidence, seen_count, frame_id, map_version,
             valid_from_unix_ns, valid_to_unix_ns, last_seen_unix_ns)
         VALUES (:object_id, :label, :room_id, :cx, :cy, :cz, :x0, :y0, :x1, :y1, :x2, :y2, :x3, :y3,
             :long_m, :short_m, :height_m, :top_m, :score, :confidence, :seen_count, :frame_id, :map_version,
             :captured, NULL, :captured)",
        named_params! {
            ":object_id": update.object_id.0 as i64, ":label": update.label, ":room_id": room,
            ":cx": update.centre.x_m, ":cy": update.centre.y_m, ":cz": update.centre.z_m,
            ":x0": c0.x_m, ":y0": c0.y_m, ":x1": c1.x_m, ":y1": c1.y_m, ":x2": c2.x_m, ":y2": c2.y_m, ":x3": c3.x_m, ":y3": c3.y_m,
            ":long_m": update.size.long_m, ":short_m": update.size.short_m, ":height_m": update.size.height_m,
            ":top_m": update.top_m, ":score": update.score, ":confidence": update.confidence,
            ":seen_count": update.seen_count, ":frame_id": update.frame_id.0 as i64,
            ":map_version": update.map_version.0 as i64, ":captured": update.captured.0,
        },
    )?;
    let history_id = connection.last_insert_rowid();
    write_extent(connection, history_id, update, false)
}

fn update_row(connection: &Connection, history_id: i64, update: &ObjectUpdate, room: Option<i64>) -> rusqlite::Result<()> {
    let [c0, c1, c2, c3] = update.footprint.0;
    connection.execute(
        "UPDATE object SET label = :label, room_id = :room_id, centre_x_m = :cx, centre_y_m = :cy, centre_z_m = :cz,
             corner0_x_m = :x0, corner0_y_m = :y0, corner1_x_m = :x1, corner1_y_m = :y1,
             corner2_x_m = :x2, corner2_y_m = :y2, corner3_x_m = :x3, corner3_y_m = :y3,
             long_m = :long_m, short_m = :short_m, height_m = :height_m, top_m = :top_m, score = :score,
             confidence = :confidence, seen_count = :seen_count, frame_id = :frame_id, map_version = :map_version,
             last_seen_unix_ns = max(last_seen_unix_ns, :captured)
         WHERE history_id = :history_id",
        named_params! {
            ":history_id": history_id, ":label": update.label, ":room_id": room,
            ":cx": update.centre.x_m, ":cy": update.centre.y_m, ":cz": update.centre.z_m,
            ":x0": c0.x_m, ":y0": c0.y_m, ":x1": c1.x_m, ":y1": c1.y_m, ":x2": c2.x_m, ":y2": c2.y_m, ":x3": c3.x_m, ":y3": c3.y_m,
            ":long_m": update.size.long_m, ":short_m": update.size.short_m, ":height_m": update.size.height_m,
            ":top_m": update.top_m, ":score": update.score, ":confidence": update.confidence,
            ":seen_count": update.seen_count, ":frame_id": update.frame_id.0 as i64,
            ":map_version": update.map_version.0 as i64, ":captured": update.captured.0,
        },
    )?;
    write_extent(connection, history_id, update, true)
}

fn write_extent(connection: &Connection, history_id: i64, update: &ObjectUpdate, replace: bool) -> rusqlite::Result<()> {
    let (lo, hi) = bounds(update.footprint.0).expect("a footprint has four corners");
    let verb = if replace { "INSERT OR REPLACE" } else { "INSERT" };
    connection.execute(
        &format!("{verb} INTO object_extent VALUES (?1, ?2, ?3, ?4, ?5)"),
        params![history_id, lo.x_m, hi.x_m, lo.y_m, hi.y_m],
    )?;
    Ok(())
}

fn close_row(connection: &Connection, history_id: i64, at: i64) -> rusqlite::Result<()> {
    connection.execute("UPDATE object SET valid_to_unix_ns = ?2 WHERE history_id = ?1", params![history_id, at])?;
    connection.execute("DELETE FROM object_extent WHERE history_id = ?1", [history_id])?;
    Ok(())
}

fn forget_relations(connection: &Connection, object: ObjectId) -> rusqlite::Result<()> {
    connection.execute("DELETE FROM object_relation WHERE object_id = ?1 OR other_object_id = ?1", [object.0 as i64])?;
    Ok(())
}

const SOLID_COLUMNS: &str = "o.object_id, o.centre_x_m, o.centre_y_m, o.corner0_x_m, o.corner0_y_m, o.corner1_x_m,
    o.corner1_y_m, o.corner2_x_m, o.corner2_y_m, o.corner3_x_m, o.corner3_y_m, o.top_m, o.height_m";

fn solid_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<(ObjectId, Solid)> {
    let p = |i: usize| -> rusqlite::Result<Point> { Ok(Point { x_m: row.get(i)?, y_m: row.get(i + 1)? }) };
    let (top_m, height_m): (f32, f32) = (row.get(11)?, row.get(12)?);
    Ok((
        ObjectId(row.get::<_, i64>(0)? as u64),
        Solid { centre: p(1)?, footprint: [p(3)?, p(5)?, p(7)?, p(9)?], bottom_m: top_m - height_m, top_m },
    ))
}

/// Recomputes the relations between an object and the objects around it.
fn relate(connection: &Connection, object: ObjectId, at: UnixNanos) -> rusqlite::Result<()> {
    forget_relations(connection, object)?;
    let (_, subject) = connection.query_row(
        &format!("SELECT {SOLID_COLUMNS} FROM object o WHERE o.object_id = ?1 AND o.valid_to_unix_ns IS NULL"),
        [object.0 as i64],
        solid_from_row,
    )?;
    let (lo, hi) = bounds(subject.footprint).expect("a footprint has four corners");
    let mut nearby = connection.prepare(&format!(
        "SELECT {SOLID_COLUMNS} FROM object_extent e JOIN object o ON o.history_id = e.history_id
         WHERE e.max_x_m >= ?1 AND e.min_x_m <= ?2 AND e.max_y_m >= ?3 AND e.min_y_m <= ?4 AND o.object_id != ?5"
    ))?;
    let others = nearby
        .query_map(
            params![lo.x_m - NEXT_TO_M, hi.x_m + NEXT_TO_M, lo.y_m - NEXT_TO_M, hi.y_m + NEXT_TO_M, object.0 as i64],
            solid_from_row,
        )?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut insert = connection.prepare("INSERT OR IGNORE INTO object_relation VALUES (?1, ?2, ?3, ?4)")?;
    for (other, solid) in others {
        if let Some(kind) = relation(&subject, &solid) {
            insert.execute(params![object.0 as i64, kind.as_str(), other.0 as i64, at.0])?;
        }
        if let Some(kind) = relation(&solid, &subject) {
            insert.execute(params![other.0 as i64, kind.as_str(), object.0 as i64, at.0])?;
        }
    }
    Ok(())
}

fn set_caption(connection: &Connection, object: ObjectId, caption: &Caption, at: UnixNanos) -> Applied {
    let known: bool = connection.query_row("SELECT EXISTS (SELECT 1 FROM object WHERE object_id = ?1)", [object.0 as i64], |r| r.get(0))?;
    if !known {
        return Ok(Err(Rejection::UnknownObject(object)));
    }
    connection.execute("INSERT OR REPLACE INTO object_caption VALUES (?1, ?2, ?3)", params![object.0 as i64, caption.0, at.0])?;
    Ok(Ok(()))
}

fn apply_topology(connection: &Connection, map: &RwLock<MapState>, topology: Topology, now: UnixNanos) -> Applied {
    let version = topology.map_version.0 as i64;
    note_version(connection, map, topology.map_version, now)?;
    for table in ["place_edge", "place", "doorway", "room_vertex", "room", "topology"] {
        connection.execute(&format!("DELETE FROM {table} WHERE map_version = ?1"), [version])?;
    }
    connection.execute("INSERT INTO topology VALUES (?1, ?2)", params![version, topology.built.0])?;
    for room in &topology.rooms {
        connection.execute("INSERT INTO room VALUES (?1, ?2, ?3, ?4)", params![version, room.id.0 as i64, room.floor_m, room.ceiling_m])?;
        for (seq, vertex) in room.polygon.iter().enumerate() {
            connection.execute(
                "INSERT INTO room_vertex VALUES (?1, ?2, ?3, ?4, ?5)",
                params![version, room.id.0 as i64, seq as i64, vertex.x_m, vertex.y_m],
            )?;
        }
    }
    for d in &topology.doorways {
        connection.execute(
            "INSERT INTO doorway VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![version, d.id.0 as i64, d.room_a.0 as i64, d.room_b.0 as i64, d.centre.x_m, d.centre.y_m, d.width_m],
        )?;
    }
    for p in &topology.places {
        connection.execute(
            "INSERT INTO place VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![version, p.id.0 as i64, p.room.0 as i64, p.position.x_m, p.position.y_m, p.clearance_m],
        )?;
    }
    for e in &topology.edges {
        connection.execute(
            "INSERT INTO place_edge VALUES (?1, ?2, ?3, ?4, ?5)",
            params![version, e.a.0 as i64, e.b.0 as i64, e.length_m, e.min_clearance_m],
        )?;
    }
    let mut state = map.write().expect("the map lock is never poisoned");
    state.set_topology(topology);
    let mut current = connection.prepare("SELECT history_id, centre_x_m, centre_y_m FROM object WHERE valid_to_unix_ns IS NULL")?;
    let rows = current
        .query_map([], |row| Ok((row.get::<_, i64>(0)?, Point { x_m: row.get(1)?, y_m: row.get(2)? })))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut assign = connection.prepare("UPDATE object SET room_id = ?2 WHERE history_id = ?1")?;
    for (history_id, centre) in rows {
        assign.execute(params![history_id, state.room_at(centre).map(|r| r.0 as i64)])?;
    }
    Ok(Ok(()))
}

fn apply_free_space(connection: &Connection, map: &RwLock<MapState>, free_space: FreeSpace, now: UnixNanos) -> Applied {
    note_version(connection, map, free_space.map_version(), now)?;
    let cells: Vec<u8> = free_space.clearance().iter().flat_map(|c| c.0.to_le_bytes()).collect();
    connection.execute(
        "INSERT OR REPLACE INTO free_space VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            free_space.map_version().0 as i64,
            free_space.built().0,
            free_space.origin().x_m,
            free_space.origin().y_m,
            free_space.resolution_m(),
            free_space.width(),
            free_space.height(),
            free_space.band().min_m,
            free_space.band().max_m,
            cells,
        ],
    )?;
    map.write().expect("the map lock is never poisoned").set_free_space(free_space);
    Ok(Ok(()))
}

fn record_robot_pose(connection: &Connection, map: &RwLock<MapState>, sample: RobotPoseSample, now: UnixNanos) -> Applied {
    note_version(connection, map, sample.map_version, now)?;
    connection.execute(
        "INSERT INTO robot_pose (captured_unix_ns, map_version, x_m, y_m, yaw_rad) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![sample.captured.0, sample.map_version.0 as i64, sample.pose.x_m, sample.pose.y_m, sample.pose.yaw_rad],
    )?;
    Ok(Ok(()))
}

fn record_goal(connection: &Connection, issued: &IssuedGoal) -> Applied {
    let goal = &issued.goal;
    let exists: bool = connection.query_row("SELECT EXISTS (SELECT 1 FROM goal WHERE goal_id = ?1)", [goal.goal_id.0 as i64], |r| r.get(0))?;
    if exists {
        return Ok(Err(Rejection::GoalExists(goal.goal_id)));
    }
    let (target_id, pose) = match goal.target {
        GoalTarget::Pose(pose) | GoalTarget::LookAt(LookTarget::Point(pose)) => (None, Some(pose)),
        GoalTarget::Object(id) | GoalTarget::LookAt(LookTarget::Object(id)) => (Some(id.0), None),
        GoalTarget::Place(id) => (Some(id.0), None),
        GoalTarget::FollowTrack(id) => (Some(id.0), None),
        GoalTarget::Stop => (None, None),
    };
    connection.execute(
        "INSERT INTO goal VALUES (:goal_id, :agent, :model, :kind, :target_id, :x, :y, :yaw, :standoff, :speed, :deadline,
             :map_version, 'sent', NULL, :issued, :issued)",
        named_params! {
            ":goal_id": goal.goal_id.0 as i64, ":agent": issued.agent.0, ":model": issued.model.0,
            ":kind": goal.target.kind().wire_name(), ":target_id": target_id.map(|id| id as i64),
            ":x": pose.map(|p| p.x_m), ":y": pose.map(|p| p.y_m), ":yaw": pose.map(|p| p.yaw_rad),
            ":standoff": goal.standoff_m.get(), ":speed": goal.max_speed_m_s.get(), ":deadline": goal.deadline.0,
            ":map_version": goal.map_version.0 as i64, ":issued": issued.issued.0,
        },
    )?;
    connection.execute(
        "INSERT INTO event (kind, detected_unix_ns, earliest_unix_ns, goal_id, goal_state) VALUES ('goal_sent', ?1, ?1, ?2, 'sent')",
        params![issued.issued.0, goal.goal_id.0 as i64],
    )?;
    Ok(Ok(()))
}

fn record_goal_status(connection: &Connection, status: GoalStatus) -> Applied {
    let goal = status.goal_id.0 as i64;
    let (state, reason) = (status.state.wire_name(), status.state.reason().map(|r| r.wire_name()));
    let changed = connection.execute(
        "UPDATE goal SET state = ?2, reason = ?3, updated_unix_ns = ?4 WHERE goal_id = ?1",
        params![goal, state, reason, status.updated.0],
    )?;
    if changed == 0 {
        return Ok(Err(Rejection::UnknownGoal(status.goal_id)));
    }
    connection.execute(
        "INSERT INTO event (kind, detected_unix_ns, earliest_unix_ns, goal_id, goal_state, goal_reason)
         VALUES ('goal_status', ?1, ?1, ?2, ?3, ?4)",
        params![status.updated.0, goal, state, reason],
    )?;
    Ok(Ok(()))
}

fn name_room(connection: &Connection, map: &RwLock<MapState>, room: RoomId, name: &RoomName, now: UnixNanos) -> Applied {
    let mut state = map.write().expect("the map lock is never poisoned");
    if !state.has_room(room) {
        return Ok(Err(Rejection::UnknownRoom(room)));
    }
    if name.0.trim().is_empty() {
        return Ok(Err(Rejection::EmptyName));
    }
    if name.0 == UNMAPPED_ROOM || parse_unnamed_room(name).is_some() {
        return Ok(Err(Rejection::ReservedName(name.clone())));
    }
    if let Some((by, _)) = state.room_names.iter().find(|(id, n)| *n == name && **id != room) {
        return Ok(Err(Rejection::NameTaken { name: name.clone(), by: *by }));
    }
    let previous = state.room_names.get(&room).cloned();
    connection.execute("INSERT OR REPLACE INTO room_name VALUES (?1, ?2, ?3)", params![room.0 as i64, name.0, now.0])?;
    connection.execute(
        "INSERT INTO event (kind, detected_unix_ns, earliest_unix_ns, room_id, room_name, previous_room_name)
         VALUES ('room_renamed', ?1, ?1, ?2, ?3, ?4)",
        params![now.0, room.0 as i64, name.0, previous.map(|p| p.0)],
    )?;
    state.room_names.insert(room, name.clone());
    Ok(Ok(()))
}
