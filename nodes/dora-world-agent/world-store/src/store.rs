//! Opening the store, and the readers: a small pool of read-only connections and the in-memory map, answering the
//! `WorldModel` questions.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use perception_wire::navigation::{GoalTarget, LookTarget, MapPose, MapVersion, ObjectId, PlaceId, TrackId, UnixNanos};
use rusqlite::{params, params_from_iter, Connection, OpenFlags, OptionalExtension, Row};
use world_model::clock::seconds_between;
use world_model::{
    Change, ChangeKind, Clearance, Doorway, EntityRef, FindQuery, FoundObject, Landmark, ObjectDescription, ObjectSummary,
    PersonTrack, PlaceSummary, Placement, PngImage, Point, ReadOnlySql, Relation, RobotState, RoomName, RoomSummary, Route,
    Sighting, Size, SqlRows, StaleSource, Stamp, TrackLiveness, ViewKind, ViewRequest, WorldError, WorldModel, WorldSnapshot,
};

use crate::geometry::{area_centroid, bounds, gap_between, polygon_contains, square_around};
use crate::inputs::{Arrivals, Clock};
use crate::map::{MapState, RoomFilter};
use crate::render::{self, Canvas};
use crate::schema::{migrate, MigrationError};
use crate::sql::{open_guarded, run_read_only};
use crate::types::{
    Cell, ClearanceMm, DoorwayId, DoorwaySpan, FreeSpace, HeightBand, InputSource, PlaceEdge, PlaceNode, RoomId, RoomLabels,
    RoomShape, Topology,
};
use crate::writer::WorldWriter;

/// The robot's footprint and speed, for clearances and route times.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobotGeometry {
    pub width_m: f32,
    pub margin_m: f32,
    pub cruise_speed_m_s: f32,
}

/// How long each input may stay silent before the digest calls it stale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StalenessBounds {
    pub robot_pose: Duration,
    pub objects: Duration,
    pub topology: Duration,
    pub free_space: Duration,
}

impl StalenessBounds {
    fn of(&self, source: InputSource) -> Duration {
        match source {
            InputSource::RobotPose => self.robot_pose,
            InputSource::Objects => self.objects,
            InputSource::Topology => self.topology,
            InputSource::FreeSpace => self.free_space,
        }
    }
}

/// How long a people track counts as live, and how long a lost one stays in the snapshot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackBounds {
    pub live: Duration,
    pub forget: Duration,
}

/// Where and how often the store copies itself with `VACUUM INTO`.
#[derive(Clone, Debug, PartialEq)]
pub struct BackupConfig {
    pub path: PathBuf,
    pub every: Duration,
}

#[derive(Clone, Debug)]
pub struct StoreConfig {
    pub path: PathBuf,
    pub backup: Option<BackupConfig>,
    /// How long the writer gathers commands into one transaction.
    pub batch_interval: Duration,
    pub readers: NonZeroUsize,
    pub sql_time_limit: Duration,
    pub robot: RobotGeometry,
    pub staleness: StalenessBounds,
    pub tracks: TrackBounds,
    /// How many changes the snapshot carries.
    pub recent_changes: usize,
    pub find_limit: NonZeroUsize,
    pub clock: Clock,
}

impl StoreConfig {
    /// The defaults, for a database at `path`, without backups.
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            backup: None,
            batch_interval: Duration::from_millis(100),
            readers: NonZeroUsize::new(4).expect("four is not zero"),
            sql_time_limit: Duration::from_millis(200),
            robot: RobotGeometry { width_m: 0.38, margin_m: 0.10, cruise_speed_m_s: 0.3 },
            staleness: StalenessBounds {
                robot_pose: Duration::from_secs(2),
                objects: Duration::from_secs(60),
                topology: Duration::from_secs(300),
                free_space: Duration::from_secs(300),
            },
            tracks: TrackBounds { live: Duration::from_secs(2), forget: Duration::from_secs(600) },
            recent_changes: 10,
            find_limit: NonZeroUsize::new(5).expect("five is not zero"),
            clock: Clock::System,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OpenError {
    #[error("the world database at {path} failed its integrity check: {problems:?}")]
    Integrity { path: PathBuf, problems: Vec<String> },
    #[error("the world database at {path} did not switch to WAL mode; it is in {mode} mode")]
    NotWal { path: PathBuf, mode: String },
    #[error("the stored free-space map of {version} is malformed: {reason}")]
    BadFreeSpace { version: MapVersion, reason: String },
    #[error(transparent)]
    Migration(#[from] MigrationError),
    #[error(transparent)]
    Sql(#[from] rusqlite::Error),
    #[error("could not start a world store thread: {0}")]
    Thread(#[from] std::io::Error),
}

fn store(error: rusqlite::Error) -> WorldError {
    WorldError::Store(error.to_string())
}

/// Reads the world from the database and the in-memory map. Cheap to clone; clones share the connections.
#[derive(Clone)]
pub struct WorldStore {
    inner: Arc<Inner>,
}

struct Inner {
    path: PathBuf,
    readers: Vec<Mutex<Connection>>,
    next_reader: AtomicUsize,
    sql: Mutex<Connection>,
    map: Arc<RwLock<MapState>>,
    arrivals: Arc<Arrivals>,
    config: StoreConfig,
    opened: UnixNanos,
    /// Dropping it stops the backup thread.
    _stop_backups: Option<Sender<()>>,
}

fn integrity_problems(connection: &Connection) -> rusqlite::Result<Vec<String>> {
    let mut statement = connection.prepare("PRAGMA integrity_check")?;
    let rows = statement.query_map([], |row| row.get::<_, String>(0))?.collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(match rows.as_slice() {
        [ok] if ok == "ok" => Vec::new(),
        _ => rows,
    })
}

fn open_reader(path: &Path) -> rusqlite::Result<Connection> {
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)?;
    connection.busy_timeout(Duration::from_secs(5))?;
    Ok(connection)
}

/// Copies the database to `target` through a temporary file beside it, so `target` is always a whole copy.
pub fn back_up(connection: &Connection, target: &Path) -> rusqlite::Result<()> {
    let partial = target.with_extension("partial");
    if partial.exists() {
        std::fs::remove_file(&partial).map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
    }
    let partial_text = partial.to_str().ok_or_else(|| rusqlite::Error::InvalidPath(partial.clone()))?;
    connection.execute("VACUUM INTO ?1", [partial_text])?;
    std::fs::rename(&partial, target).map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
}

fn start_backups(path: PathBuf, backup: BackupConfig) -> std::io::Result<Sender<()>> {
    let (stop, stopped) = mpsc::channel::<()>();
    std::thread::Builder::new().name("world-store-backup".to_owned()).spawn(move || loop {
        match stopped.recv_timeout(backup.every) {
            Err(RecvTimeoutError::Timeout) => {}
            Ok(()) | Err(RecvTimeoutError::Disconnected) => return,
        }
        let outcome = open_reader(&path).and_then(|connection| back_up(&connection, &backup.path));
        match outcome {
            Ok(()) => tracing::info!(target = %backup.path.display(), "backed up the world store"),
            Err(error) => tracing::error!(%error, target = %backup.path.display(), "backing up the world store failed"),
        }
    })?;
    Ok(stop)
}

fn load_map(connection: &Connection) -> Result<MapState, OpenError> {
    let mut map = MapState::default();
    let latest: Option<i64> = connection.query_row("SELECT max(map_version) FROM map_version", [], |r| r.get(0))?;
    if let Some(version) = latest {
        map.note_version(MapVersion(version as u64));
    }
    let topology: Option<(i64, i64)> = connection
        .query_row("SELECT map_version, built_unix_ns FROM topology ORDER BY map_version DESC LIMIT 1", [], |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()?;
    if let Some((version, built)) = topology {
        let mut rooms_statement = connection.prepare("SELECT room_id, floor_m, ceiling_m FROM room WHERE map_version = ?1 ORDER BY room_id")?;
        let mut vertices = connection.prepare("SELECT x_m, y_m FROM room_vertex WHERE map_version = ?1 AND room_id = ?2 ORDER BY seq")?;
        let mut rooms = Vec::new();
        for room in rooms_statement.query_map([version], |r| Ok((r.get::<_, i64>(0)?, r.get(1)?, r.get(2)?)))? {
            let (id, floor_m, ceiling_m) = room?;
            let polygon = vertices
                .query_map(params![version, id], |r| Ok(Point { x_m: r.get(0)?, y_m: r.get(1)? }))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rooms.push(RoomShape { id: RoomId(id as u64), polygon, floor_m, ceiling_m });
        }
        let doorways = connection
            .prepare("SELECT doorway_id, room_a, room_b, x_m, y_m, width_m FROM doorway WHERE map_version = ?1 ORDER BY doorway_id")?
            .query_map([version], |r| {
                Ok(DoorwaySpan {
                    id: DoorwayId(r.get::<_, i64>(0)? as u64),
                    room_a: RoomId(r.get::<_, i64>(1)? as u64),
                    room_b: RoomId(r.get::<_, i64>(2)? as u64),
                    centre: Point { x_m: r.get(3)?, y_m: r.get(4)? },
                    width_m: r.get(5)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let places = connection
            .prepare("SELECT place_id, room_id, x_m, y_m, clearance_m FROM place WHERE map_version = ?1 ORDER BY place_id")?
            .query_map([version], |r| {
                Ok(PlaceNode {
                    id: PlaceId(r.get::<_, i64>(0)? as u64),
                    room: RoomId(r.get::<_, i64>(1)? as u64),
                    position: Point { x_m: r.get(2)?, y_m: r.get(3)? },
                    clearance_m: r.get(4)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let edges = connection
            .prepare("SELECT place_a, place_b, length_m, min_clearance_m FROM place_edge WHERE map_version = ?1")?
            .query_map([version], |r| {
                Ok(PlaceEdge {
                    a: PlaceId(r.get::<_, i64>(0)? as u64),
                    b: PlaceId(r.get::<_, i64>(1)? as u64),
                    length_m: r.get(2)?,
                    min_clearance_m: r.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        map.set_topology(Topology { map_version: MapVersion(version as u64), built: UnixNanos(built), rooms, doorways, places, edges });
    }
    let free_space = connection
        .query_row(
            "SELECT map_version, built_unix_ns, origin_x_m, origin_y_m, resolution_m, width, height, band_min_m, band_max_m,
                 clearance_mm FROM free_space ORDER BY map_version DESC LIMIT 1",
            [],
            |r| {
                Ok((
                    MapVersion(r.get::<_, i64>(0)? as u64),
                    UnixNanos(r.get(1)?),
                    Point { x_m: r.get(2)?, y_m: r.get(3)? },
                    r.get::<_, f32>(4)?,
                    r.get::<_, u32>(5)?,
                    r.get::<_, u32>(6)?,
                    HeightBand { min_m: r.get(7)?, max_m: r.get(8)? },
                    r.get::<_, Vec<u8>>(9)?,
                ))
            },
        )
        .optional()?;
    if let Some((version, built, origin, resolution_m, width, height, band, bytes)) = free_space {
        let bad = |reason: String| OpenError::BadFreeSpace { version, reason };
        if bytes.len() % 2 != 0 {
            return Err(bad(format!("{} bytes is not a whole number of cells", bytes.len())));
        }
        let cells = bytes.chunks_exact(2).map(|pair| ClearanceMm(u16::from_le_bytes([pair[0], pair[1]]))).collect();
        let grid = FreeSpace::new(version, built, origin, resolution_m, width, height, band, cells).map_err(|e| bad(e.to_string()))?;
        map.set_free_space(grid);
    }
    let mut names = connection.prepare("SELECT room_id, name FROM room_name")?;
    for row in names.query_map([], |r| Ok((RoomId(r.get::<_, i64>(0)? as u64), RoomName(r.get(1)?))))? {
        let (id, name) = row?;
        map.room_names.insert(id, name);
    }
    Ok(map)
}

impl WorldStore {
    /// Opens (or creates) the database, checks its integrity, migrates it, and starts the writer and backup threads.
    /// The store must be the only process with the file open.
    pub fn open(config: StoreConfig) -> Result<(WorldStore, WorldWriter), OpenError> {
        let path = config.path.clone();
        let mut connection = Connection::open(&path)?;
        let mode: String = connection.query_row("PRAGMA journal_mode = WAL", [], |r| r.get(0))?;
        if !mode.eq_ignore_ascii_case("wal") {
            return Err(OpenError::NotWal { path, mode });
        }
        connection.pragma_update(None, "synchronous", "FULL")?;
        connection.pragma_update(None, "foreign_keys", true)?;
        let problems = integrity_problems(&connection)?;
        if !problems.is_empty() {
            return Err(OpenError::Integrity { path, problems });
        }
        migrate(&mut connection)?;
        let map = Arc::new(RwLock::new(load_map(&connection)?));
        let arrivals = Arc::new(Arrivals::default());
        let readers = (0..config.readers.get()).map(|_| open_reader(&path).map(Mutex::new)).collect::<rusqlite::Result<Vec<_>>>()?;
        let sql = Mutex::new(open_guarded(&path)?);
        let stop_backups = config.backup.clone().map(|backup| start_backups(path.clone(), backup)).transpose()?;
        let writer = WorldWriter::start(connection, map.clone(), arrivals.clone(), config.clock.clone(), config.batch_interval)?;
        let opened = config.clock.now();
        let inner = Inner {
            path,
            readers,
            next_reader: AtomicUsize::new(0),
            sql,
            map,
            arrivals,
            config,
            opened,
            _stop_backups: stop_backups,
        };
        Ok((WorldStore { inner: Arc::new(inner) }, writer))
    }

    /// Copies the database to `target` now.
    pub fn back_up_to(&self, target: &Path) -> Result<(), WorldError> {
        let connection = open_reader(&self.inner.path).map_err(store)?;
        back_up(&connection, target).map_err(store)
    }

    async fn read<T: Send + 'static>(
        &self,
        work: impl FnOnce(&Inner, &Connection, &MapState) -> Result<T, WorldError> + Send + 'static,
    ) -> Result<T, WorldError> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let connection = inner.reader();
            let map = inner.map.read().expect("the map lock is never poisoned");
            work(&inner, &connection, &map)
        })
        .await
        .map_err(|e| WorldError::Store(format!("a read did not finish: {e}")))?
    }

    /// Where an object stood at a time: the history row whose interval holds it, or None if it was not in the map.
    pub async fn placement_at(&self, id: ObjectId, at: UnixNanos) -> Result<Option<Placement>, WorldError> {
        self.read(move |_, connection, map| {
            latest_row(connection, id)?.ok_or(WorldError::UnknownObject(id))?;
            connection
                .query_row(
                    &format!(
                        "SELECT {ROW_COLUMNS} FROM object WHERE object_id = ?1 AND valid_from_unix_ns <= ?2
                         AND (valid_to_unix_ns IS NULL OR valid_to_unix_ns > ?2) ORDER BY valid_from_unix_ns DESC LIMIT 1"
                    ),
                    params![id.0 as i64, at.0],
                    ObjectRow::read,
                )
                .optional()
                .map_err(store)
                .map(|row| row.map(|row| row.placement(map)))
        })
        .await
    }
}

impl Inner {
    fn reader(&self) -> std::sync::MutexGuard<'_, Connection> {
        for connection in &self.readers {
            if let Ok(guard) = connection.try_lock() {
                return guard;
            }
        }
        let index = self.next_reader.fetch_add(1, Ordering::Relaxed) % self.readers.len();
        self.readers[index].lock().expect("a reader lock is never poisoned")
    }

    fn now(&self) -> UnixNanos {
        self.config.clock.now()
    }

    fn stamp(&self, map: &MapState) -> Result<Stamp, WorldError> {
        let map_version = map.latest_version.ok_or(WorldError::NotMapped("map"))?;
        Ok(Stamp { as_of: self.now(), map_version })
    }

    fn robot_needs_m(&self) -> f32 {
        self.config.robot.width_m + self.config.robot.margin_m
    }

    fn stale_sources(&self, now: UnixNanos) -> Vec<StaleSource> {
        InputSource::ALL
            .into_iter()
            .filter_map(|source| {
                let silent_s = seconds_between(self.arrivals.last(source).unwrap_or(self.opened), now);
                (silent_s > self.config.staleness.of(source).as_secs_f32()).then(|| StaleSource { name: source.name().to_owned(), silent_s })
            })
            .collect()
    }
}

const ROW_COLUMNS: &str = "object_id, label, room_id, centre_x_m, centre_y_m, long_m, short_m, height_m, last_seen_unix_ns,
    valid_from_unix_ns, valid_to_unix_ns, corner0_x_m, corner0_y_m, corner1_x_m, corner1_y_m, corner2_x_m, corner2_y_m,
    corner3_x_m, corner3_y_m";

/// One history row of an object.
struct ObjectRow {
    id: ObjectId,
    label: String,
    room: Option<RoomId>,
    position: Point,
    size: Size,
    last_seen: UnixNanos,
    valid_from: UnixNanos,
    valid_to: Option<UnixNanos>,
    footprint: [Point; 4],
}

impl ObjectRow {
    fn read(row: &Row<'_>) -> rusqlite::Result<Self> {
        let corner = |i: usize| -> rusqlite::Result<Point> { Ok(Point { x_m: row.get(i)?, y_m: row.get(i + 1)? }) };
        Ok(Self {
            id: ObjectId(row.get::<_, i64>(0)? as u64),
            label: row.get(1)?,
            room: row.get::<_, Option<i64>>(2)?.map(|r| RoomId(r as u64)),
            position: Point { x_m: row.get(3)?, y_m: row.get(4)? },
            size: Size { long_m: row.get(5)?, short_m: row.get(6)?, height_m: row.get(7)? },
            last_seen: UnixNanos(row.get(8)?),
            valid_from: UnixNanos(row.get(9)?),
            valid_to: row.get::<_, Option<i64>>(10)?.map(UnixNanos),
            footprint: [corner(11)?, corner(13)?, corner(15)?, corner(17)?],
        })
    }

    fn summary(&self, map: &MapState) -> ObjectSummary {
        ObjectSummary {
            id: self.id,
            label: self.label.clone(),
            room: map.room_name(self.room),
            position: self.position,
            size: self.size,
            last_seen: self.last_seen,
        }
    }

    fn placement(&self, map: &MapState) -> Placement {
        Placement { room: map.room_name(self.room), position: self.position, from: self.valid_from, until: self.valid_to }
    }
}

fn current_row(connection: &Connection, id: ObjectId) -> Result<Option<ObjectRow>, WorldError> {
    connection
        .query_row(
            &format!("SELECT {ROW_COLUMNS} FROM object WHERE object_id = ?1 AND valid_to_unix_ns IS NULL"),
            [id.0 as i64],
            ObjectRow::read,
        )
        .optional()
        .map_err(store)
}

fn latest_row(connection: &Connection, id: ObjectId) -> Result<Option<ObjectRow>, WorldError> {
    connection
        .query_row(
            &format!("SELECT {ROW_COLUMNS} FROM object WHERE object_id = ?1 ORDER BY valid_from_unix_ns DESC, history_id DESC LIMIT 1"),
            [id.0 as i64],
            ObjectRow::read,
        )
        .optional()
        .map_err(store)
}

fn current_rows(connection: &Connection) -> Result<Vec<ObjectRow>, WorldError> {
    let mut statement = connection
        .prepare_cached(&format!("SELECT {ROW_COLUMNS} FROM object WHERE valid_to_unix_ns IS NULL ORDER BY object_id"))
        .map_err(store)?;
    let rows = statement.query_map([], ObjectRow::read).map_err(store)?;
    rows.collect::<rusqlite::Result<Vec<_>>>().map_err(store)
}

fn latest_robot(connection: &Connection) -> Result<(UnixNanos, MapPose), WorldError> {
    connection
        .query_row("SELECT captured_unix_ns, x_m, y_m, yaw_rad FROM robot_pose ORDER BY captured_unix_ns DESC, pose_id DESC LIMIT 1", [], |r| {
            Ok((UnixNanos(r.get(0)?), MapPose { x_m: r.get(1)?, y_m: r.get(2)?, yaw_rad: r.get(3)? }))
        })
        .optional()
        .map_err(store)?
        .ok_or(WorldError::NotMapped("robot pose"))
}

fn on_floor(pose: MapPose) -> Point {
    Point { x_m: pose.x_m, y_m: pose.y_m }
}

struct TrackRow {
    id: TrackId,
    position: Point,
    heading_rad: f32,
    speed_m_s: f32,
    last_seen: UnixNanos,
}

fn read_track(row: &Row<'_>) -> rusqlite::Result<TrackRow> {
    Ok(TrackRow {
        id: TrackId(row.get::<_, i64>(0)? as u64),
        last_seen: UnixNanos(row.get(1)?),
        position: Point { x_m: row.get(2)?, y_m: row.get(3)? },
        heading_rad: row.get(4)?,
        speed_m_s: row.get(5)?,
    })
}

const TRACK_QUERY: &str = "SELECT t.track_id, p.captured_unix_ns, p.x_m, p.y_m, p.heading_rad, p.speed_m_s
    FROM track t JOIN track_pose p ON p.track_id = t.track_id AND p.captured_unix_ns = t.last_seen_unix_ns";

fn track(connection: &Connection, id: TrackId) -> Result<TrackRow, WorldError> {
    connection
        .query_row(&format!("{TRACK_QUERY} WHERE t.track_id = ?1"), [id.0 as i64], read_track)
        .optional()
        .map_err(store)?
        .ok_or(WorldError::UnknownTrack(id))
}

fn liveness(inner: &Inner, last_seen: UnixNanos, now: UnixNanos) -> TrackLiveness {
    match seconds_between(last_seen, now) <= inner.config.tracks.live.as_secs_f32() {
        true => TrackLiveness::Live,
        false => TrackLiveness::Lost,
    }
}

fn people(inner: &Inner, connection: &Connection, map: &MapState, now: UnixNanos) -> Result<Vec<PersonTrack>, WorldError> {
    let since = now.0 - inner.config.tracks.forget.as_nanos() as i64;
    let mut statement =
        connection.prepare(&format!("{TRACK_QUERY} WHERE t.last_seen_unix_ns >= ?1 ORDER BY t.track_id")).map_err(store)?;
    let rows = statement.query_map([since], read_track).map_err(store)?.collect::<rusqlite::Result<Vec<_>>>().map_err(store)?;
    Ok(rows
        .into_iter()
        .map(|t| PersonTrack {
            id: t.id,
            room: map.room_name(map.room_at(t.position)),
            position: t.position,
            heading_deg: t.heading_rad.to_degrees(),
            speed_m_s: t.speed_m_s,
            last_seen: t.last_seen,
            liveness: liveness(inner, t.last_seen, now),
        })
        .collect())
}

/// One change event as stored.
struct ChangeRow {
    kind: String,
    detected: UnixNanos,
    earliest: UnixNanos,
    object: ObjectId,
    label: String,
    room_id: Option<i64>,
    from_room_id: Option<i64>,
}

/// Which room's changes to list: none means every room.
fn changes(
    connection: &Connection,
    map: &MapState,
    since: UnixNanos,
    room: Option<RoomFilter>,
    limit: Option<usize>,
) -> Result<Vec<Change>, WorldError> {
    let room_clause = match room {
        None => "",
        Some(RoomFilter::Room(_)) => "AND (room_id = ?2 OR from_room_id = ?2)",
        Some(RoomFilter::Unmapped) => "AND room_id IS NULL",
    };
    let limit_clause = limit.map_or(String::new(), |n| format!("LIMIT {n}"));
    let sql = format!(
        "SELECT kind, detected_unix_ns, earliest_unix_ns, object_id, label, room_id, from_room_id FROM event
         WHERE kind IN ('appeared', 'moved', 'disappeared') AND detected_unix_ns >= ?1 {room_clause}
         ORDER BY detected_unix_ns DESC, event_id DESC {limit_clause}"
    );
    let mut statement = connection.prepare(&sql).map_err(store)?;
    let read = |r: &Row<'_>| -> rusqlite::Result<ChangeRow> {
        Ok(ChangeRow {
            kind: r.get(0)?,
            detected: UnixNanos(r.get(1)?),
            earliest: UnixNanos(r.get(2)?),
            object: ObjectId(r.get::<_, i64>(3)? as u64),
            label: r.get(4)?,
            room_id: r.get(5)?,
            from_room_id: r.get(6)?,
        })
    };
    let rows = match room {
        Some(RoomFilter::Room(id)) => statement.query_map(params![since.0, id.0 as i64], read),
        _ => statement.query_map(params![since.0], read),
    }
    .map_err(store)?
    .collect::<rusqlite::Result<Vec<_>>>()
    .map_err(store)?;
    let room_name = |id: Option<i64>| map.room_name(id.map(|r| RoomId(r as u64)));
    rows.into_iter()
        .map(|row| {
            let kind = match row.kind.as_str() {
                "appeared" => ChangeKind::Appeared,
                "moved" => ChangeKind::Moved { from: room_name(row.from_room_id), to: room_name(row.room_id) },
                "disappeared" => ChangeKind::Disappeared,
                other => return Err(WorldError::Store(format!("event kind {other} is not a change"))),
            };
            Ok(Change {
                detected: row.detected,
                earliest: row.earliest,
                object: row.object,
                label: row.label,
                room: room_name(row.room_id),
                kind,
            })
        })
        .collect()
}

fn resolve_room(map: &MapState, name: &RoomName) -> Result<RoomFilter, WorldError> {
    map.resolve_room(name).ok_or_else(|| WorldError::UnknownRoom(name.clone()))
}

/// A landmark's position, outline on the floor and room.
struct Located {
    position: Point,
    outline: Vec<Point>,
    room: Option<RoomId>,
}

fn locate(inner: &Inner, connection: &Connection, map: &MapState, landmark: Landmark) -> Result<Located, WorldError> {
    match landmark {
        Landmark::Robot => {
            let (_, pose) = latest_robot(connection)?;
            let position = on_floor(pose);
            Ok(Located { position, outline: square_around(position, inner.config.robot.width_m, pose.yaw_rad).to_vec(), room: map.room_at(position) })
        }
        Landmark::Object(id) => {
            let row = current_row(connection, id)?.ok_or(WorldError::UnknownObject(id))?;
            Ok(Located { position: row.position, outline: row.footprint.to_vec(), room: row.room })
        }
        Landmark::Place(id) => {
            let place = map.place(id).ok_or(WorldError::UnknownPlace(id))?;
            Ok(Located { position: place.position, outline: vec![place.position], room: Some(place.room) })
        }
    }
}

fn landmark_ref(landmark: Landmark) -> EntityRef {
    match landmark {
        Landmark::Robot => EntityRef::Robot,
        Landmark::Object(id) => EntityRef::Object(id),
        Landmark::Place(id) => EntityRef::Place(id),
    }
}

/// The narrowest corridor along the straight line from `a` to `b`, outside both outlines: twice the smallest
/// clearance of the cells the line crosses, and where that is.
fn corridor_along(free_space: &FreeSpace, a: &Located, b: &Located) -> Result<(f32, Point), WorldError> {
    let length = a.position.distance_to(b.position);
    let steps = ((length / (free_space.resolution_m() / 2.0)).ceil() as usize).max(1);
    let mut narrowest: Option<(f32, Point)> = None;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let p = Point {
            x_m: a.position.x_m + (b.position.x_m - a.position.x_m) * t,
            y_m: a.position.y_m + (b.position.y_m - a.position.y_m) * t,
        };
        if polygon_contains(&a.outline, p) || polygon_contains(&b.outline, p) {
            continue;
        }
        let clearance_m = match free_space.at(p) {
            None | Some(Cell::Unknown) => return Err(WorldError::Unmapped(p)),
            Some(Cell::Occupied) => 0.0,
            Some(Cell::Free { clearance_m }) => clearance_m,
        };
        if narrowest.is_none_or(|(best, _)| clearance_m < best) {
            narrowest = Some((clearance_m, p));
        }
    }
    let (clearance_m, at) = narrowest.ok_or(WorldError::Unmapped(a.position))?;
    Ok((2.0 * clearance_m, at))
}

/// The doorway places in front of a heading, nearest first, each as `p6 (doorway between hallway and bedroom)`.
fn exits_ahead(map: &MapState, position: Point, heading_rad: f32) -> Vec<String> {
    let (hy, hx) = heading_rad.sin_cos();
    let mut ahead: Vec<&PlaceNode> = map
        .doorway_places
        .iter()
        .filter_map(|d| map.place(d.place))
        .filter(|p| (p.position.x_m - position.x_m) * hx + (p.position.y_m - position.y_m) * hy > 0.0)
        .collect();
    ahead.sort_by(|a, b| a.position.distance_to(position).total_cmp(&b.position.distance_to(position)));
    ahead.iter().map(|p| format!("{} ({})", p.id, map.place_name(p))).collect()
}

fn escape_like(word: &str) -> String {
    word.chars().fold(String::new(), |mut escaped, c| {
        if matches!(c, '%' | '_' | '\\') {
            escaped.push('\\');
        }
        escaped.push(c);
        escaped
    })
}

fn find(inner: &Inner, connection: &Connection, map: &MapState, query: FindQuery) -> Result<Vec<FoundObject>, WorldError> {
    let room = query.room.as_ref().map(|name| resolve_room(map, name)).transpose()?;
    let (_, robot_pose) = latest_robot(connection)?;
    let robot = on_floor(robot_pose);
    let near = query.near.unwrap_or(robot);
    let words: Vec<String> = query.text.iter().flat_map(|t| t.split_whitespace()).map(str::to_lowercase).collect();
    let mut values: Vec<rusqlite::types::Value> = Vec::new();
    let hits = match words.is_empty() {
        true => "1".to_owned(),
        false => {
            values.extend(words.iter().map(|w| format!("%{}%", escape_like(w)).into()));
            vec!["((o.label || ' ' || coalesce(c.caption, '')) LIKE ? ESCAPE '\\')"; words.len()].join(" + ")
        }
    };
    let mut filters = String::new();
    if let Some(label) = &query.label {
        filters.push_str(" AND o.label = ? COLLATE NOCASE");
        values.push(label.clone().into());
    }
    match room {
        Some(RoomFilter::Room(id)) => {
            filters.push_str(" AND o.room_id = ?");
            values.push((id.0 as i64).into());
        }
        Some(RoomFilter::Unmapped) => filters.push_str(" AND o.room_id IS NULL"),
        None => {}
    }
    values.extend([near.x_m, near.x_m, near.y_m, near.y_m].map(|v| f64::from(v).into()));
    let limit = query.limit.unwrap_or(inner.config.find_limit).get();
    values.push((limit as i64).into());
    let sql = format!(
        "SELECT {columns}, hits FROM (
             SELECT o.*, ({hits}) AS hits FROM object o LEFT JOIN object_caption c USING (object_id)
             WHERE o.valid_to_unix_ns IS NULL {filters})
         WHERE hits > 0
         ORDER BY hits DESC, (centre_x_m - ?) * (centre_x_m - ?) + (centre_y_m - ?) * (centre_y_m - ?), object_id
         LIMIT ?",
        columns = ROW_COLUMNS,
    );
    let mut statement = connection.prepare(&sql).map_err(store)?;
    let rows = statement
        .query_map(params_from_iter(values), |r| Ok((ObjectRow::read(r)?, r.get::<_, i64>(19)?)))
        .map_err(store)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(store)?;
    let word_count = words.len().max(1) as f32;
    Ok(rows
        .into_iter()
        .map(|(row, hits)| FoundObject {
            id: row.id,
            label: row.label.clone(),
            room: map.room_name(row.room),
            position: row.position,
            distance_from_robot_m: row.position.distance_to(robot),
            last_seen: row.last_seen,
            score: match words.is_empty() {
                true => 1.0,
                false => hits as f32 / word_count,
            },
        })
        .collect())
}

/// The mean position of a group of objects; the group is never empty.
fn mean_position(objects: &[ObjectSummary]) -> Point {
    let n = objects.len() as f32;
    Point {
        x_m: objects.iter().map(|o| o.position.x_m).sum::<f32>() / n,
        y_m: objects.iter().map(|o| o.position.y_m).sum::<f32>() / n,
    }
}

fn snapshot(inner: &Inner, connection: &Connection, map: &MapState) -> Result<WorldSnapshot, WorldError> {
    let stamp = inner.stamp(map)?;
    let map_fused = map.fused().ok_or(WorldError::NotMapped("fused map"))?;
    let (_, pose) = latest_robot(connection)?;
    let robot = RobotState { room: map.room_name(map.room_at(on_floor(pose))), pose };
    let mut by_room: BTreeMap<Option<RoomId>, Vec<ObjectSummary>> = BTreeMap::new();
    for row in current_rows(connection)? {
        by_room.entry(row.room).or_default().push(row.summary(map));
    }
    let mut rooms: Vec<RoomSummary> = Vec::new();
    if let Some(topology) = &map.topology {
        for room in &topology.rooms {
            let doorways = map
                .doorway_places
                .iter()
                .filter_map(|d| match (d.room_a == room.id, d.room_b == room.id) {
                    (true, _) => Some(Doorway { place: d.place, to: map.room_name(Some(d.room_b)) }),
                    (_, true) => Some(Doorway { place: d.place, to: map.room_name(Some(d.room_a)) }),
                    _ => None,
                })
                .collect();
            let places = topology
                .places
                .iter()
                .filter(|p| p.room == room.id && map.doorway_place(p.id).is_none())
                .map(|p| PlaceSummary { id: p.id, name: map.place_name(p), position: p.position })
                .collect();
            let objects = by_room.remove(&Some(room.id)).unwrap_or_default();
            rooms.push(RoomSummary {
                id: Some(room.id),
                name: map.room_name(Some(room.id)),
                centroid: area_centroid(&room.polygon),
                doorways,
                places,
                objects,
            });
        }
    }
    for (room, objects) in by_room {
        let centroid = mean_position(&objects);
        rooms.push(RoomSummary { id: room, name: map.room_name(room), centroid, doorways: Vec::new(), places: Vec::new(), objects });
    }
    Ok(WorldSnapshot {
        stamp,
        map_fused,
        robot,
        people: people(inner, connection, map, stamp.as_of)?,
        rooms,
        recent_changes: changes(connection, map, UnixNanos(i64::MIN), None, Some(inner.config.recent_changes))?,
        stale_sources: inner.stale_sources(stamp.as_of),
    })
}

fn describe_object(connection: &Connection, map: &MapState, id: ObjectId) -> Result<ObjectDescription, WorldError> {
    let latest = latest_row(connection, id)?.ok_or(WorldError::UnknownObject(id))?;
    let caption: Option<String> = connection
        .query_row("SELECT caption FROM object_caption WHERE object_id = ?1", [id.0 as i64], |r| r.get(0))
        .optional()
        .map_err(store)?;
    let relations = connection
        .prepare("SELECT relation, other_object_id FROM object_relation WHERE object_id = ?1 ORDER BY relation, other_object_id")
        .map_err(store)?
        .query_map([id.0 as i64], |r| Ok(Relation { relation: r.get(0)?, other: ObjectId(r.get::<_, i64>(1)? as u64) }))
        .map_err(store)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(store)?;
    let history = connection
        .prepare(&format!("SELECT {ROW_COLUMNS} FROM object WHERE object_id = ?1 ORDER BY valid_from_unix_ns, history_id"))
        .map_err(store)?
        .query_map([id.0 as i64], ObjectRow::read)
        .map_err(store)?
        .map(|row| row.map(|row| row.placement(map)))
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(store)?;
    let crops = connection
        .prepare("SELECT jpeg FROM object_view WHERE object_id = ?1 ORDER BY score DESC, view_id DESC")
        .map_err(store)?
        .query_map([id.0 as i64], |r| r.get::<_, Vec<u8>>(0))
        .map_err(store)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(store)?;
    let views = crops
        .iter()
        .map(|jpeg| render::jpeg_to_png(jpeg).map(PngImage).map_err(|e| WorldError::Store(format!("a stored crop of {id} does not decode: {e}"))))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ObjectDescription { summary: latest.summary(map), caption, relations, history, views })
}

fn clearance_between(inner: &Inner, connection: &Connection, map: &MapState, a: Landmark, b: Landmark) -> Result<Clearance, WorldError> {
    let (a, b) = (locate(inner, connection, map, a)?, locate(inner, connection, map, b)?);
    let (min_gap_m, pinch_point) = match a.outline.len() > 1 && b.outline.len() > 1 {
        true => gap_between(&a.outline, &b.outline),
        false => corridor_along(map.free_space.as_ref().ok_or(WorldError::NotMapped("free-space map"))?, &a, &b)?,
    };
    let robot_needs_m = inner.robot_needs_m();
    Ok(Clearance { min_gap_m, pinch_point, robot_needs_m, passable: min_gap_m >= robot_needs_m })
}

/// The place a landmark joins the graph at: itself for a place, otherwise the nearest place serving its room.
fn entry_place<'m>(map: &'m MapState, landmark: Landmark, located: &Located) -> Option<&'m PlaceNode> {
    if let Landmark::Place(id) = landmark {
        return map.place(id);
    }
    map.places_serving(located.room?)
        .into_iter()
        .min_by(|a, b| a.position.distance_to(located.position).total_cmp(&b.position.distance_to(located.position)))
}

fn route(inner: &Inner, connection: &Connection, map: &MapState, from: Landmark, to: Landmark) -> Result<Route, WorldError> {
    if map.topology.is_none() {
        return Err(WorldError::NotMapped("place graph"));
    }
    let (start, end) = (locate(inner, connection, map, from)?, locate(inner, connection, map, to)?);
    let no_route = || WorldError::NoRoute { from: landmark_ref(from), to: landmark_ref(to) };
    let (first, last) = (entry_place(map, from, &start).ok_or_else(no_route)?, entry_place(map, to, &end).ok_or_else(no_route)?);
    let (path, edges) = map.shortest_path(first.id, last.id).ok_or_else(no_route)?;
    let places: Vec<&PlaceNode> = path.iter().map(|id| map.place(*id).expect("paths run through places")).collect();
    let length_m = start.position.distance_to(first.position)
        + edges.iter().map(|e| e.length_m).sum::<f32>()
        + last.position.distance_to(end.position);
    let mut rooms: Vec<RoomName> = Vec::new();
    let room_ids = start.room.into_iter().chain(places.iter().map(|p| p.room)).chain(end.room);
    for name in room_ids.map(|id| map.room_name(Some(id))) {
        if !rooms.contains(&name) {
            rooms.push(name);
        }
    }
    let min_clearance_m = edges
        .iter()
        .map(|e| e.min_clearance_m)
        .chain(places.iter().map(|p| p.clearance_m))
        .fold(f32::INFINITY, f32::min);
    Ok(Route {
        length_m,
        via: path.iter().map(ToString::to_string).collect(),
        rooms,
        min_clearance_m,
        estimated_s: length_m / inner.config.robot.cruise_speed_m_s,
    })
}

fn last_seen(inner: &Inner, connection: &Connection, map: &MapState, subject: EntityRef) -> Result<Sighting, WorldError> {
    let now = inner.now();
    let sighting = |room, position: Point, at: UnixNanos, heading_rad: Option<f32>| Sighting {
        subject: subject.to_string(),
        room,
        position,
        at,
        seconds_ago: seconds_between(at, now),
        heading_deg: heading_rad.map(f32::to_degrees),
        exits_ahead: heading_rad.map_or_else(Vec::new, |h| exits_ahead(map, position, h)),
    };
    match subject {
        EntityRef::Object(id) => {
            let row = latest_row(connection, id)?.ok_or(WorldError::UnknownObject(id))?;
            Ok(sighting(map.room_name(row.room), row.position, row.last_seen, None))
        }
        EntityRef::Robot => {
            let (at, pose) = latest_robot(connection)?;
            let position = on_floor(pose);
            Ok(sighting(map.room_name(map.room_at(position)), position, at, Some(pose.yaw_rad)))
        }
        EntityRef::Track(id) => {
            let track = track(connection, id)?;
            Ok(sighting(map.room_name(map.room_at(track.position)), track.position, track.last_seen, Some(track.heading_rad)))
        }
        EntityRef::Place(_) => Err(WorldError::Unsupported("say when a place was last seen: places do not move")),
    }
}

const VIEW_MARGIN_M: f32 = 0.3;

fn render_view(inner: &Inner, connection: &Connection, map: &MapState, request: ViewRequest) -> Result<PngImage, WorldError> {
    if request.kind == ViewKind::Egocentric {
        return Err(WorldError::Unsupported("render an egocentric view"));
    }
    let rooms = map.topology.as_ref().map_or(&[][..], |t| t.rooms.as_slice());
    let area = match &request.room {
        Some(name) => match resolve_room(map, name)? {
            RoomFilter::Room(id) => bounds(rooms.iter().filter(|r| r.id == id).flat_map(|r| r.polygon.iter().copied())),
            RoomFilter::Unmapped => return Err(WorldError::Unsupported("draw the unmapped area: it has no outline")),
        },
        None => {
            let grid = map.free_space.as_ref().map(|g| [g.origin(), g.extent()]).into_iter().flatten();
            bounds(rooms.iter().flat_map(|r| r.polygon.iter().copied()).chain(grid))
        }
    };
    let (lo, hi) = area.ok_or(WorldError::NotMapped("map to draw"))?;
    let margin = |p: Point, by: f32| Point { x_m: p.x_m + by, y_m: p.y_m + by };
    let objects = current_rows(connection)?;
    for id in &request.highlight {
        if !objects.iter().any(|o| o.id == *id) {
            return Err(WorldError::UnknownObject(*id));
        }
    }
    let mut canvas = Canvas::new(margin(lo, -VIEW_MARGIN_M), margin(hi, VIEW_MARGIN_M));
    if let Some(free_space) = &map.free_space {
        canvas.paint_free_space(free_space);
    }
    for room in rooms {
        canvas.outline_polygon(&room.polygon, render::WALL);
    }
    for doorway in &map.doorway_places {
        if let Some(place) = map.place(doorway.place) {
            canvas.fill_polygon(&square_around(place.position, 0.15, 0.0), render::DOORWAY);
        }
    }
    for object in &objects {
        let colour = if request.highlight.contains(&object.id) { render::HIGHLIGHT } else { render::OBJECT };
        canvas.fill_polygon(&object.footprint, colour);
    }
    let now = inner.now();
    for person in people(inner, connection, map, now)?.into_iter().filter(|p| p.liveness == TrackLiveness::Live) {
        canvas.fill_polygon(&square_around(person.position, 0.4, 0.0), render::PERSON);
        canvas.label(person.position, &person.id.to_string(), render::LABEL);
    }
    let (_, pose) = latest_robot(connection)?;
    let robot = on_floor(pose);
    canvas.fill_polygon(&square_around(robot, inner.config.robot.width_m, pose.yaw_rad), render::ROBOT);
    let (sin, cos) = pose.yaw_rad.sin_cos();
    canvas.line(robot, Point { x_m: robot.x_m + 0.4 * cos, y_m: robot.y_m + 0.4 * sin }, render::ROBOT);
    for object in &objects {
        canvas.label(object.position, &object.id.to_string(), render::LABEL);
    }
    Ok(PngImage(canvas.encode()))
}

fn check_target(inner: &Inner, connection: &Connection, map: &MapState, target: GoalTarget) -> Result<(), WorldError> {
    match target {
        GoalTarget::Object(id) | GoalTarget::LookAt(LookTarget::Object(id)) => {
            current_row(connection, id)?.ok_or(WorldError::UnknownObject(id)).map(drop)
        }
        GoalTarget::Place(id) => map.place(id).ok_or(WorldError::UnknownPlace(id)).map(drop),
        GoalTarget::FollowTrack(id) => match liveness(inner, track(connection, id)?.last_seen, inner.now()) {
            TrackLiveness::Live => Ok(()),
            TrackLiveness::Lost => Err(WorldError::UnknownTrack(id)),
        },
        GoalTarget::Pose(pose) | GoalTarget::LookAt(LookTarget::Point(pose)) => {
            if map.topology.is_none() {
                return Err(WorldError::NotMapped("rooms"));
            }
            match map.inside_any_room(on_floor(pose)) {
                true => Ok(()),
                false => Err(WorldError::Unsupported("send the robot outside the mapped rooms")),
            }
        }
        GoalTarget::Stop => Ok(()),
    }
}

impl WorldModel for WorldStore {
    async fn snapshot(&self) -> Result<WorldSnapshot, WorldError> {
        self.read(snapshot).await
    }

    async fn stamp(&self) -> Result<Stamp, WorldError> {
        self.read(|inner, _, map| inner.stamp(map)).await
    }

    async fn find(&self, query: FindQuery) -> Result<Vec<FoundObject>, WorldError> {
        self.read(move |inner, connection, map| find(inner, connection, map, query)).await
    }

    async fn describe_object(&self, id: ObjectId) -> Result<ObjectDescription, WorldError> {
        self.read(move |_, connection, map| describe_object(connection, map, id)).await
    }

    async fn clearance_between(&self, a: Landmark, b: Landmark) -> Result<Clearance, WorldError> {
        self.read(move |inner, connection, map| clearance_between(inner, connection, map, a, b)).await
    }

    async fn route(&self, from: Landmark, to: Landmark) -> Result<Route, WorldError> {
        self.read(move |inner, connection, map| route(inner, connection, map, from, to)).await
    }

    async fn last_seen(&self, subject: EntityRef) -> Result<Sighting, WorldError> {
        self.read(move |inner, connection, map| last_seen(inner, connection, map, subject)).await
    }

    async fn changes_since(&self, since: UnixNanos, room: Option<RoomName>) -> Result<Vec<Change>, WorldError> {
        self.read(move |_, connection, map| {
            let filter = room.as_ref().map(|name| resolve_room(map, name)).transpose()?;
            changes(connection, map, since, filter, None)
        })
        .await
    }

    async fn render_view(&self, request: ViewRequest) -> Result<PngImage, WorldError> {
        self.read(move |inner, connection, map| render_view(inner, connection, map, request)).await
    }

    async fn query_sql(&self, sql: ReadOnlySql, limit: NonZeroUsize) -> Result<SqlRows, WorldError> {
        self.query_text(sql.as_str().to_owned(), limit).await
    }

    async fn check_target(&self, target: GoalTarget) -> Result<(), WorldError> {
        self.read(move |inner, connection, map| check_target(inner, connection, map, target)).await
    }
}

impl WorldStore {
    /// Every room of the current topology, in topology order, with its given name and the labels of the objects in it.
    pub async fn room_labels(&self) -> Result<Vec<RoomLabels>, WorldError> {
        self.read(|_, connection, map| {
            let Some(topology) = &map.topology else { return Ok(Vec::new()) };
            let rows = current_rows(connection)?;
            Ok(topology
                .rooms
                .iter()
                .map(|room| RoomLabels {
                    room: room.id,
                    name: map.room_names.get(&room.id).cloned(),
                    labels: rows.iter().filter(|row| row.room == Some(room.id)).map(|row| row.label.clone()).collect(),
                })
                .collect())
        })
        .await
    }

    /// Runs SQL text on the guarded connection without `ReadOnlySql`'s keyword gate, so the connection's own guards
    /// are what stand between the text and the data.
    pub async fn query_text(&self, sql: String, limit: NonZeroUsize) -> Result<SqlRows, WorldError> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let connection = inner.sql.lock().expect("the SQL connection lock is never poisoned");
            run_read_only(&connection, &sql, limit, inner.config.sql_time_limit)
        })
        .await
        .map_err(|e| WorldError::Store(format!("a query did not finish: {e}")))?
    }
}
