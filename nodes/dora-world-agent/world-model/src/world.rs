//! What the agents may ask of the robot's world model. Every answer is a number, an id or an image computed by the
//! world model; the agent never does geometry itself.
//!
//! Coordinates are the map frame: x and y in metres, headings in degrees counter-clockwise from x.

use std::fmt;
use std::future::Future;
use std::num::NonZeroUsize;
use std::str::FromStr;

use perception_wire::navigation::{GoalTarget, MapPose, MapVersion, ObjectId, PlaceId, RoomId, TrackId, UnixNanos};
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct RoomName(pub String);

impl fmt::Display for RoomName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Point {
    pub x_m: f32,
    pub y_m: f32,
}

impl Point {
    pub fn distance_to(self, other: Point) -> f32 {
        (self.x_m - other.x_m).hypot(self.y_m - other.y_m)
    }
}

/// An object's footprint on the floor: its long and short sides and its height.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Size {
    pub long_m: f32,
    pub short_m: f32,
    pub height_m: f32,
}

/// Something with a place in the map that distances and routes run between.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Landmark {
    Robot,
    Object(ObjectId),
    Place(PlaceId),
}

/// An id as it appears in the digest: `o42` an object, `p3` a place, `t7` a people track, `robot` the robot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EntityRef {
    Robot,
    Object(ObjectId),
    Place(PlaceId),
    Track(TrackId),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0:?} is not an id: expected robot, o<number>, p<number> or t<number>")]
pub struct BadEntityRef(pub String);

impl FromStr for EntityRef {
    type Err = BadEntityRef;

    fn from_str(text: &str) -> Result<Self, BadEntityRef> {
        let bad = || BadEntityRef(text.to_owned());
        if text == "robot" {
            return Ok(EntityRef::Robot);
        }
        let (prefix, digits) = text.split_at_checked(1).ok_or_else(bad)?;
        let number: u64 = digits.parse().map_err(|_| bad())?;
        match prefix {
            "o" => Ok(EntityRef::Object(ObjectId(number))),
            "p" => Ok(EntityRef::Place(PlaceId(number))),
            "t" => Ok(EntityRef::Track(TrackId(number))),
            _ => Err(bad()),
        }
    }
}

impl fmt::Display for EntityRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntityRef::Robot => f.write_str("robot"),
            EntityRef::Object(id) => id.fmt(f),
            EntityRef::Place(id) => id.fmt(f),
            EntityRef::Track(id) => id.fmt(f),
        }
    }
}

impl EntityRef {
    /// Tracks move too fast to route to; everything else is a landmark.
    pub fn landmark(self) -> Option<Landmark> {
        match self {
            EntityRef::Robot => Some(Landmark::Robot),
            EntityRef::Object(id) => Some(Landmark::Object(id)),
            EntityRef::Place(id) => Some(Landmark::Place(id)),
            EntityRef::Track(_) => None,
        }
    }
}

/// When a fact was true and which map it belongs to; every answer carries one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stamp {
    pub as_of: UnixNanos,
    pub map_version: MapVersion,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RobotState {
    pub room: RoomName,
    pub pose: MapPose,
}

/// Whether a people track is being seen now.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrackLiveness {
    Live,
    Lost,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PersonTrack {
    pub id: TrackId,
    pub room: RoomName,
    pub position: Point,
    pub heading_deg: f32,
    pub speed_m_s: f32,
    pub last_seen: UnixNanos,
    pub liveness: TrackLiveness,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Doorway {
    pub place: PlaceId,
    pub to: RoomName,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObjectSummary {
    pub id: ObjectId,
    pub label: String,
    pub room: RoomName,
    pub position: Point,
    pub size: Size,
    pub last_seen: UnixNanos,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlaceSummary {
    pub id: PlaceId,
    pub name: String,
    pub position: Point,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RoomSummary {
    /// None for the objects that stand where no room is mapped.
    pub id: Option<RoomId>,
    pub name: RoomName,
    /// The point that stands for the room in answers: the centroid of its floor polygon, or the mean position of its
    /// objects where no polygon is known.
    pub centroid: Point,
    pub doorways: Vec<Doorway>,
    pub places: Vec<PlaceSummary>,
    pub objects: Vec<ObjectSummary>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChangeKind {
    Appeared,
    Moved { from: RoomName, to: RoomName },
    Disappeared,
}

/// A change the world model detected. It happened between `earliest` and `detected`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Change {
    #[serde(serialize_with = "serialize_unix_nanos")]
    pub detected: UnixNanos,
    #[serde(serialize_with = "serialize_unix_nanos")]
    pub earliest: UnixNanos,
    #[serde(serialize_with = "serialize_object_id")]
    pub object: ObjectId,
    pub label: String,
    pub room: RoomName,
    #[serde(flatten)]
    pub kind: ChangeKind,
}

/// A live source the world model has not heard from within its bound, so facts from it are stale.
#[derive(Clone, Debug, PartialEq)]
pub struct StaleSource {
    pub name: String,
    pub silent_s: f32,
}

/// Everything the per-turn digest is made from.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldSnapshot {
    pub stamp: Stamp,
    /// When the map the objects come from was last fused.
    pub map_fused: UnixNanos,
    pub robot: RobotState,
    pub people: Vec<PersonTrack>,
    pub rooms: Vec<RoomSummary>,
    /// Most recent first.
    pub recent_changes: Vec<Change>,
    pub stale_sources: Vec<StaleSource>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FindQuery {
    /// Free text, matched against labels and captions.
    pub text: Option<String>,
    pub label: Option<String>,
    pub room: Option<RoomName>,
    pub near: Option<Point>,
    pub limit: Option<NonZeroUsize>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FoundObject {
    #[serde(serialize_with = "serialize_object_id")]
    pub id: ObjectId,
    pub label: String,
    pub room: RoomName,
    pub position: Point,
    pub distance_from_robot_m: f32,
    #[serde(serialize_with = "serialize_unix_nanos")]
    pub last_seen: UnixNanos,
    /// 0 to 1, higher is a better match.
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Relation {
    pub relation: String,
    #[serde(serialize_with = "serialize_object_id")]
    pub other: ObjectId,
}

/// Where an object stood over one interval; `until` is None while it still stands there.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Placement {
    pub room: RoomName,
    pub position: Point,
    #[serde(serialize_with = "serialize_unix_nanos")]
    pub from: UnixNanos,
    #[serde(serialize_with = "serialize_optional_unix_nanos")]
    pub until: Option<UnixNanos>,
}

#[derive(Clone, PartialEq)]
pub struct PngImage(pub Vec<u8>);

impl fmt::Debug for PngImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PngImage({} bytes)", self.0.len())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObjectDescription {
    pub summary: ObjectSummary,
    pub caption: Option<String>,
    pub relations: Vec<Relation>,
    pub history: Vec<Placement>,
    /// The best views of the object, cropped from keyframes.
    pub views: Vec<PngImage>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Clearance {
    pub min_gap_m: f32,
    /// Where the gap is narrowest.
    pub pinch_point: Point,
    /// The robot's width plus its margin.
    pub robot_needs_m: f32,
    pub passable: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Route {
    pub length_m: f32,
    pub via: Vec<String>,
    pub rooms: Vec<RoomName>,
    pub min_clearance_m: f32,
    pub estimated_s: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Sighting {
    pub subject: String,
    pub room: RoomName,
    pub position: Point,
    #[serde(serialize_with = "serialize_unix_nanos")]
    pub at: UnixNanos,
    pub seconds_ago: f32,
    pub heading_deg: Option<f32>,
    /// Doorways ahead in the direction of travel, nearest first.
    pub exits_ahead: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewKind {
    TopDown,
    Egocentric,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ViewRequest {
    pub kind: ViewKind,
    pub room: Option<RoomName>,
    pub highlight: Vec<ObjectId>,
}

/// A query `ReadOnlySql::parse` accepted: one SELECT (or WITH ... SELECT) statement that names no writing keyword.
/// Implementations still run it on a read-only connection; the parse is a cheap first gate, not the guarantee.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadOnlySql(String);

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SqlRejected {
    #[error("the query is empty")]
    Empty,
    #[error("only one statement is allowed")]
    SeveralStatements,
    #[error("only SELECT queries are allowed")]
    NotASelect,
    #[error("{0} is not allowed in a read-only query")]
    Writes(String),
}

const WRITING_KEYWORDS: [&str; 13] = [
    "INSERT", "UPDATE", "DELETE", "REPLACE", "UPSERT", "DROP", "CREATE", "ALTER", "ATTACH", "DETACH", "PRAGMA", "VACUUM",
    "REINDEX",
];

impl ReadOnlySql {
    pub fn parse(text: &str) -> Result<Self, SqlRejected> {
        let statement = text.trim().trim_end_matches(';').trim();
        if statement.is_empty() {
            return Err(SqlRejected::Empty);
        }
        if statement.contains(';') {
            return Err(SqlRejected::SeveralStatements);
        }
        let words: Vec<String> = statement
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .filter(|w| !w.is_empty())
            .map(str::to_ascii_uppercase)
            .collect();
        if !matches!(words.first().map(String::as_str), Some("SELECT" | "WITH")) {
            return Err(SqlRejected::NotASelect);
        }
        if let Some(word) = words.iter().find(|w| WRITING_KEYWORDS.contains(&w.as_str())) {
            return Err(SqlRejected::Writes(word.clone()));
        }
        Ok(Self(statement.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum SqlValue {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SqlRows {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<SqlValue>>,
    /// True when more rows matched than the limit allowed.
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum WorldError {
    #[error("no object {0}")]
    UnknownObject(ObjectId),
    #[error("no place {0}")]
    UnknownPlace(PlaceId),
    #[error("no people track {0}")]
    UnknownTrack(TrackId),
    #[error("no room named {0}")]
    UnknownRoom(RoomName),
    #[error("{0} has no position to measure from")]
    NotALandmark(EntityRef),
    #[error("no route from {from} to {to}")]
    NoRoute { from: EntityRef, to: EntityRef },
    #[error("this world model cannot {0}")]
    Unsupported(&'static str),
    #[error("the world model has no {0} yet")]
    NotMapped(&'static str),
    #[error("the free-space map has no data at ({:.2}, {:.2})", .0.x_m, .0.y_m)]
    Unmapped(Point),
    #[error("the query failed: {0}")]
    BadQuery(String),
    #[error("the query ran past its {limit_ms} ms limit")]
    QueryTooSlow { limit_ms: u64 },
    #[error("the world store failed: {0}")]
    Store(String),
}

/// The world model the agents read. The supervisor and the MCP tools depend only on this trait; storage behind it
/// is the implementation's business. Implementations answer from current state at the moment of the call.
pub trait WorldModel: Send + Sync + 'static {
    fn snapshot(&self) -> impl Future<Output = Result<WorldSnapshot, WorldError>> + Send;

    fn stamp(&self) -> impl Future<Output = Result<Stamp, WorldError>> + Send;

    fn find(&self, query: FindQuery) -> impl Future<Output = Result<Vec<FoundObject>, WorldError>> + Send;

    fn describe_object(&self, id: ObjectId) -> impl Future<Output = Result<ObjectDescription, WorldError>> + Send;

    fn clearance_between(&self, a: Landmark, b: Landmark) -> impl Future<Output = Result<Clearance, WorldError>> + Send;

    fn route(&self, from: Landmark, to: Landmark) -> impl Future<Output = Result<Route, WorldError>> + Send;

    fn last_seen(&self, subject: EntityRef) -> impl Future<Output = Result<Sighting, WorldError>> + Send;

    fn changes_since(
        &self,
        since: UnixNanos,
        room: Option<RoomName>,
    ) -> impl Future<Output = Result<Vec<Change>, WorldError>> + Send;

    fn render_view(&self, request: ViewRequest) -> impl Future<Output = Result<PngImage, WorldError>> + Send;

    /// Runs the query on a read-only connection and returns at most `limit` rows.
    fn query_sql(&self, sql: ReadOnlySql, limit: NonZeroUsize) -> impl Future<Output = Result<SqlRows, WorldError>> + Send;

    /// Whether the target of a goal exists now, so a goal with an invented id fails at once.
    fn check_target(&self, target: GoalTarget) -> impl Future<Output = Result<(), WorldError>> + Send;
}

pub fn serialize_unix_nanos<S: serde::Serializer>(time: &UnixNanos, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&crate::clock::iso_time(*time))
}

pub fn serialize_optional_unix_nanos<S: serde::Serializer>(time: &Option<UnixNanos>, serializer: S) -> Result<S::Ok, S::Error> {
    match time {
        Some(time) => serialize_unix_nanos(time, serializer),
        None => serializer.serialize_none(),
    }
}

pub fn serialize_object_id<S: serde::Serializer>(id: &ObjectId, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_str(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_ids_parse_to_their_kind() {
        assert_eq!("o42".parse(), Ok(EntityRef::Object(ObjectId(42))));
        assert_eq!("p3".parse(), Ok(EntityRef::Place(PlaceId(3))));
        assert_eq!("t7".parse(), Ok(EntityRef::Track(TrackId(7))));
        assert_eq!("robot".parse(), Ok(EntityRef::Robot));
        for bad in ["", "o", "x4", "o-1", "table-3", "42"] {
            assert_eq!(bad.parse::<EntityRef>(), Err(BadEntityRef(bad.to_owned())), "{bad}");
        }
    }

    #[test]
    fn a_single_select_is_accepted() {
        let sql = ReadOnlySql::parse("  select label, count(*) from object group by label; ").unwrap();
        assert_eq!(sql.as_str(), "select label, count(*) from object group by label");
        assert!(ReadOnlySql::parse("WITH k AS (SELECT 1) SELECT * FROM k").is_ok());
    }

    #[test]
    fn anything_that_could_write_is_rejected() {
        assert_eq!(ReadOnlySql::parse(" ; "), Err(SqlRejected::Empty));
        assert_eq!(ReadOnlySql::parse("SELECT 1; SELECT 2"), Err(SqlRejected::SeveralStatements));
        assert_eq!(ReadOnlySql::parse("DELETE FROM object"), Err(SqlRejected::NotASelect));
        assert_eq!(
            ReadOnlySql::parse("WITH gone AS (SELECT 1) DELETE FROM object"),
            Err(SqlRejected::Writes("DELETE".to_owned()))
        );
        assert_eq!(ReadOnlySql::parse("select * from pragma_table_info('x')"), Ok(ReadOnlySql("select * from pragma_table_info('x')".to_owned())));
    }
}
