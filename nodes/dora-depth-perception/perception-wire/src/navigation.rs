//! The navigation messages of lib/dora-rig-messages/dora_rig_messages/navigation.py, as single-row Arrow structs.
//! Field names, types and nullability must match that module exactly; the tests read its golden fixtures.
//!
//! Poses are in the map frame: x and y in metres, yaw in radians counter-clockwise from x.

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use dora_node_api::arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Float32Array, Int64Array, ListArray, StringArray, StructArray, UInt32Array,
    UInt64Array,
};
use dora_node_api::arrow::buffer::{NullBuffer, OffsetBuffer};
use dora_node_api::arrow::datatypes::{DataType, Field, Fields, Float32Type, Int64Type, UInt32Type, UInt64Type};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GoalId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObjectId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PlaceId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TrackId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MapVersion(pub u64);

/// A unix time in nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UnixNanos(pub i64);

impl fmt::Display for GoalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "g{}", self.0)
    }
}

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "o{}", self.0)
    }
}

impl fmt::Display for PlaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "p{}", self.0)
    }
}

impl fmt::Display for TrackId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t{}", self.0)
    }
}

impl fmt::Display for MapVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// A distance or speed that is finite and at least zero.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct NonNegative(f32);

impl NonNegative {
    pub const ZERO: Self = Self(0.0);

    pub fn new(field: &'static str, value: f32) -> Result<Self, WireError> {
        if value.is_finite() && value >= 0.0 {
            Ok(Self(value))
        } else {
            Err(WireError::Negative { field, value })
        }
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MapPose {
    pub x_m: f32,
    pub y_m: f32,
    pub yaw_rad: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LookTarget {
    Object(ObjectId),
    Point(MapPose),
}

/// What a goal is for, which fixes what its target is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GoalTarget {
    /// Drive to a map pose.
    Pose(MapPose),
    /// Drive to the standoff distance from an object.
    Object(ObjectId),
    /// Drive to a place node.
    Place(PlaceId),
    /// Hand a people track to the follower.
    FollowTrack(TrackId),
    /// Turn on the spot to face an object or a map point.
    LookAt(LookTarget),
    /// Stop and drop whatever goal is running.
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GoalKind {
    Pose,
    Object,
    Place,
    FollowTrack,
    LookAt,
    Stop,
}

impl GoalKind {
    pub const ALL: [GoalKind; 6] =
        [GoalKind::Pose, GoalKind::Object, GoalKind::Place, GoalKind::FollowTrack, GoalKind::LookAt, GoalKind::Stop];

    pub fn wire_name(self) -> &'static str {
        match self {
            GoalKind::Pose => "pose",
            GoalKind::Object => "object",
            GoalKind::Place => "place",
            GoalKind::FollowTrack => "follow_track",
            GoalKind::LookAt => "look_at",
            GoalKind::Stop => "stop",
        }
    }
}

impl FromStr for GoalKind {
    type Err = WireError;

    fn from_str(name: &str) -> Result<Self, WireError> {
        GoalKind::ALL
            .into_iter()
            .find(|kind| kind.wire_name() == name)
            .ok_or_else(|| WireError::UnknownName { field: "kind", name: name.to_owned() })
    }
}

impl GoalTarget {
    pub fn kind(&self) -> GoalKind {
        match self {
            GoalTarget::Pose(_) => GoalKind::Pose,
            GoalTarget::Object(_) => GoalKind::Object,
            GoalTarget::Place(_) => GoalKind::Place,
            GoalTarget::FollowTrack(_) => GoalKind::FollowTrack,
            GoalTarget::LookAt(_) => GoalKind::LookAt,
            GoalTarget::Stop => GoalKind::Stop,
        }
    }

    fn wire_id(&self) -> Option<u64> {
        match self {
            GoalTarget::Object(ObjectId(id)) | GoalTarget::LookAt(LookTarget::Object(ObjectId(id))) => Some(*id),
            GoalTarget::Place(PlaceId(id)) => Some(*id),
            GoalTarget::FollowTrack(TrackId(id)) => Some(*id),
            GoalTarget::Pose(_) | GoalTarget::LookAt(LookTarget::Point(_)) | GoalTarget::Stop => None,
        }
    }

    fn wire_pose(&self) -> Option<MapPose> {
        match self {
            GoalTarget::Pose(pose) | GoalTarget::LookAt(LookTarget::Point(pose)) => Some(*pose),
            _ => None,
        }
    }

    fn from_wire(kind: GoalKind, id: Option<u64>, pose: Option<MapPose>) -> Result<Self, WireError> {
        match (kind, id, pose) {
            (GoalKind::Pose, None, Some(pose)) => Ok(GoalTarget::Pose(pose)),
            (GoalKind::Object, Some(id), None) => Ok(GoalTarget::Object(ObjectId(id))),
            (GoalKind::Place, Some(id), None) => Ok(GoalTarget::Place(PlaceId(id))),
            (GoalKind::FollowTrack, Some(id), None) => Ok(GoalTarget::FollowTrack(TrackId(id))),
            (GoalKind::LookAt, Some(id), None) => Ok(GoalTarget::LookAt(LookTarget::Object(ObjectId(id)))),
            (GoalKind::LookAt, None, Some(pose)) => Ok(GoalTarget::LookAt(LookTarget::Point(pose))),
            (GoalKind::Stop, None, None) => Ok(GoalTarget::Stop),
            (kind, _, _) => Err(WireError::TargetDoesNotFitKind(kind)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavGoal {
    pub goal_id: GoalId,
    pub target: GoalTarget,
    pub standoff_m: NonNegative,
    /// The robot caps this again with its own limits.
    pub max_speed_m_s: NonNegative,
    /// The robot drops the goal when this passes.
    pub deadline: UnixNanos,
    /// The map the agent read the target from.
    pub map_version: MapVersion,
}

/// Why a goal is blocked or failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GoalReason {
    UnknownTarget,
    NoPath,
    TargetLost,
    PersonInPath,
    StaleMap,
    OperatorOverride,
    DeadlinePassed,
    Preempted,
}

impl GoalReason {
    pub const ALL: [GoalReason; 8] = [
        GoalReason::UnknownTarget,
        GoalReason::NoPath,
        GoalReason::TargetLost,
        GoalReason::PersonInPath,
        GoalReason::StaleMap,
        GoalReason::OperatorOverride,
        GoalReason::DeadlinePassed,
        GoalReason::Preempted,
    ];

    pub fn wire_name(self) -> &'static str {
        match self {
            GoalReason::UnknownTarget => "unknown_target",
            GoalReason::NoPath => "no_path",
            GoalReason::TargetLost => "target_lost",
            GoalReason::PersonInPath => "person_in_path",
            GoalReason::StaleMap => "stale_map",
            GoalReason::OperatorOverride => "operator_override",
            GoalReason::DeadlinePassed => "deadline_passed",
            GoalReason::Preempted => "preempted",
        }
    }
}

impl FromStr for GoalReason {
    type Err = WireError;

    fn from_str(name: &str) -> Result<Self, WireError> {
        GoalReason::ALL
            .into_iter()
            .find(|reason| reason.wire_name() == name)
            .ok_or_else(|| WireError::UnknownName { field: "reason", name: name.to_owned() })
    }
}

/// Where a goal stands; blocked and failed say why.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GoalState {
    Accepted,
    Active,
    Blocked(GoalReason),
    Succeeded,
    Failed(GoalReason),
}

impl GoalState {
    pub fn wire_name(self) -> &'static str {
        match self {
            GoalState::Accepted => "accepted",
            GoalState::Active => "active",
            GoalState::Blocked(_) => "blocked",
            GoalState::Succeeded => "succeeded",
            GoalState::Failed(_) => "failed",
        }
    }

    pub fn reason(self) -> Option<GoalReason> {
        match self {
            GoalState::Blocked(reason) | GoalState::Failed(reason) => Some(reason),
            GoalState::Accepted | GoalState::Active | GoalState::Succeeded => None,
        }
    }

    /// Whether the robot has finished with the goal.
    pub fn is_final(self) -> bool {
        matches!(self, GoalState::Succeeded | GoalState::Failed(_))
    }

    fn from_wire(name: &str, reason: Option<GoalReason>) -> Result<Self, WireError> {
        match (name, reason) {
            ("accepted", None) => Ok(GoalState::Accepted),
            ("active", None) => Ok(GoalState::Active),
            ("succeeded", None) => Ok(GoalState::Succeeded),
            ("blocked", Some(reason)) => Ok(GoalState::Blocked(reason)),
            ("failed", Some(reason)) => Ok(GoalState::Failed(reason)),
            ("accepted" | "active" | "succeeded", Some(_)) => Err(WireError::ReasonDoesNotFitState(name.to_owned())),
            ("blocked" | "failed", None) => Err(WireError::ReasonDoesNotFitState(name.to_owned())),
            (other, _) => Err(WireError::UnknownName { field: "state", name: other.to_owned() }),
        }
    }
}

impl fmt::Display for GoalState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.reason() {
            Some(reason) => write!(f, "{} ({})", self.wire_name(), reason.wire_name()),
            None => f.write_str(self.wire_name()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GoalStatus {
    pub goal_id: GoalId,
    pub state: GoalState,
    pub updated: UnixNanos,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum WireError {
    #[error("expected the {message} wire type {expected:?}, received {received:?}")]
    WrongType { message: &'static str, expected: DataType, received: DataType },
    #[error("expected one {message} row, received {rows}")]
    NotOneRow { message: &'static str, rows: usize },
    #[error("{field} is null")]
    Null { field: &'static str },
    #[error("unknown {field} {name:?}")]
    UnknownName { field: &'static str, name: String },
    #[error("a {} goal takes a different target", .0.wire_name())]
    TargetDoesNotFitKind(GoalKind),
    #[error("{field} is {value}, expected a finite value of at least 0")]
    Negative { field: &'static str, value: f32 },
    #[error("a {0} goal status has a reason exactly when blocked or failed")]
    ReasonDoesNotFitState(String),
    #[error("a {} reference has an id exactly when it is not a pose", .0.wire_name())]
    IdDoesNotFitReference(ReferenceKind),
}

fn field(name: &str, data_type: DataType) -> Field {
    Field::new(name, data_type, true)
}

fn map_pose_fields() -> Fields {
    Fields::from(vec![
        field("x_m", DataType::Float32),
        field("y_m", DataType::Float32),
        field("yaw_rad", DataType::Float32),
    ])
}

pub fn nav_goal_fields() -> Fields {
    Fields::from(vec![
        field("goal_id", DataType::UInt64),
        field("kind", DataType::Utf8),
        field("target_id", DataType::UInt64),
        field("target_pose", DataType::Struct(map_pose_fields())),
        field("standoff_m", DataType::Float32),
        field("max_speed_m_s", DataType::Float32),
        field("deadline_unix_ns", DataType::Int64),
        field("map_version", DataType::UInt64),
    ])
}

pub fn goal_status_fields() -> Fields {
    Fields::from(vec![
        field("goal_id", DataType::UInt64),
        field("state", DataType::Utf8),
        field("reason", DataType::Utf8),
        field("updated_unix_ns", DataType::Int64),
    ])
}

fn checked<'a>(array: &'a dyn Array, fields: Fields, message: &'static str) -> Result<&'a StructArray, WireError> {
    let expected = DataType::Struct(fields);
    if array.data_type() != &expected {
        return Err(WireError::WrongType { message, expected, received: array.data_type().clone() });
    }
    if array.len() != 1 {
        return Err(WireError::NotOneRow { message, rows: array.len() });
    }
    Ok(array.as_struct())
}

fn column<'a>(row: &'a StructArray, name: &str) -> &'a ArrayRef {
    row.column_by_name(name).expect("checked wire type")
}

fn required<T>(value: Option<T>, field: &'static str) -> Result<T, WireError> {
    value.ok_or(WireError::Null { field })
}

fn optional_u64(row: &StructArray, name: &str) -> Option<u64> {
    let values = column(row, name).as_primitive::<UInt64Type>();
    values.is_valid(0).then(|| values.value(0))
}

fn optional_i64(row: &StructArray, name: &str) -> Option<i64> {
    let values = column(row, name).as_primitive::<Int64Type>();
    values.is_valid(0).then(|| values.value(0))
}

fn optional_f32(array: &ArrayRef) -> Option<f32> {
    let values = array.as_primitive::<Float32Type>();
    values.is_valid(0).then(|| values.value(0))
}

fn optional_str<'a>(row: &'a StructArray, name: &str) -> Option<&'a str> {
    let values = column(row, name).as_string::<i32>();
    values.is_valid(0).then(|| values.value(0))
}

fn non_negative(row: &StructArray, field: &'static str) -> Result<NonNegative, WireError> {
    NonNegative::new(field, required(optional_f32(column(row, field)), field)?)
}

fn map_pose(row: &StructArray) -> Result<Option<MapPose>, WireError> {
    let pose = column(row, "target_pose").as_struct();
    if pose.is_null(0) {
        return Ok(None);
    }
    let part = |name: &'static str| required(optional_f32(pose.column_by_name(name).expect("checked wire type")), name);
    Ok(Some(MapPose { x_m: part("x_m")?, y_m: part("y_m")?, yaw_rad: part("yaw_rad")? }))
}

pub fn decode_nav_goal(array: &dyn Array) -> Result<NavGoal, WireError> {
    let row = checked(array, nav_goal_fields(), "nav goal")?;
    let kind: GoalKind = required(optional_str(row, "kind"), "kind")?.parse()?;
    Ok(NavGoal {
        goal_id: GoalId(required(optional_u64(row, "goal_id"), "goal_id")?),
        target: GoalTarget::from_wire(kind, optional_u64(row, "target_id"), map_pose(row)?)?,
        standoff_m: non_negative(row, "standoff_m")?,
        max_speed_m_s: non_negative(row, "max_speed_m_s")?,
        deadline: UnixNanos(required(optional_i64(row, "deadline_unix_ns"), "deadline_unix_ns")?),
        map_version: MapVersion(required(optional_u64(row, "map_version"), "map_version")?),
    })
}

fn floats(value: f32) -> ArrayRef {
    Arc::new(Float32Array::from(vec![value]))
}

pub fn encode_nav_goal(goal: &NavGoal) -> StructArray {
    let pose = goal.target.wire_pose();
    let parts = pose.unwrap_or(MapPose { x_m: 0.0, y_m: 0.0, yaw_rad: 0.0 });
    let pose_column = StructArray::new(
        map_pose_fields(),
        vec![floats(parts.x_m), floats(parts.y_m), floats(parts.yaw_rad)],
        Some(NullBuffer::from(vec![pose.is_some()])),
    );
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(vec![goal.goal_id.0])),
        Arc::new(StringArray::from(vec![goal.target.kind().wire_name()])),
        Arc::new(UInt64Array::from(vec![goal.target.wire_id()])),
        Arc::new(pose_column),
        floats(goal.standoff_m.get()),
        floats(goal.max_speed_m_s.get()),
        Arc::new(Int64Array::from(vec![goal.deadline.0])),
        Arc::new(UInt64Array::from(vec![goal.map_version.0])),
    ];
    StructArray::new(nav_goal_fields(), columns, None)
}

pub fn decode_goal_status(array: &dyn Array) -> Result<GoalStatus, WireError> {
    let row = checked(array, goal_status_fields(), "goal status")?;
    let reason = optional_str(row, "reason").map(str::parse::<GoalReason>).transpose()?;
    Ok(GoalStatus {
        goal_id: GoalId(required(optional_u64(row, "goal_id"), "goal_id")?),
        state: GoalState::from_wire(required(optional_str(row, "state"), "state")?, reason)?,
        updated: UnixNanos(required(optional_i64(row, "updated_unix_ns"), "updated_unix_ns")?),
    })
}

pub fn encode_goal_status(status: &GoalStatus) -> StructArray {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(vec![status.goal_id.0])),
        Arc::new(StringArray::from(vec![status.state.wire_name()])),
        Arc::new(StringArray::from(vec![status.state.reason().map(GoalReason::wire_name)])),
        Arc::new(Int64Array::from(vec![status.updated.0])),
    ];
    StructArray::new(goal_status_fields(), columns, None)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QuestionId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RoomId(pub u64);

/// Which message for a question this is: 0 for the first answer, then one more for each later message.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Revision(pub u32);

impl Revision {
    pub const FIRST: Revision = Revision(0);

    pub fn next(self) -> Revision {
        Revision(self.0 + 1)
    }
}

impl fmt::Display for QuestionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "q{}", self.0)
    }
}

impl fmt::Display for RoomId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.0)
    }
}

/// Who asked a question: the name of the agent or person, for the logs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Asker(pub String);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpaceQuestion {
    pub question_id: QuestionId,
    pub asked: UnixNanos,
    pub asker: Asker,
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReferenceKind {
    Object,
    Place,
    Room,
    Track,
    Pose,
}

impl ReferenceKind {
    pub const ALL: [ReferenceKind; 5] =
        [ReferenceKind::Object, ReferenceKind::Place, ReferenceKind::Room, ReferenceKind::Track, ReferenceKind::Pose];

    pub fn wire_name(self) -> &'static str {
        match self {
            ReferenceKind::Object => "object",
            ReferenceKind::Place => "place",
            ReferenceKind::Room => "room",
            ReferenceKind::Track => "track",
            ReferenceKind::Pose => "pose",
        }
    }
}

impl FromStr for ReferenceKind {
    type Err = WireError;

    fn from_str(name: &str) -> Result<Self, WireError> {
        ReferenceKind::ALL
            .into_iter()
            .find(|kind| kind.wire_name() == name)
            .ok_or_else(|| WireError::UnknownName { field: "references.kind", name: name.to_owned() })
    }
}

/// What an answer refers to: something with an id, or a bare map point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Referent {
    Object(ObjectId),
    Place(PlaceId),
    Room(RoomId),
    Track(TrackId),
    Pose,
}

impl Referent {
    pub fn kind(self) -> ReferenceKind {
        match self {
            Referent::Object(_) => ReferenceKind::Object,
            Referent::Place(_) => ReferenceKind::Place,
            Referent::Room(_) => ReferenceKind::Room,
            Referent::Track(_) => ReferenceKind::Track,
            Referent::Pose => ReferenceKind::Pose,
        }
    }

    fn wire_id(self) -> Option<u64> {
        match self {
            Referent::Object(ObjectId(id)) | Referent::Place(PlaceId(id)) => Some(id),
            Referent::Room(RoomId(id)) | Referent::Track(TrackId(id)) => Some(id),
            Referent::Pose => None,
        }
    }

    fn from_wire(kind: ReferenceKind, id: Option<u64>) -> Result<Self, WireError> {
        match (kind, id) {
            (ReferenceKind::Object, Some(id)) => Ok(Referent::Object(ObjectId(id))),
            (ReferenceKind::Place, Some(id)) => Ok(Referent::Place(PlaceId(id))),
            (ReferenceKind::Room, Some(id)) => Ok(Referent::Room(RoomId(id))),
            (ReferenceKind::Track, Some(id)) => Ok(Referent::Track(TrackId(id))),
            (ReferenceKind::Pose, None) => Ok(Referent::Pose),
            (kind, _) => Err(WireError::IdDoesNotFitReference(kind)),
        }
    }
}

/// Something an answer is about, and where it is in the map frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reference {
    pub referent: Referent,
    pub x_m: f32,
    pub y_m: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpaceAnswer {
    pub question_id: QuestionId,
    pub revision: Revision,
    /// Set on the last message for the question.
    pub is_final: bool,
    /// The model whose answer this is.
    pub model: String,
    pub text: String,
    pub references: Vec<Reference>,
    /// The nav goal the robot is on because of this answer.
    pub goal_id: Option<GoalId>,
}

pub fn space_question_fields() -> Fields {
    Fields::from(vec![
        field("question_id", DataType::UInt64),
        field("asked_unix_ns", DataType::Int64),
        field("asker", DataType::Utf8),
        field("text", DataType::Utf8),
    ])
}

fn reference_fields() -> Fields {
    Fields::from(vec![
        field("kind", DataType::Utf8),
        field("id", DataType::UInt64),
        field("x_m", DataType::Float32),
        field("y_m", DataType::Float32),
    ])
}

fn reference_item() -> Arc<Field> {
    Arc::new(Field::new("item", DataType::Struct(reference_fields()), false))
}

pub fn space_answer_fields() -> Fields {
    Fields::from(vec![
        field("question_id", DataType::UInt64),
        field("revision", DataType::UInt32),
        field("is_final", DataType::Boolean),
        field("model", DataType::Utf8),
        field("text", DataType::Utf8),
        field("references", DataType::List(reference_item())),
        field("goal_id", DataType::UInt64),
    ])
}

fn required_str(row: &StructArray, name: &'static str) -> Result<String, WireError> {
    required(optional_str(row, name), name).map(str::to_owned)
}

pub fn decode_space_question(array: &dyn Array) -> Result<SpaceQuestion, WireError> {
    let row = checked(array, space_question_fields(), "space question")?;
    Ok(SpaceQuestion {
        question_id: QuestionId(required(optional_u64(row, "question_id"), "question_id")?),
        asked: UnixNanos(required(optional_i64(row, "asked_unix_ns"), "asked_unix_ns")?),
        asker: Asker(required_str(row, "asker")?),
        text: required_str(row, "text")?,
    })
}

pub fn encode_space_question(question: &SpaceQuestion) -> StructArray {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(vec![question.question_id.0])),
        Arc::new(Int64Array::from(vec![question.asked.0])),
        Arc::new(StringArray::from(vec![question.asker.0.as_str()])),
        Arc::new(StringArray::from(vec![question.text.as_str()])),
    ];
    StructArray::new(space_question_fields(), columns, None)
}

fn decode_references(row: &StructArray) -> Result<Vec<Reference>, WireError> {
    let list = column(row, "references").as_list::<i32>();
    if list.is_null(0) {
        return Err(WireError::Null { field: "references" });
    }
    let items = list.value(0);
    let items = items.as_struct();
    let part = |name: &str| items.column_by_name(name).expect("checked wire type");
    let (kinds, ids) = (part("kind").as_string::<i32>(), part("id").as_primitive::<UInt64Type>());
    let (xs, ys) = (part("x_m").as_primitive::<Float32Type>(), part("y_m").as_primitive::<Float32Type>());
    (0..items.len())
        .map(|i| {
            let kind: ReferenceKind = required(kinds.is_valid(i).then(|| kinds.value(i)), "references.kind")?.parse()?;
            Ok(Reference {
                referent: Referent::from_wire(kind, ids.is_valid(i).then(|| ids.value(i)))?,
                x_m: required(xs.is_valid(i).then(|| xs.value(i)), "references.x_m")?,
                y_m: required(ys.is_valid(i).then(|| ys.value(i)), "references.y_m")?,
            })
        })
        .collect()
}

pub fn decode_space_answer(array: &dyn Array) -> Result<SpaceAnswer, WireError> {
    let row = checked(array, space_answer_fields(), "space answer")?;
    let revision = column(row, "revision").as_primitive::<UInt32Type>();
    let is_final = column(row, "is_final").as_boolean();
    Ok(SpaceAnswer {
        question_id: QuestionId(required(optional_u64(row, "question_id"), "question_id")?),
        revision: Revision(required(revision.is_valid(0).then(|| revision.value(0)), "revision")?),
        is_final: required(is_final.is_valid(0).then(|| is_final.value(0)), "is_final")?,
        model: required_str(row, "model")?,
        text: required_str(row, "text")?,
        references: decode_references(row)?,
        goal_id: optional_u64(row, "goal_id").map(GoalId),
    })
}

fn references_column(references: &[Reference]) -> ArrayRef {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(references.iter().map(|r| r.referent.kind().wire_name()))),
        Arc::new(references.iter().map(|r| r.referent.wire_id()).collect::<UInt64Array>()),
        Arc::new(references.iter().map(|r| r.x_m).collect::<Float32Array>()),
        Arc::new(references.iter().map(|r| r.y_m).collect::<Float32Array>()),
    ];
    let items = StructArray::new(reference_fields(), columns, None);
    Arc::new(ListArray::new(reference_item(), OffsetBuffer::from_lengths([references.len()]), Arc::new(items), None))
}

pub fn encode_space_answer(answer: &SpaceAnswer) -> StructArray {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(vec![answer.question_id.0])),
        Arc::new(UInt32Array::from(vec![answer.revision.0])),
        Arc::new(BooleanArray::from(vec![answer.is_final])),
        Arc::new(StringArray::from(vec![answer.model.as_str()])),
        Arc::new(StringArray::from(vec![answer.text.as_str()])),
        references_column(&answer.references),
        Arc::new(UInt64Array::from(vec![answer.goal_id.map(|g| g.0)])),
    ];
    StructArray::new(space_answer_fields(), columns, None)
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use dora_node_api::arrow::ipc::reader::FileReader;

    use super::*;

    fn fixture(name: &str) -> StructArray {
        let path = format!("{}/../../../lib/dora-rig-messages/dora_rig_messages/fixtures/{name}.arrow", env!("CARGO_MANIFEST_DIR"));
        let mut reader = FileReader::try_new(File::open(&path).unwrap_or_else(|e| panic!("{path}: {e}")), None).unwrap();
        StructArray::from(reader.next().unwrap().unwrap())
    }

    fn metres(value: f32) -> NonNegative {
        NonNegative::new("test", value).unwrap()
    }

    #[test]
    fn the_object_goal_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("nav_goal_object");
        let goal = decode_nav_goal(&golden).unwrap();
        assert_eq!(
            goal,
            NavGoal {
                goal_id: GoalId(7),
                target: GoalTarget::Object(ObjectId(42)),
                standoff_m: metres(0.5),
                max_speed_m_s: metres(0.25),
                deadline: UnixNanos(1_762_560_060_000_000_000),
                map_version: MapVersion(412),
            }
        );
        assert_eq!(encode_nav_goal(&goal), golden);
    }

    #[test]
    fn the_pose_goal_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("nav_goal_pose");
        let goal = decode_nav_goal(&golden).unwrap();
        assert_eq!(goal.target, GoalTarget::Pose(MapPose { x_m: 3.25, y_m: -0.75, yaw_rad: 1.5 }));
        assert_eq!(encode_nav_goal(&goal), golden);
    }

    #[test]
    fn the_stop_goal_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("nav_goal_stop");
        let goal = decode_nav_goal(&golden).unwrap();
        assert_eq!((goal.goal_id, goal.target), (GoalId(9), GoalTarget::Stop));
        assert_eq!(encode_nav_goal(&goal), golden);
    }

    #[test]
    fn the_goal_status_fixtures_decode_and_encode_back_to_themselves() {
        for (name, state) in [
            ("goal_status_active", GoalState::Active),
            ("goal_status_blocked", GoalState::Blocked(GoalReason::PersonInPath)),
        ] {
            let golden = fixture(name);
            let status = decode_goal_status(&golden).unwrap();
            assert_eq!((status.goal_id, status.state), (GoalId(7), state), "{name}");
            assert_eq!(encode_goal_status(&status), golden, "{name}");
        }
    }

    #[test]
    fn every_goal_target_survives_the_trip() {
        let pose = MapPose { x_m: 1.0, y_m: -2.0, yaw_rad: 0.5 };
        for target in [
            GoalTarget::Pose(pose),
            GoalTarget::Object(ObjectId(3)),
            GoalTarget::Place(PlaceId(4)),
            GoalTarget::FollowTrack(TrackId(5)),
            GoalTarget::LookAt(LookTarget::Object(ObjectId(6))),
            GoalTarget::LookAt(LookTarget::Point(pose)),
            GoalTarget::Stop,
        ] {
            let goal = NavGoal {
                goal_id: GoalId(1),
                target,
                standoff_m: metres(0.25),
                max_speed_m_s: metres(0.5),
                deadline: UnixNanos(10),
                map_version: MapVersion(2),
            };
            assert_eq!(decode_nav_goal(&encode_nav_goal(&goal)).unwrap(), goal);
        }
    }

    #[test]
    fn a_goal_status_of_another_wire_type_is_rejected() {
        let error = decode_goal_status(&fixture("nav_goal_object")).unwrap_err();
        assert!(matches!(error, WireError::WrongType { message: "goal status", .. }), "{error}");
    }

    #[test]
    fn a_negative_standoff_is_rejected() {
        assert_eq!(NonNegative::new("standoff_m", -0.5), Err(WireError::Negative { field: "standoff_m", value: -0.5 }));
        assert!(NonNegative::new("standoff_m", f32::NAN).is_err());
    }

    #[test]
    fn a_reason_on_a_state_that_takes_none_is_rejected() {
        assert_eq!(
            GoalState::from_wire("succeeded", Some(GoalReason::NoPath)),
            Err(WireError::ReasonDoesNotFitState("succeeded".to_owned()))
        );
        assert_eq!(GoalState::from_wire("failed", None), Err(WireError::ReasonDoesNotFitState("failed".to_owned())));
    }

    #[test]
    fn an_unknown_kind_is_rejected() {
        assert_eq!(
            "teleport".parse::<GoalKind>(),
            Err(WireError::UnknownName { field: "kind", name: "teleport".to_owned() })
        );
    }

    #[test]
    fn the_space_question_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("space_question");
        let question = decode_space_question(&golden).unwrap();
        assert_eq!(
            question,
            SpaceQuestion {
                question_id: QuestionId(31),
                asked: UnixNanos(1_762_560_000_250_000_000),
                asker: Asker("planner".to_owned()),
                text: "Where is my blue mug?".to_owned(),
            }
        );
        assert_eq!(encode_space_question(&question), golden);
    }

    #[test]
    fn the_first_space_answer_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("space_answer_first");
        let answer = decode_space_answer(&golden).unwrap();
        assert_eq!(
            answer,
            SpaceAnswer {
                question_id: QuestionId(31),
                revision: Revision::FIRST,
                is_final: false,
                model: "claude-haiku-5-5".to_owned(),
                text: "The blue mug o51 is on the kitchen table o42; I am heading to it.".to_owned(),
                references: vec![
                    Reference { referent: Referent::Object(ObjectId(51)), x_m: 2.25, y_m: 1.5 },
                    Reference { referent: Referent::Object(ObjectId(42)), x_m: 2.0, y_m: 1.5 },
                ],
                goal_id: Some(GoalId(17)),
            }
        );
        assert_eq!(encode_space_answer(&answer), golden);
    }

    #[test]
    fn the_final_space_answer_fixture_carries_a_room_a_pose_and_no_goal() {
        let golden = fixture("space_answer_final");
        let answer = decode_space_answer(&golden).unwrap();
        assert_eq!((answer.revision, answer.is_final, answer.goal_id), (Revision(2), true, None));
        assert_eq!(
            answer.references,
            vec![
                Reference { referent: Referent::Room(RoomId(3)), x_m: 1.5, y_m: 2.75 },
                Reference { referent: Referent::Pose, x_m: 0.75, y_m: 2.5 },
            ]
        );
        assert_eq!(encode_space_answer(&answer), golden);
    }

    #[test]
    fn an_answer_without_references_survives_the_trip() {
        let answer = SpaceAnswer {
            question_id: QuestionId(1),
            revision: Revision::FIRST.next(),
            is_final: true,
            model: "claude-opus-5-5".to_owned(),
            text: "Nothing like that here.".to_owned(),
            references: Vec::new(),
            goal_id: None,
        };
        assert_eq!(decode_space_answer(&encode_space_answer(&answer)).unwrap(), answer);
    }

    #[test]
    fn a_reference_whose_id_does_not_fit_its_kind_is_rejected() {
        assert_eq!(Referent::from_wire(ReferenceKind::Pose, Some(3)), Err(WireError::IdDoesNotFitReference(ReferenceKind::Pose)));
        assert_eq!(Referent::from_wire(ReferenceKind::Room, None), Err(WireError::IdDoesNotFitReference(ReferenceKind::Room)));
        assert!("doorway".parse::<ReferenceKind>().is_err());
    }

    #[test]
    fn a_space_answer_of_another_wire_type_is_rejected() {
        let error = decode_space_answer(&fixture("space_question")).unwrap_err();
        assert!(matches!(error, WireError::WrongType { message: "space answer", .. }), "{error}");
    }
}
