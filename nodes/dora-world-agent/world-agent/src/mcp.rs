//! The tools the agents call, served over streamable HTTP on 127.0.0.1 from inside the supervisor's process.
//! Every request must carry `X-Agent-Id` and `Authorization: Bearer <token>` matching a registered claude process.
//! Every text answer is a JSON object that starts with `as_of` and `map_version`. Only a process answering a question
//! may call tools; the fast answerer's goals go to the robot at once, a deliberate answerer's are kept as proposals.

use std::net::{Ipv4Addr, SocketAddr};
use std::num::NonZeroUsize;
use std::sync::Arc;

use axum::extract::{Request, State};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use base64::Engine;
use perception_wire::navigation::{GoalId, GoalKind, GoalTarget, LookTarget, MapPose, NonNegative, QuestionId};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::tool::Extension;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, ServerCapabilities, ServerConfig};
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{tool, tool_handler, tool_router, ErrorData, ServerHandler};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use crate::clock;
use crate::config::{AnswererRole, GoalDefaults};
use crate::goals::{
    AnswerGoal, Approach, GoalBook, GoalBookError, GoalLedger, GoalOrigin, GoalSink, GoalSinkClosed, PlannedGoal, SentGoal,
};
use crate::identity::{AgentId, AgentRegistry, Authenticated};
use crate::model::{AnswererId, Duty};
use crate::world::{
    BadEntityRef, EntityRef, FindQuery, Landmark, Point, ReadOnlySql, RoomName, SqlRejected, Stamp, ViewKind,
    ViewRequest, WorldError, WorldModel,
};

/// The name of the server in `--mcp-config`; tools reach the model as `mcp__world__<tool>`.
pub const SERVER_NAME: &str = "world";
pub const AGENT_ID_HEADER: &str = "x-agent-id";
pub const TOOL_NAMES: [&str; 10] = [
    "find",
    "describe_object",
    "clearance_between",
    "route",
    "last_seen",
    "changes_since",
    "render_view",
    "query_sql",
    "send_goal",
    "cancel_goal",
];

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error(transparent)]
    World(#[from] WorldError),
    #[error(transparent)]
    BadId(#[from] BadEntityRef),
    #[error("{0}")]
    BadArgument(String),
    #[error(transparent)]
    Sql(#[from] SqlRejected),
    #[error(transparent)]
    Goal(#[from] GoalBookError),
    #[error(transparent)]
    Sink(#[from] GoalSinkClosed),
}

/// A tool's answer before it becomes MCP content.
#[derive(Debug, PartialEq)]
pub enum ToolAnswer {
    Json(Value),
    /// A JSON caption followed by PNG images.
    Images { caption: Value, pngs: Vec<Vec<u8>> },
}

impl ToolAnswer {
    fn into_result(self) -> CallToolResult {
        match self {
            ToolAnswer::Json(value) => CallToolResult::success(vec![ContentBlock::text(value.to_string())]),
            ToolAnswer::Images { caption, pngs } => {
                let images = pngs
                    .into_iter()
                    .map(|png| ContentBlock::image(base64::engine::general_purpose::STANDARD.encode(png), "image/png"));
                CallToolResult::success(std::iter::once(ContentBlock::text(caption.to_string())).chain(images).collect())
            }
        }
    }
}

fn stamped(stamp: Stamp, key: &str, body: impl Serialize) -> Value {
    let mut object = Map::new();
    object.insert("as_of".to_owned(), Value::String(clock::iso_time(stamp.as_of)));
    object.insert("map_version".to_owned(), Value::String(stamp.map_version.to_string()));
    object.insert(key.to_owned(), serde_json::to_value(body).expect("tool answers serialise"));
    Value::Object(object)
}

fn object_id(text: &str, tool: &str) -> Result<perception_wire::navigation::ObjectId, ToolError> {
    match text.parse::<EntityRef>()? {
        EntityRef::Object(id) => Ok(id),
        other => Err(ToolError::BadArgument(format!("{tool} takes an object id like o42, not {other}"))),
    }
}

fn landmark(text: &str) -> Result<Landmark, ToolError> {
    let entity: EntityRef = text.parse()?;
    entity.landmark().ok_or_else(|| ToolError::BadArgument(format!("{entity} is a people track; use last_seen for tracks")))
}

fn goal_id(text: &str) -> Result<GoalId, ToolError> {
    text.strip_prefix('g')
        .and_then(|digits| digits.parse().ok())
        .map(GoalId)
        .ok_or_else(|| ToolError::BadArgument(format!("{text:?} is not a goal id like g17")))
}

fn non_negative(field: &'static str, value: Option<f32>, default: NonNegative) -> Result<NonNegative, ToolError> {
    value.map_or(Ok(default), |v| NonNegative::new(field, v).map_err(|e| ToolError::BadArgument(e.to_string())))
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FindParams {
    /// Free text describing the object, matched against labels and captions.
    pub text: Option<String>,
    /// An exact label, such as "chair".
    pub label: Option<String>,
    /// Only objects in this room.
    pub room: Option<String>,
    /// Rank by distance from this map x instead of from the robot; needs near_y_m.
    pub near_x_m: Option<f32>,
    pub near_y_m: Option<f32>,
    /// How many to return, 5 when not given.
    pub limit: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ObjectParams {
    /// An object id from the digest, such as o42.
    pub object: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PairParams {
    /// An object id (o42), a place id (p3) or robot.
    pub a: String,
    /// An object id (o42), a place id (p3) or robot.
    pub b: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RouteParams {
    /// An object id (o42), a place id (p3) or robot.
    pub from: String,
    /// An object id (o42), a place id (p3) or robot.
    pub to: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct LastSeenParams {
    /// A people track id (t3) or an object id (o42).
    pub subject: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ChangesSinceParams {
    /// A UTC time like 2026-11-14T07:00:00Z; the digest's as_of is the current time.
    pub since: String,
    pub room: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ViewKindParam {
    #[default]
    TopDown,
    Egocentric,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RenderViewParams {
    #[serde(default)]
    pub kind: ViewKindParam,
    /// Only this room; the whole map when not given.
    pub room: Option<String>,
    /// Object ids to draw in red.
    #[serde(default)]
    pub highlight: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct QuerySqlParams {
    /// One SELECT statement.
    pub sql: String,
}

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GoalKindParam {
    /// Drive to x_m, y_m, facing yaw_deg.
    Pose,
    /// Drive to standoff_m from the target object.
    Object,
    /// Drive to the target place.
    Place,
    /// Follow the target people track.
    FollowTrack,
    /// Turn to face the target object, or the point x_m, y_m.
    LookAt,
    /// Stop the robot.
    Stop,
}

impl From<GoalKindParam> for GoalKind {
    fn from(kind: GoalKindParam) -> Self {
        match kind {
            GoalKindParam::Pose => GoalKind::Pose,
            GoalKindParam::Object => GoalKind::Object,
            GoalKindParam::Place => GoalKind::Place,
            GoalKindParam::FollowTrack => GoalKind::FollowTrack,
            GoalKindParam::LookAt => GoalKind::LookAt,
            GoalKindParam::Stop => GoalKind::Stop,
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SendGoalParams {
    pub kind: GoalKindParam,
    /// The id the kind needs: an object (o42), a place (p3) or a people track (t7).
    pub target: Option<String>,
    pub x_m: Option<f32>,
    pub y_m: Option<f32>,
    pub yaw_deg: Option<f32>,
    pub standoff_m: Option<f32>,
    pub max_speed_m_s: Option<f32>,
    /// Seconds the robot may take before it gives up.
    pub deadline_s: Option<f32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CancelGoalParams {
    /// The goal id send_goal returned, such as g17.
    pub goal: String,
}

/// A tool call from a process answering a question.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnswerCall {
    pub agent: AgentId,
    pub question: QuestionId,
    pub answerer: AnswererId,
    pub role: AnswererRole,
}

/// What the tools work on: the world model, the goals sent and proposed, and where goals go.
pub struct ToolContext<W, G> {
    pub world: Arc<W>,
    pub goals: Arc<GoalBook>,
    pub ledger: Arc<GoalLedger>,
    pub sink: Arc<G>,
    pub goal_defaults: GoalDefaults,
    pub sql_row_limit: NonZeroUsize,
}

impl<W: WorldModel, G: GoalSink> ToolContext<W, G> {
    pub async fn find(&self, params: FindParams) -> Result<ToolAnswer, ToolError> {
        let near = match (params.near_x_m, params.near_y_m) {
            (Some(x_m), Some(y_m)) => Some(Point { x_m, y_m }),
            (None, None) => None,
            _ => return Err(ToolError::BadArgument("near needs both near_x_m and near_y_m".to_owned())),
        };
        let limit = params.limit.map(|l| NonZeroUsize::new(l as usize).ok_or_else(|| ToolError::BadArgument("limit must be at least 1".to_owned()))).transpose()?;
        let query = FindQuery { text: params.text, label: params.label, room: params.room.map(RoomName), near, limit };
        let stamp = self.world.stamp().await?;
        Ok(ToolAnswer::Json(stamped(stamp, "objects", self.world.find(query).await?)))
    }

    pub async fn describe_object(&self, params: ObjectParams) -> Result<ToolAnswer, ToolError> {
        let id = object_id(&params.object, "describe_object")?;
        let stamp = self.world.stamp().await?;
        let description = self.world.describe_object(id).await?;
        let s = &description.summary;
        let caption = stamped(
            stamp,
            "object",
            serde_json::json!({
                "id": s.id.to_string(),
                "label": s.label,
                "caption": description.caption,
                "room": s.room,
                "position": s.position,
                "size": s.size,
                "last_seen": clock::iso_time(s.last_seen),
                "relations": description.relations,
                "history": description.history,
                "views": description.views.len(),
            }),
        );
        Ok(match description.views.is_empty() {
            true => ToolAnswer::Json(caption),
            false => ToolAnswer::Images { caption, pngs: description.views.into_iter().map(|v| v.0).collect() },
        })
    }

    pub async fn clearance_between(&self, params: PairParams) -> Result<ToolAnswer, ToolError> {
        let (a, b) = (landmark(&params.a)?, landmark(&params.b)?);
        let stamp = self.world.stamp().await?;
        Ok(ToolAnswer::Json(stamped(stamp, "clearance", self.world.clearance_between(a, b).await?)))
    }

    pub async fn route(&self, params: RouteParams) -> Result<ToolAnswer, ToolError> {
        let (from, to) = (landmark(&params.from)?, landmark(&params.to)?);
        let stamp = self.world.stamp().await?;
        Ok(ToolAnswer::Json(stamped(stamp, "route", self.world.route(from, to).await?)))
    }

    pub async fn last_seen(&self, params: LastSeenParams) -> Result<ToolAnswer, ToolError> {
        let subject: EntityRef = params.subject.parse()?;
        let stamp = self.world.stamp().await?;
        Ok(ToolAnswer::Json(stamped(stamp, "sighting", self.world.last_seen(subject).await?)))
    }

    pub async fn changes_since(&self, params: ChangesSinceParams) -> Result<ToolAnswer, ToolError> {
        let since = clock::parse_iso_time(&params.since)
            .ok_or_else(|| ToolError::BadArgument(format!("{:?} is not a UTC time like 2026-11-14T07:00:00Z", params.since)))?;
        let stamp = self.world.stamp().await?;
        Ok(ToolAnswer::Json(stamped(stamp, "changes", self.world.changes_since(since, params.room.map(RoomName)).await?)))
    }

    pub async fn render_view(&self, params: RenderViewParams) -> Result<ToolAnswer, ToolError> {
        let highlight = params.highlight.iter().map(|id| object_id(id, "render_view highlight")).collect::<Result<Vec<_>, _>>()?;
        let kind = match params.kind {
            ViewKindParam::TopDown => ViewKind::TopDown,
            ViewKindParam::Egocentric => ViewKind::Egocentric,
        };
        let request = ViewRequest { kind, room: params.room.map(RoomName), highlight };
        let stamp = self.world.stamp().await?;
        let png = self.world.render_view(request).await?;
        let caption = stamped(stamp, "view", "north up; rooms grey, objects brown, highlighted red, people green, robot blue");
        Ok(ToolAnswer::Images { caption, pngs: vec![png.0] })
    }

    pub async fn query_sql(&self, params: QuerySqlParams) -> Result<ToolAnswer, ToolError> {
        let sql = ReadOnlySql::parse(&params.sql)?;
        let stamp = self.world.stamp().await?;
        Ok(ToolAnswer::Json(stamped(stamp, "rows", self.world.query_sql(sql, self.sql_row_limit).await?)))
    }

    fn goal_target(params: &SendGoalParams) -> Result<GoalTarget, ToolError> {
        let entity = params.target.as_deref().map(str::parse::<EntityRef>).transpose()?;
        let point = match (params.x_m, params.y_m) {
            (Some(x_m), Some(y_m)) => Some((x_m, y_m)),
            (None, None) => None,
            _ => return Err(ToolError::BadArgument("a point needs both x_m and y_m".to_owned())),
        };
        let kind = GoalKind::from(params.kind);
        let mismatch = || ToolError::BadArgument(format!("a {} goal takes {}", kind.wire_name(), match kind {
            GoalKind::Pose => "x_m, y_m and yaw_deg",
            GoalKind::Object => "an object id as target",
            GoalKind::Place => "a place id as target",
            GoalKind::FollowTrack => "a people track id as target",
            GoalKind::LookAt => "an object id as target, or x_m and y_m",
            GoalKind::Stop => "no target",
        }));
        match (kind, entity, point, params.yaw_deg) {
            (GoalKind::Pose, None, Some((x_m, y_m)), Some(yaw_deg)) => {
                Ok(GoalTarget::Pose(MapPose { x_m, y_m, yaw_rad: yaw_deg.to_radians() }))
            }
            (GoalKind::Object, Some(EntityRef::Object(id)), None, None) => Ok(GoalTarget::Object(id)),
            (GoalKind::Place, Some(EntityRef::Place(id)), None, None) => Ok(GoalTarget::Place(id)),
            (GoalKind::FollowTrack, Some(EntityRef::Track(id)), None, None) => Ok(GoalTarget::FollowTrack(id)),
            (GoalKind::LookAt, Some(EntityRef::Object(id)), None, None) => Ok(GoalTarget::LookAt(LookTarget::Object(id))),
            (GoalKind::LookAt, None, Some((x_m, y_m)), None) => {
                Ok(GoalTarget::LookAt(LookTarget::Point(MapPose { x_m, y_m, yaw_rad: 0.0 })))
            }
            (GoalKind::Stop, None, None, None) => Ok(GoalTarget::Stop),
            _ => Err(mismatch()),
        }
    }

    /// The fast answerer's goal goes to the robot now; a deliberate answerer's is recorded as its proposal.
    fn act(&self, call: &AnswerCall, plan: PlannedGoal) -> Result<Value, ToolError> {
        let kind = plan.target.kind().wire_name();
        match call.role {
            AnswererRole::Fast => {
                let goal = self.goals.issue(plan, call.question);
                self.sink.send(SentGoal { goal, origin: GoalOrigin::Answerer(call.answerer) })?;
                self.ledger.record(call.question, call.answerer, AnswerGoal::Sent(goal));
                tracing::info!(agent = %call.agent, question = %call.question, goal = %goal.goal_id, kind, "goal sent");
                Ok(serde_json::json!({
                    "id": goal.goal_id.to_string(),
                    "kind": kind,
                    "standoff_m": goal.standoff_m.get(),
                    "max_speed_m_s": goal.max_speed_m_s.get(),
                    "deadline": clock::iso_time(goal.deadline),
                    "state": "sent; the robot is acting on it",
                }))
            }
            AnswererRole::Deliberate => {
                self.ledger.record(call.question, call.answerer, AnswerGoal::Proposed(plan));
                tracing::info!(agent = %call.agent, question = %call.question, kind, "goal proposed");
                Ok(serde_json::json!({
                    "kind": kind,
                    "state": "recorded as your proposal; it moves the robot only if your answer is adopted",
                }))
            }
        }
    }

    pub async fn send_goal(&self, call: &AnswerCall, params: SendGoalParams) -> Result<ToolAnswer, ToolError> {
        let target = Self::goal_target(&params)?;
        self.world.check_target(target).await?;
        let defaults = self.goal_defaults;
        let within = match params.deadline_s {
            Some(s) => std::time::Duration::try_from_secs_f32(s)
                .map_err(|e| ToolError::BadArgument(format!("deadline_s {s}: {e}")))?,
            None => defaults.deadline,
        };
        let approach = Approach {
            standoff_m: non_negative("standoff_m", params.standoff_m, defaults.standoff_m)?,
            max_speed_m_s: non_negative("max_speed_m_s", params.max_speed_m_s, defaults.max_speed_m_s)?,
            within,
        };
        let stamp = self.world.stamp().await?;
        let goal = self.act(call, PlannedGoal { target, approach, map_version: stamp.map_version })?;
        Ok(ToolAnswer::Json(stamped(stamp, "goal", goal)))
    }

    pub async fn cancel_goal(&self, call: &AnswerCall, params: CancelGoalParams) -> Result<ToolAnswer, ToolError> {
        let id = goal_id(&params.goal)?;
        self.goals.check_running(id)?;
        let stamp = self.world.stamp().await?;
        let plan = PlannedGoal { target: GoalTarget::Stop, approach: self.goal_defaults.approach(), map_version: stamp.map_version };
        let stop = self.act(call, plan)?;
        Ok(ToolAnswer::Json(stamped(stamp, "cancelled", serde_json::json!({"goal": id.to_string(), "stop": stop}))))
    }
}

fn caller(parts: &http::request::Parts) -> Result<AnswerCall, ErrorData> {
    let authenticated = parts
        .extensions
        .get::<Authenticated>()
        .cloned()
        .ok_or_else(|| ErrorData::invalid_request("the request carries no authenticated agent", None))?;
    let caller = authenticated.caller().map_err(|error| ErrorData::invalid_request(error.to_string(), None))?;
    match caller.duty {
        Duty::Answer { question, answerer, role } => Ok(AnswerCall { agent: caller.agent, question, answerer, role }),
        other => Err(ErrorData::invalid_request(format!("agent {} serves {other:?}, which has no tools", caller.agent), None)),
    }
}

fn answer(result: Result<ToolAnswer, ToolError>) -> Result<CallToolResult, ErrorData> {
    Ok(match result {
        Ok(answer) => answer.into_result(),
        Err(error) => CallToolResult::error(vec![ContentBlock::text(error.to_string())]),
    })
}

/// The MCP face of `ToolContext`.
pub struct WorldTools<W, G> {
    context: Arc<ToolContext<W, G>>,
    tool_router: ToolRouter<Self>,
}

impl<W, G> Clone for WorldTools<W, G> {
    fn clone(&self) -> Self {
        Self { context: self.context.clone(), tool_router: self.tool_router.clone() }
    }
}

#[tool_router]
impl<W: WorldModel, G: GoalSink> WorldTools<W, G> {
    pub fn new(context: Arc<ToolContext<W, G>>) -> Self {
        Self { context, tool_router: Self::tool_router() }
    }

    #[tool(description = "Search objects by free text, label, room or nearness; returns ids, rooms, positions, distance from the robot and when last seen.")]
    async fn find(&self, Extension(parts): Extension<http::request::Parts>, Parameters(params): Parameters<FindParams>) -> Result<CallToolResult, ErrorData> {
        caller(&parts)?;
        answer(self.context.find(params).await)
    }

    #[tool(description = "An object's geometry, caption, relations to other objects, where it stood over time, and its best views as images.")]
    async fn describe_object(&self, Extension(parts): Extension<http::request::Parts>, Parameters(params): Parameters<ObjectParams>) -> Result<CallToolResult, ErrorData> {
        caller(&parts)?;
        answer(self.context.describe_object(params).await)
    }

    #[tool(description = "The narrowest free gap between two objects, places or the robot, where it is, what the robot needs, and whether it fits.")]
    async fn clearance_between(&self, Extension(parts): Extension<http::request::Parts>, Parameters(params): Parameters<PairParams>) -> Result<CallToolResult, ErrorData> {
        caller(&parts)?;
        answer(self.context.clearance_between(params).await)
    }

    #[tool(description = "The path between two objects, places or the robot: length, places and rooms passed, narrowest clearance, estimated time.")]
    async fn route(&self, Extension(parts): Extension<http::request::Parts>, Parameters(params): Parameters<RouteParams>) -> Result<CallToolResult, ErrorData> {
        caller(&parts)?;
        answer(self.context.route(params).await)
    }

    #[tool(description = "Where and when a people track or an object was last seen, with the heading and the doorways ahead for a person.")]
    async fn last_seen(&self, Extension(parts): Extension<http::request::Parts>, Parameters(params): Parameters<LastSeenParams>) -> Result<CallToolResult, ErrorData> {
        caller(&parts)?;
        answer(self.context.last_seen(params).await)
    }

    #[tool(description = "Objects that appeared, moved or disappeared since a time, optionally in one room, each with the window it happened in.")]
    async fn changes_since(&self, Extension(parts): Extension<http::request::Parts>, Parameters(params): Parameters<ChangesSinceParams>) -> Result<CallToolResult, ErrorData> {
        caller(&parts)?;
        answer(self.context.changes_since(params).await)
    }

    #[tool(description = "A picture of the map, top-down or from the robot's camera, with chosen objects highlighted.")]
    async fn render_view(&self, Extension(parts): Extension<http::request::Parts>, Parameters(params): Parameters<RenderViewParams>) -> Result<CallToolResult, ErrorData> {
        caller(&parts)?;
        answer(self.context.render_view(params).await)
    }

    #[tool(description = "Run one read-only SELECT on the world store, for questions the other tools do not answer.")]
    async fn query_sql(&self, Extension(parts): Extension<http::request::Parts>, Parameters(params): Parameters<QuerySqlParams>) -> Result<CallToolResult, ErrorData> {
        caller(&parts)?;
        answer(self.context.query_sql(params).await)
    }

    #[tool(description = "Send the robot a navigation goal by id: an object, a place, a people track to follow, a point to look at, a pose, or stop.")]
    async fn send_goal(&self, Extension(parts): Extension<http::request::Parts>, Parameters(params): Parameters<SendGoalParams>) -> Result<CallToolResult, ErrorData> {
        let caller = caller(&parts)?;
        answer(self.context.send_goal(&caller, params).await)
    }

    #[tool(description = "Stop the robot working on a goal that is still running.")]
    async fn cancel_goal(&self, Extension(parts): Extension<http::request::Parts>, Parameters(params): Parameters<CancelGoalParams>) -> Result<CallToolResult, ErrorData> {
        let caller = caller(&parts)?;
        answer(self.context.cancel_goal(&caller, params).await)
    }
}

#[tool_handler]
impl<W: WorldModel, G: GoalSink> ServerHandler for WorldTools<W, G> {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }
}

fn bearer(request: &Request) -> Option<&str> {
    request.headers().get(http::header::AUTHORIZATION)?.to_str().ok()?.strip_prefix("Bearer ")
}

async fn authenticate(State(registry): State<Arc<AgentRegistry>>, mut request: Request, next: Next) -> Response {
    let agent = request.headers().get(AGENT_ID_HEADER).and_then(|v| v.to_str().ok());
    match registry.verify(agent, bearer(&request)) {
        Ok(authenticated) => {
            request.extensions_mut().insert(authenticated);
            next.run(request).await
        }
        Err(error) => {
            tracing::warn!(%error, "MCP request refused");
            (http::StatusCode::UNAUTHORIZED, error.to_string()).into_response()
        }
    }
}

/// The MCP service behind the authentication check, mounted at `/mcp`.
pub fn router<W: WorldModel, G: GoalSink>(context: Arc<ToolContext<W, G>>, registry: Arc<AgentRegistry>, cancel: CancellationToken) -> axum::Router {
    let config = StreamableHttpServerConfig::default().with_cancellation_token(cancel);
    let service: StreamableHttpService<WorldTools<W, G>, LocalSessionManager> =
        StreamableHttpService::new(move || Ok(WorldTools::new(context.clone())), Default::default(), config);
    axum::Router::new()
        .nest_service("/mcp", service)
        .layer(middleware::from_fn_with_state(registry, authenticate))
}

/// Where the claude processes reach the tools.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpUrl(pub String);

/// The running endpoint; it stops when dropped.
pub struct McpEndpoint {
    pub url: McpUrl,
    cancel: CancellationToken,
}

impl McpEndpoint {
    pub async fn start(router: axum::Router, port: u16, cancel: CancellationToken) -> std::io::Result<Self> {
        let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, port))).await?;
        let address = listener.local_addr()?;
        let shutdown = cancel.clone();
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, router).with_graceful_shutdown(shutdown.cancelled_owned()).await {
                tracing::error!(%error, "the MCP endpoint stopped");
            }
        });
        Ok(Self { url: McpUrl(format!("http://{address}/mcp")), cancel })
    }
}

impl Drop for McpEndpoint {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use http_body_util::BodyExt;
    use perception_wire::navigation::{GoalState, GoalStatus, ObjectId, TrackId};
    use tokio::sync::mpsc;
    use tower::ServiceExt;

    use super::*;
    use crate::fake_world::FakeWorld;
    use crate::goals::ChannelGoalSink;

    fn context() -> (Arc<ToolContext<FakeWorld, ChannelGoalSink>>, mpsc::UnboundedReceiver<SentGoal>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        let context = ToolContext {
            world: Arc::new(FakeWorld::house_at(perception_wire::navigation::UnixNanos(1_794_610_000_000_000_000))),
            goals: Arc::new(GoalBook::default()),
            ledger: Arc::new(GoalLedger::default()),
            sink: Arc::new(ChannelGoalSink(sender)),
            goal_defaults: GoalDefaults {
                standoff_m: NonNegative::new("standoff_m", 0.5).unwrap(),
                max_speed_m_s: NonNegative::new("max_speed_m_s", 0.3).unwrap(),
                deadline: Duration::from_secs(120),
            },
            sql_row_limit: NonZeroUsize::new(10).unwrap(),
        };
        (Arc::new(context), receiver)
    }

    const QUESTION: QuestionId = QuestionId(31);

    fn caller() -> AnswerCall {
        AnswerCall { agent: AgentId("agent-1".to_owned()), question: QUESTION, answerer: AnswererId(0), role: AnswererRole::Fast }
    }

    fn deliberate() -> AnswerCall {
        AnswerCall { agent: AgentId("agent-2".to_owned()), question: QUESTION, answerer: AnswererId(1), role: AnswererRole::Deliberate }
    }

    fn json(answer: ToolAnswer) -> Value {
        match answer {
            ToolAnswer::Json(value) => value,
            other => panic!("expected JSON, got {other:?}"),
        }
    }

    fn send_params(kind: GoalKindParam, target: Option<&str>) -> SendGoalParams {
        SendGoalParams {
            kind,
            target: target.map(str::to_owned),
            x_m: None,
            y_m: None,
            yaw_deg: None,
            standoff_m: None,
            max_speed_m_s: None,
            deadline_s: None,
        }
    }

    #[tokio::test]
    async fn every_answer_starts_with_its_time_and_map_version() {
        let (context, _) = context();
        let found = json(context.find(FindParams { text: Some("sofa".to_owned()), label: None, room: None, near_x_m: None, near_y_m: None, limit: None }).await.unwrap());
        let keys: Vec<&String> = found.as_object().unwrap().keys().collect();
        assert_eq!(keys, ["as_of", "map_version", "objects"]);
        assert_eq!(found["map_version"], "v412");
        assert_eq!(found["objects"][0]["id"], "o12");
    }

    #[tokio::test]
    async fn the_route_tool_answers_from_the_world_model() {
        let (context, _) = context();
        let route = json(context.route(RouteParams { from: "robot".to_owned(), to: "o12".to_owned() }).await.unwrap());
        assert_eq!(route["route"]["via"], serde_json::json!(["p2", "p3", "p4", "p5"]));
    }

    #[tokio::test]
    async fn clearance_takes_landmarks_but_not_tracks() {
        let (context, _) = context();
        let gap = json(context.clearance_between(PairParams { a: "o12".to_owned(), b: "o13".to_owned() }).await.unwrap());
        assert_eq!(gap["clearance"]["passable"], false);
        let error = context.clearance_between(PairParams { a: "t7".to_owned(), b: "o13".to_owned() }).await.unwrap_err();
        assert_eq!(error.to_string(), "t7 is a people track; use last_seen for tracks");
    }

    #[tokio::test]
    async fn describing_an_unknown_object_is_a_tool_error() {
        let (context, _) = context();
        let error = context.describe_object(ObjectParams { object: "o999".to_owned() }).await.unwrap_err();
        assert_eq!(error.to_string(), "no object o999");
        let error = context.describe_object(ObjectParams { object: "p3".to_owned() }).await.unwrap_err();
        assert_eq!(error.to_string(), "describe_object takes an object id like o42, not p3");
    }

    #[tokio::test]
    async fn changes_since_takes_a_utc_time() {
        let (context, _) = context();
        let changes = json(context.changes_since(ChangesSinceParams { since: "2026-11-13T21:00:00Z".to_owned(), room: None }).await.unwrap());
        assert_eq!(changes["changes"].as_array().unwrap().len(), 2);
        let error = context.changes_since(ChangesSinceParams { since: "this morning".to_owned(), room: None }).await.unwrap_err();
        assert!(matches!(error, ToolError::BadArgument(_)), "{error}");
    }

    #[tokio::test]
    async fn render_view_answers_with_a_png_image() {
        let (context, _) = context();
        let answer = context.render_view(RenderViewParams { kind: ViewKindParam::TopDown, room: Some("kitchen".to_owned()), highlight: vec!["o42".to_owned()] }).await.unwrap();
        let ToolAnswer::Images { pngs, .. } = &answer else { panic!("{answer:?}") };
        assert!(pngs[0].starts_with(b"\x89PNG\r\n\x1a\n"));
        let result = answer.into_result();
        assert!(matches!(&result.content[1], ContentBlock::Image(image) if image.mime_type == "image/png"));
    }

    #[tokio::test]
    async fn query_sql_refuses_writes_before_reaching_the_store() {
        let (context, _) = context();
        let error = context.query_sql(QuerySqlParams { sql: "DELETE FROM object".to_owned() }).await.unwrap_err();
        assert!(matches!(error, ToolError::Sql(SqlRejected::NotASelect)));
        let error = context.query_sql(QuerySqlParams { sql: "SELECT 1".to_owned() }).await.unwrap_err();
        assert!(matches!(error, ToolError::World(WorldError::Unsupported(_))), "{error}");
    }

    #[tokio::test]
    async fn send_goal_sends_a_nav_goal_for_a_known_object() {
        let (context, mut goals) = context();
        let answer = json(context.send_goal(&caller(), send_params(GoalKindParam::Object, Some("o42"))).await.unwrap());
        let sent = goals.try_recv().unwrap();
        assert_eq!(sent.origin, GoalOrigin::Answerer(AnswererId(0)));
        let goal = sent.goal;
        assert_eq!(goal.target, GoalTarget::Object(ObjectId(42)));
        assert_eq!(goal.standoff_m.get(), 0.5);
        assert_eq!(goal.map_version, perception_wire::navigation::MapVersion(412));
        assert_eq!(answer["goal"]["id"], goal.goal_id.to_string());
        assert_eq!(context.goals.running().unwrap().question, QUESTION);
        assert_eq!(context.ledger.take(QUESTION, AnswererId(0)), Some(AnswerGoal::Sent(goal)));
    }

    #[tokio::test]
    async fn a_deliberate_answerers_goal_is_only_a_proposal() {
        let (context, mut goals) = context();
        let answer = json(context.send_goal(&deliberate(), send_params(GoalKindParam::Place, Some("p5"))).await.unwrap());
        assert!(goals.try_recv().is_err());
        assert_eq!(context.goals.running(), None);
        assert!(answer["goal"]["state"].as_str().unwrap().starts_with("recorded as your proposal"), "{answer}");
        let Some(AnswerGoal::Proposed(plan)) = context.ledger.take(QUESTION, AnswererId(1)) else { panic!("no proposal") };
        assert_eq!(plan.target, GoalTarget::Place(perception_wire::navigation::PlaceId(5)));
        assert_eq!(plan.map_version, perception_wire::navigation::MapVersion(412));
    }

    #[tokio::test]
    async fn a_deliberate_answerers_cancel_proposes_a_stop_and_sends_nothing() {
        let (context, mut goals) = context();
        context.send_goal(&caller(), send_params(GoalKindParam::Object, Some("o42"))).await.unwrap();
        let goal = goals.try_recv().unwrap().goal;
        context.cancel_goal(&deliberate(), CancelGoalParams { goal: goal.goal_id.to_string() }).await.unwrap();
        assert!(goals.try_recv().is_err());
        assert_eq!(context.goals.running().unwrap().goal, goal);
        assert_eq!(context.ledger.take(QUESTION, AnswererId(1)).unwrap().target(), GoalTarget::Stop);
    }

    #[tokio::test]
    async fn send_goal_refuses_invented_ids_and_mismatched_targets() {
        let (context, mut goals) = context();
        let error = context.send_goal(&caller(), send_params(GoalKindParam::Object, Some("o999"))).await.unwrap_err();
        assert_eq!(error.to_string(), "no object o999");
        let error = context.send_goal(&caller(), send_params(GoalKindParam::FollowTrack, Some("o42"))).await.unwrap_err();
        assert_eq!(error.to_string(), "a follow_track goal takes a people track id as target");
        let error = context.send_goal(&caller(), send_params(GoalKindParam::Pose, None)).await.unwrap_err();
        assert_eq!(error.to_string(), "a pose goal takes x_m, y_m and yaw_deg");
        assert!(goals.try_recv().is_err());
        let followed = context.send_goal(&caller(), send_params(GoalKindParam::FollowTrack, Some("t7"))).await;
        assert!(followed.is_ok());
        assert_eq!(goals.try_recv().unwrap().goal.target, GoalTarget::FollowTrack(TrackId(7)));
    }

    #[tokio::test]
    async fn cancel_goal_sends_a_stop_only_for_the_running_goal() {
        let (context, mut goals) = context();
        context.send_goal(&caller(), send_params(GoalKindParam::Object, Some("o42"))).await.unwrap();
        let goal = goals.try_recv().unwrap().goal;
        let answer = json(context.cancel_goal(&caller(), CancelGoalParams { goal: goal.goal_id.to_string() }).await.unwrap());
        let stop = goals.try_recv().unwrap().goal;
        assert_eq!(stop.target, GoalTarget::Stop);
        assert_eq!(answer["cancelled"]["stop"]["id"], stop.goal_id.to_string());
        let error = context.cancel_goal(&caller(), CancelGoalParams { goal: goal.goal_id.to_string() }).await.unwrap_err();
        assert_eq!(error.to_string(), format!("goal {} is not running", goal.goal_id));
    }

    #[tokio::test]
    async fn a_finished_goal_cannot_be_cancelled() {
        let (context, mut goals) = context();
        context.send_goal(&caller(), send_params(GoalKindParam::Place, Some("p5"))).await.unwrap();
        let goal = goals.try_recv().unwrap().goal;
        context.goals.update(GoalStatus { goal_id: goal.goal_id, state: GoalState::Succeeded, updated: clock::now() }).unwrap();
        let error = context.cancel_goal(&caller(), CancelGoalParams { goal: goal.goal_id.to_string() }).await.unwrap_err();
        assert!(matches!(error, ToolError::Goal(GoalBookError::NotRunning(_))));
    }

    fn initialize(agent: Option<&str>, token: Option<&str>) -> http::Request<axum::body::Body> {
        let mut request = http::Request::post("/mcp")
            .header(http::header::HOST, "127.0.0.1")
            .header(http::header::CONTENT_TYPE, "application/json")
            .header(http::header::ACCEPT, "application/json, text/event-stream");
        if let Some(agent) = agent {
            request = request.header(AGENT_ID_HEADER, agent);
        }
        if let Some(token) = token {
            request = request.header(http::header::AUTHORIZATION, format!("Bearer {token}"));
        }
        let body = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}"#;
        request.body(axum::body::Body::from(body)).unwrap()
    }

    #[test]
    fn only_a_process_answering_a_question_may_call_tools() {
        let (mut parts, _) = http::Request::new(()).into_parts();
        assert!(super::caller(&parts).is_err());
        let agent = AgentId("agent-1".to_owned());
        parts.extensions.insert(Authenticated { agent: agent.clone(), duty: None });
        assert_eq!(super::caller(&parts).unwrap_err().message, "agent agent-1 serves no question yet");
        parts.extensions.insert(Authenticated { agent: agent.clone(), duty: Some(Duty::Reconcile { question: QUESTION }) });
        assert!(super::caller(&parts).is_err());
        let duty = Duty::Answer { question: QUESTION, answerer: AnswererId(0), role: AnswererRole::Fast };
        parts.extensions.insert(Authenticated { agent, duty: Some(duty) });
        assert_eq!(super::caller(&parts).unwrap(), caller());
    }

    #[tokio::test]
    async fn the_endpoint_admits_only_a_registered_agent_with_its_own_token() {
        let (context, _) = context();
        let registry = Arc::new(AgentRegistry::default());
        let agent = AgentId("agent-1".to_owned());
        let token = registry.register(agent);
        let other_token = registry.register(AgentId("agent-2".to_owned()));
        let app = router(context, registry.clone(), CancellationToken::new());

        let status = |response: Response| response.status();
        assert_eq!(status(app.clone().oneshot(initialize(None, None)).await.unwrap()), http::StatusCode::UNAUTHORIZED);
        assert_eq!(status(app.clone().oneshot(initialize(Some("agent-1"), Some(other_token.expose()))).await.unwrap()), http::StatusCode::UNAUTHORIZED);
        assert_eq!(status(app.clone().oneshot(initialize(Some("agent-3"), Some(token.expose()))).await.unwrap()), http::StatusCode::UNAUTHORIZED);

        // A spare connects before it serves a slot.
        let response = app.oneshot(initialize(Some("agent-1"), Some(token.expose()))).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert!(String::from_utf8_lossy(&body).contains("\"tools\""), "{}", String::from_utf8_lossy(&body));
    }
}
