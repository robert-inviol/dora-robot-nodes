//! The spatial agent wired together: the MCP endpoint, one Claude Code client per configured model, the question desk
//! and the room namer. The dora node and the console both run this.

use std::path::Path;
use std::sync::Arc;

use perception_wire::navigation::{GoalStatus, SpaceAnswer, SpaceQuestion};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::claude_code::ClaudeCodeClient;
use crate::config::{SpatialConfig, WatchdogConfig};
use crate::goals::{GoalBook, GoalBookError, GoalLedger, GoalSink};
use crate::identity::AgentRegistry;
use crate::mcp::{router, McpEndpoint, ToolContext};
use crate::model::AnswererId;
use crate::process::{Launcher, ProcessProfile, ToolAccess};
use crate::questions::{Answerer, DeskSettings, QuestionDesk};
use crate::room_names::{RoomContents, RoomNameSink, RoomNamer};
use crate::world::WorldModel;

#[derive(Debug, thiserror::Error)]
pub enum StartError {
    #[error("starting the MCP endpoint: {0}")]
    Mcp(#[source] std::io::Error),
    #[error("preparing the work directory {path}: {source}")]
    WorkDir { path: String, source: std::io::Error },
    #[error("the work directory {0} is not empty")]
    WorkDirNotEmpty(String),
}

fn prepare_work_dir(path: &Path) -> Result<(), StartError> {
    let failed = |source| StartError::WorkDir { path: path.display().to_string(), source };
    std::fs::create_dir_all(path).map_err(failed)?;
    match std::fs::read_dir(path).map_err(failed)?.next() {
        None => Ok(()),
        Some(_) => Err(StartError::WorkDirNotEmpty(path.display().to_string())),
    }
}

pub struct SpatialService<W, G, S> {
    desk: Arc<QuestionDesk<W, ClaudeCodeClient, G>>,
    namer: Arc<RoomNamer<ClaudeCodeClient, S>>,
    goals: Arc<GoalBook>,
    clients: Vec<Arc<ClaudeCodeClient>>,
    _mcp: McpEndpoint,
}

impl<W: WorldModel, G: GoalSink, S: RoomNameSink> SpatialService<W, G, S> {
    /// Starts the endpoint and a warm process for every model. Answers go out on `answers`.
    pub async fn start(
        config: &SpatialConfig,
        world: Arc<W>,
        goal_sink: G,
        room_names: S,
        answers: mpsc::UnboundedSender<SpaceAnswer>,
    ) -> Result<Self, StartError> {
        prepare_work_dir(&config.claude.work_dir)?;
        let registry = Arc::new(AgentRegistry::default());
        let goals = Arc::new(GoalBook::default());
        let ledger = Arc::new(GoalLedger::default());
        let sink = Arc::new(goal_sink);
        let context = Arc::new(ToolContext {
            world: world.clone(),
            goals: goals.clone(),
            ledger: ledger.clone(),
            sink: sink.clone(),
            goal_defaults: config.goals,
            sql_row_limit: config.mcp.sql_row_limit,
        });
        let cancel = CancellationToken::new();
        let mcp = McpEndpoint::start(router(context, registry.clone(), cancel.clone()), config.mcp.port, cancel)
            .await
            .map_err(StartError::Mcp)?;
        tracing::info!(url = mcp.url.0, "MCP endpoint up");

        let watchdog: WatchdogConfig = config.watchdog;
        let client = |profile: ProcessProfile| {
            let launcher = Launcher::new(config.claude.clone(), profile, registry.clone(), watchdog.startup);
            Arc::new(ClaudeCodeClient::start(launcher, registry.clone(), watchdog))
        };
        let answerers: Vec<Answerer<ClaudeCodeClient>> = config
            .answering
            .answerers
            .iter()
            .enumerate()
            .map(|(index, answerer)| Answerer {
                id: AnswererId(index),
                role: answerer.role,
                client: client(ProcessProfile {
                    model: answerer.model.clone(),
                    effort: answerer.effort,
                    system_prompt: config.answering.system_prompt.clone(),
                    tools: ToolAccess::World(mcp.url.clone()),
                }),
            })
            .collect();
        let reconciler = client(ProcessProfile::without_tools(&config.reconciler));
        let namer_client = client(ProcessProfile::without_tools(&config.room_namer.profile()));
        let mut clients: Vec<Arc<ClaudeCodeClient>> = answerers.iter().map(|a| a.client.clone()).collect();
        clients.extend([reconciler.clone(), namer_client.clone()]);

        let settings = DeskSettings { deadline: config.answering.deadline, stop_approach: config.goals.approach() };
        let desk = Arc::new(QuestionDesk::new(world, goals.clone(), ledger, sink, answerers, reconciler, settings, answers));
        let namer = Arc::new(RoomNamer::new(namer_client, Arc::new(room_names), config.room_namer.rename_after_change));
        Ok(Self { desk, namer, goals, clients, _mcp: mcp })
    }

    pub fn ask(&self, question: SpaceQuestion) {
        self.desk.ask(question);
    }

    /// Records the robot's report on a goal, for the digest.
    pub fn goal_status(&self, status: GoalStatus) -> Result<(), GoalBookError> {
        self.goals.update(status)
    }

    /// The world store calls this when its rooms or their objects change.
    pub fn rooms_changed(&self, rooms: Vec<RoomContents>) {
        let started = self.namer.rooms_changed(rooms);
        if !started.is_empty() {
            tracing::info!(?started, "naming rooms");
        }
    }

    /// Stops every question in flight and every claude process.
    pub async fn shutdown(self) {
        self.desk.shutdown().await;
        for client in &self.clients {
            client.stop();
        }
    }
}
