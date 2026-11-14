//! What the node does with each input and each goal, apart from dora: the dora loop in `main` and the tests drive
//! the same calls.

use std::sync::Arc;

use dora_node_api::arrow::array::Array;
use eyre::{Result, WrapErr};
use perception_wire::navigation::{decode_goal_status, decode_space_question, NavGoal, SpaceAnswer};
use tokio::sync::mpsc;
use world_agent::clock;
use world_agent::config::{ModelName, SpatialConfig};
use world_agent::goals::{ChannelGoalSink, GoalBookError, GoalOrigin, SentGoal};
use world_agent::model::AnswererId;
use world_agent::service::SpatialService;
use world_store::{AgentName, IssuedGoal};

use crate::world::{store_input, Ingested, StoreInput, WorldStore};

pub const SPACE_QUESTION_INPUT: &str = "space_question";
pub const SPACE_ANSWER_OUTPUT: &str = "space_answer";
pub const NAV_GOAL_OUTPUT: &str = "nav_goal";

/// The models behind each goal origin, from the settings.
struct GoalAuthors {
    answerers: Vec<ModelName>,
    reconciler: ModelName,
}

impl GoalAuthors {
    fn of(config: &SpatialConfig) -> Self {
        Self {
            answerers: config.answering.answerers.iter().map(|a| a.model.clone()).collect(),
            reconciler: config.reconciler.model.clone(),
        }
    }

    fn model(&self, origin: GoalOrigin) -> &ModelName {
        match origin {
            GoalOrigin::Answerer(AnswererId(index)) => &self.answerers[index],
            GoalOrigin::Reconciler => &self.reconciler,
        }
    }
}

/// Where the node's outputs come out: every answer and every goal, in the order they were made.
pub struct Outputs {
    pub answers: mpsc::UnboundedReceiver<SpaceAnswer>,
    pub goals: mpsc::UnboundedReceiver<SentGoal>,
}

pub struct SpatialNode<W> {
    world: Arc<W>,
    service: SpatialService<W, ChannelGoalSink, W>,
    authors: GoalAuthors,
}

impl<W: WorldStore + Clone> SpatialNode<W> {
    /// Starts the spatial agent on the world; its answers and goals come out of `Outputs`.
    pub async fn start(config: &SpatialConfig, world: W) -> Result<(Self, Outputs)> {
        let (goal_sender, goals) = mpsc::unbounded_channel();
        let (answer_sender, answers) = mpsc::unbounded_channel();
        let world = Arc::new(world);
        let service =
            SpatialService::start(config, world.clone(), ChannelGoalSink(goal_sender), world.as_ref().clone(), answer_sender).await?;
        Ok((Self { world, service, authors: GoalAuthors::of(config) }, Outputs { answers, goals }))
    }

    /// Handles one dora input: a question starts answering, everything else goes to the world store.
    pub async fn on_input(&self, id: &str, data: &dyn Array) -> Result<()> {
        if id == SPACE_QUESTION_INPUT {
            self.service.ask(decode_space_question(data).wrap_err("decoding a space question")?);
            return Ok(());
        }
        let input = store_input(id)?;
        if input == StoreInput::GoalStatus {
            let status = decode_goal_status(data).wrap_err("decoding a goal status")?;
            match self.service.goal_status(status) {
                Ok(()) => {}
                Err(GoalBookError::Unknown(goal)) => tracing::warn!(%goal, "a status for a goal this node did not send"),
                Err(error) => return Err(error.into()),
            }
        }
        if self.world.ingest(input, data)? == Ingested::RoomsChanged {
            self.service.rooms_changed(self.world.rooms().await?);
        }
        Ok(())
    }

    /// Records a goal the agent sent in the world store, and returns it for the `nav_goal` output.
    pub fn on_goal(&self, sent: SentGoal) -> Result<NavGoal> {
        let model = self.authors.model(sent.origin);
        self.world.record_goal(IssuedGoal {
            goal: sent.goal,
            agent: AgentName(sent.origin.to_string()),
            model: world_store::ModelName(model.0.clone()),
            issued: clock::now(),
        })?;
        Ok(sent.goal)
    }

    /// Stops every question in flight and every claude process, then commits what the store has pending.
    pub async fn shutdown(self) -> Result<()> {
        self.service.shutdown().await;
        self.world.close()
    }
}
