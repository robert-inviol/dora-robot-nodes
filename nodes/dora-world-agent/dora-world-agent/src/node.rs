//! What the node does with each input and each goal, apart from dora: the dora loop in `main` and the tests drive
//! the same calls.

use std::sync::Arc;

use dora_node_api::arrow::array::Array;
use eyre::{eyre, Result, WrapErr};
use object_descriptions::describer::{describer_profile, ClaudeDescriber};
use object_descriptions::embedder::QueriesOf;
use object_descriptions::gemini::{GeminiApiKey, GeminiEmbedder, GEMINI_API_KEY_VARIABLE};
use object_descriptions::service::{DescriptionSettings, ObjectDescriptions};
use world_agent::claude_code::ClaudeCodeClient;
use world_agent::search::QueryEmbedder;
use perception_wire::navigation::{decode_goal_status, decode_spatial_query, NavGoal, SpatialAnswer};
use tokio::sync::mpsc;
use world_agent::clock;
use world_agent::config::{ModelName, SpatialConfig};
use world_agent::goals::{ChannelGoalSink, GoalBookError, GoalOrigin, SentGoal};
use world_agent::model::AnswererId;
use world_agent::service::SpatialService;
use world_store::{AgentName, IssuedGoal};

use crate::world::{store_input, Ingested, StoreInput, WorldStore};

pub const SPATIAL_QUERY_INPUT: &str = "spatial_query";
pub const SPATIAL_ANSWER_OUTPUT: &str = "spatial_answer";
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
    pub answers: mpsc::UnboundedReceiver<SpatialAnswer>,
    pub goals: mpsc::UnboundedReceiver<SentGoal>,
}

/// The describer of object views, when the settings have `[object_descriptions]`.
type Descriptions<W> = ObjectDescriptions<ClaudeDescriber<ClaudeCodeClient>, GeminiEmbedder, W>;

pub struct SpatialNode<W> {
    world: Arc<W>,
    service: SpatialService<W, ChannelGoalSink, W>,
    descriptions: Option<Descriptions<W>>,
    authors: GoalAuthors,
}

impl<W: WorldStore + Clone> SpatialNode<W> {
    /// Starts the spatial agent on the world; its answers and goals come out of `Outputs`. With
    /// `[object_descriptions]` in the settings, object views are described and embedded with the Gemini key, which
    /// must then be given, and `find` searches by meaning.
    pub async fn start(config: &SpatialConfig, world: W, gemini_key: Option<GeminiApiKey>) -> Result<(Self, Outputs)> {
        let (goal_sender, goals) = mpsc::unbounded_channel();
        let (answer_sender, answers) = mpsc::unbounded_channel();
        let world = Arc::new(world);
        let embedder = match (&config.object_descriptions, gemini_key) {
            (Some(descriptions), Some(key)) => Some(Arc::new(GeminiEmbedder::new(key, &descriptions.embedding).wrap_err("starting the Gemini client")?)),
            (Some(_), None) => return Err(eyre!("the settings have [object_descriptions] but {GEMINI_API_KEY_VARIABLE} is not set")),
            (None, Some(_)) => {
                tracing::info!("{GEMINI_API_KEY_VARIABLE} is set, but the settings have no [object_descriptions]: objects are not described");
                None
            }
            (None, None) => None,
        };
        let query_embedder = embedder.clone().map(|e| Arc::new(QueriesOf(e)) as Arc<dyn QueryEmbedder>);
        let mut service =
            SpatialService::start(config, world.clone(), ChannelGoalSink(goal_sender), world.as_ref().clone(), answer_sender, query_embedder)
                .await?;
        let descriptions = match (&config.object_descriptions, embedder) {
            (Some(settings), Some(embedder)) => {
                let describer = ClaudeDescriber::new(service.client(describer_profile(settings)), settings.describe_timeout);
                tracing::info!(describer = %settings.describer.model, embedding = %settings.embedding.model, "object views are described and embedded");
                Some(ObjectDescriptions::start(describer, embedder, world.clone(), DescriptionSettings::of(settings)))
            }
            _ => None,
        };
        Ok((Self { world, service, descriptions, authors: GoalAuthors::of(config) }, Outputs { answers, goals }))
    }

    /// Handles one dora input: a question starts answering, everything else goes to the world store.
    pub async fn on_input(&self, id: &str, data: &dyn Array) -> Result<()> {
        if id == SPATIAL_QUERY_INPUT {
            self.service.ask(decode_spatial_query(data).wrap_err("decoding a spatial query")?);
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
        match self.world.ingest(input, data)? {
            Ingested::Recorded => {}
            Ingested::RoomsChanged => self.service.rooms_changed(self.world.rooms().await?),
            Ingested::ObjectUpdated(update) => {
                if let Some(descriptions) = &self.descriptions {
                    descriptions.offer(&update);
                }
            }
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
        if let Some(descriptions) = &self.descriptions {
            descriptions.shutdown().await;
        }
        self.service.shutdown().await;
        self.world.close()
    }
}
