//! The real `claude` CLI against the MCP endpoint, with Haiku in every role. It needs a signed-in CLI or
//! ANTHROPIC_API_KEY and costs a few model calls, so it runs only on request:
//!
//! ```text
//! cargo test -p world-agent --test real_claude -- --ignored
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use perception_wire::navigation::{Asker, GoalTarget, ObjectId, QuestionId, SpaceQuestion};
use tokio::sync::mpsc;
use world_agent::clock;
use world_agent::config::{ModelName, SpatialConfig};
use world_agent::fake_world::FakeWorld;
use world_agent::goals::ChannelGoalSink;
use world_agent::service::SpatialService;

const QUESTION_LIMIT: Duration = Duration::from_secs(120);

fn haiku_config(work_dir: &Path) -> SpatialConfig {
    let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("spatial-agent.example.toml");
    let mut config = SpatialConfig::load(&example).unwrap();
    let haiku = ModelName("haiku".to_owned());
    for answerer in &mut config.answering.answerers {
        answerer.model = haiku.clone();
    }
    config.reconciler.model = haiku;
    config.claude.work_dir = work_dir.to_owned();
    config
}

#[tokio::test]
#[ignore = "runs the real claude CLI and spends model calls"]
async fn haiku_answers_moves_the_robot_and_reconciles_through_the_mcp_endpoint() {
    let work_dir: PathBuf = std::env::temp_dir().join(format!("spatial-agent-real-{}", uuid::Uuid::new_v4()));
    let (goal_sender, mut goals) = mpsc::unbounded_channel();
    let (answer_sender, mut answers) = mpsc::unbounded_channel();
    let service = SpatialService::start(
        &haiku_config(&work_dir),
        Arc::new(FakeWorld::house()),
        ChannelGoalSink(goal_sender),
        FakeWorld::house(),
        answer_sender,
    )
    .await
    .unwrap();

    service.ask(SpaceQuestion {
        question_id: QuestionId(1),
        asked: clock::now(),
        asker: Asker("test".to_owned()),
        text: "Go to the kitchen table.".to_owned(),
    });
    let thread = tokio::time::timeout(QUESTION_LIMIT, async {
        let mut thread = Vec::new();
        while let Some(answer) = answers.recv().await {
            println!("{answer:?}");
            let is_final = answer.is_final;
            thread.push(answer);
            if is_final {
                return thread;
            }
        }
        thread
    })
    .await
    .expect("a final answer");
    let goal = goals.try_recv().expect("the fast model sent a goal").goal;
    assert_eq!(goal.target, GoalTarget::Object(ObjectId(42)), "{thread:?}");
    assert_eq!(thread[0].goal_id, Some(goal.goal_id));

    service.shutdown().await;
    std::fs::remove_dir_all(&work_dir).unwrap();
}
