//! Asks the proposal's example questions of the in-memory house through the real `claude` CLI, with the models,
//! roles and reconciler of a settings file, and prints for each question how long the first and final answers
//! took, every revision, and the goals sent. It needs a signed-in CLI and spends model calls.
//!
//! ```text
//! cargo run -p world-agent --example eval_questions -- [settings.toml]
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use perception_wire::navigation::{Asker, QuestionId, SpaceAnswer, SpaceQuestion};
use tokio::sync::mpsc;
use world_agent::clock;
use world_agent::config::SpatialConfig;
use world_agent::fake_world::FakeWorld;
use world_agent::goals::{ChannelGoalSink, SentGoal};
use world_agent::service::SpatialService;

const QUESTIONS: [&str; 8] = [
    "Go to the kitchen table.",
    "Follow the person in the living room.",
    "Where did the person in the hallway go?",
    "Is there room for the robot between the sofa and the armchair?",
    "How far is the bed from the kitchen table, going through the house?",
    "What changed in the last hour?",
    "How many chairs are in the kitchen?",
    "Find my blue mug.",
];

const QUESTION_LIMIT: Duration = Duration::from_secs(180);

struct Timed<T> {
    after: Duration,
    item: T,
}

#[tokio::main]
async fn main() {
    let settings = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("spatial-agent.example.toml")
    });
    let mut config = SpatialConfig::load(&settings).expect("the settings file loads");
    config.claude.work_dir = std::env::temp_dir().join(format!("spatial-agent-eval-{}", uuid::Uuid::new_v4()));
    let models: Vec<String> = config.answering.answerers.iter().map(|a| format!("{} ({:?})", a.model.0, a.role)).collect();
    println!("answerers: {}; reconciler: {}", models.join(", "), config.reconciler.model.0);

    let (goal_sender, mut goals) = mpsc::unbounded_channel::<SentGoal>();
    let (answer_sender, mut answers) = mpsc::unbounded_channel::<SpaceAnswer>();
    let service = SpatialService::start(
        &config,
        Arc::new(FakeWorld::house()),
        ChannelGoalSink(goal_sender),
        FakeWorld::house(),
        answer_sender,
    )
    .await
    .expect("the service starts");

    for (index, text) in QUESTIONS.iter().enumerate() {
        let question_id = QuestionId(index as u64 + 1);
        println!("\n=== Q{}: {text}", question_id.0);
        let asked = Instant::now();
        service.ask(SpaceQuestion { question_id, asked: clock::now(), asker: Asker("eval".to_owned()), text: (*text).to_owned() });
        let mut thread: Vec<Timed<SpaceAnswer>> = Vec::new();
        let finished = tokio::time::timeout(QUESTION_LIMIT, async {
            while let Some(answer) = answers.recv().await {
                if answer.question_id != question_id {
                    continue;
                }
                let is_final = answer.is_final;
                thread.push(Timed { after: asked.elapsed(), item: answer });
                if is_final {
                    return;
                }
            }
        })
        .await;
        let mut sent: Vec<Timed<SentGoal>> = Vec::new();
        while let Ok(goal) = goals.try_recv() {
            sent.push(Timed { after: asked.elapsed(), item: goal });
        }
        for Timed { after, item } in &thread {
            let refs: Vec<String> = item.references.iter().map(|r| format!("{:?}", r.referent)).collect();
            println!(
                "  r{} {:>5.1}s {}{} [{}]: {}",
                item.revision.0,
                after.as_secs_f32(),
                item.model,
                if item.is_final { " FINAL" } else { "" },
                refs.join(" "),
                item.text.replace('\n', " ")
            );
        }
        for Timed { item, .. } in &sent {
            println!("  goal {:?} from {:?}: {:?}", item.goal.goal_id, item.origin, item.goal.target);
        }
        if finished.is_err() {
            println!("  no final answer within {}s", QUESTION_LIMIT.as_secs());
        }
    }
    service.shutdown().await;
    std::fs::remove_dir_all(&config.claude.work_dir).ok();
}
