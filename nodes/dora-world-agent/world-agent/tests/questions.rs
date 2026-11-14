//! Questions fanned out to `fake-claude` playing the fast model, two deliberate models and the reconciler, through
//! the whole service: the MCP endpoint, the goal tools, the desk and the answers that go out.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{config, WorkDir, FAST, NOW, RECONCILER, SLOW_A, SLOW_B};
use perception_wire::navigation::{
    Asker, GoalTarget, NavGoal, ObjectId, PlaceId, QuestionId, Referent, Revision, SpaceAnswer, SpaceQuestion,
};
use tokio::sync::mpsc;
use world_agent::config::SpatialConfig;
use world_agent::digest::DIGEST_HEADING;
use world_agent::fake_world::FakeWorld;
use world_agent::goals::{ChannelGoalSink, SentGoal};
use world_agent::service::{SpatialService, StartError};

const QUESTION: QuestionId = QuestionId(31);

struct Harness {
    service: SpatialService<FakeWorld, ChannelGoalSink, FakeWorld>,
    answers: mpsc::UnboundedReceiver<SpaceAnswer>,
    goals: mpsc::UnboundedReceiver<SentGoal>,
    _work_dir: WorkDir,
}

impl Harness {
    async fn start(deadline: Duration) -> Self {
        Self::start_with(deadline, |_| {}).await
    }

    /// With calls that may stay silent until the deadline.
    async fn patient(deadline: Duration) -> Self {
        Self::start_with(deadline, |config| config.watchdog.silent_turn = Duration::from_secs(30)).await
    }

    async fn start_with(deadline: Duration, adjust: impl FnOnce(&mut SpatialConfig)) -> Self {
        let work_dir = WorkDir::new();
        let (goal_sender, goals) = mpsc::unbounded_channel();
        let (answer_sender, answers) = mpsc::unbounded_channel();
        let mut config = config(&work_dir.0, deadline);
        adjust(&mut config);
        let service = SpatialService::start(
            &config,
            Arc::new(FakeWorld::house_at(NOW)),
            ChannelGoalSink(goal_sender),
            FakeWorld::house_at(NOW),
            answer_sender,
        )
        .await
        .unwrap();
        Self { service, answers, goals, _work_dir: work_dir }
    }

    /// Every message for the question, up to the final one.
    async fn ask(&mut self, text: &str) -> Vec<SpaceAnswer> {
        self.service.ask(SpaceQuestion {
            question_id: QUESTION,
            asked: NOW,
            asker: Asker("planner".to_owned()),
            text: text.to_owned(),
        });
        let mut thread = Vec::new();
        tokio::time::timeout(Duration::from_secs(20), async {
            while let Some(answer) = self.answers.recv().await {
                let is_final = answer.is_final;
                thread.push(answer);
                if is_final {
                    return;
                }
            }
        })
        .await
        .unwrap_or_else(|_| panic!("no final answer; so far {thread:#?}"));
        thread
    }

    fn goals_sent(&mut self) -> Vec<NavGoal> {
        std::iter::from_fn(|| self.goals.try_recv().ok().map(|sent| sent.goal)).collect()
    }

    async fn stop(self) {
        self.service.shutdown().await;
    }
}

fn summary(thread: &[SpaceAnswer]) -> Vec<(u32, bool, &str, &str)> {
    thread.iter().map(|a| (a.revision.0, a.is_final, a.model.as_str(), a.text.as_str())).collect()
}

fn script(lines: &[&str]) -> String {
    std::iter::once("Where is my mug?".to_owned()).chain(lines.iter().map(|l| format!("fake {l}"))).collect::<Vec<_>>().join("\n")
}

#[tokio::test]
async fn every_model_gets_the_digest_then_the_question_in_a_fresh_session() {
    let mut harness = Harness::start(Duration::from_secs(10)).await;
    let thread = harness.ask("Where is my mug?").await;
    let first = &thread[0];
    assert_eq!((first.revision, first.model.as_str()), (Revision::FIRST, FAST));
    assert!(first.text.starts_with(&format!("echo: {DIGEST_HEADING} as_of ")), "{}", first.text);
    assert!(first.text.contains("\n  o51 mug 0.1x0.1"), "{}", first.text);
    assert!(first.text.ends_with("\n\nWhere is my mug?"), "{}", first.text);
    harness.stop().await;
}

#[tokio::test]
async fn the_fast_answer_goes_out_first_and_kept_answers_end_in_a_final_confirmation() {
    let mut harness = Harness::start(Duration::from_secs(10)).await;
    let thread = harness
        .ask(&script(&[
            &format!("{FAST} say The mug o51 is on the table o42."),
            &format!("{SLOW_A} wait 100"),
            &format!("{SLOW_A} say The mug o51 is on the kitchen table."),
            &format!("{SLOW_B} wait 300"),
            &format!("{SLOW_B} say It is on o42."),
        ]))
        .await;
    assert_eq!(
        summary(&thread),
        [(0, false, FAST, "The mug o51 is on the table o42."), (1, true, FAST, "The mug o51 is on the table o42.")]
    );
    let referents: Vec<Referent> = thread[0].references.iter().map(|r| r.referent).collect();
    assert_eq!(referents, [Referent::Object(ObjectId(51)), Referent::Object(ObjectId(42))]);
    assert_eq!(thread[0].goal_id, None);
    harness.stop().await;
}

#[tokio::test]
async fn a_slower_answer_the_reconciler_prefers_goes_out_as_a_revision() {
    let mut harness = Harness::start(Duration::from_secs(10)).await;
    let revise = r#"{"decision": "revise", "text": "The mug o51 is by the sink, near p1.", "goal": "sent"}"#;
    let thread = harness
        .ask(&script(&[
            &format!("{FAST} say The mug is on the table o42."),
            &format!("{SLOW_A} wait 100"),
            &format!("{SLOW_A} say It is by the sink."),
            &format!("{SLOW_B} wait 600"),
            &format!("{SLOW_B} say On the table."),
            &format!("{RECONCILER} after {SLOW_A} say {revise}"),
        ]))
        .await;
    assert_eq!(
        summary(&thread),
        [
            (0, false, FAST, "The mug is on the table o42."),
            (1, false, SLOW_A, "The mug o51 is by the sink, near p1."),
            (2, true, SLOW_A, "The mug o51 is by the sink, near p1."),
        ]
    );
    harness.stop().await;
}

#[tokio::test]
async fn the_last_revision_is_final_itself() {
    let mut harness = Harness::start(Duration::from_secs(10)).await;
    let revise = r#"{"decision": "revise", "text": "By the sink.", "goal": "sent"}"#;
    let thread = harness
        .ask(&script(&[
            &format!("{FAST} say On the table."),
            &format!("{SLOW_A} wait 100"),
            &format!("{SLOW_A} say Not sure."),
            &format!("{SLOW_B} wait 500"),
            &format!("{SLOW_B} say By the sink."),
            &format!("{RECONCILER} after {SLOW_B} say {revise}"),
        ]))
        .await;
    assert_eq!(summary(&thread), [(0, false, FAST, "On the table."), (1, true, SLOW_B, "By the sink.")]);
    harness.stop().await;
}

#[tokio::test]
async fn the_fast_goal_moves_the_robot_at_once_and_a_revised_goal_cancels_and_replaces_it() {
    let mut harness = Harness::start(Duration::from_secs(10)).await;
    let revise = r#"{"decision": "revise", "text": "The mug is by p1; going there instead.", "goal": "proposed"}"#;
    let thread = harness
        .ask(&script(&[
            &format!(r#"{FAST} call send_goal {{"kind": "object", "target": "o42"}}"#),
            &format!("{FAST} say Going to the table o42."),
            &format!("{SLOW_A} wait 200"),
            &format!(r#"{SLOW_A} call send_goal {{"kind": "place", "target": "p1"}}"#),
            &format!("{SLOW_A} say The mug is by p1."),
            &format!("{SLOW_B} wait 600"),
            &format!("{SLOW_B} say Agreed."),
            &format!("{RECONCILER} after {SLOW_A} say {revise}"),
        ]))
        .await;
    let goals = harness.goals_sent();
    let targets: Vec<GoalTarget> = goals.iter().map(|g| g.target).collect();
    assert_eq!(targets, [GoalTarget::Object(ObjectId(42)), GoalTarget::Stop, GoalTarget::Place(PlaceId(1))]);
    assert_eq!(thread[0].goal_id, Some(goals[0].goal_id));
    assert_eq!((thread[1].revision, thread[1].model.as_str()), (Revision(1), SLOW_A));
    assert_eq!(thread[1].goal_id, Some(goals[2].goal_id));
    assert_eq!(thread.last().unwrap().goal_id, Some(goals[2].goal_id));
    let referents: Vec<Referent> = thread[1].references.iter().map(|r| r.referent).collect();
    assert_eq!(referents, [Referent::Place(PlaceId(1))]);
    harness.stop().await;
}

#[tokio::test]
async fn a_revision_to_no_goal_stops_the_robot() {
    let mut harness = Harness::start(Duration::from_secs(10)).await;
    let revise = r#"{"decision": "revise", "text": "There is no mug here.", "goal": "none"}"#;
    let thread = harness
        .ask(&script(&[
            &format!(r#"{FAST} call send_goal {{"kind": "object", "target": "o51"}}"#),
            &format!("{FAST} say Going to the mug o51."),
            &format!("{SLOW_A} wait 200"),
            &format!("{SLOW_A} say That is a cup, not your mug."),
            &format!("{SLOW_B} say Same."),
            &format!("{RECONCILER} after {SLOW_A} say {revise}"),
        ]))
        .await;
    let targets: Vec<GoalTarget> = harness.goals_sent().iter().map(|g| g.target).collect();
    assert_eq!(targets, [GoalTarget::Object(ObjectId(51)), GoalTarget::Stop]);
    let revised = thread.iter().find(|a| a.model == SLOW_A).expect("a revision");
    assert_eq!((revised.text.as_str(), revised.goal_id), ("There is no mug here.", None));
    harness.stop().await;
}

#[tokio::test]
async fn proposals_from_deliberate_models_never_move_the_robot_unless_adopted() {
    let mut harness = Harness::start(Duration::from_secs(10)).await;
    let thread = harness
        .ask(&script(&[
            &format!("{FAST} say The mug o51 is in the kitchen."),
            &format!(r#"{SLOW_A} call send_goal {{"kind": "place", "target": "p5"}}"#),
            &format!("{SLOW_A} say Let us look in the living room p5."),
            &format!(r#"{SLOW_B} call send_goal {{"kind": "object", "target": "o42"}}"#),
            &format!("{SLOW_B} say It is on o42."),
        ]))
        .await;
    assert!(thread.last().unwrap().is_final);
    assert!(thread.iter().all(|a| a.goal_id.is_none()), "{thread:#?}");
    assert_eq!(harness.goals_sent(), []);
    harness.stop().await;
}

#[tokio::test]
async fn the_deadline_ends_the_question_with_the_answer_standing() {
    let mut harness = Harness::patient(Duration::from_millis(1500)).await;
    let started = tokio::time::Instant::now();
    let thread = harness
        .ask(&script(&[&format!("{FAST} say On the table o42."), &format!("{SLOW_A} deaf"), &format!("{SLOW_B} hang")]))
        .await;
    assert_eq!(summary(&thread), [(0, false, FAST, "On the table o42."), (1, true, FAST, "On the table o42.")]);
    assert!(started.elapsed() >= Duration::from_millis(1400), "{:?}", started.elapsed());
    harness.stop().await;
}

#[tokio::test]
async fn when_no_model_answers_in_time_the_final_message_says_so() {
    let mut harness = Harness::patient(Duration::from_millis(1000)).await;
    let thread = harness.ask(&script(&[&format!("{FAST} deaf"), &format!("{SLOW_A} deaf"), &format!("{SLOW_B} deaf")])).await;
    assert_eq!(thread.len(), 1);
    assert_eq!((thread[0].revision, thread[0].is_final, thread[0].model.as_str()), (Revision::FIRST, true, FAST));
    assert!(thread[0].text.starts_with("No model answered within"), "{}", thread[0].text);
    harness.stop().await;
}

#[tokio::test]
async fn a_failed_fast_model_is_reported_and_the_slower_answer_can_replace_it() {
    let mut harness = Harness::start(Duration::from_secs(10)).await;
    let revise = r#"{"decision": "revise", "text": "On the table o42.", "goal": "sent"}"#;
    let thread = harness
        .ask(&script(&[
            &format!(r#"{FAST} call send_goal {{"kind": "object", "target": "o999"}}"#),
            &format!("{SLOW_A} wait 100"),
            &format!("{SLOW_A} say On the table o42."),
            &format!("{SLOW_B} crash"),
            &format!("{RECONCILER} after {SLOW_A} say {revise}"),
        ]))
        .await;
    assert!(thread[0].text.starts_with("The fast model could not answer"), "{}", thread[0].text);
    assert!(thread[0].text.contains("no object o999"), "{}", thread[0].text);
    let failed = &thread[1];
    assert_eq!((failed.revision, failed.is_final, failed.model.as_str()), (Revision(1), false, FAST));
    assert!(failed.text.starts_with(&format!("{SLOW_B} could not answer (")), "{}", failed.text);
    assert!(failed.text.contains("so the standing answer was not checked against it. The fast model could not answer"), "{}", failed.text);
    assert_eq!(summary(&thread)[2..], [(2, true, SLOW_A, "On the table o42.")]);
    harness.stop().await;
}

#[tokio::test]
async fn a_reply_that_is_not_a_verdict_ends_the_question_saying_the_answer_was_not_checked() {
    let mut harness = Harness::start(Duration::from_secs(10)).await;
    let thread = harness
        .ask(&script(&[
            &format!(r#"{FAST} call send_goal {{"kind": "object", "target": "o42"}}"#),
            &format!("{FAST} say The mug o51 is on the table o42."),
            &format!("{SLOW_A} wait 100"),
            &format!("{SLOW_A} say It is by the sink."),
            &format!("{SLOW_B} wait 1500"),
            &format!("{SLOW_B} say By the sink."),
            &format!("{RECONCILER} after {SLOW_A} say Keep it, I think."),
        ]))
        .await;
    assert_eq!(thread.len(), 2, "{thread:#?}");
    let last = &thread[1];
    assert_eq!((last.revision, last.is_final, last.model.as_str(), last.goal_id), (Revision(1), true, FAST, thread[0].goal_id));
    assert!(last.text.starts_with("Reconciliation failed (the reconciler's reply is not a verdict"), "{}", last.text);
    assert!(
        last.text.ends_with(&format!("so the standing answer was not checked against {SLOW_A}'s later answer. The mug o51 is on the table o42.")),
        "{}",
        last.text
    );
    let referents: Vec<Referent> = last.references.iter().map(|r| r.referent).collect();
    assert_eq!(referents, [Referent::Object(ObjectId(51)), Referent::Object(ObjectId(42))]);
    assert_eq!(harness.goals_sent().len(), 1, "no stop and no other goal");
    harness.stop().await;
}

#[tokio::test]
async fn a_failed_reconciler_call_ends_the_question_saying_the_answer_was_not_checked() {
    let mut harness = Harness::start(Duration::from_secs(10)).await;
    let thread = harness
        .ask(&script(&[
            &format!("{FAST} say On the table o42."),
            &format!("{SLOW_A} wait 100"),
            &format!("{SLOW_A} say By the sink."),
            &format!("{SLOW_B} wait 1500"),
            &format!("{SLOW_B} say By the sink."),
            &format!("{RECONCILER} after {SLOW_A} crash"),
        ]))
        .await;
    assert_eq!(thread.len(), 2, "{thread:#?}");
    assert!(thread[1].is_final);
    assert!(thread[1].text.starts_with("Reconciliation failed (the reconciler failed: "), "{}", thread[1].text);
    assert!(thread[1].text.ends_with("later answer. On the table o42."), "{}", thread[1].text);
    harness.stop().await;
}

#[tokio::test]
async fn a_work_dir_that_is_not_empty_is_refused() {
    let work_dir = WorkDir::new();
    std::fs::create_dir_all(&work_dir.0).unwrap();
    std::fs::write(work_dir.0.join("CLAUDE.md"), "stray instructions").unwrap();
    let (goals, _) = mpsc::unbounded_channel();
    let (answers, _) = mpsc::unbounded_channel();
    let started = SpatialService::start(
        &config(&work_dir.0, Duration::from_secs(1)),
        Arc::new(FakeWorld::house_at(NOW)),
        ChannelGoalSink(goals),
        FakeWorld::house_at(NOW),
        answers,
    )
    .await;
    assert!(matches!(started, Err(StartError::WorkDirNotEmpty(_))));
}
