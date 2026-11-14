//! Answers each space question by fanning it out to every configured model at once, each in a fresh session with
//! the same digest and the same tools.
//!
//! The fast model's answer goes out at once as revision 0. Every later answer goes to the reconciler with the answer
//! already sent; a revision goes out as the next revision. When the reconciler revises to another goal, the goal
//! running is cancelled and the new one sent. The last message for a question carries `is_final`: once every model
//! has answered and been reconciled, or when the question's deadline passes, whichever is first.
//!
//! Failures are sent, never swallowed. A failed fast model still sends revision 0, saying so. A failed deliberate
//! model sends a revision saying the standing answer was not checked against it. A failed reconciliation (the call
//! fails, its reply is not a verdict, or the verdict cannot be applied) ends the question: its final revision says the
//! standing answer was not checked, and the remaining answers are dropped.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use perception_wire::navigation::{GoalTarget, NavGoal, QuestionId, Revision, SpaceAnswer, SpaceQuestion};
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::config::{AnswererRole, ModelName};
use crate::digest::{compose_question, render, Digest};
use crate::goals::{AnswerGoal, Approach, GoalBook, GoalLedger, GoalOrigin, GoalSink, GoalSinkClosed, PlannedGoal, SentGoal};
use crate::model::{AnswererId, Duty, ModelClient, ModelReply, Prompt};
use crate::reconcile::{parse_verdict, reconcile_prompt, Candidate, GoalChoice, Verdict, VerdictError};
use crate::references::references;
use crate::world::{WorldModel, WorldSnapshot};

/// One configured model that answers every question.
pub struct Answerer<C> {
    pub id: AnswererId,
    pub role: AnswererRole,
    pub client: Arc<C>,
}

#[derive(Clone, Copy, Debug)]
pub struct DeskSettings {
    /// When a question gets its final message, whatever is still running.
    pub deadline: Duration,
    /// How a stop that cancels a goal is driven.
    pub stop_approach: Approach,
}

/// The answer that stands for a question: the last one sent.
#[derive(Clone, Debug, PartialEq)]
struct Standing {
    model: ModelName,
    text: String,
    goal: Option<NavGoal>,
}

/// A deliberate answerer's answer, waiting for the reconciler.
#[derive(Clone, Debug, PartialEq)]
struct LaterAnswer {
    model: ModelName,
    text: String,
    goal: Option<AnswerGoal>,
}

/// What a deliberate answerer came back with, waiting until there is a standing answer to weigh it against.
#[derive(Clone, Debug, PartialEq)]
enum Later {
    Answered(LaterAnswer),
    /// The call failed; the message says why.
    Failed { model: ModelName, error: String },
}

#[derive(Debug, thiserror::Error)]
enum ReconcileError {
    #[error("the reconciler failed: {0}")]
    Call(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error(transparent)]
    Verdict(#[from] VerdictError),
    #[error("the reconciler chose the proposed goal, but the later answer proposed none")]
    NoProposal,
    #[error(transparent)]
    Sink(#[from] GoalSinkClosed),
}

/// The standing answer, resent with the reason a later answer could not be weighed against it.
fn unchecked(current: &Standing, reason: String) -> Standing {
    Standing { model: current.model.clone(), text: format!("{reason} {}", current.text), goal: current.goal }
}

/// The messages of one question, counted.
struct Thread {
    question: QuestionId,
    snapshot: Option<WorldSnapshot>,
    next: Revision,
    final_sent: bool,
    out: mpsc::UnboundedSender<SpaceAnswer>,
}

impl Thread {
    fn publish(&mut self, standing: &Standing, is_final: bool) {
        assert!(!self.final_sent, "a message after the final one for {}", self.question);
        let goal_target = standing.goal.map(|g| g.target);
        let answer = SpaceAnswer {
            question_id: self.question,
            revision: self.next,
            is_final,
            model: standing.model.0.clone(),
            text: standing.text.clone(),
            references: self.snapshot.as_ref().map_or_else(Vec::new, |s| references(&standing.text, goal_target, s)),
            goal_id: standing.goal.map(|g| g.goal_id),
        };
        tracing::info!(question = %self.question, revision = self.next.0, is_final, model = %standing.model, "answer");
        if self.out.send(answer).is_err() {
            tracing::error!(question = %self.question, "an answer after the node stopped");
        }
        self.next = self.next.next();
        self.final_sent = is_final;
    }
}

type AnswerResult<E> = (AnswererId, Result<ModelReply, E>);

pub struct QuestionDesk<W, C, G> {
    world: Arc<W>,
    goals: Arc<GoalBook>,
    ledger: Arc<GoalLedger>,
    sink: Arc<G>,
    answerers: Vec<Answerer<C>>,
    reconciler: Arc<C>,
    settings: DeskSettings,
    out: mpsc::UnboundedSender<SpaceAnswer>,
    tasks: TaskTracker,
    cancel: CancellationToken,
}

impl<W: WorldModel, C: ModelClient, G: GoalSink> QuestionDesk<W, C, G> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        world: Arc<W>,
        goals: Arc<GoalBook>,
        ledger: Arc<GoalLedger>,
        sink: Arc<G>,
        answerers: Vec<Answerer<C>>,
        reconciler: Arc<C>,
        settings: DeskSettings,
        out: mpsc::UnboundedSender<SpaceAnswer>,
    ) -> Self {
        assert_eq!(
            answerers.iter().filter(|a| a.role == AnswererRole::Fast).count(),
            1,
            "exactly one answerer is fast"
        );
        Self { world, goals, ledger, sink, answerers, reconciler, settings, out, tasks: TaskTracker::new(), cancel: CancellationToken::new() }
    }

    /// Starts answering; the answers go out on the desk's channel.
    pub fn ask(self: &Arc<Self>, question: SpaceQuestion) {
        let desk = self.clone();
        self.tasks.spawn(async move { desk.answer(question).await });
    }

    /// Stops every question in flight; their model calls end and no further messages go out.
    pub async fn shutdown(&self) {
        self.cancel.cancel();
        self.tasks.close();
        self.tasks.wait().await;
    }

    fn fast(&self) -> &Answerer<C> {
        self.answerers.iter().find(|a| a.role == AnswererRole::Fast).expect("exactly one answerer is fast")
    }

    fn sent_goal(&self, question: QuestionId, answerer: AnswererId) -> Option<NavGoal> {
        match self.ledger.take(question, answerer)? {
            AnswerGoal::Sent(goal) => Some(goal),
            AnswerGoal::Proposed(plan) => unreachable!("the fast answerer proposed {plan:?}; it only sends"),
        }
    }

    async fn answer(self: Arc<Self>, question: SpaceQuestion) {
        let id = question.question_id;
        tracing::info!(question = %id, asker = question.asker.0, text = question.text, "question");
        let mut thread = Thread { question: id, snapshot: None, next: Revision::FIRST, final_sent: false, out: self.out.clone() };
        let snapshot = match self.world.snapshot().await {
            Ok(snapshot) => snapshot,
            Err(error) => {
                tracing::error!(question = %id, %error, "no world snapshot");
                let text = format!("The world model could not be read: {error}");
                thread.publish(&Standing { model: self.fast().client.model().clone(), text, goal: None }, true);
                return;
            }
        };
        let digest = render(&snapshot, self.goals.running().as_ref());
        thread.snapshot = Some(snapshot);
        let outcome = tokio::select! {
            standing = self.converge(&question, &digest, &mut thread) => Some(standing),
            _ = self.cancel.cancelled() => None,
        };
        match outcome {
            Some(standing) if !thread.final_sent => {
                let standing = standing.unwrap_or_else(|| Standing {
                    model: self.fast().client.model().clone(),
                    text: format!("No model answered within {:?}.", self.settings.deadline),
                    goal: self.sent_goal(id, self.fast().id),
                });
                thread.publish(&standing, true);
            }
            Some(_) => {}
            None => tracing::info!(question = %id, "stopped before its final answer"),
        }
        self.ledger.forget(id);
    }

    /// Runs every answerer and the reconciler until all are done or the deadline passes, publishing as it goes, and
    /// returns the answer standing at the end; None when nothing was sent.
    async fn converge(&self, question: &SpaceQuestion, digest: &Digest, thread: &mut Thread) -> Option<Standing> {
        let id = question.question_id;
        let deadline = tokio::time::sleep(self.settings.deadline);
        tokio::pin!(deadline);
        let prompt = Prompt(compose_question(digest, &question.text));
        let mut pending: JoinSet<AnswerResult<C::Error>> = JoinSet::new();
        for answerer in &self.answerers {
            let (client, prompt) = (answerer.client.clone(), prompt.clone());
            let duty = Duty::Answer { question: id, answerer: answerer.id, role: answerer.role };
            let answerer_id = answerer.id;
            pending.spawn(async move { (answerer_id, client.ask(duty, prompt).await) });
        }
        let mut standing: Option<Standing> = None;
        let mut waiting: VecDeque<Later> = VecDeque::new();
        loop {
            while standing.is_some() && !waiting.is_empty() {
                let next = waiting.pop_front().expect("checked not empty");
                let current = standing.as_ref().expect("checked some");
                let later = match next {
                    Later::Answered(later) => later,
                    Later::Failed { model, error } => {
                        let reason = format!("{model} could not answer ({error}), so the standing answer was not checked against it.");
                        let revised = unchecked(current, reason);
                        thread.publish(&revised, pending.is_empty() && waiting.is_empty());
                        standing = Some(revised);
                        continue;
                    }
                };
                let outcome = tokio::select! {
                    outcome = self.reconcile(question, digest, current, &later) => Some(outcome),
                    _ = &mut deadline => None,
                };
                let Some(outcome) = outcome else {
                    tracing::warn!(question = %id, "deadline passed during reconciliation");
                    return standing;
                };
                match outcome {
                    Ok(Some(revised)) => {
                        thread.publish(&revised, pending.is_empty() && waiting.is_empty());
                        standing = Some(revised);
                    }
                    Ok(None) => {}
                    Err(error) => {
                        tracing::error!(question = %id, later = %later.model, %error, "reconciliation failed; the question ends unchecked");
                        let reason = format!(
                            "Reconciliation failed ({error}), so the standing answer was not checked against {}'s later answer.",
                            later.model
                        );
                        thread.publish(&unchecked(current, reason), true);
                        return None;
                    }
                }
            }
            if pending.is_empty() {
                return standing;
            }
            let joined = tokio::select! {
                joined = pending.join_next() => joined,
                _ = &mut deadline => None,
            };
            let Some(joined) = joined else {
                tracing::warn!(question = %id, still_running = pending.len(), "deadline passed");
                return standing;
            };
            let (answerer_id, result) = joined.expect("a model call does not panic");
            let answerer = &self.answerers[answerer_id.0];
            let model = answerer.client.model().clone();
            match (answerer.role, result) {
                (AnswererRole::Fast, result) => {
                    let text = match result {
                        Ok(reply) => reply.text,
                        Err(error) => {
                            tracing::error!(question = %id, %model, %error, "the fast model failed");
                            format!("The fast model could not answer: {error}")
                        }
                    };
                    let first = Standing { model, text, goal: self.sent_goal(id, answerer_id) };
                    thread.publish(&first, pending.is_empty() && waiting.is_empty());
                    standing = Some(first);
                }
                (AnswererRole::Deliberate, Ok(reply)) => {
                    let later = LaterAnswer { model, text: reply.text, goal: self.ledger.take(id, answerer_id) };
                    waiting.push_back(Later::Answered(later));
                }
                (AnswererRole::Deliberate, Err(error)) => {
                    tracing::error!(question = %id, %model, %error, "a deliberate model failed");
                    waiting.push_back(Later::Failed { model, error: error.to_string() });
                }
            }
        }
    }

    /// Weighs the later answer against the standing one and applies the verdict: the revised answer, with its goal
    /// sent, or None when the verdict keeps the standing answer.
    async fn reconcile(
        &self,
        question: &SpaceQuestion,
        digest: &Digest,
        current: &Standing,
        later: &LaterAnswer,
    ) -> Result<Option<Standing>, ReconcileError> {
        let id = question.question_id;
        let prompt = reconcile_prompt(
            digest,
            &question.text,
            Candidate { model: &current.model, text: &current.text, goal: current.goal.map(|g| g.target) },
            Candidate { model: &later.model, text: &later.text, goal: later.goal.map(|g| g.target()) },
        );
        let reply = self
            .reconciler
            .ask(Duty::Reconcile { question: id }, prompt)
            .await
            .map_err(|error| ReconcileError::Call(Box::new(error)))?;
        let verdict = parse_verdict(&reply.text)?;
        tracing::info!(question = %id, later = %later.model, ?verdict, "reconciled");
        self.apply(id, current, later, verdict)
    }

    /// Sends a stop for the goal if the robot is still on it.
    fn cancel_goal(&self, question: QuestionId, goal: Option<NavGoal>) -> Result<(), GoalSinkClosed> {
        let Some(goal) = goal else { return Ok(()) };
        if self.goals.check_running(goal.goal_id).is_err() {
            return Ok(());
        }
        let stop = PlannedGoal { target: GoalTarget::Stop, approach: self.settings.stop_approach, map_version: goal.map_version };
        let stop = self.goals.issue(stop, question);
        tracing::info!(%question, cancelled = %goal.goal_id, stop = %stop.goal_id, "goal cancelled by a revision");
        self.sink.send(SentGoal { goal: stop, origin: GoalOrigin::Reconciler })
    }

    /// The revised answer, with its goal sent; None when the verdict keeps the standing answer.
    fn apply(&self, question: QuestionId, current: &Standing, later: &LaterAnswer, verdict: Verdict) -> Result<Option<Standing>, ReconcileError> {
        let Verdict::Revise { text, goal: choice } = verdict else { return Ok(None) };
        let goal = match choice {
            GoalChoice::Sent => current.goal,
            GoalChoice::None => {
                self.cancel_goal(question, current.goal)?;
                None
            }
            GoalChoice::Proposed => {
                let Some(AnswerGoal::Proposed(plan)) = later.goal else { return Err(ReconcileError::NoProposal) };
                match current.goal {
                    Some(goal) if goal.target == plan.target => Some(goal),
                    _ => {
                        self.cancel_goal(question, current.goal)?;
                        let goal = self.goals.issue(plan, question);
                        self.sink.send(SentGoal { goal, origin: GoalOrigin::Reconciler })?;
                        tracing::info!(%question, goal = %goal.goal_id, "proposed goal sent by a revision");
                        Some(goal)
                    }
                }
            }
        };
        Ok(Some(Standing { model: later.model.clone(), text, goal }))
    }
}
