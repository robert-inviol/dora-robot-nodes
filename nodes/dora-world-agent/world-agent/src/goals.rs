//! The goals sent to the robot, what it reports about them, and the goals each answer put the robot on or proposed.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::Mutex;
use std::time::Duration;

use perception_wire::navigation::{
    GoalId, GoalStatus, GoalTarget, MapVersion, NavGoal, NonNegative, QuestionId, UnixNanos,
};
use tokio::sync::mpsc;

use crate::clock;
use crate::model::AnswererId;

/// Who put a goal on the robot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GoalOrigin {
    /// The fast answerer, through `send_goal` or `cancel_goal`.
    Answerer(AnswererId),
    /// The reconciler, through a revision: a deliberate answerer's proposal adopted, or a stop.
    Reconciler,
}

impl fmt::Display for GoalOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GoalOrigin::Answerer(answerer) => answerer.fmt(f),
            GoalOrigin::Reconciler => f.write_str("reconciler"),
        }
    }
}

/// A goal on its way to the robot, with who sent it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SentGoal {
    pub goal: NavGoal,
    pub origin: GoalOrigin,
}

/// Where nav goals go: the dataflow in the node, a channel in tests and the console.
pub trait GoalSink: Send + Sync + 'static {
    fn send(&self, goal: SentGoal) -> Result<(), GoalSinkClosed>;
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("the link to the robot is closed; the goal was not sent")]
pub struct GoalSinkClosed;

/// Sends each goal down a channel, for whatever owns the receiver to publish.
#[derive(Clone)]
pub struct ChannelGoalSink(pub mpsc::UnboundedSender<SentGoal>);

impl GoalSink for ChannelGoalSink {
    fn send(&self, goal: SentGoal) -> Result<(), GoalSinkClosed> {
        self.0.send(goal).map_err(|_| GoalSinkClosed)
    }
}

/// How a goal is driven: how close to stop, how fast to go and how long it may take.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Approach {
    pub standoff_m: NonNegative,
    pub max_speed_m_s: NonNegative,
    pub within: Duration,
}

/// A goal an answer wants, before it has an id: a deliberate answerer's proposal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlannedGoal {
    pub target: GoalTarget,
    pub approach: Approach,
    /// The map the answerer read the target from.
    pub map_version: MapVersion,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GoalRecord {
    pub goal: NavGoal,
    /// The question whose answer sent it.
    pub question: QuestionId,
    pub issued: UnixNanos,
    /// The robot's latest report; None until it acknowledges the goal.
    pub status: Option<GoalStatus>,
}

impl GoalRecord {
    pub fn is_open(&self) -> bool {
        self.goal.target != GoalTarget::Stop && !self.status.is_some_and(|s| s.state.is_final())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GoalBookError {
    #[error("no goal {0} was sent")]
    Unknown(GoalId),
    #[error("goal {0} is not running")]
    NotRunning(GoalId),
}

/// Every goal sent and its latest status, newest last. Goal ids start from the clock, so they do not repeat across
/// restarts of the process.
pub struct GoalBook {
    inner: Mutex<BookInner>,
}

struct BookInner {
    next_id: u64,
    goals: BTreeMap<GoalId, GoalRecord>,
}

const KEPT_GOALS: usize = 64;

impl Default for GoalBook {
    fn default() -> Self {
        let first_id = u64::try_from(clock::now().0 / 1_000_000).expect("the clock is after 1970");
        Self { inner: Mutex::new(BookInner { next_id: first_id, goals: BTreeMap::new() }) }
    }
}

impl GoalBook {
    /// Gives the planned goal an id and records it as sent; the caller sends it.
    pub fn issue(&self, plan: PlannedGoal, question: QuestionId) -> NavGoal {
        let mut inner = self.inner.lock().expect("goal book lock");
        let goal_id = GoalId(inner.next_id);
        inner.next_id += 1;
        let issued = clock::now();
        let goal = NavGoal {
            goal_id,
            target: plan.target,
            standoff_m: plan.approach.standoff_m,
            max_speed_m_s: plan.approach.max_speed_m_s,
            deadline: clock::after(issued, plan.approach.within),
            map_version: plan.map_version,
        };
        inner.goals.insert(goal_id, GoalRecord { goal, question, issued, status: None });
        while inner.goals.len() > KEPT_GOALS {
            inner.goals.pop_first();
        }
        goal
    }

    pub fn update(&self, status: GoalStatus) -> Result<(), GoalBookError> {
        let mut inner = self.inner.lock().expect("goal book lock");
        let record = inner.goals.get_mut(&status.goal_id).ok_or(GoalBookError::Unknown(status.goal_id))?;
        record.status = Some(status);
        Ok(())
    }

    /// The goal the robot is working on: the newest one sent, unless it was a stop or has finished.
    pub fn running(&self) -> Option<GoalRecord> {
        let inner = self.inner.lock().expect("goal book lock");
        inner.goals.values().next_back().filter(|r| r.is_open()).cloned()
    }

    pub fn check_running(&self, goal_id: GoalId) -> Result<GoalRecord, GoalBookError> {
        let inner = self.inner.lock().expect("goal book lock");
        if !inner.goals.contains_key(&goal_id) {
            return Err(GoalBookError::Unknown(goal_id));
        }
        inner
            .goals
            .values()
            .next_back()
            .filter(|r| r.goal.goal_id == goal_id && r.is_open())
            .cloned()
            .ok_or(GoalBookError::NotRunning(goal_id))
    }
}

/// The goal an answer stands for: one the fast answerer sent, or one a deliberate answerer proposed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AnswerGoal {
    Sent(NavGoal),
    Proposed(PlannedGoal),
}

impl AnswerGoal {
    pub fn target(&self) -> GoalTarget {
        match self {
            AnswerGoal::Sent(goal) => goal.target,
            AnswerGoal::Proposed(plan) => plan.target,
        }
    }
}

/// The latest goal each answerer of each open question sent or proposed, until the question's desk takes it.
#[derive(Default)]
pub struct GoalLedger {
    goals: Mutex<HashMap<(QuestionId, AnswererId), AnswerGoal>>,
}

impl GoalLedger {
    pub fn record(&self, question: QuestionId, answerer: AnswererId, goal: AnswerGoal) {
        self.goals.lock().expect("ledger lock").insert((question, answerer), goal);
    }

    pub fn take(&self, question: QuestionId, answerer: AnswererId) -> Option<AnswerGoal> {
        self.goals.lock().expect("ledger lock").remove(&(question, answerer))
    }

    /// Drops what is left for a question that has had its final answer.
    pub fn forget(&self, question: QuestionId) {
        self.goals.lock().expect("ledger lock").retain(|(q, _), _| *q != question);
    }
}

#[cfg(test)]
mod tests {
    use perception_wire::navigation::{GoalState, ObjectId, PlaceId};

    use super::*;

    fn plan(target: GoalTarget) -> PlannedGoal {
        PlannedGoal {
            target,
            approach: Approach {
                standoff_m: NonNegative::new("standoff_m", 0.5).unwrap(),
                max_speed_m_s: NonNegative::new("max_speed_m_s", 0.3).unwrap(),
                within: Duration::from_secs(60),
            },
            map_version: MapVersion(1),
        }
    }

    fn status(goal_id: GoalId, state: GoalState) -> GoalStatus {
        GoalStatus { goal_id, state, updated: clock::now() }
    }

    #[test]
    fn the_newest_goal_runs_until_it_finishes() {
        let book = GoalBook::default();
        let first = book.issue(plan(GoalTarget::Object(ObjectId(42))), QuestionId(1));
        let second = book.issue(plan(GoalTarget::Place(PlaceId(3))), QuestionId(2));
        assert_eq!(second.goal_id.0, first.goal_id.0 + 1);
        assert_eq!(book.running().unwrap().goal, second);
        assert_eq!(book.running().unwrap().question, QuestionId(2));
        assert_eq!(book.check_running(first.goal_id), Err(GoalBookError::NotRunning(first.goal_id)));
        book.update(status(second.goal_id, GoalState::Succeeded)).unwrap();
        assert_eq!(book.running(), None);
    }

    #[test]
    fn a_status_for_an_unknown_goal_is_an_error_and_a_stop_leaves_nothing_running() {
        let book = GoalBook::default();
        assert_eq!(book.update(status(GoalId(1), GoalState::Active)), Err(GoalBookError::Unknown(GoalId(1))));
        book.issue(plan(GoalTarget::Object(ObjectId(42))), QuestionId(1));
        book.issue(plan(GoalTarget::Stop), QuestionId(1));
        assert_eq!(book.running(), None);
    }

    #[test]
    fn the_ledger_keeps_the_latest_goal_per_answerer_until_taken() {
        let ledger = GoalLedger::default();
        let (question, answerer) = (QuestionId(5), AnswererId(1));
        ledger.record(question, answerer, AnswerGoal::Proposed(plan(GoalTarget::Place(PlaceId(3)))));
        ledger.record(question, answerer, AnswerGoal::Proposed(plan(GoalTarget::Place(PlaceId(4)))));
        ledger.record(question, AnswererId(2), AnswerGoal::Proposed(plan(GoalTarget::Stop)));
        assert_eq!(ledger.take(question, answerer).unwrap().target(), GoalTarget::Place(PlaceId(4)));
        assert_eq!(ledger.take(question, answerer), None);
        ledger.forget(question);
        assert_eq!(ledger.take(question, AnswererId(2)), None);
    }
}
