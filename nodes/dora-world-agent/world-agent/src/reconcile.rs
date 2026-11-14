//! The reconciler's user message and its verdict. The reconciler sees the digest, the question, the answer already
//! sent with the goal it put the robot on, and one later answer with the goal it proposes, and replies with JSON:
//! keep the sent answer, or revise it with new text and a choice of goal.

use perception_wire::navigation::GoalTarget;
use serde::Deserialize;

use crate::config::ModelName;
use crate::digest::{describe_target, Digest};
use crate::model::Prompt;

pub const QUESTION_HEADING: &str = "QUESTION";
pub const ANSWER_SENT_HEADING: &str = "ANSWER SENT";
pub const LATER_ANSWER_HEADING: &str = "LATER ANSWER";

/// One answer as the reconciler sees it.
#[derive(Clone, Copy, Debug)]
pub struct Candidate<'a> {
    pub model: &'a ModelName,
    pub text: &'a str,
    pub goal: Option<GoalTarget>,
}

/// The line that opens the later answer of a reconciler prompt.
pub fn later_answer_marker(model: &ModelName) -> String {
    format!("{LATER_ANSWER_HEADING} ({model}):")
}

fn goal_text(goal: Option<GoalTarget>) -> String {
    goal.map_or("none".to_owned(), describe_target)
}

pub fn reconcile_prompt(digest: &Digest, question: &str, sent: Candidate, later: Candidate) -> Prompt {
    Prompt(format!(
        "{digest}\n\n{QUESTION_HEADING}: {question}\n\n{ANSWER_SENT_HEADING} ({}): {}\nGOAL SENT: {}\n\n{} {}\nGOAL PROPOSED: {}",
        sent.model,
        sent.text,
        goal_text(sent.goal),
        later_answer_marker(later.model),
        later.text,
        goal_text(later.goal),
    ))
}

/// Which goal the robot should be on after a revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalChoice {
    /// The goal already sent stays.
    Sent,
    /// The later answer's proposal replaces it.
    Proposed,
    /// The robot stops.
    None,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum Verdict {
    Keep,
    Revise { text: String, goal: GoalChoice },
}

#[derive(Debug, thiserror::Error)]
#[error("the reconciler's reply is not a verdict ({source}): {reply:?}")]
pub struct VerdictError {
    pub reply: String,
    pub source: serde_json::Error,
}

/// The verdict in a reply that is one JSON object, bare or in one fenced code block.
pub fn parse_verdict(reply: &str) -> Result<Verdict, VerdictError> {
    let trimmed = reply.trim();
    let json = match trimmed.strip_prefix("```") {
        Some(fenced) => {
            let body = fenced.split_once('\n').map_or("", |(_, body)| body);
            body.trim_end().strip_suffix("```").unwrap_or(body)
        }
        None => trimmed,
    };
    serde_json::from_str(json).map_err(|source| VerdictError { reply: reply.to_owned(), source })
}

#[cfg(test)]
mod tests {
    use perception_wire::navigation::{ObjectId, PlaceId};

    use super::*;

    #[test]
    fn the_prompt_holds_the_digest_question_and_both_answers_with_their_goals() {
        let (fast, slow) = (ModelName("claude-haiku-5-5".to_owned()), ModelName("claude-opus-5-5".to_owned()));
        let prompt = reconcile_prompt(
            &Digest::from_text("WORLD DIGEST as_of now"),
            "Where is my mug?",
            Candidate { model: &fast, text: "On the table o42.", goal: Some(GoalTarget::Object(ObjectId(42))) },
            Candidate { model: &slow, text: "In the sink, by p1.", goal: Some(GoalTarget::Place(PlaceId(1))) },
        );
        assert_eq!(
            prompt.0,
            "WORLD DIGEST as_of now\n\nQUESTION: Where is my mug?\n\n\
             ANSWER SENT (claude-haiku-5-5): On the table o42.\nGOAL SENT: object o42\n\n\
             LATER ANSWER (claude-opus-5-5): In the sink, by p1.\nGOAL PROPOSED: place p1"
        );
    }

    #[test]
    fn verdicts_parse_bare_or_fenced() {
        assert_eq!(parse_verdict(r#" {"decision": "keep"} "#).unwrap(), Verdict::Keep);
        let fenced = "```json\n{\"decision\": \"revise\", \"text\": \"In the sink.\", \"goal\": \"proposed\"}\n```";
        assert_eq!(
            parse_verdict(fenced).unwrap(),
            Verdict::Revise { text: "In the sink.".to_owned(), goal: GoalChoice::Proposed }
        );
    }

    #[test]
    fn anything_but_a_verdict_is_an_error() {
        for reply in ["Keep it.", r#"{"decision": "maybe"}"#, r#"{"decision": "revise", "text": "x"}"#] {
            assert!(parse_verdict(reply).is_err(), "{reply}");
        }
    }
}
