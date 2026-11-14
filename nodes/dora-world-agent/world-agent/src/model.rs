//! The one seam between the spatial agent and whatever runs the models. Tools, the digest, goals, the question desk
//! and the room namer depend only on `ModelClient`; `claude_code::ClaudeCodeClient` runs the Claude Code CLI, and a
//! backend on the Messages API would implement the same trait, calling the tools of `mcp::ToolContext` itself.

use std::fmt;
use std::future::Future;

use perception_wire::navigation::{QuestionId, RoomId};

use crate::config::{AnswererRole, ModelName};

/// Which answerer of the configured list a model call serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AnswererId(pub usize);

impl fmt::Display for AnswererId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "answerer {}", self.0)
    }
}

/// What a model call is for. It decides which tools the call may use and how `send_goal` behaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Duty {
    Answer { question: QuestionId, answerer: AnswererId, role: AnswererRole },
    Reconcile { question: QuestionId },
    NameRoom { room: RoomId },
}

/// The user message of one model call. Each call is a fresh session holding only this message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prompt(pub String);

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spend {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelReply {
    pub text: String,
    pub spend: Spend,
}

/// One configured model. Every `ask` is independent: nothing carries over from one call to the next.
pub trait ModelClient: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn model(&self) -> &ModelName;

    fn ask(&self, duty: Duty, prompt: Prompt) -> impl Future<Output = Result<ModelReply, Self::Error>> + Send;
}
