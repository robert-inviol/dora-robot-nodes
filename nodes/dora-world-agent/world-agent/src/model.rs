//! The one seam between the spatial agent and whatever runs the models. Tools, the digest, goals, the question desk
//! and the room namer depend only on `ModelClient`; `claude_code::ClaudeCodeClient` runs the Claude Code CLI, and a
//! backend on the Messages API would implement the same trait, calling the tools of `mcp::ToolContext` itself.

use std::fmt;
use std::future::Future;

use perception_wire::navigation::{ObjectId, QuestionId, RoomId};
use serde_json::Value;

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
    DescribeObject { object: ObjectId },
}

/// An image a prompt shows the model.
#[derive(Clone, PartialEq, Eq)]
pub enum PromptImage {
    Jpeg(Vec<u8>),
}

impl PromptImage {
    pub fn media_type(&self) -> &'static str {
        match self {
            PromptImage::Jpeg(_) => "image/jpeg",
        }
    }

    pub fn bytes(&self) -> &[u8] {
        match self {
            PromptImage::Jpeg(bytes) => bytes,
        }
    }
}

impl fmt::Debug for PromptImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PromptImage({}, {} bytes)", self.media_type(), self.bytes().len())
    }
}

/// The user message of one model call: an image, if the call is about one, then the text. Each call is a fresh
/// session holding only this message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prompt {
    pub image: Option<PromptImage>,
    pub text: String,
}

impl Prompt {
    pub fn text(text: impl Into<String>) -> Self {
        Self { image: None, text: text.into() }
    }

    pub fn about_image(image: PromptImage, text: impl Into<String>) -> Self {
        Self { image: Some(image), text: text.into() }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spend {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelReply {
    pub text: String,
    /// The reply in the shape of the client's JSON schema, for a client given one.
    pub structured: Option<Value>,
    pub spend: Spend,
}

/// One configured model. Every `ask` is independent: nothing carries over from one call to the next.
pub trait ModelClient: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn model(&self) -> &ModelName;

    fn ask(&self, duty: Duty, prompt: Prompt) -> impl Future<Output = Result<ModelReply, Self::Error>> + Send;
}
