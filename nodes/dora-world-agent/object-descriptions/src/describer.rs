//! The description seam, and the describer on a Claude Code client: one call per view, the image and one line of
//! facts in the user message, the reply's shape set by a JSON schema rather than by instructions.

use std::future::Future;
use std::time::Duration;

use serde_json::json;
use world_agent::config::{ModelName, ObjectDescriptionsConfig};
use world_agent::model::{Duty, ModelClient, Prompt, PromptImage};
use world_agent::process::{ProcessProfile, ReplySchema, ReplyShape, ToolAccess};
use world_model::ObjectCaption;
use world_store::JpegImage;

use crate::facts::ObjectFacts;
use perception_wire::navigation::ObjectId;

/// The whole instruction of a description call; the reply's shape comes from `caption_schema`.
pub const DESCRIBE_INSTRUCTION: &str = "Describe this object.";

#[derive(Debug, thiserror::Error)]
pub enum DescribeError {
    #[error("the description call failed: {0}")]
    Call(Box<dyn std::error::Error + Send + Sync>),
    #[error("the description call took longer than {0:?}")]
    TimedOut(Duration),
    #[error("the reply is not a description: {0}")]
    NotADescription(String),
}

/// A vision model that describes one view of an object.
pub trait Describer: Send + Sync + 'static {
    fn model(&self) -> &ModelName;

    fn describe(&self, object: ObjectId, view: &JpegImage, facts: &ObjectFacts) -> impl Future<Output = Result<ObjectCaption, DescribeError>> + Send;
}

impl<D: Describer> Describer for std::sync::Arc<D> {
    fn model(&self) -> &ModelName {
        self.as_ref().model()
    }

    fn describe(&self, object: ObjectId, view: &JpegImage, facts: &ObjectFacts) -> impl Future<Output = Result<ObjectCaption, DescribeError>> + Send {
        self.as_ref().describe(object, view, facts)
    }
}

/// The schema of `ObjectCaption`, which the describer's reply must match.
pub fn caption_schema() -> ReplySchema {
    let text = |description: &str| json!({"type": "string", "description": description});
    ReplySchema(json!({
        "type": "object",
        "properties": {
            "name": text("What the object is, in a few words."),
            "category": text("The kind of thing it is."),
            "description": text("One or two sentences."),
            "colour": text("Its main colour or colours."),
            "material": text("What it is mostly made of."),
        },
        "required": ["name", "category", "description", "colour", "material"],
        "additionalProperties": false,
    }))
}

/// How the describer's claude processes run: the configured model with no tools, replying in `caption_schema`.
pub fn describer_profile(config: &ObjectDescriptionsConfig) -> ProcessProfile {
    ProcessProfile {
        model: config.describer.model.clone(),
        effort: config.describer.effort,
        system_prompt: config.describer.system_prompt.clone(),
        tools: ToolAccess::None,
        reply: ReplyShape::Structured(caption_schema()),
    }
}

/// The user message of a description call: the view, then the facts line and the instruction.
pub fn describe_prompt(view: &JpegImage, facts: &ObjectFacts) -> Prompt {
    Prompt::about_image(PromptImage::Jpeg(view.0.clone()), format!("{}\n{DESCRIBE_INSTRUCTION}", facts.line()))
}

/// `Describer` on a model client set up with `describer_profile`.
pub struct ClaudeDescriber<C> {
    client: std::sync::Arc<C>,
    timeout: Duration,
}

impl<C: ModelClient> ClaudeDescriber<C> {
    pub fn new(client: std::sync::Arc<C>, timeout: Duration) -> Self {
        Self { client, timeout }
    }
}

impl<C: ModelClient> Describer for ClaudeDescriber<C> {
    fn model(&self) -> &ModelName {
        self.client.model()
    }

    async fn describe(&self, object: ObjectId, view: &JpegImage, facts: &ObjectFacts) -> Result<ObjectCaption, DescribeError> {
        let call = self.client.ask(Duty::DescribeObject { object }, describe_prompt(view, facts));
        let reply = tokio::time::timeout(self.timeout, call)
            .await
            .map_err(|_| DescribeError::TimedOut(self.timeout))?
            .map_err(|error| DescribeError::Call(Box::new(error)))?;
        let structured = reply.structured.ok_or_else(|| DescribeError::NotADescription("no structured output".to_owned()))?;
        serde_json::from_value(structured).map_err(|error| DescribeError::NotADescription(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use world_model::Size;

    use super::*;

    #[test]
    fn the_prompt_is_the_view_then_the_facts_and_the_instruction() {
        let facts = ObjectFacts { size: Size { long_m: 0.12, short_m: 0.08, height_m: 0.1 }, labels: vec!["cup".to_owned()] };
        let prompt = describe_prompt(&JpegImage(vec![0xFF, 0xD8]), &facts);
        assert_eq!(prompt.image, Some(PromptImage::Jpeg(vec![0xFF, 0xD8])));
        assert_eq!(prompt.text, "About 12 x 8 x 10 cm (long x short x high). Segmented as: cup.\nDescribe this object.");
    }

    #[test]
    fn a_reply_that_fills_the_schema_is_a_caption() {
        let schema = caption_schema().0;
        let required: Vec<&str> = schema["required"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        let reply = json!({"name": "red mug", "category": "kitchenware", "description": "A red mug.", "colour": "red", "material": "ceramic"});
        assert_eq!(required.len(), reply.as_object().unwrap().len());
        assert!(required.iter().all(|field| reply.get(field).is_some()));
        let caption: ObjectCaption = serde_json::from_value(reply).unwrap();
        assert_eq!(caption.material, "ceramic");
    }
}
