//! `Embedder` on the Gemini API's `embedContent` method, over HTTPS.
//!
//! One request per input: `POST {endpoint}/models/{model}:embedContent` with the key in the `x-goog-api-key` header
//! and the body `{"content": {"parts": [part]}, "output_dimensionality": n}`, where the part is `{"text": ...}` or, for
//! an image, `{"inline_data": {"mime_type": "image/jpeg", "data": base64}}`. The answer is
//! `{"embedding": {"values": [...]}}`. gemini-embedding-2 takes no task type: a query and a document are told apart by
//! a prefix on their text, and an image goes without one. Every request has a timeout; an answer that is not a
//! success, or not a vector of the requested size, is an error.

use std::ffi::OsString;
use std::fmt;
use std::time::Duration;

use base64::Engine;
use serde_json::{json, Value};
use world_agent::config::{EmbeddingConfig, EmbeddingDimensions};
use world_model::{BadEmbedding, Embedding, EmbeddingModel};

use crate::embedder::{EmbedError, Embedder, EmbeddingInput};

/// The environment variable the Gemini API key is read from.
pub const GEMINI_API_KEY_VARIABLE: &str = "GEMINI_API_KEY";

/// The Gemini API, version v1beta, which serves `embedContent`.
pub const GEMINI_ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta";

/// How much of an error answer's body goes into the error, enough to read Google's message.
const ERROR_BODY_CHARS: usize = 500;

/// A Gemini API key. It never appears in logs or debug output.
#[derive(Clone)]
pub struct GeminiApiKey(String);

impl GeminiApiKey {
    /// The key in an environment variable's value; none when it is unset or blank.
    pub fn from_variable(value: Option<OsString>) -> Option<Self> {
        let value = value?.into_string().ok()?;
        let key = value.trim();
        (!key.is_empty()).then(|| Self(key.to_owned()))
    }
}

impl fmt::Debug for GeminiApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("GeminiApiKey(..)")
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GeminiError {
    #[error("the embedding request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("the Gemini API answered {status}: {body}")]
    Status { status: reqwest::StatusCode, body: String },
    #[error("the Gemini API's answer is not an embedding: {0}")]
    NotAnEmbedding(String),
    #[error("the Gemini API gave {received} values, not the {requested} asked for")]
    WrongSize { requested: u32, received: usize },
    #[error("the Gemini API gave an unusable vector: {0}")]
    BadVector(#[from] BadEmbedding),
}

/// The text gemini-embedding-2's documentation gives for a retrieval query.
fn query_text(query: &str) -> String {
    format!("task: search result | query: {query}")
}

/// The text gemini-embedding-2's documentation gives for a document searched for, titled with the object's name.
fn description_text(caption: &world_model::ObjectCaption) -> String {
    format!(
        "title: {} | text: {} Colour: {}. Material: {}.",
        caption.name, caption.description, caption.colour, caption.material
    )
}

/// The body of one `embedContent` request.
pub fn embed_content_request(input: EmbeddingInput<'_>, dimensions: EmbeddingDimensions) -> Value {
    let part = match input {
        EmbeddingInput::Query(query) => json!({"text": query_text(query)}),
        EmbeddingInput::Description(caption) => json!({"text": description_text(caption)}),
        EmbeddingInput::View(jpeg) => json!({"inline_data": {
            "mime_type": "image/jpeg",
            "data": base64::engine::general_purpose::STANDARD.encode(&jpeg.0),
        }}),
    };
    json!({"content": {"parts": [part]}, "output_dimensionality": dimensions.get()})
}

/// The vector of one `embedContent` answer, which must have the requested size.
pub fn embedding_from_answer(body: &str, dimensions: EmbeddingDimensions) -> Result<Embedding, GeminiError> {
    let answer: Value = serde_json::from_str(body).map_err(|e| GeminiError::NotAnEmbedding(e.to_string()))?;
    let values = answer["embedding"]["values"]
        .as_array()
        .ok_or_else(|| GeminiError::NotAnEmbedding("no embedding.values".to_owned()))?
        .iter()
        .map(|v| v.as_f64().map(|v| v as f32).ok_or_else(|| GeminiError::NotAnEmbedding(format!("{v} is not a number"))))
        .collect::<Result<Vec<f32>, _>>()?;
    if values.len() != dimensions.get() as usize {
        return Err(GeminiError::WrongSize { requested: dimensions.get(), received: values.len() });
    }
    Ok(Embedding::new(values)?)
}

pub struct GeminiEmbedder {
    http: reqwest::Client,
    key: GeminiApiKey,
    endpoint: String,
    model: EmbeddingModel,
    dimensions: EmbeddingDimensions,
}

impl GeminiEmbedder {
    pub fn new(key: GeminiApiKey, config: &EmbeddingConfig) -> Result<Self, reqwest::Error> {
        Self::at_endpoint(GEMINI_ENDPOINT, key, &config.model, config.dimensions, config.request_timeout)
    }

    /// The embedder on another endpoint serving the same API, such as a test server.
    pub fn at_endpoint(
        endpoint: &str,
        key: GeminiApiKey,
        model: &EmbeddingModel,
        dimensions: EmbeddingDimensions,
        request_timeout: Duration,
    ) -> Result<Self, reqwest::Error> {
        let http = reqwest::Client::builder().timeout(request_timeout).build()?;
        Ok(Self { http, key, endpoint: endpoint.trim_end_matches('/').to_owned(), model: model.clone(), dimensions })
    }

    async fn embed_content(&self, input: EmbeddingInput<'_>) -> Result<Embedding, GeminiError> {
        let url = format!("{}/models/{}:embedContent", self.endpoint, self.model.0);
        let response = self
            .http
            .post(url)
            .header("x-goog-api-key", &self.key.0)
            .json(&embed_content_request(input, self.dimensions))
            .send()
            .await?;
        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            return Err(GeminiError::Status { status, body: body.chars().take(ERROR_BODY_CHARS).collect() });
        }
        embedding_from_answer(&body, self.dimensions)
    }
}

impl Embedder for GeminiEmbedder {
    fn model(&self) -> &EmbeddingModel {
        &self.model
    }

    async fn embed(&self, input: EmbeddingInput<'_>) -> Result<Embedding, EmbedError> {
        Ok(self.embed_content(input).await?)
    }
}

#[cfg(test)]
mod tests {
    use world_model::ObjectCaption;
    use world_store::JpegImage;

    use super::*;

    fn dimensions() -> EmbeddingDimensions {
        EmbeddingDimensions::new(128).unwrap()
    }

    #[test]
    fn a_query_is_embedded_as_a_search_query() {
        assert_eq!(
            embed_content_request(EmbeddingInput::Query("red cup"), dimensions()),
            json!({"content": {"parts": [{"text": "task: search result | query: red cup"}]}, "output_dimensionality": 128})
        );
    }

    #[test]
    fn a_description_is_embedded_as_a_document_titled_with_the_name() {
        let caption = ObjectCaption {
            name: "red mug".to_owned(),
            category: "kitchenware".to_owned(),
            description: "A red ceramic mug on a table.".to_owned(),
            colour: "red".to_owned(),
            material: "ceramic".to_owned(),
        };
        let request = embed_content_request(EmbeddingInput::Description(&caption), dimensions());
        assert_eq!(
            request["content"]["parts"][0]["text"],
            "title: red mug | text: A red ceramic mug on a table. Colour: red. Material: ceramic."
        );
    }

    #[test]
    fn a_view_is_embedded_as_inline_jpeg_bytes_without_a_prefix() {
        let request = embed_content_request(EmbeddingInput::View(&JpegImage(vec![0xFF, 0xD8, 0xFF])), dimensions());
        assert_eq!(request["content"]["parts"][0], json!({"inline_data": {"mime_type": "image/jpeg", "data": "/9j/"}}));
        assert_eq!(request["output_dimensionality"], 128);
    }

    #[test]
    fn an_answer_of_the_requested_size_is_an_embedding() {
        let values: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let body = json!({"embedding": {"values": values}}).to_string();
        assert_eq!(embedding_from_answer(&body, dimensions()).unwrap().dimensions(), 128);
    }

    #[test]
    fn an_answer_of_another_size_or_shape_is_an_error() {
        let short = json!({"embedding": {"values": [0.5, 0.5]}}).to_string();
        assert!(matches!(embedding_from_answer(&short, dimensions()), Err(GeminiError::WrongSize { requested: 128, received: 2 })));
        assert!(matches!(embedding_from_answer(r#"{"embeddings": []}"#, dimensions()), Err(GeminiError::NotAnEmbedding(_))));
        let empty = json!({"embedding": {"values": []}}).to_string();
        assert!(matches!(embedding_from_answer(&empty, dimensions()), Err(GeminiError::WrongSize { .. })));
    }

    #[test]
    fn a_key_is_read_from_a_set_variable_and_never_shown() {
        assert!(GeminiApiKey::from_variable(None).is_none());
        assert!(GeminiApiKey::from_variable(Some(" \n".into())).is_none());
        let key = GeminiApiKey::from_variable(Some("secret-value\n".into())).unwrap();
        assert_eq!(format!("{key:?}"), "GeminiApiKey(..)");
    }
}
