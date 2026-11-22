//! The Gemini embedder against a local server standing in for the API: what it sends, and what it makes of answers.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, Uri};
use axum::response::IntoResponse;
use object_descriptions::embedder::{Embedder, EmbeddingInput};
use object_descriptions::gemini::{GeminiApiKey, GeminiEmbedder};
use serde_json::{json, Value};
use world_agent::config::EmbeddingDimensions;
use world_model::EmbeddingModel;

/// One request as the stand-in received it.
#[derive(Clone, Debug)]
struct Received {
    path: String,
    key: Option<String>,
    body: Value,
}

/// What the stand-in answers: status, body, and how long it waits first.
type Answer = (StatusCode, String, Duration);

/// What the stand-in answers, and every request it received.
#[derive(Clone)]
struct StandIn {
    answer: Arc<dyn Fn() -> Answer + Send + Sync>,
    received: Arc<Mutex<Vec<Received>>>,
}

async fn answer(State(stand_in): State<StandIn>, uri: Uri, headers: HeaderMap, body: Bytes) -> impl IntoResponse {
    let key = headers.get("x-goog-api-key").map(|v| v.to_str().unwrap().to_owned());
    stand_in.received.lock().unwrap().push(Received { path: uri.path().to_owned(), key, body: serde_json::from_slice(&body).unwrap() });
    let (status, body, delay) = (stand_in.answer)();
    tokio::time::sleep(delay).await;
    (status, body)
}

async fn serve(answer_with: impl Fn() -> Answer + Send + Sync + 'static) -> (GeminiEmbedder, StandIn) {
    let stand_in = StandIn { answer: Arc::new(answer_with), received: Arc::default() };
    let app = axum::Router::new().fallback(answer).with_state(stand_in.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let embedder = GeminiEmbedder::at_endpoint(
        &format!("http://{address}/v1beta"),
        GeminiApiKey::from_variable(Some("test-key".into())).unwrap(),
        &EmbeddingModel("gemini-embedding-2".to_owned()),
        EmbeddingDimensions::new(128).unwrap(),
        Duration::from_millis(300),
    )
    .unwrap();
    (embedder, stand_in)
}

fn values(n: usize) -> String {
    json!({"embedding": {"values": vec![0.25; n]}}).to_string()
}

#[tokio::test]
async fn a_query_is_posted_to_embed_content_with_the_key_in_its_header() {
    let (embedder, stand_in) = serve(|| (StatusCode::OK, values(128), Duration::ZERO)).await;
    let embedding = embedder.embed(EmbeddingInput::Query("red cup")).await.unwrap();
    assert_eq!(embedding.dimensions(), 128);
    let received = stand_in.received.lock().unwrap().clone();
    let Received { path, key, body } = &received[0];
    assert_eq!(path, "/v1beta/models/gemini-embedding-2:embedContent");
    assert_eq!(key.as_deref(), Some("test-key"));
    assert_eq!(body["content"]["parts"][0]["text"], "task: search result | query: red cup");
    assert_eq!(body["output_dimensionality"], 128);
}

#[tokio::test]
async fn an_error_answer_is_an_error_that_carries_its_status_and_message() {
    let (embedder, _) = serve(|| (StatusCode::TOO_MANY_REQUESTS, r#"{"error": {"message": "Resource exhausted"}}"#.to_owned(), Duration::ZERO)).await;
    let error = embedder.embed(EmbeddingInput::Query("red cup")).await.unwrap_err().to_string();
    assert!(error.contains("429") && error.contains("Resource exhausted"), "{error}");
}

#[tokio::test]
async fn an_answer_slower_than_the_timeout_is_an_error() {
    let (embedder, _) = serve(|| (StatusCode::OK, values(128), Duration::from_secs(2))).await;
    let started = std::time::Instant::now();
    let error = embedder.embed(EmbeddingInput::Query("red cup")).await.unwrap_err();
    assert!(started.elapsed() < Duration::from_secs(1), "{error}");
}

#[tokio::test]
async fn a_vector_of_another_size_is_an_error_not_an_embedding() {
    let (embedder, _) = serve(|| (StatusCode::OK, values(3072), Duration::ZERO)).await;
    let error = embedder.embed(EmbeddingInput::Query("red cup")).await.unwrap_err().to_string();
    assert!(error.contains("3072 values, not the 128"), "{error}");
}
