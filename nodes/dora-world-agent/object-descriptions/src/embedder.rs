//! The embedding seam: what is embedded, and the trait the Gemini client and the tests' fakes implement.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use world_agent::search::{QueryEmbedder, QueryEmbeddingError};
use world_model::{Embedding, EmbeddingModel, ObjectCaption, QueryEmbedding};
use world_store::JpegImage;

/// One input to embed. Each kind is embedded the way the model's documentation asks for its role in a search.
#[derive(Clone, Copy, Debug)]
pub enum EmbeddingInput<'a> {
    /// The text of a `find` query.
    Query(&'a str),
    /// An object's description, searched for.
    Description(&'a ObjectCaption),
    /// The view that was described, searched for.
    View(&'a JpegImage),
}

impl EmbeddingInput<'_> {
    pub fn kind(&self) -> &'static str {
        match self {
            EmbeddingInput::Query(_) => "query",
            EmbeddingInput::Description(_) => "description",
            EmbeddingInput::View(_) => "view",
        }
    }
}

pub type EmbedError = Box<dyn std::error::Error + Send + Sync>;

/// An embedding model that maps text and images into one vector space.
pub trait Embedder: Send + Sync + 'static {
    fn model(&self) -> &EmbeddingModel;

    fn embed(&self, input: EmbeddingInput<'_>) -> impl Future<Output = Result<Embedding, EmbedError>> + Send;
}

/// An embedder as the `find` tool's query embedder.
pub struct QueriesOf<E>(pub Arc<E>);

impl<E: Embedder> QueryEmbedder for QueriesOf<E> {
    fn embed_query<'a>(&'a self, text: &'a str) -> Pin<Box<dyn Future<Output = Result<QueryEmbedding, QueryEmbeddingError>> + Send + 'a>> {
        Box::pin(async move {
            let vector = self.0.embed(EmbeddingInput::Query(text)).await?;
            Ok(QueryEmbedding { model: self.0.model().clone(), vector })
        })
    }
}
