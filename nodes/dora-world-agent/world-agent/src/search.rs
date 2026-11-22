//! Searching objects by meaning: `find` embeds its query text with the model the objects' descriptions and views were
//! embedded with, and the world model compares the vectors.

use std::future::Future;
use std::pin::Pin;

use crate::world::QueryEmbedding;

pub type QueryEmbeddingError = Box<dyn std::error::Error + Send + Sync>;

/// Embeds the text of a `find` query in the space of the stored object embeddings.
pub trait QueryEmbedder: Send + Sync + 'static {
    fn embed_query<'a>(&'a self, text: &'a str) -> Pin<Box<dyn Future<Output = Result<QueryEmbedding, QueryEmbeddingError>> + Send + 'a>>;
}
