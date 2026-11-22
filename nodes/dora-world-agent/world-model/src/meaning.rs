//! What an object is, in words and as vectors: the description a vision model gave its best view, and the embeddings
//! that let a question find it by meaning rather than by its segmentation label.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A vision model's description of one view of an object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectCaption {
    /// What the object is, in a few words.
    pub name: String,
    /// The kind of thing it is.
    pub category: String,
    /// One or two sentences.
    pub description: String,
    pub colour: String,
    pub material: String,
}

/// The embedding model a vector came from, by the id its API takes. Vectors of different models are never compared.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EmbeddingModel(pub String);

impl fmt::Display for EmbeddingModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BadEmbedding {
    #[error("an embedding has no values")]
    Empty,
    #[error("an embedding value is not finite")]
    NotFinite,
    #[error("an embedding is all zeros and points nowhere")]
    Zero,
    #[error("{0} bytes are not a whole number of float32 values")]
    NotWholeFloats(usize),
}

/// A vector an embedding model gave one input. Inputs that mean the same point the same way.
#[derive(Clone, PartialEq)]
pub struct Embedding(Vec<f32>);

impl Embedding {
    pub fn new(values: Vec<f32>) -> Result<Self, BadEmbedding> {
        if values.is_empty() {
            return Err(BadEmbedding::Empty);
        }
        if !values.iter().all(|v| v.is_finite()) {
            return Err(BadEmbedding::NotFinite);
        }
        if values.iter().all(|&v| v == 0.0) {
            return Err(BadEmbedding::Zero);
        }
        Ok(Self(values))
    }

    pub fn values(&self) -> &[f32] {
        &self.0
    }

    pub fn dimensions(&self) -> usize {
        self.0.len()
    }

    /// The cosine of the angle between the two vectors. Both must have the same number of dimensions.
    pub fn similarity(&self, other: &Embedding) -> Similarity {
        assert_eq!(self.dimensions(), other.dimensions(), "only embeddings of one model are compared");
        let dot: f64 = self.0.iter().zip(&other.0).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
        let norm = |values: &[f32]| values.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
        Similarity((dot / (norm(&self.0) * norm(&other.0))).clamp(-1.0, 1.0) as f32)
    }

    /// Little-endian f32s, as the world store keeps them.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        self.0.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, BadEmbedding> {
        if !bytes.len().is_multiple_of(4) {
            return Err(BadEmbedding::NotWholeFloats(bytes.len()));
        }
        Self::new(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

impl fmt::Debug for Embedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Embedding({} dimensions)", self.0.len())
    }
}

/// The text of a question, embedded by the model the stored vectors came from.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryEmbedding {
    pub model: EmbeddingModel,
    pub vector: Embedding,
}

/// Cosine similarity between two embeddings, from -1 to 1; higher means closer in meaning.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct Similarity(f32);

impl Similarity {
    pub fn get(self) -> f32 {
        self.0
    }
}

/// The share of a query's words found in an object's label or description, from 0 to 1.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct WordShare(f32);

/// The words of a text: its runs of letters and digits, lower-cased, so "Air-intake" holds "air" and "intake".
pub fn words_of(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric()).filter(|word| !word.is_empty()).map(str::to_lowercase)
}

impl WordShare {
    pub const NONE: WordShare = WordShare(0.0);
    pub const ALL: WordShare = WordShare(1.0);

    /// The share of the query's words that are whole words of any of the texts, case ignored: "red" is not in
    /// "colored" and "cup" is not in "cupboard". A query with no words matches nothing.
    pub fn of_query(query: &str, texts: &[&str]) -> Self {
        let held: std::collections::HashSet<String> = texts.iter().flat_map(|text| words_of(text)).collect();
        let asked: Vec<String> = words_of(query).collect();
        match asked.len() {
            0 => WordShare::NONE,
            n => WordShare(asked.iter().filter(|word| held.contains(*word)).count() as f32 / n as f32),
        }
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

/// What made `find` return an object: the words of the query it holds, and how close the query's meaning is to its
/// description and to its best view, where those have been embedded.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct MatchScores {
    pub words: WordShare,
    pub description: Option<Similarity>,
    pub view: Option<Similarity>,
}

/// With a query embedding, a found object's score is its closest similarity, to its description or to its view (0
/// when neither is embedded yet), plus this much times the share of query words in its label or description: words
/// separate objects whose meanings are equally close, and never lift a word-only match above a clear match by meaning.
pub const WORD_MATCH_WEIGHT: f32 = 0.1;

impl MatchScores {
    pub fn by_words(words: WordShare) -> Self {
        Self { words, description: None, view: None }
    }

    /// Whether the object matched at all: some words, or a vector to compare with.
    pub fn is_match(&self) -> bool {
        self.words.get() > 0.0 || self.description.is_some() || self.view.is_some()
    }

    /// The closest similarity, to the description or to the view, if either is embedded.
    pub fn closest(&self) -> Option<Similarity> {
        [self.description, self.view].into_iter().flatten().reduce(|a, b| if b.get() > a.get() { b } else { a })
    }

    /// The score of a query that has an embedding: the closest similarity, or 0, plus `WORD_MATCH_WEIGHT` times the
    /// share of words found.
    pub fn score_by_meaning(&self) -> f32 {
        self.closest().map_or(0.0, Similarity::get) + WORD_MATCH_WEIGHT * self.words.get()
    }

    /// The score of a query without an embedding: the share of words found.
    pub fn score_by_words(&self) -> f32 {
        self.words.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedding(values: &[f32]) -> Embedding {
        Embedding::new(values.to_vec()).unwrap()
    }

    #[test]
    fn similarity_is_the_cosine_of_the_angle_between_vectors() {
        let x = embedding(&[2.0, 0.0]);
        assert!((x.similarity(&embedding(&[5.0, 0.0])).get() - 1.0).abs() < 1e-6);
        assert!(x.similarity(&embedding(&[0.0, 3.0])).get().abs() < 1e-6);
        assert!((x.similarity(&embedding(&[-1.0, 0.0])).get() + 1.0).abs() < 1e-6);
    }

    #[test]
    fn an_embedding_round_trips_through_its_stored_bytes() {
        let vector = embedding(&[0.25, -1.5, 3.0]);
        assert_eq!(Embedding::from_le_bytes(&vector.to_le_bytes()).unwrap(), vector);
    }

    #[test]
    fn an_empty_or_zero_vector_is_not_an_embedding() {
        assert_eq!(Embedding::new(Vec::new()), Err(BadEmbedding::Empty));
        assert_eq!(Embedding::new(vec![0.0, 0.0]), Err(BadEmbedding::Zero));
        assert_eq!(Embedding::new(vec![f32::NAN]), Err(BadEmbedding::NotFinite));
    }

    #[test]
    fn query_words_match_whole_words_only_whatever_their_case() {
        assert_eq!(WordShare::of_query("red", &["a light-colored duck"]), WordShare::NONE);
        assert_eq!(WordShare::of_query("cup", &["cupboard"]), WordShare::NONE);
        assert_eq!(WordShare::of_query("Red CUP", &["mug", "A red cup."]), WordShare::ALL);
        assert_eq!(WordShare::of_query("air", &["Air-intake grille"]), WordShare::ALL);
        assert_eq!(WordShare::of_query("somewhere to put rubbish", &["a bin to put rubbish in"]).get(), 0.75);
        assert_eq!(WordShare::of_query("%", &["anything"]), WordShare::NONE, "a query with no words matches nothing");
    }

    #[test]
    fn a_match_by_meaning_outscores_a_match_by_words_alone() {
        let by_meaning = MatchScores { words: WordShare::NONE, description: Some(Similarity(0.8)), view: Some(Similarity(0.6)) };
        let by_words = MatchScores { words: WordShare::of_query("red cup", &["cup"]), description: Some(Similarity(0.3)), view: None };
        let not_embedded = MatchScores::by_words(WordShare::ALL);
        assert!((by_meaning.score_by_meaning() - 0.8).abs() < 1e-6);
        assert!((by_words.score_by_meaning() - 0.35).abs() < 1e-6);
        assert!((not_embedded.score_by_meaning() - WORD_MATCH_WEIGHT).abs() < 1e-6);
        assert!(by_meaning.score_by_meaning() > by_words.score_by_meaning());
        assert!((MatchScores::by_words(WordShare::of_query("red cup", &["a cup"])).score_by_words() - 0.5).abs() < 1e-6);
        assert!(!MatchScores::by_words(WordShare::NONE).is_match());
    }
}
