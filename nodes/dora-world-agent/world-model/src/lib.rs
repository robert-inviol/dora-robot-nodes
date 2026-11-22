//! The world model the spatial agents' tools read: its value types, the `WorldModel` trait, the clock they share, and
//! the descriptions and embeddings that let objects be found by meaning.

pub mod clock;
mod meaning;
mod world;

pub use meaning::*;
pub use world::*;
