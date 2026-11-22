//! Describing and embedding the objects of the world store, so the spatial agent can find them by meaning.
//!
//! Each view of an object the object map sends (on appearing, and again for a clearly better view) is described by a
//! vision model (`describer`, Claude through the Claude Code CLI) from the image and one line of facts (`facts`); the
//! description's text and the view's image are each embedded by Gemini (`embedder`, `gemini`), into one vector space
//! that the text of a `find` query is embedded into as well. `service::ObjectDescriptions` runs this in the
//! background behind a bounded queue and keeps the results in the world store. People are never described.

pub mod describer;
pub mod embedder;
pub mod facts;
pub mod gemini;
pub mod service;
pub mod try_cli;
