//! The spatial agent: answers questions about the robot's surroundings with several models at once, each in a small,
//! fresh Claude Code session holding the world digest and the question, reconciles their answers, moves the robot
//! through nav goals, and names rooms. The world model's tools are served to the models over MCP from this process.

pub use world_model::clock;
pub use world_model as world;
pub mod claude_code;
pub mod config;
pub mod digest;
pub mod fake_world;
pub mod goals;
pub mod identity;
pub mod mcp;
pub mod model;
pub mod process;
pub mod protocol;
pub mod questions;
pub mod reconcile;
pub mod references;
pub mod room_names;
pub mod service;
