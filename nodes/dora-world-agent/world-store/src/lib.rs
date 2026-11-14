//! The world model in SQLite, behind the `WorldModel` trait.
//!
//! One writer thread owns the only read-write connection and commits ingest commands in batches; a small pool of
//! read-only connections and the in-memory map (topology, free-space grid, room names) answer the agents' tools.
//! Objects are kept as history rows, every change is an append-only event, and goals are recorded with their status.

mod geometry;
mod inputs;
mod map;
mod relations;
mod render;
pub mod schema;
mod sql;
mod store;
mod types;
mod wire;
mod writer;

pub use inputs::{Clock, ManualClock};
pub use map::{parse_unnamed_room, unnamed_room_name, UNMAPPED_ROOM};
pub use store::{
    back_up, BackupConfig, OpenError, RobotGeometry, StalenessBounds, StoreConfig, TrackBounds, WorldStore,
};
pub use types::*;
pub use wire::BadObjectUpdate;
pub use writer::{IngestError, Rejection, WorldWriter, WriterStats};
