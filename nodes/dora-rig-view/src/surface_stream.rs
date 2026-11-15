//! Following the dense map's `mesh_blocks` messages. They are a stream of changes, and a full dora input queue
//! drops its oldest message without a word, so each message's run and sequence say whether the surface held is
//! still the dense map's. A missed message, or a run joined after its first message, leaves the held surface
//! untrusted until a snapshot of the whole surface replaces it. One snapshot request is outstanding at a time; it is
//! sent again only once [`SNAPSHOT_REQUEST_DEADLINE`] passes without a snapshot.

use std::fmt;
use std::time::{Duration, Instant};

use perception_wire::navigation::UnixNanos;
use perception_wire::space_map::{MeshBlocks, MeshBlocksKind, MeshSequence};

/// How long a snapshot request may go unanswered before it is sent again.
pub const SNAPSHOT_REQUEST_DEADLINE: Duration = Duration::from_secs(3);

/// Why the surface held may not be the dense map's, and so why a snapshot is asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Untrusted {
    /// Messages of the run were missed: `expected` was next and `came` came.
    MissedMessages { expected: MeshSequence, came: MeshSequence },
    /// The first message seen of a run was not its first.
    JoinedLate { first_seen: MeshSequence },
}

impl fmt::Display for Untrusted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Untrusted::MissedMessages { expected, came } if came.0 - expected.0 == 1 => {
                write!(f, "missed surface message {expected}; the surface drawn may be out of date")
            }
            Untrusted::MissedMessages { expected, came } => {
                write!(f, "missed surface messages {expected} to {}; the surface drawn may be out of date", came.0 - 1)
            }
            Untrusted::JoinedLate { first_seen } => {
                write!(f, "joined the dense map's run at surface message {first_seen}; blocks sent before it are missing")
            }
        }
    }
}

/// Why a snapshot request goes out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotRequestReason {
    /// The surface held has just become untrusted.
    Untrusted(Untrusted),
    /// The outstanding request went unanswered for [`SNAPSHOT_REQUEST_DEADLINE`].
    DeadlinePassed,
}

/// What to do with one message's mesh blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceApplication {
    /// Lay them over the surface held: the message holds changes.
    Changes,
    /// Put them in place of the surface held: the message is a snapshot or the first of a new run.
    Replace,
}

/// The stream as the page shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceStreamStatus {
    /// Set while a snapshot is awaited.
    pub untrusted: Option<Untrusted>,
    /// Gaps in the sequence seen, each one or more missed messages.
    pub gaps: u64,
    /// Snapshots that replaced the surface held.
    pub snapshots: u64,
}

/// What one message means for the surface held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceStep {
    pub application: SurfaceApplication,
    /// A snapshot request to send now.
    pub request: Option<SnapshotRequestReason>,
    pub status: SurfaceStreamStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SurfaceStreamError {
    #[error("a mesh_blocks message of the dense map run started at {came:?} came after one of the later run started at {held:?}")]
    RunWentBack { held: UnixNanos, came: UnixNanos },
    #[error("mesh_blocks went back from {expected} to {came} within one run")]
    SequenceWentBack { expected: MeshSequence, came: MeshSequence },
}

/// Where in the dense map's messages the stream is.
#[derive(Clone, Copy, Debug)]
struct Position {
    run_started: UnixNanos,
    next: MeshSequence,
}

#[derive(Clone, Copy, Debug)]
struct AwaitedSnapshot {
    requested_at: Instant,
    untrusted: Untrusted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Continuity {
    FirstOfRun,
    Next,
    Gap { expected: MeshSequence },
}

#[derive(Default)]
pub struct SurfaceStream {
    position: Option<Position>,
    awaited: Option<AwaitedSnapshot>,
    gaps: u64,
    missed: u64,
    snapshots: u64,
    requests: u64,
}

impl SurfaceStream {
    pub fn status(&self) -> SurfaceStreamStatus {
        SurfaceStreamStatus { untrusted: self.awaited.map(|awaited| awaited.untrusted), gaps: self.gaps, snapshots: self.snapshots }
    }

    /// Checks one message's run and sequence against the messages before it.
    pub fn on_message(&mut self, surface: &MeshBlocks, now: Instant) -> Result<SurfaceStep, SurfaceStreamError> {
        let continuity = match self.position {
            None => Continuity::FirstOfRun,
            Some(held) if surface.run_started < held.run_started => {
                return Err(SurfaceStreamError::RunWentBack { held: held.run_started, came: surface.run_started });
            }
            Some(held) if surface.run_started > held.run_started => {
                tracing::warn!(
                    held_run_started_unix_ns = held.run_started.0,
                    run_started_unix_ns = surface.run_started.0,
                    "the dense map started a new run; its surface replaces the one held"
                );
                Continuity::FirstOfRun
            }
            Some(held) if surface.sequence < held.next => {
                return Err(SurfaceStreamError::SequenceWentBack { expected: held.next, came: surface.sequence });
            }
            Some(held) if surface.sequence > held.next => Continuity::Gap { expected: held.next },
            Some(_) => Continuity::Next,
        };
        self.position = Some(Position { run_started: surface.run_started, next: surface.sequence.next() });
        if continuity == Continuity::FirstOfRun {
            // A request to an earlier run is answered by no one.
            self.awaited = None;
        }
        if let Continuity::Gap { expected } = continuity {
            self.gaps += 1;
            self.missed += surface.sequence.0 - expected.0;
            tracing::warn!(
                expected = expected.0,
                came = surface.sequence.0,
                gaps = self.gaps,
                missed = self.missed,
                "missed mesh_blocks messages; the surface held is out of date until a snapshot comes"
            );
        }
        let mut application = match continuity {
            Continuity::FirstOfRun => SurfaceApplication::Replace,
            Continuity::Next | Continuity::Gap { .. } => SurfaceApplication::Changes,
        };
        let mut request = None;
        match surface.kind {
            MeshBlocksKind::Snapshot => {
                application = SurfaceApplication::Replace;
                self.snapshots += 1;
                let waited = self.awaited.take().map(|awaited| now.duration_since(awaited.requested_at));
                tracing::info!(
                    sequence = surface.sequence.0,
                    blocks = surface.blocks.len(),
                    snapshots = self.snapshots,
                    waited_ms = waited.map(|waited| waited.as_millis() as u64),
                    "a mesh snapshot replaced the surface held"
                );
            }
            MeshBlocksKind::Changes => {
                let untrusted = match continuity {
                    Continuity::FirstOfRun if surface.sequence != MeshSequence(0) => Some(Untrusted::JoinedLate { first_seen: surface.sequence }),
                    Continuity::Gap { expected } => Some(Untrusted::MissedMessages { expected, came: surface.sequence }),
                    Continuity::FirstOfRun | Continuity::Next => None,
                };
                if let Some(untrusted) = untrusted.filter(|_| self.awaited.is_none()) {
                    self.awaited = Some(AwaitedSnapshot { requested_at: now, untrusted });
                    request = Some(SnapshotRequestReason::Untrusted(untrusted));
                    self.note_request();
                }
            }
        }
        Ok(SurfaceStep { application, request, status: self.status() })
    }

    /// A snapshot request to send again, when the outstanding one has gone unanswered past the deadline.
    pub fn on_time(&mut self, now: Instant) -> Option<SnapshotRequestReason> {
        let awaited = self.awaited.as_mut().filter(|awaited| now.duration_since(awaited.requested_at) >= SNAPSHOT_REQUEST_DEADLINE)?;
        awaited.requested_at = now;
        self.note_request();
        tracing::warn!(deadline_s = SNAPSHOT_REQUEST_DEADLINE.as_secs(), requests = self.requests, "no mesh snapshot came in time; asking again");
        Some(SnapshotRequestReason::DeadlinePassed)
    }

    fn note_request(&mut self) {
        self.requests += 1;
    }
}

#[cfg(test)]
mod tests {
    use perception_wire::navigation::MapVersion;

    use super::*;

    const RUN: UnixNanos = UnixNanos(1_000);

    fn message(run_started: UnixNanos, sequence: u64, kind: MeshBlocksKind) -> MeshBlocks {
        MeshBlocks {
            map_version: MapVersion(1),
            built: UnixNanos(2_000),
            run_started,
            sequence: MeshSequence(sequence),
            kind,
            voxel_m: 0.05,
            block_voxels: 8,
            blocks: Vec::new(),
        }
    }

    fn changes(sequence: u64) -> MeshBlocks {
        message(RUN, sequence, MeshBlocksKind::Changes)
    }

    #[test]
    fn a_contiguous_stream_asks_for_nothing() {
        let (mut stream, now) = (SurfaceStream::default(), Instant::now());
        let first = stream.on_message(&changes(0), now).unwrap();
        assert_eq!((first.application, first.request), (SurfaceApplication::Replace, None), "the run's first message starts the surface");
        for sequence in 1..20 {
            let step = stream.on_message(&changes(sequence), now).unwrap();
            assert_eq!((step.application, step.request), (SurfaceApplication::Changes, None));
        }
        assert_eq!(stream.on_time(now + 10 * SNAPSHOT_REQUEST_DEADLINE), None);
        assert_eq!(stream.status(), SurfaceStreamStatus { untrusted: None, gaps: 0, snapshots: 0 });
    }

    #[test]
    fn a_dropped_message_asks_for_exactly_one_snapshot_until_the_deadline_passes() {
        let (mut stream, now) = (SurfaceStream::default(), Instant::now());
        stream.on_message(&changes(0), now).unwrap();
        stream.on_message(&changes(1), now).unwrap();
        let missed = Untrusted::MissedMessages { expected: MeshSequence(2), came: MeshSequence(3) };
        let step = stream.on_message(&changes(3), now).unwrap();
        assert_eq!(step.request, Some(SnapshotRequestReason::Untrusted(missed)));
        assert_eq!(step.application, SurfaceApplication::Changes, "the surface held is still drawn, with the new mesh blocks");
        assert_eq!(step.status, SurfaceStreamStatus { untrusted: Some(missed), gaps: 1, snapshots: 0 });
        let later = stream.on_message(&changes(5), now + Duration::from_millis(200)).unwrap();
        assert_eq!(later.request, None, "one request is outstanding at a time");
        assert_eq!(later.status.gaps, 2);
        assert_eq!(stream.on_time(now + SNAPSHOT_REQUEST_DEADLINE - Duration::from_millis(1)), None);
        assert_eq!(stream.on_time(now + SNAPSHOT_REQUEST_DEADLINE), Some(SnapshotRequestReason::DeadlinePassed));
        assert_eq!(stream.on_time(now + SNAPSHOT_REQUEST_DEADLINE + Duration::from_millis(1)), None, "the deadline starts again");
    }

    #[test]
    fn a_snapshot_replaces_the_surface_and_trusts_it_again() {
        let (mut stream, now) = (SurfaceStream::default(), Instant::now());
        stream.on_message(&changes(0), now).unwrap();
        stream.on_message(&changes(2), now).unwrap();
        let snapshot = stream.on_message(&message(RUN, 3, MeshBlocksKind::Snapshot), now).unwrap();
        assert_eq!((snapshot.application, snapshot.request), (SurfaceApplication::Replace, None));
        assert_eq!(snapshot.status, SurfaceStreamStatus { untrusted: None, gaps: 1, snapshots: 1 });
        assert_eq!(stream.on_time(now + 10 * SNAPSHOT_REQUEST_DEADLINE), None, "nothing is awaited");
        let step = stream.on_message(&changes(4), now).unwrap();
        assert_eq!((step.application, step.request), (SurfaceApplication::Changes, None));
    }

    #[test]
    fn starting_after_the_dense_map_asks_for_a_snapshot() {
        let (mut stream, now) = (SurfaceStream::default(), Instant::now());
        let step = stream.on_message(&changes(41), now).unwrap();
        assert_eq!(step.application, SurfaceApplication::Replace);
        assert_eq!(step.request, Some(SnapshotRequestReason::Untrusted(Untrusted::JoinedLate { first_seen: MeshSequence(41) })));
        assert_eq!(step.status.gaps, 0, "joining late is not a gap");
    }

    #[test]
    fn a_restarted_dense_map_starts_the_surface_again() {
        let (mut stream, now) = (SurfaceStream::default(), Instant::now());
        stream.on_message(&changes(0), now).unwrap();
        stream.on_message(&changes(2), now).unwrap();
        let restarted = UnixNanos(RUN.0 + 1);
        let step = stream.on_message(&message(restarted, 0, MeshBlocksKind::Changes), now).unwrap();
        assert_eq!((step.application, step.request, step.status.untrusted), (SurfaceApplication::Replace, None, None));
        assert_eq!(stream.on_time(now + 10 * SNAPSHOT_REQUEST_DEADLINE), None, "the request to the old run is dropped");
        let joined_late = stream.on_message(&message(UnixNanos(RUN.0 + 2), 7, MeshBlocksKind::Changes), now).unwrap();
        assert_eq!(joined_late.application, SurfaceApplication::Replace);
        assert_eq!(joined_late.request, Some(SnapshotRequestReason::Untrusted(Untrusted::JoinedLate { first_seen: MeshSequence(7) })));
    }

    #[test]
    fn a_run_or_sequence_that_goes_back_is_refused() {
        let (mut stream, now) = (SurfaceStream::default(), Instant::now());
        stream.on_message(&changes(3), now).unwrap();
        assert_eq!(
            stream.on_message(&changes(3), now),
            Err(SurfaceStreamError::SequenceWentBack { expected: MeshSequence(4), came: MeshSequence(3) })
        );
        assert_eq!(
            stream.on_message(&message(UnixNanos(RUN.0 - 1), 0, MeshBlocksKind::Changes), now),
            Err(SurfaceStreamError::RunWentBack { held: RUN, came: UnixNanos(RUN.0 - 1) })
        );
    }
}
