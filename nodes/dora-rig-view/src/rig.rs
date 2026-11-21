//! The rig view as a whole: the messages it takes, and where the work for each happens. The robot, the people and
//! the other light messages change the scene as they arrive. Keyframes and mesh blocks go to one background
//! thread, in the order they came with the keyframe poses: it decodes and encodes them with no lock held, and takes
//! a lock only to keep the result and send it, so a keyframe never holds up the people.
//!
//! Each surface message's run and sequence are checked as it arrives (see [`crate::surface_stream`]); when the
//! surface held can no longer be trusted, the rig view asks the dense map for a snapshot, and when the snapshot or a
//! new run's first message replaces the surface, every page is sent the whole scene again, as a page that connects
//! is.

use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::time::Instant;

use axum::Router;
use perception_wire::localization::{FrameId, KeyframePose, PoseGraphUpdate, MapToOdom};
use perception_wire::objects::ObjectUpdate;
use perception_wire::odometry::OdomPose;
use perception_wire::people::PeopleTracks;
use perception_wire::space_map::{MeshBlocks, Topology};
use perception_wire::Keyframe;

use crate::cloud::{CloudError, CloudSettings};
use crate::clouds::{CloudStore, KeyframeMapPose, PreparedKeyframe};
use crate::scene::Scene;
use crate::server::{router, SharedView};
use crate::surface::{BlockGrid, MeshCache, SurfaceError};
use crate::surface_stream::{SnapshotRequestReason, SurfaceApplication, SurfaceStream, SurfaceStreamError, SurfaceStreamStatus};
use crate::view::Outgoing;

/// Keyframes, keyframe poses and surface messages the background thread may be behind by before the node stops.
const BACKGROUND_BACKLOG: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigViewSettings {
    pub cloud: CloudSettings,
    /// The newest this many keyframe clouds, and keyframe poses, are kept.
    pub max_keyframes: usize,
}

impl RigViewSettings {
    pub const DEFAULT: RigViewSettings =
        RigViewSettings { cloud: CloudSettings::DEFAULT, max_keyframes: 600 };
}

/// One message for the page, decoded.
pub enum RigInput {
    Keyframe(Keyframe),
    KeyframePose(KeyframePose),
    PoseGraphUpdate(PoseGraphUpdate),
    OdomPose(OdomPose),
    MapToOdom(MapToOdom),
    PeopleTracks(PeopleTracks),
    MeshBlocks(MeshBlocks),
    ObjectUpdate(ObjectUpdate),
    Topology(Topology),
}

#[derive(Debug, thiserror::Error)]
pub enum RigViewError {
    #[error(transparent)]
    Cloud(#[from] CloudError),
    #[error(transparent)]
    Surface(#[from] SurfaceError),
    #[error(transparent)]
    SurfaceStream(#[from] SurfaceStreamError),
    #[error("the keyframe and surface work is {BACKGROUND_BACKLOG} messages behind")]
    BackgroundBehind,
    #[error("the keyframe and surface work has stopped")]
    BackgroundStopped,
}

/// What the background thread works on, in the order it arrived.
enum BackgroundInput {
    Keyframe(Keyframe),
    MapPose(FrameId, KeyframeMapPose),
    PoseGraphUpdate(PoseGraphUpdate),
    MeshBlocks { surface: MeshBlocks, application: SurfaceApplication, status: SurfaceStreamStatus },
}

pub struct RigView {
    shared: SharedView,
    background: SyncSender<BackgroundInput>,
    surface_stream: SurfaceStream,
}

impl RigView {
    /// The rig view and its background thread, which runs until the rig view is dropped or a keyframe or surface
    /// message cannot be drawn.
    pub fn start(settings: RigViewSettings) -> (RigView, tokio::task::JoinHandle<Result<(), RigViewError>>) {
        let shared = SharedView::new(Scene::new(), CloudStore::new(settings.cloud, settings.max_keyframes));
        let (sender, receiver) = mpsc::sync_channel(BACKGROUND_BACKLOG);
        let worker = shared.clone();
        let background = tokio::task::spawn_blocking(move || run_background(receiver, &worker, settings.cloud));
        (RigView { shared, background: sender, surface_stream: SurfaceStream::default() }, background)
    }

    pub fn router(&self) -> Router {
        router(self.shared.clone())
    }

    /// The whole surface as a page that connects now gets it, encoded; the background thread may still be working
    /// on surface messages taken before.
    pub fn whole_surface(&self) -> Option<bytes::Bytes> {
        self.shared.read_scene(|scene| scene.whole_surface().cloned())
    }

    /// Takes one message, arriving at `now`. Returns why to ask the dense map for a mesh snapshot, when one
    /// should be asked for now.
    pub fn on_input(&mut self, input: RigInput, now: Instant) -> Result<Option<SnapshotRequestReason>, RigViewError> {
        let overdue = self.surface_stream.on_time(now);
        let request = match input {
            RigInput::MeshBlocks(surface) => {
                let step = self.surface_stream.on_message(&surface, now)?;
                self.to_background(BackgroundInput::MeshBlocks { surface, application: step.application, status: step.status })?;
                step.request
            }
            RigInput::Keyframe(keyframe) => {
                self.to_background(BackgroundInput::Keyframe(keyframe))?;
                None
            }
            RigInput::KeyframePose(pose) => {
                self.to_background(BackgroundInput::MapPose(pose.frame_id, KeyframeMapPose::of_keyframe_pose(&pose)))?;
                None
            }
            RigInput::PoseGraphUpdate(moved) => {
                self.to_background(BackgroundInput::PoseGraphUpdate(moved))?;
                None
            }
            RigInput::OdomPose(pose) => self.apply_scene(|scene| scene.on_odom_pose(&pose)),
            RigInput::MapToOdom(correction) => self.apply_scene(|scene| scene.on_map_to_odom(&correction)),
            RigInput::PeopleTracks(tracks) => self.apply_scene(|scene| scene.on_people_tracks(&tracks)),
            RigInput::ObjectUpdate(update) => self.apply_scene(|scene| scene.on_object_update(&update)),
            RigInput::Topology(topology) => self.apply_scene(|scene| scene.on_topology(&topology)),
        };
        Ok(request.or(overdue))
    }

    /// Changes the scene with a light message, which asks for no snapshot.
    fn apply_scene(&self, change: impl FnOnce(&mut Scene) -> Vec<Outgoing>) -> Option<SnapshotRequestReason> {
        self.shared.apply_scene(change);
        None
    }

    fn to_background(&self, input: BackgroundInput) -> Result<(), RigViewError> {
        self.background.try_send(input).map_err(|error| match error {
            TrySendError::Full(_) => RigViewError::BackgroundBehind,
            TrySendError::Disconnected(_) => RigViewError::BackgroundStopped,
        })
    }
}

fn run_background(inputs: Receiver<BackgroundInput>, shared: &SharedView, settings: CloudSettings) -> Result<(), RigViewError> {
    let mut mesh_cache = MeshCache::default();
    for input in inputs {
        match input {
            BackgroundInput::Keyframe(keyframe) => {
                let prepared = PreparedKeyframe::of(&keyframe, &settings)?;
                shared.apply_clouds(|clouds| clouds.on_keyframe(prepared));
            }
            BackgroundInput::MapPose(frame_id, map_pose) => shared.apply_clouds(|clouds| clouds.set_map_pose(frame_id, map_pose)),
            BackgroundInput::PoseGraphUpdate(moved) => shared.apply_clouds(|clouds| {
                moved.keyframes.iter().flat_map(|keyframe| clouds.set_map_pose(keyframe.frame_id, clouds.moved(keyframe))).collect()
            }),
            BackgroundInput::MeshBlocks { surface, application, status } => {
                let grid = BlockGrid::of(&surface);
                let change = match application {
                    SurfaceApplication::Changes => mesh_cache.apply(grid, &surface.blocks)?,
                    SurfaceApplication::Replace => mesh_cache.replace(grid, &surface.blocks)?,
                };
                shared.apply_scene(|scene| scene.on_surface(change, application, status));
            }
        }
    }
    Ok(())
}
