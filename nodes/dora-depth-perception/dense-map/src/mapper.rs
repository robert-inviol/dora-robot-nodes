//! The dense map as a node runs it: keyframes wait for their poses, are integrated and kept, the map is rebuilt
//! from the kept keyframes in the background when the localizer publishes a new map version, and free space and
//! topology are sliced from it at most once a period.

use std::collections::VecDeque;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use perception_wire::navigation::{MapVersion, UnixNanos};
use perception_wire::space_map::{Clearance, FreeSpace, HeightBand, KeepOut, Topology};
use rustc_hash::FxHashMap;

use crate::grid::{slice_field, squared_distance_transform, Cell, CellGrid, Slice};
use crate::keep_out::{KeepOutZones, ObjectUpdate};
use crate::keyframe::{DepthKeyframe, FrameId, KeyframePose, PosedDepth, Tracking};
use crate::levels::{ceiling_points, find_floor};
use crate::settings::DenseMapSettings;
use crate::topology::{Topologist, TopologyPlan};
use crate::tsdf::DistanceField;

#[derive(Debug, thiserror::Error)]
pub enum MapperError {
    #[error("the background rebuild of map {0} panicked")]
    RebuildPanicked(MapVersion),
}

/// What happened to a keyframe or pose on arrival.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arrival {
    Integrated(FrameId),
    /// Held until its pose (or keyframe) arrives.
    Waiting,
    /// Held, pushing out the oldest one waiting, which is lost.
    WaitingDropped(FrameId),
    /// A lost pose: the keyframe is not integrated.
    Skipped(FrameId),
    /// A new pose for a keyframe already in the map; the map will be rebuilt.
    Moved(FrameId),
    /// The pose a keyframe already in the map already has.
    AlreadyPlaced(FrameId),
}

/// What a tick produced.
#[derive(Clone, Debug, PartialEq)]
pub enum MapOutput {
    FreeSpace(FreeSpace),
    Topology(Topology),
}

/// What a tick did besides its outputs, for the node to log.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TickReport {
    pub rebuilt: Option<MapVersion>,
    pub rebuild_started: Option<MapVersion>,
    /// Set when there was something new to slice but no floor seen yet.
    pub no_floor: bool,
    pub slice_time: Option<Duration>,
}

struct Kept {
    keyframe: Arc<DepthKeyframe>,
    pose: KeyframePose,
}

struct Rebuild<F> {
    thread: JoinHandle<F>,
    version: MapVersion,
    /// How many kept keyframes the rebuild integrates; later ones are integrated when it lands.
    keyframes: usize,
}

pub struct DenseMapper<F: DistanceField> {
    settings: DenseMapSettings,
    field: F,
    /// The map version of the poses the field was built from.
    field_version: Option<MapVersion>,
    awaiting_pose: VecDeque<DepthKeyframe>,
    early_poses: VecDeque<KeyframePose>,
    kept: Vec<Kept>,
    kept_index: FxHashMap<FrameId, usize>,
    rebuild: Option<Rebuild<F>>,
    rebuild_wanted: bool,
    changed_since_slice: bool,
    last_slice: Option<Instant>,
    topologist: Topologist,
    last_topology: Option<(MapVersion, TopologyPlan)>,
    keep_out: KeepOutZones,
}

impl<F: DistanceField> DenseMapper<F> {
    pub fn new(settings: DenseMapSettings, field: F) -> DenseMapper<F> {
        DenseMapper {
            topologist: Topologist::new(settings.topology),
            keep_out: KeepOutZones::new(&settings.keep_out),
            settings,
            field,
            field_version: None,
            awaiting_pose: VecDeque::new(),
            early_poses: VecDeque::new(),
            kept: Vec::new(),
            kept_index: FxHashMap::default(),
            rebuild: None,
            rebuild_wanted: false,
            changed_since_slice: false,
            last_slice: None,
            last_topology: None,
        }
    }

    pub fn field(&self) -> &F {
        &self.field
    }

    pub fn kept_keyframes(&self) -> usize {
        self.kept.len()
    }

    pub fn is_rebuilding(&self) -> bool {
        self.rebuild.is_some()
    }

    pub fn on_keyframe(&mut self, keyframe: DepthKeyframe) -> Arrival {
        if let Some(at) = self.early_poses.iter().position(|p| p.frame_id == keyframe.frame_id) {
            let pose = self.early_poses.remove(at).expect("found above");
            return self.place(keyframe, pose);
        }
        self.awaiting_pose.push_back(keyframe);
        if self.awaiting_pose.len() > self.settings.buffers.max_awaiting_pose {
            let dropped = self.awaiting_pose.pop_front().expect("over the limit");
            return Arrival::WaitingDropped(dropped.frame_id);
        }
        Arrival::Waiting
    }

    pub fn on_keyframe_pose(&mut self, pose: KeyframePose) -> Arrival {
        if self.field_version.is_some_and(|v| pose.map_version > v) {
            self.rebuild_wanted = true;
        }
        if let Some(&i) = self.kept_index.get(&pose.frame_id) {
            if pose.tracking != Tracking::Lost && self.kept[i].pose != pose {
                self.kept[i].pose = pose;
                self.rebuild_wanted = true;
                return Arrival::Moved(pose.frame_id);
            }
            return Arrival::AlreadyPlaced(pose.frame_id);
        }
        if let Some(at) = self.awaiting_pose.iter().position(|k| k.frame_id == pose.frame_id) {
            let keyframe = self.awaiting_pose.remove(at).expect("found above");
            return self.place(keyframe, pose);
        }
        self.early_poses.push_back(pose);
        if self.early_poses.len() > self.settings.buffers.max_awaiting_pose {
            let dropped = self.early_poses.pop_front().expect("over the limit");
            return Arrival::WaitingDropped(dropped.frame_id);
        }
        Arrival::Waiting
    }

    pub fn on_object_update(&mut self, update: &ObjectUpdate) -> Option<KeepOut> {
        self.keep_out.apply(update)
    }

    fn place(&mut self, keyframe: DepthKeyframe, pose: KeyframePose) -> Arrival {
        let frame_id = keyframe.frame_id;
        if pose.tracking == Tracking::Lost {
            return Arrival::Skipped(frame_id);
        }
        // The localizer places keyframes in order, so any older one still waiting was skipped and never will be.
        self.awaiting_pose.retain(|waiting| waiting.frame_id > frame_id);
        self.field.integrate(&PosedDepth::new(&keyframe, &pose));
        self.field_version.get_or_insert(pose.map_version);
        self.kept_index.insert(frame_id, self.kept.len());
        self.kept.push(Kept { keyframe: Arc::new(keyframe), pose });
        self.changed_since_slice = true;
        Arrival::Integrated(frame_id)
    }

    /// Lands a finished rebuild, starts a wanted one, and slices the map when it changed and a period has passed.
    pub fn on_tick(&mut self, now: Instant, now_unix: UnixNanos) -> Result<(Vec<MapOutput>, TickReport), MapperError> {
        self.tick(now, now_unix, false)
    }

    /// As [`DenseMapper::on_tick`], waiting for a running rebuild to finish first.
    pub fn on_tick_after_rebuild(&mut self, now: Instant, now_unix: UnixNanos) -> Result<(Vec<MapOutput>, TickReport), MapperError> {
        self.tick(now, now_unix, true)
    }

    fn tick(&mut self, now: Instant, now_unix: UnixNanos, wait: bool) -> Result<(Vec<MapOutput>, TickReport), MapperError> {
        let mut report = TickReport::default();
        if self.rebuild.as_ref().is_some_and(|r| wait || r.thread.is_finished()) {
            report.rebuilt = Some(self.land_rebuild()?);
        }
        if self.rebuild_wanted && self.rebuild.is_none() {
            report.rebuild_started = Some(self.start_rebuild());
        }
        let period = Duration::from_secs_f32(self.settings.free_space.period_s);
        let due = self.last_slice.is_none_or(|last| now.duration_since(last) >= period);
        if !(self.changed_since_slice && due) {
            return Ok((Vec::new(), report));
        }
        let started = Instant::now();
        let outputs = self.slice_now(now_unix);
        self.changed_since_slice = false;
        match outputs {
            Some(outputs) => {
                self.last_slice = Some(now);
                report.slice_time = Some(started.elapsed());
                Ok((outputs, report))
            }
            None => {
                report.no_floor = true;
                Ok((Vec::new(), report))
            }
        }
    }

    fn start_rebuild(&mut self) -> MapVersion {
        self.rebuild_wanted = false;
        let version = self.kept.iter().map(|k| k.pose.map_version).max().expect("a rebuild is only wanted once keyframes are kept");
        let snapshot: Vec<(Arc<DepthKeyframe>, KeyframePose)> = self.kept.iter().map(|k| (k.keyframe.clone(), k.pose)).collect();
        let mut field = self.field.cleared();
        let thread = std::thread::spawn(move || {
            for (keyframe, pose) in &snapshot {
                field.integrate(&PosedDepth::new(keyframe, pose));
            }
            field
        });
        self.rebuild = Some(Rebuild { thread, version, keyframes: self.kept.len() });
        version
    }

    fn land_rebuild(&mut self) -> Result<MapVersion, MapperError> {
        let rebuild = self.rebuild.take().expect("a running rebuild");
        let mut field = rebuild.thread.join().map_err(|_| MapperError::RebuildPanicked(rebuild.version))?;
        for kept in &self.kept[rebuild.keyframes..] {
            field.integrate(&PosedDepth::new(&kept.keyframe, &kept.pose));
        }
        self.field = field;
        self.field_version = Some(rebuild.version);
        self.changed_since_slice = true;
        Ok(rebuild.version)
    }

    /// Free space, and the topology when it changed, sliced from the map as it stands, or None while no floor is
    /// seen.
    pub fn slice_now(&mut self, now_unix: UnixNanos) -> Option<Vec<MapOutput>> {
        let settings = &self.settings;
        let surface = self.field.surface();
        let floor_m = find_floor(&surface, self.field.voxel_m(), &settings.floor)?;
        let band = HeightBand { min_m: settings.free_space.band_min_m, max_m: settings.free_space.band_max_m };
        let slice = Slice {
            min_z_m: floor_m + band.min_m,
            max_z_m: floor_m + band.max_m,
            occupied_below_m: settings.free_space.occupied_below_m,
        };
        let grid = slice_field(&self.field, slice)?;
        let map_version = self.field_version.expect("a field with a floor has keyframes");
        let mut outputs = vec![MapOutput::FreeSpace(free_space(&grid, band, map_version, now_unix))];
        let ceilings = ceiling_points(&surface, floor_m, &settings.floor);
        let plan = self.topologist.cut(&grid, floor_m, &ceilings, settings.floor.min_ceiling_points);
        let latest = Some((map_version, plan));
        if self.last_topology != latest {
            self.last_topology = latest;
            let (_, plan) = self.last_topology.clone().expect("set above");
            outputs.push(MapOutput::Topology(Topology {
                map_version,
                built: now_unix,
                rooms: plan.rooms,
                doorways: plan.doorways,
                places: plan.places,
                edges: plan.edges,
            }));
        }
        Some(outputs)
    }
}

/// The free-space message of a grid: each free cell's distance to the nearest occupied cell.
pub fn free_space(grid: &CellGrid, band: HeightBand, map_version: MapVersion, built: UnixNanos) -> FreeSpace {
    let geometry = &grid.geometry;
    let squared = squared_distance_transform(geometry, |i| grid.cells[i] == Cell::Occupied);
    let to_mm = 1000.0 * geometry.resolution_m as f64;
    let clearance_mm = grid
        .cells
        .iter()
        .zip(&squared)
        .map(|(cell, d2)| {
            let clearance = match cell {
                Cell::Occupied => Clearance::Occupied,
                Cell::Unknown => Clearance::Unknown,
                Cell::Free => Clearance::Free { mm: (d2.sqrt() * to_mm).round().min(Clearance::MAX_FREE_MM as f64) as u16 },
            };
            clearance.to_wire()
        })
        .collect();
    FreeSpace {
        map_version,
        built,
        origin: geometry.origin,
        resolution_m: geometry.resolution_m,
        width: geometry.width as u32,
        height: geometry.height as u32,
        band,
        clearance_mm,
    }
}
