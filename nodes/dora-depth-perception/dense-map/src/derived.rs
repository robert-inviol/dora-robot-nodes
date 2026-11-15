//! What is derived from the distance field: the surface mesh of the blocks that changed, the occupancy grid sliced above the
//! floor, and the topology cut from the occupancy grid. It runs on a snapshot of the field, off the thread that fuses depth.

use std::time::{Duration, Instant};

use glam::IVec3;
use perception_wire::navigation::{MapVersion, UnixNanos};
use perception_wire::space_map::{Clearance, OccupancyGrid, HeightBand, Topology};
use rustc_hash::FxHashSet;

use crate::grid::{slice_field, squared_distance_transform, Cell, OccupancyCells, Slice};
use crate::levels::{ceiling_points, find_floor};
use crate::settings::DenseMapSettings;
use crate::block_mesher::{MeshPass, ResendRule, BlockMesher};
use crate::topology::{TopologyBuilder, TopologyPlan};
use crate::tsdf::Tsdf;

/// The occupancy grid, and the topology when it changed.
#[derive(Clone, Debug, PartialEq)]
pub struct Sliced {
    pub occupancy_grid: OccupancyGrid,
    pub topology: Option<Topology>,
}

/// The state that meshing and slicing carry from one pass to the next: the surfaces last sent and the last topology.
pub struct DerivedMaps {
    settings: DenseMapSettings,
    mesher: BlockMesher,
    topology_builder: TopologyBuilder,
    last_topology: Option<(MapVersion, TopologyPlan)>,
}

impl DerivedMaps {
    pub fn new(settings: &DenseMapSettings) -> DerivedMaps {
        DerivedMaps {
            mesher: BlockMesher::new(ResendRule {
                min_vertex_move_m: settings.mesh.min_vertex_move_m,
                min_changed_cells: settings.mesh.min_changed_cells,
                min_block_period: Duration::from_secs_f32(settings.mesh.min_block_period_s),
            }),
            topology_builder: TopologyBuilder::new(settings.topology),
            last_topology: None,
            settings: settings.clone(),
        }
    }

    /// The surface of every block the `changed` blocks reach that differs enough from the one last sent.
    pub fn mesh(&mut self, field: &dyn Tsdf, changed: &FxHashSet<IVec3>, now: Instant) -> MeshPass {
        self.mesher.mesh_changed_blocks(field, changed, now)
    }

    /// The occupancy grid, and the topology when it changed, sliced from the field, or None while no floor is seen.
    pub fn slice(&mut self, field: &dyn Tsdf, map_version: MapVersion, now_unix: UnixNanos) -> Option<Sliced> {
        let settings = &self.settings;
        let surface = field.surface();
        let floor_m = find_floor(&surface, field.voxel_m(), &settings.floor)?;
        let band = HeightBand { min_m: settings.occupancy_grid.band_min_m, max_m: settings.occupancy_grid.band_max_m };
        let slice = Slice { min_z_m: floor_m + band.min_m, max_z_m: floor_m + band.max_m, occupied_below_m: settings.occupancy_grid.occupied_below_m };
        let grid = slice_field(field, slice)?;
        let occupancy_grid = occupancy_grid(&grid, band, map_version, now_unix);
        let ceilings = ceiling_points(&surface, floor_m, &settings.floor);
        let plan = self.topology_builder.cut(&grid, floor_m, &ceilings, settings.floor.min_ceiling_points);
        let latest = Some((map_version, plan));
        if self.last_topology == latest {
            return Some(Sliced { occupancy_grid, topology: None });
        }
        self.last_topology = latest;
        let (_, plan) = self.last_topology.clone().expect("set above");
        let topology = Topology { map_version, built: now_unix, rooms: plan.rooms, doorways: plan.doorways, places: plan.places, edges: plan.edges };
        Some(Sliced { occupancy_grid, topology: Some(topology) })
    }
}

/// The occupancy grid message of a grid: each free cell's distance to the nearest occupied cell.
pub fn occupancy_grid(grid: &OccupancyCells, band: HeightBand, map_version: MapVersion, built: UnixNanos) -> OccupancyGrid {
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
    OccupancyGrid {
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
