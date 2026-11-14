//! Rooms, doorways and places cut from the free-space grid.
//!
//! Clearance here is the distance to the nearest cell that is not known free, so the frontier of what has been
//! seen bounds rooms as walls do. Small unknown pockets bordered mostly by free cells, such as the floor under
//! where the camera stood, count as free. Doorways are pinch points of the clearance ridge (the middle of the free space)
//! narrower than a door, where the clearance rises again on both sides; cutting the free space there splits it
//! into rooms, and rooms too small to be rooms merge into the neighbour they share the widest doorway with. Ids
//! carry over from the previous cut: a room keeps the id of the room it overlaps most, doorways and places keep
//! the id of the nearest previous one close enough.

use std::collections::VecDeque;

use glam::{IVec2, Vec2, Vec3};
use perception_wire::navigation::PlaceId;
use perception_wire::space_map::{Doorway, DoorwayId, Place, PlaceEdge, RoomId, TopologyRoom};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::grid::{squared_distance_transform, Cell, CellGrid, GridGeometry};
use crate::outline::{outer_outline, simplify};
use crate::settings::TopologySettings;

const NEIGHBOURS_4: [IVec2; 4] = [IVec2::X, IVec2::Y, IVec2::NEG_X, IVec2::NEG_Y];
const RIDGE_PAIRS: [IVec2; 4] = [IVec2::X, IVec2::Y, IVec2::ONE, IVec2::new(1, -1)];

/// Everything a topology message carries apart from its stamps.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TopologyPlan {
    pub rooms: Vec<TopologyRoom>,
    pub doorways: Vec<Doorway>,
    pub places: Vec<Place>,
    pub edges: Vec<PlaceEdge>,
}

/// The room of every cell in a cut.
#[derive(Clone, Debug)]
pub struct Segmentation {
    pub geometry: GridGeometry,
    pub rooms: Vec<Option<RoomId>>,
}

impl Segmentation {
    pub fn room_at(&self, point: Vec2) -> Option<RoomId> {
        self.geometry.index(self.geometry.cell_at(point)).and_then(|i| self.rooms[i])
    }
}

/// Cuts free space into a topology, keeping ids stable from one cut to the next.
pub struct Topologist {
    settings: TopologySettings,
    previous: Option<Segmentation>,
    previous_doorways: Vec<Doorway>,
    previous_places: Vec<Place>,
    next_room: u64,
    next_doorway: u64,
    next_place: u64,
}

/// A pinch point accepted as a cut.
#[derive(Clone, Copy, Debug)]
struct Cut {
    cell: IVec2,
    /// Clearance at the pinch, in cells.
    clearance: f32,
    radius: f32,
}

/// A cut that separates two regions.
#[derive(Clone, Copy, Debug)]
struct RegionDoor {
    cut: Cut,
    regions: (u32, u32),
}

const NO_REGION: u32 = u32::MAX;

struct Field<'a> {
    geometry: GridGeometry,
    grid: &'a CellGrid,
    /// Distance in cells to the nearest cell that is not free.
    clearance: Vec<f32>,
}

impl Field<'_> {
    fn free(&self, cell: IVec2) -> Option<usize> {
        self.geometry.index(cell).filter(|&i| self.grid.cells[i] == Cell::Free)
    }

    fn clearance_at(&self, cell: IVec2) -> f32 {
        self.geometry.index(cell).map_or(0.0, |i| self.clearance[i])
    }

    fn is_ridge(&self, cell: IVec2) -> bool {
        let c = self.clearance_at(cell);
        c > 0.0
            && RIDGE_PAIRS.iter().any(|&d| {
                let (a, b) = (self.clearance_at(cell + d), self.clearance_at(cell - d));
                c >= a && c >= b && (c > a || c > b)
            })
    }
}

impl Topologist {
    pub fn new(settings: TopologySettings) -> Topologist {
        Topologist {
            settings,
            previous: None,
            previous_doorways: Vec::new(),
            previous_places: Vec::new(),
            next_room: 1,
            next_doorway: 1,
            next_place: 1,
        }
    }

    /// The most recent cut.
    pub fn segmentation(&self) -> Option<&Segmentation> {
        self.previous.as_ref()
    }

    /// Cuts the grid into rooms at a floor height, with ceiling heights from the ceiling points over each room.
    pub fn cut(&mut self, grid: &CellGrid, floor_m: f32, ceiling_points: &[Vec3], min_ceiling_points: usize) -> TopologyPlan {
        let filled = fill_holes(grid, self.settings.max_hole_area_m2);
        let grid = &filled;
        let geometry = grid.geometry;
        let squared = squared_distance_transform(&geometry, |i| grid.cells[i] != Cell::Free);
        let field = Field { geometry, grid, clearance: squared.into_par_iter().map(|d| d.sqrt() as f32).collect() };
        let cuts = self.find_cuts(&field);
        let (mut regions, doors) = regions_between_cuts(&field, &cuts);
        let (rooms_of_regions, doors) = self.merge_small_regions(&field, &mut regions, doors);
        let room_ids = self.match_room_ids(&geometry, &regions, &rooms_of_regions);
        let cell_rooms: Vec<Option<RoomId>> =
            regions.iter().map(|&r| if r == NO_REGION { None } else { room_ids.get(&r).copied() }).collect();
        let segmentation = Segmentation { geometry, rooms: cell_rooms };

        let rooms = self.room_shapes(&segmentation, floor_m, ceiling_points, min_ceiling_points);
        let doorways = self.doorways(&field, &doors, &room_ids);
        let places = self.places(&field, &segmentation, &doorways);
        let edges = self.edges(&field, &places);

        self.previous = Some(segmentation);
        self.previous_doorways = doorways.clone();
        self.previous_places = places.clone();
        TopologyPlan { rooms, doorways, places, edges }
    }

    fn cells(&self, metres: f32, geometry: &GridGeometry) -> f32 {
        metres / geometry.resolution_m
    }

    /// Ridge cells narrower than a door, where the free space locally falls apart without them, narrowest first,
    /// none within a door width of a narrower one.
    fn find_cuts(&self, field: &Field) -> Vec<Cut> {
        let geometry = &field.geometry;
        // A passage n cells wide has clearance (n + 1) / 2 cells in its middle.
        let max_clearance = (self.cells(self.settings.door_width_m, geometry) + 1.0) / 2.0;
        let look = self.cells(self.settings.door_width_m, geometry);
        let rise = self.cells(self.settings.min_rise_m, geometry);
        let mut candidates: Vec<Cut> = (0..geometry.len())
            .into_par_iter()
            .filter_map(|i| {
                let cell = geometry.cell(i);
                let c = field.clearance[i];
                let narrowest_nearby = || {
                    (-1..=1).flat_map(|y| (-1..=1).map(move |x| IVec2::new(x, y))).all(|d| {
                        let n = cell + d;
                        !field.is_ridge(n) || field.clearance_at(n) >= c
                    })
                };
                (c > 0.0 && c < max_clearance && field.is_ridge(cell) && narrowest_nearby())
                    .then_some(Cut { cell, clearance: c, radius: c + 1.5 })
                    .filter(|cut| separates(field, cut, look, rise))
            })
            .collect();
        candidates.sort_by(|a, b| a.clearance.total_cmp(&b.clearance).then((a.cell.y, a.cell.x).cmp(&(b.cell.y, b.cell.x))));
        let mut accepted: Vec<Cut> = Vec::new();
        for candidate in candidates {
            if accepted.iter().all(|a| a.cell.as_vec2().distance(candidate.cell.as_vec2()) >= look) {
                accepted.push(candidate);
            }
        }
        accepted
    }

    /// Small regions merged into the neighbour behind their widest doorway until none is left with a neighbour;
    /// regions still too small are dropped. Gives each kept region's root and the doors between kept rooms.
    fn merge_small_regions(&self, field: &Field, regions: &mut [u32], mut doors: Vec<RegionDoor>) -> (FxHashSet<u32>, Vec<RegionDoor>) {
        let cell_area = field.geometry.resolution_m * field.geometry.resolution_m;
        let mut area: FxHashMap<u32, f32> = FxHashMap::default();
        for &r in regions.iter().filter(|&&r| r != NO_REGION) {
            *area.entry(r).or_default() += cell_area;
        }
        let mut merged_into: FxHashMap<u32, u32> = FxHashMap::default();
        loop {
            let smallest = area
                .iter()
                .filter(|(r, a)| **a < self.settings.min_room_area_m2 && doors.iter().any(|d| d.regions.0 == **r || d.regions.1 == **r))
                .min_by(|a, b| a.1.total_cmp(b.1).then(a.0.cmp(b.0)))
                .map(|(r, _)| *r);
            let Some(small) = smallest else { break };
            let widest = doors
                .iter()
                .filter(|d| d.regions.0 == small || d.regions.1 == small)
                .max_by(|a, b| a.cut.clearance.total_cmp(&b.cut.clearance))
                .expect("the region has a door");
            let into = if widest.regions.0 == small { widest.regions.1 } else { widest.regions.0 };
            let small_area = area.remove(&small).expect("a region with an area");
            *area.get_mut(&into).expect("a region with an area") += small_area;
            merged_into.insert(small, into);
            for door in &mut doors {
                for end in [&mut door.regions.0, &mut door.regions.1] {
                    if *end == small {
                        *end = into;
                    }
                }
            }
            doors.retain(|d| d.regions.0 != d.regions.1);
        }
        let root = |mut r: u32| {
            while let Some(&next) = merged_into.get(&r) {
                r = next;
            }
            r
        };
        let kept: FxHashSet<u32> = area.iter().filter(|(_, a)| **a >= self.settings.min_room_area_m2).map(|(r, _)| *r).collect();
        regions.par_iter_mut().for_each(|r| {
            if *r != NO_REGION {
                let top = root(*r);
                *r = if kept.contains(&top) { top } else { NO_REGION };
            }
        });
        doors.retain(|d| kept.contains(&d.regions.0) && kept.contains(&d.regions.1));
        (kept, doors)
    }

    /// Each kept region takes the id of the previous room it overlaps most, largest overlaps first; the rest get
    /// new ids.
    fn match_room_ids(&mut self, geometry: &GridGeometry, regions: &[u32], kept: &FxHashSet<u32>) -> FxHashMap<u32, RoomId> {
        let mut overlaps: FxHashMap<(u32, RoomId), usize> = FxHashMap::default();
        if let Some(previous) = &self.previous {
            for (i, &region) in regions.iter().enumerate() {
                if region == NO_REGION {
                    continue;
                }
                if let Some(room) = previous.room_at(geometry.centre(geometry.cell(i))) {
                    *overlaps.entry((region, room)).or_default() += 1;
                }
            }
        }
        let mut ranked: Vec<((u32, RoomId), usize)> = overlaps.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut ids: FxHashMap<u32, RoomId> = FxHashMap::default();
        let mut taken: FxHashSet<RoomId> = FxHashSet::default();
        for ((region, room), _) in ranked {
            if !ids.contains_key(&region) && !taken.contains(&room) {
                ids.insert(region, room);
                taken.insert(room);
            }
        }
        let mut fresh: Vec<u32> = kept.iter().filter(|r| !ids.contains_key(r)).copied().collect();
        fresh.sort_unstable();
        for region in fresh {
            ids.insert(region, RoomId(self.next_room));
            self.next_room += 1;
        }
        ids
    }

    fn room_shapes(&self, segmentation: &Segmentation, floor_m: f32, ceiling_points: &[Vec3], min_ceiling_points: usize) -> Vec<TopologyRoom> {
        let geometry = segmentation.geometry;
        let mut cells: FxHashMap<RoomId, Vec<IVec2>> = FxHashMap::default();
        for (i, room) in segmentation.rooms.iter().enumerate() {
            if let Some(room) = room {
                cells.entry(*room).or_default().push(geometry.cell(i));
            }
        }
        let mut ceilings: FxHashMap<RoomId, Vec<f32>> = FxHashMap::default();
        for point in ceiling_points {
            if let Some(room) = segmentation.room_at(point.truncate()) {
                ceilings.entry(room).or_default().push(point.z);
            }
        }
        let mut rooms: Vec<TopologyRoom> = cells
            .into_par_iter()
            .map(|(room_id, cells)| {
                let inside = |cell: IVec2| geometry.index(cell).is_some_and(|i| segmentation.rooms[i] == Some(room_id));
                let ring = outer_outline(&cells, inside).expect("a room has cells");
                let corners: Vec<Vec2> = ring.iter().map(|c| geometry.corner(*c)).collect();
                TopologyRoom { room_id, polygon: simplify(&corners, geometry.resolution_m), floor_m, ceiling_m: None }
            })
            .collect();
        for room in &mut rooms {
            if let Some(heights) = ceilings.get_mut(&room.room_id).filter(|h| h.len() >= min_ceiling_points) {
                heights.sort_by(f32::total_cmp);
                room.ceiling_m = Some(heights[heights.len() / 2]);
            }
        }
        rooms.sort_by_key(|r| r.room_id);
        rooms
    }

    fn doorways(&mut self, field: &Field, doors: &[RegionDoor], room_ids: &FxHashMap<u32, RoomId>) -> Vec<Doorway> {
        let geometry = &field.geometry;
        let mut doorways: Vec<Doorway> = doors
            .iter()
            .filter_map(|door| {
                let width_m = (2.0 * door.cut.clearance - 1.0) * geometry.resolution_m;
                let (a, b) = (room_ids[&door.regions.0], room_ids[&door.regions.1]);
                (width_m >= self.settings.min_door_width_m).then(|| Doorway {
                    doorway_id: DoorwayId(0),
                    room_a: a.min(b),
                    room_b: a.max(b),
                    position: geometry.centre(door.cut.cell),
                    width_m,
                })
            })
            .collect();
        let previous: Vec<(u64, Vec2)> = self.previous_doorways.iter().map(|d| (d.doorway_id.0, d.position)).collect();
        let positions: Vec<Vec2> = doorways.iter().map(|d| d.position).collect();
        let ids = carry_ids(&positions, &previous, 0.5 * self.settings.door_width_m, &mut self.next_doorway);
        for (doorway, id) in doorways.iter_mut().zip(ids) {
            doorway.doorway_id = DoorwayId(id);
        }
        doorways.sort_by_key(|d| d.doorway_id);
        doorways
    }

    /// Doorway middles first, then ridge cells from the most clearance down, each at least the spacing from every
    /// place taken before it.
    fn places(&mut self, field: &Field, segmentation: &Segmentation, doorways: &[Doorway]) -> Vec<Place> {
        let geometry = &field.geometry;
        let min_clearance = self.cells(self.settings.min_place_clearance_m, geometry);
        let spacing = self.settings.place_spacing_m;
        let place = |cell: IVec2| {
            let i = geometry.index(cell)?;
            let room_id = segmentation.rooms[i]?;
            (field.clearance[i] >= min_clearance).then(|| Place {
                place_id: PlaceId(0),
                room_id,
                position: geometry.centre(cell),
                clearance_m: field.clearance[i] * geometry.resolution_m,
            })
        };
        let mut ridge: Vec<(f32, usize)> = (0..geometry.len())
            .into_par_iter()
            .filter(|&i| segmentation.rooms[i].is_some() && field.clearance[i] >= min_clearance && field.is_ridge(geometry.cell(i)))
            .map(|i| (field.clearance[i], i))
            .collect();
        ridge.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let seeds = doorways.iter().map(|d| geometry.cell_at(d.position));
        let mut places: Vec<Place> = Vec::new();
        for cell in seeds.chain(ridge.into_iter().map(|(_, i)| geometry.cell(i))) {
            let Some(candidate) = place(cell) else { continue };
            if places.iter().all(|p| p.position.distance(candidate.position) >= spacing) {
                places.push(candidate);
            }
        }
        let previous: Vec<(u64, Vec2)> = self.previous_places.iter().map(|p| (p.place_id.0, p.position)).collect();
        let positions: Vec<Vec2> = places.iter().map(|p| p.position).collect();
        let ids = carry_ids(&positions, &previous, 0.5 * spacing, &mut self.next_place);
        for (place, id) in places.iter_mut().zip(ids) {
            place.place_id = PlaceId(id);
        }
        places.sort_by_key(|p| p.place_id);
        places
    }

    /// Places that see each other across free space within the edge length, thinned to the relative neighbourhood
    /// graph: an edge goes when a third place is closer to both of its ends than they are to each other.
    fn edges(&self, field: &Field, places: &[Place]) -> Vec<PlaceEdge> {
        let max_length = self.settings.max_edge_length_m;
        let pairs: Vec<(usize, usize)> =
            (0..places.len()).flat_map(|a| (a + 1..places.len()).map(move |b| (a, b))).collect();
        let visible: Vec<(usize, usize, f32, f32)> = pairs
            .into_par_iter()
            .filter_map(|(a, b)| {
                let length = places[a].position.distance(places[b].position);
                if length > max_length {
                    return None;
                }
                let min_clearance = line_clearance(field, places[a].position, places[b].position)?;
                Some((a, b, length, min_clearance))
            })
            .collect();
        let linked: FxHashSet<(usize, usize)> = visible.iter().map(|&(a, b, ..)| (a, b)).collect();
        let joined = |a: usize, b: usize| linked.contains(&(a.min(b), a.max(b)));
        visible
            .iter()
            .filter(|&&(a, b, length, _)| {
                !(0..places.len()).any(|c| {
                    c != a
                        && c != b
                        && joined(a, c)
                        && joined(b, c)
                        && places[a].position.distance(places[c].position) < length
                        && places[b].position.distance(places[c].position) < length
                })
            })
            .map(|&(a, b, length_m, min_clearance_m)| PlaceEdge {
                place_a: places[a].place_id,
                place_b: places[b].place_id,
                length_m,
                min_clearance_m,
            })
            .collect()
    }
}

/// The grid with every unknown pocket that does not reach the grid's edge, covers at most `max_area_m2` and has
/// at least as many free cells as occupied ones around it made free.
fn fill_holes(grid: &CellGrid, max_area_m2: f32) -> CellGrid {
    let geometry = grid.geometry;
    let max_cells = (max_area_m2 / (geometry.resolution_m * geometry.resolution_m)) as usize;
    let mut cells = grid.cells.clone();
    let mut visited = vec![false; geometry.len()];
    for start in 0..geometry.len() {
        if visited[start] || grid.cells[start] != Cell::Unknown {
            continue;
        }
        visited[start] = true;
        let mut pocket = vec![start];
        let mut stack = vec![start];
        let (mut free, mut occupied, mut reaches_edge) = (0usize, 0usize, false);
        while let Some(i) = stack.pop() {
            for d in NEIGHBOURS_4 {
                match geometry.index(geometry.cell(i) + d) {
                    None => reaches_edge = true,
                    Some(n) => match grid.cells[n] {
                        Cell::Free => free += 1,
                        Cell::Occupied => occupied += 1,
                        Cell::Unknown if !visited[n] => {
                            visited[n] = true;
                            pocket.push(n);
                            stack.push(n);
                        }
                        Cell::Unknown => {}
                    },
                }
            }
        }
        if !reaches_edge && pocket.len() <= max_cells && free >= occupied {
            for i in pocket {
                cells[i] = Cell::Free;
            }
        }
    }
    CellGrid { geometry, cells }
}

/// The lowest clearance along a straight line, in metres, or None if it leaves the free space.
fn line_clearance(field: &Field, from: Vec2, to: Vec2) -> Option<f32> {
    let geometry = &field.geometry;
    let steps = (from.distance(to) / (0.5 * geometry.resolution_m)).ceil().max(1.0) as usize;
    let mut lowest = f32::INFINITY;
    for step in 0..=steps {
        let i = field.free(geometry.cell_at(from.lerp(to, step as f32 / steps as f32)))?;
        lowest = lowest.min(field.clearance[i]);
    }
    Some(lowest * geometry.resolution_m)
}

/// Whether removing a disc around a pinch point leaves at least two separate pieces of free space nearby that
/// each open out to more clearance than the pinch.
fn separates(field: &Field, cut: &Cut, look: f32, rise: f32) -> bool {
    let outer = cut.radius + look;
    let reach = outer.ceil() as i32;
    let side = (2 * reach + 1) as usize;
    let local = |cell: IVec2| {
        let d = cell - cut.cell + reach;
        (d.y as usize) * side + d.x as usize
    };
    let in_ring = |cell: IVec2| {
        let distance = (cell - cut.cell).as_vec2().length();
        distance > cut.radius && distance <= outer && field.free(cell).is_some()
    };
    let mut seen = vec![false; side * side];
    let mut opening_pieces = 0;
    for y in -reach..=reach {
        for x in -reach..=reach {
            let start = cut.cell + IVec2::new(x, y);
            if seen[local(start)] || !in_ring(start) {
                continue;
            }
            seen[local(start)] = true;
            let mut stack = vec![start];
            let mut widest = 0.0f32;
            while let Some(cell) = stack.pop() {
                widest = widest.max(field.clearance_at(cell));
                for d in NEIGHBOURS_4 {
                    let next = cell + d;
                    if (next - cut.cell).abs().max_element() <= reach && !seen[local(next)] && in_ring(next) {
                        seen[local(next)] = true;
                        stack.push(next);
                    }
                }
            }
            if widest >= cut.clearance + rise {
                opening_pieces += 1;
                if opening_pieces == 2 {
                    return true;
                }
            }
        }
    }
    false
}

/// Free cells labelled by the region they fall in once every cut disc is taken out, with the disc cells then given
/// to the nearest region; and the cuts that ended up between two different regions.
fn regions_between_cuts(field: &Field, cuts: &[Cut]) -> (Vec<u32>, Vec<RegionDoor>) {
    let geometry = &field.geometry;
    let mut in_cut = vec![false; geometry.len()];
    for cut in cuts {
        let reach = cut.radius.ceil() as i32;
        for y in -reach..=reach {
            for x in -reach..=reach {
                let offset = IVec2::new(x, y);
                if offset.as_vec2().length() <= cut.radius {
                    if let Some(i) = field.free(cut.cell + offset) {
                        in_cut[i] = true;
                    }
                }
            }
        }
    }
    let mut regions = vec![NO_REGION; geometry.len()];
    let mut next_region = 0;
    for start in 0..geometry.len() {
        if regions[start] != NO_REGION || in_cut[start] || grid_cell(field, start) != Cell::Free {
            continue;
        }
        regions[start] = next_region;
        let mut stack = vec![start];
        while let Some(i) = stack.pop() {
            for d in NEIGHBOURS_4 {
                if let Some(n) = field.free(geometry.cell(i) + d) {
                    if regions[n] == NO_REGION && !in_cut[n] {
                        regions[n] = next_region;
                        stack.push(n);
                    }
                }
            }
        }
        next_region += 1;
    }
    let mut frontier: VecDeque<usize> = (0..geometry.len()).filter(|&i| regions[i] != NO_REGION).collect();
    while let Some(i) = frontier.pop_front() {
        for d in NEIGHBOURS_4 {
            if let Some(n) = field.free(geometry.cell(i) + d) {
                if regions[n] == NO_REGION {
                    regions[n] = regions[i];
                    frontier.push_back(n);
                }
            }
        }
    }
    let doors = cuts
        .iter()
        .filter_map(|cut| {
            let reach = cut.radius.ceil() as i32 + 1;
            let mut counts: FxHashMap<u32, usize> = FxHashMap::default();
            for y in -reach..=reach {
                for x in -reach..=reach {
                    let offset = IVec2::new(x, y);
                    if offset.as_vec2().length() <= cut.radius + 1.0 {
                        if let Some(i) = field.free(cut.cell + offset).filter(|&i| regions[i] != NO_REGION) {
                            *counts.entry(regions[i]).or_default() += 1;
                        }
                    }
                }
            }
            let mut ranked: Vec<(u32, usize)> = counts.into_iter().collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            (ranked.len() >= 2).then(|| RegionDoor { cut: *cut, regions: (ranked[0].0, ranked[1].0) })
        })
        .collect();
    (regions, doors)
}

fn grid_cell(field: &Field, index: usize) -> Cell {
    field.grid.cells[index]
}

/// For each new position, the id of the nearest unclaimed previous position within `within`, nearest pairs first,
/// or a new id.
fn carry_ids(positions: &[Vec2], previous: &[(u64, Vec2)], within: f32, next: &mut u64) -> Vec<u64> {
    let mut pairs: Vec<(f32, usize, usize)> = positions
        .iter()
        .enumerate()
        .flat_map(|(i, p)| previous.iter().enumerate().map(move |(j, (_, q))| (p.distance(*q), i, j)))
        .filter(|(d, ..)| *d <= within)
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then((a.1, a.2).cmp(&(b.1, b.2))));
    let mut ids: Vec<Option<u64>> = vec![None; positions.len()];
    let mut claimed = vec![false; previous.len()];
    for (_, i, j) in pairs {
        if ids[i].is_none() && !claimed[j] {
            ids[i] = Some(previous[j].0);
            claimed[j] = true;
        }
    }
    ids.into_iter()
        .map(|id| {
            id.unwrap_or_else(|| {
                *next += 1;
                *next - 1
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RESOLUTION_M: f32 = 0.05;

    /// A plan of free rectangles in a block of occupied cells, with an unknown ring around it.
    struct Plan {
        geometry: GridGeometry,
        cells: Vec<Cell>,
    }

    impl Plan {
        fn new(width_m: f32, height_m: f32) -> Plan {
            let width = (width_m / RESOLUTION_M).round() as usize + 2;
            let height = (height_m / RESOLUTION_M).round() as usize + 2;
            let geometry = GridGeometry { origin: Vec2::splat(-RESOLUTION_M), resolution_m: RESOLUTION_M, width, height };
            let cells = (0..geometry.len())
                .map(|i| {
                    let cell = geometry.cell(i);
                    if cell.x == 0 || cell.y == 0 || cell.x as usize == width - 1 || cell.y as usize == height - 1 {
                        Cell::Unknown
                    } else {
                        Cell::Occupied
                    }
                })
                .collect();
            Plan { geometry, cells }
        }

        fn set(mut self, low: Vec2, high: Vec2, to: Cell) -> Plan {
            for i in 0..self.geometry.len() {
                let centre = self.geometry.centre(self.geometry.cell(i));
                if self.cells[i] != Cell::Unknown && centre.cmpge(low).all() && centre.cmplt(high).all() {
                    self.cells[i] = to;
                }
            }
            self
        }

        fn free(self, low: (f32, f32), high: (f32, f32)) -> Plan {
            self.set(low.into(), high.into(), Cell::Free)
        }

        fn grid(&self) -> CellGrid {
            CellGrid { geometry: self.geometry, cells: self.cells.clone() }
        }
    }

    /// Two 4 m rooms side by side, a 0.1 m wall between them with a door 0.8 m wide in the middle.
    fn two_rooms() -> Plan {
        Plan::new(8.1, 4.0).free((0.0, 0.0), (4.0, 4.0)).free((4.1, 0.0), (8.1, 4.0)).free((3.9, 1.6), (4.2, 2.4))
    }

    fn cut(topologist: &mut Topologist, plan: &Plan) -> TopologyPlan {
        topologist.cut(&plan.grid(), -1.0, &[], 1)
    }

    fn area(polygon: &[Vec2]) -> f32 {
        0.5 * polygon.iter().zip(polygon.iter().cycle().skip(1)).map(|(a, b)| a.perp_dot(*b)).sum::<f32>()
    }

    fn connected(plan: &TopologyPlan) -> bool {
        let mut reached = vec![plan.places[0].place_id];
        let mut changed = true;
        while changed {
            changed = false;
            for edge in &plan.edges {
                for (from, to) in [(edge.place_a, edge.place_b), (edge.place_b, edge.place_a)] {
                    if reached.contains(&from) && !reached.contains(&to) {
                        reached.push(to);
                        changed = true;
                    }
                }
            }
        }
        reached.len() == plan.places.len()
    }

    #[test]
    fn two_rooms_are_cut_at_the_door_between_them() {
        let plan = cut(&mut Topologist::new(TopologySettings::default()), &two_rooms());
        assert_eq!(plan.rooms.len(), 2, "{:?}", plan.rooms);
        for room in &plan.rooms {
            assert!((area(&room.polygon) - 16.0).abs() < 0.8, "{room:?}");
            assert_eq!((room.floor_m, room.ceiling_m), (-1.0, None));
        }
        assert_eq!(plan.doorways.len(), 1, "{:?}", plan.doorways);
        let door = plan.doorways[0];
        assert!(door.position.distance(Vec2::new(4.05, 2.0)) < 0.15, "{door:?}");
        assert!((door.width_m - 0.8).abs() < 0.1, "{door:?}");
        assert_eq!((door.room_a, door.room_b), (plan.rooms[0].room_id, plan.rooms[1].room_id));
    }

    #[test]
    fn places_cover_both_rooms_and_their_edges_join_through_the_door() {
        let plan = cut(&mut Topologist::new(TopologySettings::default()), &two_rooms());
        for room in &plan.rooms {
            assert!(plan.places.iter().filter(|p| p.room_id == room.room_id).count() >= 4, "{:?}", plan.places);
        }
        assert!(plan.places.iter().any(|p| p.position.distance(plan.doorways[0].position) < 0.01), "a place in the doorway");
        assert!(connected(&plan), "{:?}", plan.edges);
        for edge in &plan.edges {
            assert!(edge.length_m <= 1.5 && edge.min_clearance_m > 0.0, "{edge:?}");
        }
        for (a, b) in plan.places.iter().flat_map(|a| plan.places.iter().map(move |b| (a, b))) {
            assert!(a.place_id == b.place_id || a.position.distance(b.position) >= 0.75);
        }
    }

    #[test]
    fn room_ids_survive_a_rebuild_that_grows_a_room_and_new_rooms_get_new_ids() {
        let mut topologist = Topologist::new(TopologySettings::default());
        let first = cut(&mut topologist, &two_rooms());
        let ids_at = |plan: &TopologyPlan, points: &[Vec2]| -> Vec<RoomId> {
            points.iter().map(|p| topologist_room(plan, *p)).collect()
        };
        let left_and_right = [Vec2::new(1.0, 1.0), Vec2::new(7.0, 3.0)];
        let before = ids_at(&first, &left_and_right);

        let grown = Plan::new(12.2, 4.0)
            .free((0.0, 0.0), (4.0, 4.0))
            .free((4.1, 0.0), (8.1, 4.0))
            .free((3.9, 1.6), (4.2, 2.4))
            .free((8.2, 0.0), (12.2, 4.0))
            .free((8.0, 0.4), (8.3, 1.2));
        let second = cut(&mut topologist, &grown);
        assert_eq!(second.rooms.len(), 3, "{:?}", second.rooms);
        assert_eq!(ids_at(&second, &left_and_right), before);
        let new_room = topologist_room(&second, Vec2::new(10.0, 2.0));
        assert!(!before.contains(&new_room));
        assert_eq!(second.doorways.len(), 2);
        let first_door = first.doorways[0].doorway_id;
        assert!(second.doorways.iter().any(|d| d.doorway_id == first_door));

        let third = cut(&mut topologist, &grown);
        assert_eq!(third.rooms.iter().map(|r| r.room_id).collect::<Vec<_>>(), second.rooms.iter().map(|r| r.room_id).collect::<Vec<_>>());
        assert_eq!(third.places.iter().map(|p| p.place_id).collect::<Vec<_>>(), second.places.iter().map(|p| p.place_id).collect::<Vec<_>>());
    }

    fn topologist_room(plan: &TopologyPlan, point: Vec2) -> RoomId {
        plan.places
            .iter()
            .min_by(|a, b| a.position.distance(point).total_cmp(&b.position.distance(point)))
            .expect("places")
            .room_id
    }

    #[test]
    fn a_corridor_as_narrow_as_a_door_stays_one_room() {
        let plan = cut(&mut Topologist::new(TopologySettings::default()), &Plan::new(6.0, 1.0).free((0.0, 0.0), (6.0, 1.0)));
        assert_eq!((plan.rooms.len(), plan.doorways.len()), (1, 0), "{:?}", plan.doorways);
    }

    #[test]
    fn a_sofa_out_from_the_wall_does_not_split_the_room() {
        let plan = Plan::new(5.0, 5.0).free((0.0, 0.0), (5.0, 5.0)).set(Vec2::new(1.5, 0.6), Vec2::new(3.5, 1.4), Cell::Occupied);
        let plan = cut(&mut Topologist::new(TopologySettings::default()), &plan);
        assert_eq!((plan.rooms.len(), plan.doorways.len()), (1, 0), "{:?}", plan.doorways);
    }

    #[test]
    fn a_cupboard_smaller_than_a_room_merges_into_its_neighbour() {
        let plan = Plan::new(5.1, 4.0).free((0.0, 0.0), (4.0, 4.0)).free((4.1, 1.0), (5.1, 2.0)).free((3.9, 1.2), (4.2, 1.8));
        let plan = cut(&mut Topologist::new(TopologySettings::default()), &plan);
        assert_eq!((plan.rooms.len(), plan.doorways.len()), (1, 0), "{:?}", plan.rooms);
    }

    #[test]
    fn unknown_space_bounds_a_room_as_a_wall_does() {
        let plan = Plan::new(4.0, 4.0).free((0.0, 0.0), (4.0, 4.0)).set(Vec2::new(2.0, 0.0), Vec2::new(4.0, 4.0), Cell::Unknown);
        let plan = cut(&mut Topologist::new(TopologySettings::default()), &plan);
        assert_eq!(plan.rooms.len(), 1);
        assert!((area(&plan.rooms[0].polygon) - 8.0).abs() < 0.3, "{:?}", plan.rooms[0]);
    }

    #[test]
    fn an_unseen_patch_in_a_doorway_does_not_hide_the_door() {
        let plan = two_rooms().set(Vec2::new(3.95, 1.9), Vec2::new(4.15, 2.1), Cell::Unknown);
        let plan = cut(&mut Topologist::new(TopologySettings::default()), &plan);
        assert_eq!((plan.rooms.len(), plan.doorways.len()), (2, 1), "{:?}", plan.doorways);
    }

    #[test]
    fn the_unseen_inside_of_furniture_stays_out_of_the_room() {
        let plan = Plan::new(4.0, 4.0)
            .free((0.0, 0.0), (4.0, 4.0))
            .set(Vec2::new(1.0, 1.0), Vec2::new(2.0, 2.0), Cell::Occupied)
            .set(Vec2::new(1.1, 1.1), Vec2::new(1.9, 1.9), Cell::Unknown);
        let plan = cut(&mut Topologist::new(TopologySettings::default()), &plan);
        assert!((area(&plan.rooms[0].polygon) - 16.0).abs() < 0.5);
        assert!(plan.places.iter().all(|p| !(p.position.cmpgt(Vec2::splat(1.0)).all() && p.position.cmplt(Vec2::splat(2.0)).all())));
    }

    #[test]
    fn a_ceiling_over_a_room_gives_its_height() {
        let ceiling: Vec<Vec3> = (0..30).map(|i| Vec3::new(1.0 + 0.05 * i as f32, 1.0, 1.4)).collect();
        let plan = Topologist::new(TopologySettings::default()).cut(&two_rooms().grid(), -1.0, &ceiling, 20);
        let left = topologist_room(&plan, Vec2::new(1.0, 1.0));
        for room in &plan.rooms {
            assert_eq!(room.ceiling_m, (room.room_id == left).then_some(1.4));
        }
    }
}
