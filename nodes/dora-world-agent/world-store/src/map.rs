//! The map held in memory for the geometry tools: the latest topology, the latest free-space grid and room names.
//! The writer replaces it as updates arrive; readers take a shared lock.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

use perception_wire::navigation::{MapVersion, PlaceId, UnixNanos};
use world_model::{Point, RoomName};

use crate::geometry::{distance_to_polygon, polygon_contains};
use crate::types::{FreeSpace, PlaceEdge, PlaceNode, RoomId, Topology};

/// What objects stand in when no room polygon is known.
pub const UNMAPPED_ROOM: &str = "unmapped";

/// The name an unnamed room goes by, such as `room3`.
pub fn unnamed_room_name(id: RoomId) -> RoomName {
    RoomName(format!("room{}", id.0))
}

/// The room an unnamed room's name refers to.
pub fn parse_unnamed_room(name: &RoomName) -> Option<RoomId> {
    let digits = name.0.strip_prefix("room")?;
    match !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
        true => digits.parse().ok().map(RoomId),
        false => None,
    }
}

/// A doorway's place: the place nearest the doorway, joining two rooms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoorwayPlace {
    pub place: PlaceId,
    pub room_a: RoomId,
    pub room_b: RoomId,
}

/// Which rooms a room filter selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoomFilter {
    Room(RoomId),
    Unmapped,
}

#[derive(Clone, Debug, Default)]
pub struct MapState {
    pub topology: Option<Topology>,
    pub doorway_places: Vec<DoorwayPlace>,
    pub free_space: Option<FreeSpace>,
    pub room_names: BTreeMap<RoomId, RoomName>,
    pub latest_version: Option<MapVersion>,
}

impl MapState {
    pub fn note_version(&mut self, version: MapVersion) {
        self.latest_version = Some(self.latest_version.map_or(version, |v| v.max(version)));
    }

    pub fn set_topology(&mut self, topology: Topology) {
        self.note_version(topology.map_version);
        self.doorway_places = topology
            .doorways
            .iter()
            .filter_map(|d| {
                topology
                    .places
                    .iter()
                    .filter(|p| p.room == d.room_a || p.room == d.room_b)
                    .min_by(|a, b| a.position.distance_to(d.centre).total_cmp(&b.position.distance_to(d.centre)))
                    .map(|p| DoorwayPlace { place: p.id, room_a: d.room_a, room_b: d.room_b })
            })
            .collect();
        self.topology = Some(topology);
    }

    pub fn set_free_space(&mut self, free_space: FreeSpace) {
        self.note_version(free_space.map_version());
        self.free_space = Some(free_space);
    }

    /// When the map was last fused: the newer of the topology and free-space builds.
    pub fn fused(&self) -> Option<UnixNanos> {
        let topology = self.topology.as_ref().map(|t| t.built);
        let free_space = self.free_space.as_ref().map(FreeSpace::built);
        topology.max(free_space)
    }

    pub fn room_name(&self, room: Option<RoomId>) -> RoomName {
        match room {
            Some(id) => self.room_names.get(&id).cloned().unwrap_or_else(|| unnamed_room_name(id)),
            None => RoomName(UNMAPPED_ROOM.to_owned()),
        }
    }

    pub fn has_room(&self, id: RoomId) -> bool {
        self.topology.as_ref().is_some_and(|t| t.rooms.iter().any(|r| r.id == id))
    }

    /// The room a name refers to: a given name, or `room<N>` for room N.
    pub fn resolve_room(&self, name: &RoomName) -> Option<RoomFilter> {
        if name.0 == UNMAPPED_ROOM {
            return Some(RoomFilter::Unmapped);
        }
        if let Some((id, _)) = self.room_names.iter().find(|(_, n)| *n == name) {
            return Some(RoomFilter::Room(*id));
        }
        parse_unnamed_room(name).filter(|id| self.has_room(*id)).map(RoomFilter::Room)
    }

    /// The room whose floor holds the point, or the nearest room when the point is just outside every outline.
    pub fn room_at(&self, p: Point) -> Option<RoomId> {
        let rooms = &self.topology.as_ref()?.rooms;
        if let Some(room) = rooms.iter().find(|r| polygon_contains(&r.polygon, p)) {
            return Some(room.id);
        }
        rooms
            .iter()
            .min_by(|a, b| distance_to_polygon(&a.polygon, p).total_cmp(&distance_to_polygon(&b.polygon, p)))
            .map(|r| r.id)
    }

    pub fn inside_any_room(&self, p: Point) -> bool {
        self.topology.as_ref().is_some_and(|t| t.rooms.iter().any(|r| polygon_contains(&r.polygon, p)))
    }

    pub fn place(&self, id: PlaceId) -> Option<&PlaceNode> {
        self.topology.as_ref()?.places.iter().find(|p| p.id == id)
    }

    pub fn doorway_place(&self, id: PlaceId) -> Option<&DoorwayPlace> {
        self.doorway_places.iter().find(|d| d.place == id)
    }

    /// A place's name: the rooms its doorway joins, or the room it is in.
    pub fn place_name(&self, place: &PlaceNode) -> String {
        match self.doorway_place(place.id) {
            Some(d) => format!("doorway between {} and {}", self.room_name(Some(d.room_a)), self.room_name(Some(d.room_b))),
            None => format!("place in {}", self.room_name(Some(place.room))),
        }
    }

    /// The places a position in a room can join the graph at: the room's places and the doorways into it.
    pub fn places_serving(&self, room: RoomId) -> Vec<&PlaceNode> {
        let Some(topology) = &self.topology else { return Vec::new() };
        topology
            .places
            .iter()
            .filter(|p| p.room == room || self.doorway_place(p.id).is_some_and(|d| d.room_a == room || d.room_b == room))
            .collect()
    }

    /// The shortest path between two places over the place edges, with the edges it takes.
    pub fn shortest_path(&self, from: PlaceId, to: PlaceId) -> Option<(Vec<PlaceId>, Vec<PlaceEdge>)> {
        let edges = &self.topology.as_ref()?.edges;
        let mut best: BTreeMap<PlaceId, (u64, Option<(PlaceId, PlaceEdge)>)> = BTreeMap::from([(from, (0, None))]);
        let mut frontier = BinaryHeap::from([Reverse((0_u64, from))]);
        while let Some(Reverse((distance_um, here))) = frontier.pop() {
            if here == to {
                break;
            }
            if best.get(&here).is_some_and(|(known, _)| *known < distance_um) {
                continue;
            }
            for edge in edges {
                let next = match (edge.a == here, edge.b == here) {
                    (true, _) => edge.b,
                    (_, true) => edge.a,
                    _ => continue,
                };
                let through = distance_um + (f64::from(edge.length_m) * 1e6).round() as u64;
                if best.get(&next).is_none_or(|(known, _)| through < *known) {
                    best.insert(next, (through, Some((here, *edge))));
                    frontier.push(Reverse((through, next)));
                }
            }
        }
        let (mut path, mut taken) = (vec![to], Vec::new());
        let mut here = to;
        while here != from {
            let (previous, edge) = best.get(&here)?.1?;
            taken.push(edge);
            path.push(previous);
            here = previous;
        }
        path.reverse();
        taken.reverse();
        Some((path, taken))
    }
}
