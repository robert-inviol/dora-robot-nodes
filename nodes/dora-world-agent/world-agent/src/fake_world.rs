//! A small house held in memory, for tests and for trying the agents without a robot: four rooms, the places
//! between them, a few pieces of furniture and two people tracks. Rooms and footprints are axis-aligned rectangles.

use std::collections::{BTreeMap, BinaryHeap};
use std::num::NonZeroUsize;
use std::cmp::Reverse;

use perception_wire::navigation::{GoalTarget, LookTarget, MapPose, MapVersion, ObjectId, PlaceId, RoomId, TrackId, UnixNanos};

use crate::clock;
use crate::room_names::{RoomNameSink, RoomNameSinkError};
use crate::world::{
    Change, ChangeKind, Clearance, Doorway, EntityRef, FindQuery, FoundObject, Landmark, ObjectDescription,
    ObjectSummary, PersonTrack, Placement, PlaceSummary, Point, PngImage, ReadOnlySql, Relation, RobotState,
    RoomName, RoomSummary, Route, Sighting, Size, SqlRows, Stamp, TrackLiveness, ViewKind, ViewRequest, WorldError,
    WorldModel, WorldSnapshot,
};

const ROBOT_WIDTH_M: f32 = 0.38;
const ROBOT_MARGIN_M: f32 = 0.10;
const CRUISE_SPEED_M_S: f32 = 0.3;
const DOORWAY_CLEARANCE_M: f32 = 0.8;
const OPEN_FLOOR_CLEARANCE_M: f32 = 1.2;
const DEFAULT_FIND_LIMIT: usize = 5;
const RENDER_PX_PER_M: f32 = 20.0;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Rect {
    min: Point,
    max: Point,
}

impl Rect {
    fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self { min: Point { x_m: x0, y_m: y0 }, max: Point { x_m: x1, y_m: y1 } }
    }

    fn around(centre: Point, size: Size) -> Self {
        let (half_long, half_short) = (size.long_m / 2.0, size.short_m / 2.0);
        Self::new(centre.x_m - half_long, centre.y_m - half_short, centre.x_m + half_long, centre.y_m + half_short)
    }

    fn centre(&self) -> Point {
        Point { x_m: (self.min.x_m + self.max.x_m) / 2.0, y_m: (self.min.y_m + self.max.y_m) / 2.0 }
    }

    fn contains(&self, p: Point) -> bool {
        (self.min.x_m..=self.max.x_m).contains(&p.x_m) && (self.min.y_m..=self.max.y_m).contains(&p.y_m)
    }

    /// The shortest gap between two rectangles and the middle of it; zero where they touch or overlap.
    fn gap_to(&self, other: &Rect) -> (f32, Point) {
        let axis = |a0: f32, a1: f32, b0: f32, b1: f32| -> (f32, f32) {
            if a1 < b0 {
                (b0 - a1, (a1 + b0) / 2.0)
            } else if b1 < a0 {
                (a0 - b1, (b1 + a0) / 2.0)
            } else {
                (0.0, (a0.max(b0) + a1.min(b1)) / 2.0)
            }
        };
        let (dx, mx) = axis(self.min.x_m, self.max.x_m, other.min.x_m, other.max.x_m);
        let (dy, my) = axis(self.min.y_m, self.max.y_m, other.min.y_m, other.max.y_m);
        (dx.hypot(dy), Point { x_m: mx, y_m: my })
    }
}

#[derive(Clone, Debug)]
struct Room {
    id: RoomId,
    name: RoomName,
    bounds: Rect,
}

#[derive(Clone, Debug)]
struct Place {
    id: PlaceId,
    name: &'static str,
    room: RoomName,
    position: Point,
    doorway_to: Option<RoomName>,
}

#[derive(Clone, Debug)]
struct FakeObject {
    summary: ObjectSummary,
    caption: &'static str,
    relations: Vec<Relation>,
    history: Vec<Placement>,
}

/// A fixed house whose clock is the moment it was built, so ages in its answers stay put across calls.
#[derive(Clone, Debug)]
pub struct FakeWorld {
    now: UnixNanos,
    map_version: MapVersion,
    robot: RobotState,
    rooms: Vec<Room>,
    places: Vec<Place>,
    edges: Vec<(PlaceId, PlaceId)>,
    objects: Vec<FakeObject>,
    people: Vec<PersonTrack>,
    changes: Vec<Change>,
}

fn room(name: &str) -> RoomName {
    RoomName(name.to_owned())
}

fn point(x_m: f32, y_m: f32) -> Point {
    Point { x_m, y_m }
}

impl FakeWorld {
    pub fn house() -> Self {
        Self::house_at(clock::now())
    }

    pub fn house_at(now: UnixNanos) -> Self {
        let ago = |seconds: i64| UnixNanos(now.0 - seconds * 1_000_000_000);
        let rooms = vec![
            Room { id: RoomId(1), name: room("kitchen"), bounds: Rect::new(0.0, 0.0, 4.0, 4.0) },
            Room { id: RoomId(2), name: room("hallway"), bounds: Rect::new(4.0, 0.0, 6.0, 6.0) },
            Room { id: RoomId(3), name: room("living_room"), bounds: Rect::new(6.0, 0.0, 12.0, 6.0) },
            Room { id: RoomId(4), name: room("bedroom"), bounds: Rect::new(4.0, 6.0, 8.0, 10.0) },
        ];
        let place = |id, name, room_name: &str, x, y, doorway_to: Option<&str>| Place {
            id: PlaceId(id),
            name,
            room: room(room_name),
            position: point(x, y),
            doorway_to: doorway_to.map(room),
        };
        let places = vec![
            place(1, "kitchen middle", "kitchen", 2.0, 2.5, None),
            place(2, "kitchen door", "kitchen", 4.0, 2.0, Some("hallway")),
            place(3, "hallway middle", "hallway", 5.0, 3.0, None),
            place(4, "living room door", "hallway", 6.0, 3.0, Some("living_room")),
            place(5, "living room middle", "living_room", 9.0, 3.0, None),
            place(6, "bedroom door", "hallway", 5.0, 6.0, Some("bedroom")),
            place(7, "bedroom middle", "bedroom", 6.0, 8.0, None),
        ];
        let edges = [(1, 2), (2, 3), (3, 4), (4, 5), (3, 6), (6, 7)].map(|(a, b)| (PlaceId(a), PlaceId(b))).to_vec();
        let object = |id, label: &str, room_name: &str, x, y, (long_m, short_m, height_m), seen_s, caption| FakeObject {
            summary: ObjectSummary {
                id: ObjectId(id),
                label: label.to_owned(),
                room: room(room_name),
                position: point(x, y),
                size: Size { long_m, short_m, height_m },
                last_seen: ago(seen_s),
            },
            caption,
            relations: Vec::new(),
            history: vec![Placement { room: room(room_name), position: point(x, y), from: ago(86_400), until: None }],
        };
        let mut objects = vec![
            object(42, "table", "kitchen", 2.0, 1.5, (1.6, 0.9, 0.74), 40, "a wooden kitchen table"),
            object(43, "chair", "kitchen", 1.2, 0.6, (0.5, 0.5, 0.9), 40, "a wooden chair"),
            object(44, "chair", "kitchen", 2.8, 0.6, (0.5, 0.5, 0.9), 120, "a wooden chair"),
            object(51, "mug", "kitchen", 2.2, 1.6, (0.1, 0.1, 0.1), 300, "a blue mug"),
            object(12, "sofa", "living_room", 9.0, 5.2, (2.1, 0.9, 0.85), 180, "a grey three-seat sofa"),
            object(13, "armchair", "living_room", 7.4, 4.9, (0.9, 0.8, 0.9), 180, "a green armchair"),
            object(20, "bed", "bedroom", 6.0, 9.0, (2.0, 1.6, 0.5), 3600, "a double bed"),
            object(21, "laundry basket", "hallway", 4.6, 5.0, (0.6, 0.4, 0.5), 1500, "a wicker laundry basket"),
        ];
        let mug = objects.iter_mut().find(|o| o.summary.id == ObjectId(51)).expect("the mug is in the house");
        mug.relations.push(Relation { relation: "on".to_owned(), other: ObjectId(42) });
        mug.history = vec![Placement { room: room("kitchen"), position: point(2.2, 1.6), from: ago(420), until: None }];
        let basket = objects.iter_mut().find(|o| o.summary.id == ObjectId(21)).expect("the basket is in the house");
        basket.history = vec![
            Placement { room: room("bedroom"), position: point(7.0, 7.0), from: ago(86_400), until: Some(ago(2400)) },
            Placement { room: room("hallway"), position: point(4.6, 5.0), from: ago(1500), until: None },
        ];
        let people = vec![
            PersonTrack {
                id: TrackId(7),
                room: room("living_room"),
                position: point(8.1, 2.0),
                heading_deg: 180.0,
                speed_m_s: 0.6,
                last_seen: ago(0),
                liveness: TrackLiveness::Live,
            },
            PersonTrack {
                id: TrackId(3),
                room: room("hallway"),
                position: point(5.0, 4.2),
                heading_deg: 90.0,
                speed_m_s: 0.8,
                last_seen: ago(120),
                liveness: TrackLiveness::Lost,
            },
        ];
        let changes = vec![
            Change {
                detected: ago(300),
                earliest: ago(420),
                object: ObjectId(51),
                label: "mug".to_owned(),
                room: room("kitchen"),
                kind: ChangeKind::Appeared,
            },
            Change {
                detected: ago(1500),
                earliest: ago(2400),
                object: ObjectId(21),
                label: "laundry basket".to_owned(),
                room: room("hallway"),
                kind: ChangeKind::Moved { from: room("bedroom"), to: room("hallway") },
            },
        ];
        Self {
            now,
            map_version: MapVersion(412),
            robot: RobotState { room: room("kitchen"), pose: MapPose { x_m: 3.2, y_m: 2.5, yaw_rad: std::f32::consts::FRAC_PI_2 } },
            rooms,
            places,
            edges,
            objects,
            people,
            changes,
        }
    }

    fn stamp_now(&self) -> Stamp {
        Stamp { as_of: self.now, map_version: self.map_version }
    }

    fn robot_point(&self) -> Point {
        point(self.robot.pose.x_m, self.robot.pose.y_m)
    }

    fn object(&self, id: ObjectId) -> Result<&FakeObject, WorldError> {
        self.objects.iter().find(|o| o.summary.id == id).ok_or(WorldError::UnknownObject(id))
    }

    fn place(&self, id: PlaceId) -> Result<&Place, WorldError> {
        self.places.iter().find(|p| p.id == id).ok_or(WorldError::UnknownPlace(id))
    }

    fn track(&self, id: TrackId) -> Result<&PersonTrack, WorldError> {
        self.people.iter().find(|t| t.id == id).ok_or(WorldError::UnknownTrack(id))
    }

    fn check_room(&self, name: &RoomName) -> Result<(), WorldError> {
        match self.rooms.iter().any(|r| &r.name == name) {
            true => Ok(()),
            false => Err(WorldError::UnknownRoom(name.clone())),
        }
    }

    fn footprint(&self, landmark: Landmark) -> Result<Rect, WorldError> {
        Ok(match landmark {
            Landmark::Robot => Rect::around(self.robot_point(), Size { long_m: ROBOT_WIDTH_M, short_m: ROBOT_WIDTH_M, height_m: 0.0 }),
            Landmark::Object(id) => {
                let summary = &self.object(id)?.summary;
                Rect::around(summary.position, summary.size)
            }
            Landmark::Place(id) => {
                let position = self.place(id)?.position;
                Rect { min: position, max: position }
            }
        })
    }

    fn position_and_room(&self, landmark: Landmark) -> Result<(Point, RoomName), WorldError> {
        Ok(match landmark {
            Landmark::Robot => (self.robot_point(), self.robot.room.clone()),
            Landmark::Object(id) => {
                let summary = &self.object(id)?.summary;
                (summary.position, summary.room.clone())
            }
            Landmark::Place(id) => {
                let place = self.place(id)?;
                (place.position, place.room.clone())
            }
        })
    }

    /// The place a landmark joins the graph at: itself for a place, otherwise the nearest place in its room.
    fn entry_place(&self, landmark: Landmark) -> Result<&Place, WorldError> {
        if let Landmark::Place(id) = landmark {
            return self.place(id);
        }
        let (position, room_name) = self.position_and_room(landmark)?;
        Ok(self
            .places
            .iter()
            .filter(|p| p.room == room_name || p.doorway_to.as_ref() == Some(&room_name))
            .min_by(|a, b| a.position.distance_to(position).total_cmp(&b.position.distance_to(position)))
            .expect("every room of the house has a place"))
    }

    fn shortest_path(&self, from: PlaceId, to: PlaceId) -> Option<Vec<PlaceId>> {
        let length = |a: PlaceId, b: PlaceId| -> f32 {
            let (pa, pb) = (self.place(a).expect("edges join places"), self.place(b).expect("edges join places"));
            pa.position.distance_to(pb.position)
        };
        let mut best: BTreeMap<PlaceId, (u32, Option<PlaceId>)> = BTreeMap::from([(from, (0, None))]);
        let mut frontier = BinaryHeap::from([Reverse((0_u32, from))]);
        while let Some(Reverse((distance_mm, here))) = frontier.pop() {
            if here == to {
                break;
            }
            if best.get(&here).is_some_and(|(known, _)| *known < distance_mm) {
                continue;
            }
            let neighbours = self.edges.iter().filter_map(|&(a, b)| match (a == here, b == here) {
                (true, _) => Some(b),
                (_, true) => Some(a),
                _ => None,
            });
            for next in neighbours {
                let through = distance_mm + (length(here, next) * 1000.0).round() as u32;
                if best.get(&next).is_none_or(|(known, _)| through < *known) {
                    best.insert(next, (through, Some(here)));
                    frontier.push(Reverse((through, next)));
                }
            }
        }
        let mut path = vec![to];
        let mut here = to;
        while here != from {
            here = best.get(&here)?.1?;
            path.push(here);
        }
        path.reverse();
        Some(path)
    }

    fn render(&self, request: &ViewRequest) -> Result<PngImage, WorldError> {
        if request.kind == ViewKind::Egocentric {
            return Err(WorldError::Unsupported("render an egocentric view"));
        }
        let area = match &request.room {
            Some(name) => self.rooms.iter().find(|r| &r.name == name).ok_or_else(|| WorldError::UnknownRoom(name.clone()))?.bounds,
            None => self.rooms.iter().map(|r| r.bounds).reduce(|a, b| Rect {
                min: point(a.min.x_m.min(b.min.x_m), a.min.y_m.min(b.min.y_m)),
                max: point(a.max.x_m.max(b.max.x_m), a.max.y_m.max(b.max.y_m)),
            }).expect("the house has rooms"),
        };
        for id in &request.highlight {
            self.object(*id)?;
        }
        let mut canvas = Canvas::new(area);
        for r in &self.rooms {
            canvas.fill(r.bounds, [235, 235, 230]);
            canvas.outline(r.bounds, [90, 90, 90]);
        }
        for o in &self.objects {
            let colour = if request.highlight.contains(&o.summary.id) { [220, 40, 40] } else { [120, 100, 80] };
            canvas.fill(Rect::around(o.summary.position, o.summary.size), colour);
        }
        for t in self.people.iter().filter(|t| t.liveness == TrackLiveness::Live) {
            canvas.fill(Rect::around(t.position, Size { long_m: 0.4, short_m: 0.4, height_m: 0.0 }), [40, 160, 60]);
        }
        canvas.fill(Rect::around(self.robot_point(), Size { long_m: ROBOT_WIDTH_M, short_m: ROBOT_WIDTH_M, height_m: 0.0 }), [40, 80, 220]);
        Ok(PngImage(canvas.encode()))
    }
}

/// An RGB raster of part of the map, north up.
struct Canvas {
    area: Rect,
    width: usize,
    height: usize,
    pixels: Vec<[u8; 3]>,
}

impl Canvas {
    fn new(area: Rect) -> Self {
        let width = ((area.max.x_m - area.min.x_m) * RENDER_PX_PER_M).ceil() as usize + 1;
        let height = ((area.max.y_m - area.min.y_m) * RENDER_PX_PER_M).ceil() as usize + 1;
        Self { area, width, height, pixels: vec![[255, 255, 255]; width * height] }
    }

    fn pixel_range(&self, rect: Rect) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
        let to_px = |v: f32, origin: f32, limit: usize| (((v - origin) * RENDER_PX_PER_M).round().max(0.0) as usize).min(limit);
        let columns = to_px(rect.min.x_m, self.area.min.x_m, self.width)..to_px(rect.max.x_m, self.area.min.x_m, self.width - 1) + 1;
        // Rows count down from the top of the image, which is the largest y.
        let rows = to_px(self.area.max.y_m - rect.max.y_m, 0.0, self.height)..to_px(self.area.max.y_m - rect.min.y_m, 0.0, self.height - 1) + 1;
        (columns, rows)
    }

    fn fill(&mut self, rect: Rect, colour: [u8; 3]) {
        let (columns, rows) = self.pixel_range(rect);
        for row in rows {
            for column in columns.clone() {
                self.pixels[row * self.width + column] = colour;
            }
        }
    }

    fn outline(&mut self, rect: Rect, colour: [u8; 3]) {
        let (columns, rows) = self.pixel_range(rect);
        let (last_column, last_row) = (columns.end.saturating_sub(1), rows.end.saturating_sub(1));
        for row in rows.clone() {
            for column in columns.clone() {
                if row == rows.start || row == last_row || column == columns.start || column == last_column {
                    self.pixels[row * self.width + column] = colour;
                }
            }
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut encoder = png::Encoder::new(&mut bytes, self.width as u32, self.height as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("writing a PNG header to memory");
        writer.write_image_data(self.pixels.as_flattened()).expect("writing PNG data to memory");
        writer.finish().expect("finishing a PNG in memory");
        bytes
    }
}

/// The house's rooms are named when it is built; it keeps no other names.
impl RoomNameSink for FakeWorld {
    fn name_room(&self, room: RoomId, name: RoomName) -> Result<(), RoomNameSinkError> {
        Err(RoomNameSinkError(format!("the in-memory house cannot rename {room} to {name}")))
    }
}

impl WorldModel for FakeWorld {
    async fn snapshot(&self) -> Result<WorldSnapshot, WorldError> {
        let rooms = self
            .rooms
            .iter()
            .map(|r| RoomSummary {
                id: Some(r.id),
                name: r.name.clone(),
                centroid: r.bounds.centre(),
                doorways: self
                    .places
                    .iter()
                    .filter_map(|p| match &p.doorway_to {
                        Some(to) if p.room == r.name => Some(Doorway { place: p.id, to: to.clone() }),
                        Some(_) if p.doorway_to.as_ref() == Some(&r.name) => Some(Doorway { place: p.id, to: p.room.clone() }),
                        _ => None,
                    })
                    .collect(),
                places: self
                    .places
                    .iter()
                    .filter(|p| p.room == r.name && p.doorway_to.is_none())
                    .map(|p| PlaceSummary { id: p.id, name: p.name.to_owned(), position: p.position })
                    .collect(),
                objects: self.objects.iter().filter(|o| o.summary.room == r.name).map(|o| o.summary.clone()).collect(),
            })
            .collect();
        Ok(WorldSnapshot {
            stamp: self.stamp_now(),
            map_fused: UnixNanos(self.now.0 - 17_000_000_000),
            robot: self.robot.clone(),
            people: self.people.clone(),
            rooms,
            recent_changes: self.changes.clone(),
            stale_sources: Vec::new(),
        })
    }

    async fn stamp(&self) -> Result<Stamp, WorldError> {
        Ok(self.stamp_now())
    }

    async fn find(&self, query: FindQuery) -> Result<Vec<FoundObject>, WorldError> {
        if let Some(name) = &query.room {
            self.check_room(name)?;
        }
        let words: Vec<String> = query.text.iter().flat_map(|t| t.split_whitespace()).map(str::to_lowercase).collect();
        let robot = self.robot_point();
        let mut found: Vec<FoundObject> = self
            .objects
            .iter()
            .filter(|o| query.room.as_ref().is_none_or(|r| &o.summary.room == r))
            .filter(|o| query.label.as_ref().is_none_or(|l| o.summary.label.eq_ignore_ascii_case(l)))
            .filter_map(|o| {
                let haystack = format!("{} {}", o.summary.label, o.caption).to_lowercase();
                let hits = words.iter().filter(|w| haystack.contains(w.as_str())).count();
                let score = if words.is_empty() { 1.0 } else { hits as f32 / words.len() as f32 };
                (score > 0.0).then(|| FoundObject {
                    id: o.summary.id,
                    label: o.summary.label.clone(),
                    room: o.summary.room.clone(),
                    position: o.summary.position,
                    distance_from_robot_m: o.summary.position.distance_to(robot),
                    last_seen: o.summary.last_seen,
                    score,
                })
            })
            .collect();
        let near = query.near.unwrap_or(robot);
        found.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.position.distance_to(near).total_cmp(&b.position.distance_to(near))));
        found.truncate(query.limit.map_or(DEFAULT_FIND_LIMIT, NonZeroUsize::get));
        Ok(found)
    }

    async fn describe_object(&self, id: ObjectId) -> Result<ObjectDescription, WorldError> {
        let object = self.object(id)?;
        Ok(ObjectDescription {
            summary: object.summary.clone(),
            caption: Some(object.caption.to_owned()),
            relations: object.relations.clone(),
            history: object.history.clone(),
            views: Vec::new(),
        })
    }

    async fn clearance_between(&self, a: Landmark, b: Landmark) -> Result<Clearance, WorldError> {
        let (gap, pinch_point) = self.footprint(a)?.gap_to(&self.footprint(b)?);
        let robot_needs_m = ROBOT_WIDTH_M + ROBOT_MARGIN_M;
        Ok(Clearance { min_gap_m: gap, pinch_point, robot_needs_m, passable: gap >= robot_needs_m })
    }

    async fn route(&self, from: Landmark, to: Landmark) -> Result<Route, WorldError> {
        let (start, _) = self.position_and_room(from)?;
        let (end, _) = self.position_and_room(to)?;
        let (first, last) = (self.entry_place(from)?, self.entry_place(to)?);
        let path = self.shortest_path(first.id, last.id).ok_or(WorldError::NoRoute {
            from: landmark_ref(from),
            to: landmark_ref(to),
        })?;
        let points: Vec<Point> = std::iter::once(start)
            .chain(path.iter().map(|id| self.place(*id).expect("paths run through places").position))
            .chain(std::iter::once(end))
            .collect();
        let length_m: f32 = points.windows(2).map(|w| w[0].distance_to(w[1])).sum();
        let mut rooms: Vec<RoomName> = Vec::new();
        let mut min_clearance_m = OPEN_FLOOR_CLEARANCE_M;
        for id in &path {
            let place = self.place(*id).expect("paths run through places");
            for name in std::iter::once(&place.room).chain(place.doorway_to.iter()) {
                if rooms.last() != Some(name) && !rooms.contains(name) {
                    rooms.push(name.clone());
                }
            }
            if place.doorway_to.is_some() {
                min_clearance_m = min_clearance_m.min(DOORWAY_CLEARANCE_M);
            }
        }
        Ok(Route {
            length_m,
            via: path.iter().map(ToString::to_string).collect(),
            rooms,
            min_clearance_m,
            estimated_s: length_m / CRUISE_SPEED_M_S,
        })
    }

    async fn last_seen(&self, subject: EntityRef) -> Result<Sighting, WorldError> {
        match subject {
            EntityRef::Track(id) => {
                let track = self.track(id)?;
                let heading = track.heading_deg.to_radians();
                let (hx, hy) = (heading.cos(), heading.sin());
                let mut ahead: Vec<&Place> = self
                    .places
                    .iter()
                    .filter(|p| p.doorway_to.is_some())
                    .filter(|p| (p.position.x_m - track.position.x_m) * hx + (p.position.y_m - track.position.y_m) * hy > 0.0)
                    .collect();
                ahead.sort_by(|a, b| a.position.distance_to(track.position).total_cmp(&b.position.distance_to(track.position)));
                Ok(Sighting {
                    subject: subject.to_string(),
                    room: track.room.clone(),
                    position: track.position,
                    at: track.last_seen,
                    seconds_ago: clock::seconds_between(track.last_seen, self.now),
                    heading_deg: Some(track.heading_deg),
                    exits_ahead: ahead.iter().map(|p| format!("{} ({})", p.id, p.name)).collect(),
                })
            }
            EntityRef::Object(id) => {
                let summary = &self.object(id)?.summary;
                Ok(Sighting {
                    subject: subject.to_string(),
                    room: summary.room.clone(),
                    position: summary.position,
                    at: summary.last_seen,
                    seconds_ago: clock::seconds_between(summary.last_seen, self.now),
                    heading_deg: None,
                    exits_ahead: Vec::new(),
                })
            }
            EntityRef::Robot | EntityRef::Place(_) => Err(WorldError::Unsupported("say when the robot or a place was last seen")),
        }
    }

    async fn changes_since(&self, since: UnixNanos, room_name: Option<RoomName>) -> Result<Vec<Change>, WorldError> {
        if let Some(name) = &room_name {
            self.check_room(name)?;
        }
        Ok(self
            .changes
            .iter()
            .filter(|c| c.detected >= since)
            .filter(|c| room_name.as_ref().is_none_or(|r| &c.room == r))
            .cloned()
            .collect())
    }

    async fn render_view(&self, request: ViewRequest) -> Result<PngImage, WorldError> {
        self.render(&request)
    }

    async fn query_sql(&self, _sql: ReadOnlySql, _limit: NonZeroUsize) -> Result<SqlRows, WorldError> {
        Err(WorldError::Unsupported("run SQL: it holds the house in memory, not in a database"))
    }

    async fn check_target(&self, target: GoalTarget) -> Result<(), WorldError> {
        match target {
            GoalTarget::Object(id) | GoalTarget::LookAt(LookTarget::Object(id)) => self.object(id).map(drop),
            GoalTarget::Place(id) => self.place(id).map(drop),
            GoalTarget::FollowTrack(id) => match self.track(id)?.liveness {
                TrackLiveness::Live => Ok(()),
                TrackLiveness::Lost => Err(WorldError::UnknownTrack(id)),
            },
            GoalTarget::Pose(pose) | GoalTarget::LookAt(LookTarget::Point(pose)) => {
                let p = point(pose.x_m, pose.y_m);
                match self.rooms.iter().any(|r| r.bounds.contains(p)) {
                    true => Ok(()),
                    false => Err(WorldError::Unsupported("send the robot outside the mapped rooms")),
                }
            }
            GoalTarget::Stop => Ok(()),
        }
    }
}

fn landmark_ref(landmark: Landmark) -> EntityRef {
    match landmark {
        Landmark::Robot => EntityRef::Robot,
        Landmark::Object(id) => EntityRef::Object(id),
        Landmark::Place(id) => EntityRef::Place(id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: UnixNanos = UnixNanos(1_794_610_000_000_000_000);

    fn house() -> FakeWorld {
        FakeWorld::house_at(NOW)
    }

    #[tokio::test]
    async fn find_ranks_text_matches_and_filters_by_room() {
        let world = house();
        let found = world.find(FindQuery { text: Some("table".to_owned()), ..FindQuery::default() }).await.unwrap();
        assert_eq!(found[0].id, ObjectId(42));
        let chairs = world
            .find(FindQuery { label: Some("chair".to_owned()), room: Some(room("kitchen")), ..FindQuery::default() })
            .await
            .unwrap();
        assert_eq!(chairs.iter().map(|f| f.id).collect::<Vec<_>>(), vec![ObjectId(44), ObjectId(43)]);
        let error = world.find(FindQuery { room: Some(room("garage")), ..FindQuery::default() }).await.unwrap_err();
        assert_eq!(error, WorldError::UnknownRoom(room("garage")));
    }

    #[tokio::test]
    async fn the_gap_between_the_sofa_and_the_armchair_is_measured_between_footprints() {
        let clearance = house().clearance_between(Landmark::Object(ObjectId(12)), Landmark::Object(ObjectId(13))).await.unwrap();
        // The sofa spans x 7.95..10.05, the armchair x 6.95..7.85; their y ranges overlap.
        assert!((clearance.min_gap_m - 0.1).abs() < 1e-4, "{clearance:?}");
        assert!(!clearance.passable);
    }

    #[tokio::test]
    async fn a_route_from_the_robot_to_the_sofa_runs_through_the_hallway() {
        let route = house().route(Landmark::Robot, Landmark::Object(ObjectId(12))).await.unwrap();
        assert_eq!(route.via, vec!["p2", "p3", "p4", "p5"]);
        assert_eq!(route.rooms, vec![room("kitchen"), room("hallway"), room("living_room")]);
        assert_eq!(route.min_clearance_m, DOORWAY_CLEARANCE_M);
        assert!((route.estimated_s - route.length_m / CRUISE_SPEED_M_S).abs() < 1e-4);
    }

    #[tokio::test]
    async fn a_lost_person_heading_north_has_the_bedroom_door_ahead() {
        let sighting = house().last_seen(EntityRef::Track(TrackId(3))).await.unwrap();
        assert_eq!(sighting.exits_ahead, vec!["p6 (bedroom door)"]);
        assert_eq!(sighting.seconds_ago, 120.0);
    }

    #[tokio::test]
    async fn changes_since_a_time_leave_out_older_ones() {
        let world = house();
        let recent = world.changes_since(UnixNanos(NOW.0 - 600_000_000_000), None).await.unwrap();
        assert_eq!(recent.iter().map(|c| c.object).collect::<Vec<_>>(), vec![ObjectId(51)]);
        let in_hallway = world.changes_since(UnixNanos(0), Some(room("hallway"))).await.unwrap();
        assert_eq!(in_hallway.len(), 1);
    }

    #[tokio::test]
    async fn the_top_down_view_is_a_png_of_the_house() {
        let image = house()
            .render_view(ViewRequest { kind: ViewKind::TopDown, room: None, highlight: vec![ObjectId(42)] })
            .await
            .unwrap();
        let decoder = png::Decoder::new(std::io::Cursor::new(image.0));
        let reader = decoder.read_info().unwrap();
        assert_eq!((reader.info().width, reader.info().height), (241, 201));
    }

    #[tokio::test]
    async fn goals_to_invented_or_lost_targets_are_refused() {
        let world = house();
        assert_eq!(world.check_target(GoalTarget::Object(ObjectId(999))).await, Err(WorldError::UnknownObject(ObjectId(999))));
        assert_eq!(world.check_target(GoalTarget::FollowTrack(TrackId(3))).await, Err(WorldError::UnknownTrack(TrackId(3))));
        assert_eq!(world.check_target(GoalTarget::FollowTrack(TrackId(7))).await, Ok(()));
        assert_eq!(world.check_target(GoalTarget::Place(PlaceId(5))).await, Ok(()));
    }
}
