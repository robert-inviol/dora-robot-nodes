//! The store against a seeded house: the same four rooms, places and furniture as world-agent's fake world, put in
//! through the ingest API, then asked every `WorldModel` question.

use std::num::NonZeroUsize;
use std::path::Path;
use std::time::Duration;

use perception_wire::navigation::{
    GoalId, GoalReason, GoalState, GoalStatus, GoalTarget, MapPose, MapVersion, NavGoal, NonNegative, ObjectId, PlaceId,
    TrackId, UnixNanos,
};
use world_model::{
    ChangeKind, EntityRef, FindQuery, Landmark, Point, ReadOnlySql, RoomName, Size, SqlValue, ViewKind, ViewRequest,
    WorldError, WorldModel,
};
use world_store::{
    AgentName, Caption, ClearanceMm, Clock, DoorwayId, DoorwaySpan, FrameId, FreeSpace, HeightBand, IngestError, IssuedGoal,
    JpegImage, ManualClock, ModelName, ObjectChange, ObjectUpdate, OpenError, PlaceEdge, PlaceNode, Position3, Footprint,
    Rejection, RobotPoseSample, RoomId, RoomShape, StoreConfig, Topology, WorldStore, WorldWriter,
};

const NOW: UnixNanos = UnixNanos(1_794_610_000_000_000_000);
const VERSION: MapVersion = MapVersion(412);
const RESOLUTION_M: f32 = 0.1;

fn ago(seconds: i64) -> UnixNanos {
    UnixNanos(NOW.0 - seconds * 1_000_000_000)
}

fn p(x_m: f32, y_m: f32) -> Point {
    Point { x_m, y_m }
}

fn room(name: &str) -> RoomName {
    RoomName(name.to_owned())
}

fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<Point> {
    vec![p(x0, y0), p(x1, y0), p(x1, y1), p(x0, y1)]
}

const ROOMS: [(u64, &str, [f32; 4]); 4] = [
    (1, "kitchen", [0.0, 0.0, 4.0, 4.0]),
    (2, "hallway", [4.0, 0.0, 6.0, 6.0]),
    (3, "living_room", [6.0, 0.0, 12.0, 6.0]),
    (4, "bedroom", [4.0, 6.0, 8.0, 10.0]),
];
const DOORWAYS: [(u64, u64, u64, f32, f32); 3] = [(1, 1, 2, 4.0, 2.0), (2, 2, 3, 6.0, 3.0), (3, 2, 4, 5.0, 6.0)];
const DOORWAY_WIDTH_M: f32 = 0.9;
const PLACES: [(u64, u64, f32, f32); 7] =
    [(1, 1, 2.0, 2.5), (2, 1, 4.0, 2.0), (3, 2, 5.0, 3.0), (4, 2, 6.0, 3.0), (5, 3, 9.0, 3.0), (6, 2, 5.0, 6.0), (7, 4, 6.0, 8.0)];
const DOORWAY_PLACES: [u64; 3] = [2, 4, 6];
const EDGES: [(u64, u64); 6] = [(1, 2), (2, 3), (3, 4), (4, 5), (3, 6), (6, 7)];
const DOORWAY_CLEARANCE_M: f32 = 0.8;
const OPEN_FLOOR_CLEARANCE_M: f32 = 1.2;

fn topology() -> Topology {
    let place = |id: u64| PLACES.iter().find(|(p, ..)| *p == id).map(|&(_, _, x, y)| p(x, y)).unwrap();
    let clearance = |id: u64| if DOORWAY_PLACES.contains(&id) { DOORWAY_CLEARANCE_M } else { OPEN_FLOOR_CLEARANCE_M };
    Topology {
        map_version: VERSION,
        built: ago(17),
        rooms: ROOMS
            .iter()
            .map(|&(id, _, [x0, y0, x1, y1])| RoomShape { id: RoomId(id), polygon: rect(x0, y0, x1, y1), floor_m: 0.0, ceiling_m: Some(2.4) })
            .collect(),
        doorways: DOORWAYS
            .iter()
            .map(|&(id, a, b, x, y)| DoorwaySpan { id: DoorwayId(id), room_a: RoomId(a), room_b: RoomId(b), centre: p(x, y), width_m: DOORWAY_WIDTH_M })
            .collect(),
        places: PLACES
            .iter()
            .map(|&(id, room, x, y)| PlaceNode { id: PlaceId(id), room: RoomId(room), position: p(x, y), clearance_m: clearance(id) })
            .collect(),
        edges: EDGES
            .iter()
            .map(|&(a, b)| PlaceEdge {
                a: PlaceId(a),
                b: PlaceId(b),
                length_m: place(a).distance_to(place(b)),
                min_clearance_m: clearance(a).min(clearance(b)),
            })
            .collect(),
    }
}

struct Furniture {
    id: u64,
    label: &'static str,
    at: (f32, f32),
    size: (f32, f32, f32),
    resting_on_m: f32,
    caption: &'static str,
}

const FURNITURE: [Furniture; 8] = [
    Furniture { id: 42, label: "table", at: (2.0, 1.5), size: (1.6, 0.9, 0.74), resting_on_m: 0.0, caption: "a wooden kitchen table" },
    Furniture { id: 43, label: "chair", at: (1.2, 0.6), size: (0.5, 0.5, 0.9), resting_on_m: 0.0, caption: "a wooden chair" },
    Furniture { id: 44, label: "chair", at: (2.8, 0.6), size: (0.5, 0.5, 0.9), resting_on_m: 0.0, caption: "a wooden chair" },
    Furniture { id: 51, label: "mug", at: (2.2, 1.6), size: (0.1, 0.1, 0.1), resting_on_m: 0.74, caption: "a blue mug" },
    Furniture { id: 12, label: "sofa", at: (9.0, 5.2), size: (2.1, 0.9, 0.85), resting_on_m: 0.0, caption: "a grey three-seat sofa" },
    Furniture { id: 13, label: "armchair", at: (7.4, 4.9), size: (0.9, 0.8, 0.9), resting_on_m: 0.0, caption: "a green armchair" },
    Furniture { id: 20, label: "bed", at: (6.0, 9.0), size: (2.0, 1.6, 0.5), resting_on_m: 0.0, caption: "a double bed" },
    Furniture { id: 21, label: "laundry basket", at: (7.0, 7.0), size: (0.6, 0.4, 0.5), resting_on_m: 0.0, caption: "a wicker laundry basket" },
];

fn footprint(at: (f32, f32), size: (f32, f32, f32)) -> Footprint {
    let (hl, hs) = (size.0 / 2.0, size.1 / 2.0);
    Footprint([p(at.0 - hl, at.1 - hs), p(at.0 + hl, at.1 - hs), p(at.0 + hl, at.1 + hs), p(at.0 - hl, at.1 + hs)])
}

fn crop() -> JpegImage {
    let mut bytes = Vec::new();
    let pixels: Vec<u8> = (0..16 * 8).flat_map(|i| [(i * 2) as u8, 100, 200]).collect();
    jpeg_encoder::Encoder::new(&mut bytes, 90).encode(&pixels, 16, 8, jpeg_encoder::ColorType::Rgb).unwrap();
    JpegImage(bytes)
}

fn update(f: &Furniture, change: ObjectChange, at: (f32, f32), captured: UnixNanos) -> ObjectUpdate {
    let top_m = f.resting_on_m + f.size.2;
    ObjectUpdate {
        change,
        object_id: ObjectId(f.id),
        label: f.label.to_owned(),
        score: 0.9,
        confidence: 0.8,
        seen_count: 5,
        frame_id: FrameId(1),
        captured,
        map_version: VERSION,
        centre: Position3 { x_m: at.0, y_m: at.1, z_m: f.resting_on_m + f.size.2 / 2.0 },
        footprint: footprint(at, f.size),
        size: Size { long_m: f.size.0, short_m: f.size.1, height_m: f.size.2 },
        top_m,
        crop: (f.id == 42).then(crop),
    }
}

fn last_seen_ago(id: u64) -> i64 {
    match id {
        42 | 43 => 40,
        44 => 120,
        51 => 300,
        12 | 13 => 180,
        20 => 3600,
        _ => 0,
    }
}

/// The free-space grid of the house: walls along every room outline except the doorways, the furniture, and
/// unknown ground outside the rooms. Clearances are exact distances to the nearest occupied cell, capped at 1.5 m.
fn free_space() -> FreeSpace {
    let (width, height) = (120_u32, 100_u32);
    let centre = |column: u32, row: u32| p((column as f32 + 0.5) * RESOLUTION_M, (row as f32 + 0.5) * RESOLUTION_M);
    let inside_rooms = |q: Point| ROOMS.iter().any(|(_, _, [x0, y0, x1, y1])| (*x0..=*x1).contains(&q.x_m) && (*y0..=*y1).contains(&q.y_m));
    let on_wall = |q: Point| {
        let near_edge = ROOMS.iter().any(|(_, _, [x0, y0, x1, y1])| {
            let within_y = (*y0 - 0.1..=*y1 + 0.1).contains(&q.y_m);
            let within_x = (*x0 - 0.1..=*x1 + 0.1).contains(&q.x_m);
            (within_y && ((q.x_m - x0).abs() < 0.1 || (q.x_m - x1).abs() < 0.1))
                || (within_x && ((q.y_m - y0).abs() < 0.1 || (q.y_m - y1).abs() < 0.1))
        });
        let in_doorway = DOORWAYS.iter().any(|&(_, _, _, x, y)| q.distance_to(p(x, y)) < DOORWAY_WIDTH_M / 2.0);
        near_edge && !in_doorway
    };
    let in_furniture = |q: Point| {
        FURNITURE.iter().filter(|f| f.id != 21).any(|f| {
            let fp = footprint(f.at, f.size).0;
            (fp[0].x_m..=fp[2].x_m).contains(&q.x_m) && (fp[0].y_m..=fp[2].y_m).contains(&q.y_m)
        })
    };
    let mut state = vec![0_u8; (width * height) as usize]; // 0 free, 1 occupied, 2 unknown
    for row in 0..height {
        for column in 0..width {
            let q = centre(column, row);
            state[(row * width + column) as usize] = match (on_wall(q) || in_furniture(q), inside_rooms(q)) {
                (true, _) => 1,
                (false, true) => 0,
                (false, false) => 2,
            };
        }
    }
    let reach = 15_i32;
    let mut cells = Vec::with_capacity(state.len());
    for row in 0..height as i32 {
        for column in 0..width as i32 {
            let index = (row * width as i32 + column) as usize;
            cells.push(match state[index] {
                1 => ClearanceMm::OCCUPIED,
                2 => ClearanceMm::UNKNOWN,
                _ => {
                    let mut nearest = reach as f32 * RESOLUTION_M;
                    for dy in -reach..=reach {
                        for dx in -reach..=reach {
                            let (c, r) = (column + dx, row + dy);
                            if c >= 0 && r >= 0 && c < width as i32 && r < height as i32 && state[(r * width as i32 + c) as usize] == 1 {
                                nearest = nearest.min(((dx * dx + dy * dy) as f32).sqrt() * RESOLUTION_M);
                            }
                        }
                    }
                    ClearanceMm((nearest * 1000.0).round() as u16)
                }
            });
        }
    }
    FreeSpace::new(VERSION, ago(17), p(0.0, 0.0), RESOLUTION_M, width, height, HeightBand { min_m: 0.05, max_m: 1.2 }, cells).unwrap()
}

fn config(dir: &Path) -> (StoreConfig, ManualClock) {
    let clock = ManualClock::starting_at(NOW);
    let mut config = StoreConfig::at(dir.join("world.sqlite"));
    config.clock = Clock::Manual(clock.clone());
    config.batch_interval = Duration::from_millis(20);
    (config, clock)
}

fn seed(writer: &WorldWriter) {
    writer.set_topology(topology()).unwrap();
    for &(id, name, _) in &ROOMS {
        writer.name_room(RoomId(id), room(name)).unwrap();
    }
    writer.set_free_space(free_space()).unwrap();
    for f in &FURNITURE {
        let appeared = if f.id == 51 { ago(300) } else { ago(86_400) };
        writer.apply_object_update(update(f, ObjectChange::Appeared, f.at, appeared)).unwrap();
        writer.set_caption(ObjectId(f.id), Caption(f.caption.to_owned())).unwrap();
    }
    for f in FURNITURE.iter().filter(|f| ![21, 51].contains(&f.id)) {
        writer.apply_object_update(update(f, ObjectChange::Updated, f.at, ago(last_seen_ago(f.id)))).unwrap();
    }
    let basket = FURNITURE.iter().find(|f| f.id == 21).unwrap();
    writer.apply_object_update(update(basket, ObjectChange::Updated, basket.at, ago(2400))).unwrap();
    writer.apply_object_update(update(basket, ObjectChange::Moved, (4.6, 5.0), ago(1500))).unwrap();
    writer
        .record_robot_pose(RobotPoseSample {
            captured: NOW,
            map_version: VERSION,
            pose: MapPose { x_m: 3.2, y_m: 2.5, yaw_rad: std::f32::consts::FRAC_PI_2 },
        })
        .unwrap();
    writer.flush().unwrap();
}

struct House {
    store: WorldStore,
    writer: WorldWriter,
    clock: ManualClock,
    _dir: tempfile::TempDir,
}

fn house() -> House {
    let dir = tempfile::tempdir().unwrap();
    let (config, clock) = config(dir.path());
    let (store, writer) = WorldStore::open(config).unwrap();
    seed(&writer);
    House { store, writer, clock, _dir: dir }
}

fn ids(found: &[world_model::FoundObject]) -> Vec<ObjectId> {
    found.iter().map(|f| f.id).collect()
}

#[tokio::test]
async fn find_ranks_text_matches_and_filters_by_room() {
    let House { store, _dir, .. } = house();
    let found = store.find(FindQuery { text: Some("table".to_owned()), ..FindQuery::default() }).await.unwrap();
    assert_eq!(found[0].id, ObjectId(42));
    let chairs = store
        .find(FindQuery { label: Some("chair".to_owned()), room: Some(room("kitchen")), ..FindQuery::default() })
        .await
        .unwrap();
    assert_eq!(ids(&chairs), vec![ObjectId(44), ObjectId(43)]);
    let error = store.find(FindQuery { room: Some(room("garage")), ..FindQuery::default() }).await.unwrap_err();
    assert_eq!(error, WorldError::UnknownRoom(room("garage")));
}

#[tokio::test]
async fn find_matches_captions_and_ranks_by_distance_from_a_point() {
    let House { store, _dir, .. } = house();
    let blue = store.find(FindQuery { text: Some("blue mug".to_owned()), ..FindQuery::default() }).await.unwrap();
    assert_eq!(ids(&blue), vec![ObjectId(51)]);
    assert_eq!(blue[0].score, 1.0);
    let half = store.find(FindQuery { text: Some("wooden sofa".to_owned()), ..FindQuery::default() }).await.unwrap();
    assert!(half.iter().all(|f| f.score == 0.5), "{half:?}");
    let near_bed = store
        .find(FindQuery { near: Some(p(6.0, 9.0)), limit: NonZeroUsize::new(2), ..FindQuery::default() })
        .await
        .unwrap();
    assert_eq!(ids(&near_bed), vec![ObjectId(20), ObjectId(21)]);
    let unnamed = store.find(FindQuery { room: Some(room("room3")), ..FindQuery::default() }).await.unwrap();
    assert_eq!(ids(&unnamed), vec![ObjectId(13), ObjectId(12)]);
    let literal = store.find(FindQuery { text: Some("%".to_owned()), ..FindQuery::default() }).await.unwrap();
    assert!(literal.is_empty());
}

#[tokio::test]
async fn the_gap_between_the_sofa_and_the_armchair_is_measured_between_footprints() {
    let House { store, _dir, .. } = house();
    let clearance = store.clearance_between(Landmark::Object(ObjectId(12)), Landmark::Object(ObjectId(13))).await.unwrap();
    assert!((clearance.min_gap_m - 0.1).abs() < 1e-4, "{clearance:?}");
    assert!((clearance.robot_needs_m - 0.48).abs() < 1e-6);
    assert!(!clearance.passable);
}

#[tokio::test]
async fn clearance_along_a_line_comes_from_the_free_space_grid() {
    let House { store, _dir, .. } = house();
    let open = store.clearance_between(Landmark::Place(PlaceId(2)), Landmark::Place(PlaceId(4))).await.unwrap();
    assert!(open.passable, "{open:?}");
    let through_doorway = store.clearance_between(Landmark::Place(PlaceId(3)), Landmark::Place(PlaceId(5))).await.unwrap();
    assert!((0.7..=1.0).contains(&through_doorway.min_gap_m), "{through_doorway:?}");
    assert!((through_doorway.pinch_point.x_m - 6.0).abs() < 0.2, "{through_doorway:?}");
    let through_wall = store.clearance_between(Landmark::Place(PlaceId(3)), Landmark::Place(PlaceId(7))).await.unwrap();
    assert_eq!(through_wall.min_gap_m, 0.0);
    assert!(!through_wall.passable);
    let off_the_map = store.clearance_between(Landmark::Place(PlaceId(1)), Landmark::Place(PlaceId(7))).await.unwrap_err();
    assert!(matches!(off_the_map, WorldError::Unmapped(_)), "{off_the_map}");
}

#[tokio::test]
async fn a_route_from_the_robot_to_the_sofa_runs_through_the_hallway() {
    let House { store, _dir, .. } = house();
    let route = store.route(Landmark::Robot, Landmark::Object(ObjectId(12))).await.unwrap();
    assert_eq!(route.via, vec!["p2", "p3", "p4", "p5"]);
    assert_eq!(route.rooms, vec![room("kitchen"), room("hallway"), room("living_room")]);
    assert_eq!(route.min_clearance_m, DOORWAY_CLEARANCE_M);
    assert!((route.estimated_s - route.length_m / 0.3).abs() < 1e-3);
    let to_bed = store.route(Landmark::Place(PlaceId(1)), Landmark::Object(ObjectId(20))).await.unwrap();
    assert_eq!(to_bed.via, vec!["p1", "p2", "p3", "p6", "p7"]);
    let unknown = store.route(Landmark::Robot, Landmark::Place(PlaceId(99))).await.unwrap_err();
    assert_eq!(unknown, WorldError::UnknownPlace(PlaceId(99)));
}

#[tokio::test]
async fn the_robot_heading_north_has_the_living_room_and_bedroom_doors_ahead() {
    let House { store, _dir, .. } = house();
    let sighting = store.last_seen(EntityRef::Robot).await.unwrap();
    assert_eq!(sighting.room, room("kitchen"));
    assert!((sighting.heading_deg.unwrap() - 90.0).abs() < 1e-3);
    assert_eq!(
        sighting.exits_ahead,
        vec!["p4 (doorway between hallway and living_room)", "p6 (doorway between hallway and bedroom)"]
    );
}

#[tokio::test]
async fn an_object_was_last_seen_where_it_last_stood() {
    let House { store, clock, _dir, .. } = house();
    let basket = store.last_seen(EntityRef::Object(ObjectId(21))).await.unwrap();
    assert_eq!((basket.room, basket.at, basket.seconds_ago), (room("hallway"), ago(1500), 1500.0));
    clock.advance(Duration::from_secs(60));
    let table = store.last_seen(EntityRef::Object(ObjectId(42))).await.unwrap();
    assert_eq!(table.seconds_ago, 100.0);
    assert_eq!(store.last_seen(EntityRef::Track(TrackId(3))).await.unwrap_err(), WorldError::UnknownTrack(TrackId(3)));
    assert!(matches!(store.last_seen(EntityRef::Place(PlaceId(1))).await, Err(WorldError::Unsupported(_))));
}

#[tokio::test]
async fn changes_since_a_time_leave_out_older_ones() {
    let House { store, _dir, .. } = house();
    let recent = store.changes_since(ago(600), None).await.unwrap();
    assert_eq!(recent.iter().map(|c| c.object).collect::<Vec<_>>(), vec![ObjectId(51)]);
    let in_hallway = store.changes_since(UnixNanos(0), Some(room("hallway"))).await.unwrap();
    assert_eq!(in_hallway.len(), 1);
    let moved = &in_hallway[0];
    assert_eq!(moved.kind, ChangeKind::Moved { from: room("bedroom"), to: room("hallway") });
    assert_eq!((moved.earliest, moved.detected), (ago(2400), ago(1500)));
    let everything = store.changes_since(UnixNanos(0), None).await.unwrap();
    assert_eq!(everything.len(), 9, "eight appearances and a move");
    assert!(everything.windows(2).all(|w| w[0].detected >= w[1].detected));
}

#[tokio::test]
async fn a_move_keeps_where_the_object_stood_before() {
    let House { store, _dir, .. } = house();
    let before = store.placement_at(ObjectId(21), ago(2000)).await.unwrap().unwrap();
    assert_eq!((before.room, before.position, before.until), (room("bedroom"), p(7.0, 7.0), Some(ago(1500))));
    let after = store.placement_at(ObjectId(21), ago(1000)).await.unwrap().unwrap();
    assert_eq!((after.room, after.until), (room("hallway"), None));
    assert_eq!(store.placement_at(ObjectId(21), ago(100_000)).await.unwrap(), None);
    assert_eq!(store.placement_at(ObjectId(999), NOW).await.unwrap_err(), WorldError::UnknownObject(ObjectId(999)));
    let described = store.describe_object(ObjectId(21)).await.unwrap();
    assert_eq!(described.history.iter().map(|h| h.room.0.as_str()).collect::<Vec<_>>(), vec!["bedroom", "hallway"]);
}

#[tokio::test]
async fn describing_an_object_gives_its_caption_relations_and_views() {
    let House { store, _dir, .. } = house();
    let mug = store.describe_object(ObjectId(51)).await.unwrap();
    assert_eq!(mug.caption.as_deref(), Some("a blue mug"));
    assert_eq!(mug.relations.iter().map(|r| (r.relation.as_str(), r.other)).collect::<Vec<_>>(), vec![("on", ObjectId(42))]);
    assert!(mug.views.is_empty());
    let table = store.describe_object(ObjectId(42)).await.unwrap();
    assert_eq!(table.summary.room, room("kitchen"));
    let next_to: Vec<ObjectId> = table.relations.iter().filter(|r| r.relation == "next to").map(|r| r.other).collect();
    assert_eq!(next_to, vec![ObjectId(43), ObjectId(44)]);
    assert_eq!(table.views.len(), 2, "the appearance and the update each brought a crop");
    let decoder = png::Decoder::new(std::io::Cursor::new(table.views[0].0.clone()));
    let reader = decoder.read_info().unwrap();
    assert_eq!((reader.info().width, reader.info().height), (16, 8));
    assert_eq!(store.describe_object(ObjectId(999)).await.unwrap_err(), WorldError::UnknownObject(ObjectId(999)));
}

#[tokio::test]
async fn the_top_down_view_is_a_png_of_the_house_or_a_room() {
    let House { store, _dir, .. } = house();
    let size = |png_bytes: Vec<u8>| {
        let reader = png::Decoder::new(std::io::Cursor::new(png_bytes)).read_info().unwrap();
        (reader.info().width, reader.info().height)
    };
    let whole = store.render_view(ViewRequest { kind: ViewKind::TopDown, room: None, highlight: vec![ObjectId(42)] }).await.unwrap();
    let (width, height) = size(whole.0);
    assert!((500..=510).contains(&width) && (420..=430).contains(&height), "{width}x{height}");
    let kitchen = store.render_view(ViewRequest { kind: ViewKind::TopDown, room: Some(room("kitchen")), highlight: vec![] }).await.unwrap();
    let (width, height) = size(kitchen.0);
    assert!((180..=190).contains(&width) && (180..=190).contains(&height), "{width}x{height}");
    let unknown = store.render_view(ViewRequest { kind: ViewKind::TopDown, room: None, highlight: vec![ObjectId(7)] }).await;
    assert_eq!(unknown.unwrap_err(), WorldError::UnknownObject(ObjectId(7)));
    let egocentric = store.render_view(ViewRequest { kind: ViewKind::Egocentric, room: None, highlight: vec![] }).await;
    assert!(matches!(egocentric, Err(WorldError::Unsupported(_))));
}

#[tokio::test]
async fn goals_to_invented_or_lost_targets_are_refused() {
    let House { store, _dir, .. } = house();
    assert_eq!(store.check_target(GoalTarget::Object(ObjectId(999))).await, Err(WorldError::UnknownObject(ObjectId(999))));
    assert_eq!(store.check_target(GoalTarget::FollowTrack(TrackId(3))).await, Err(WorldError::UnknownTrack(TrackId(3))));
    assert_eq!(store.check_target(GoalTarget::Place(PlaceId(5))).await, Ok(()));
    assert_eq!(store.check_target(GoalTarget::Place(PlaceId(55))).await, Err(WorldError::UnknownPlace(PlaceId(55))));
    assert_eq!(store.check_target(GoalTarget::Object(ObjectId(42))).await, Ok(()));
    let inside = MapPose { x_m: 1.0, y_m: 3.0, yaw_rad: 0.0 };
    let outside = MapPose { x_m: 1.0, y_m: 8.0, yaw_rad: 0.0 };
    assert_eq!(store.check_target(GoalTarget::Pose(inside)).await, Ok(()));
    assert!(matches!(store.check_target(GoalTarget::Pose(outside)).await, Err(WorldError::Unsupported(_))));
}

#[tokio::test]
async fn room_labels_list_every_room_with_its_name_and_objects() {
    let House { store, _dir, .. } = house();
    let rooms = store.room_labels().await.unwrap();
    assert_eq!(rooms.iter().map(|r| r.room).collect::<Vec<_>>(), ROOMS.map(|(id, ..)| RoomId(id)));
    assert_eq!(rooms[0].name, Some(room("kitchen")));
    let mut kitchen = rooms[0].labels.clone();
    kitchen.sort();
    assert_eq!(kitchen, ["chair", "chair", "mug", "table"]);
}

#[tokio::test]
async fn the_snapshot_holds_rooms_objects_changes_and_stale_sources() {
    let House { store, clock, _dir, .. } = house();
    let snapshot = store.snapshot().await.unwrap();
    assert_eq!(snapshot.stamp.map_version, VERSION);
    assert_eq!(snapshot.stamp.as_of, NOW);
    assert_eq!(snapshot.map_fused, ago(17));
    assert_eq!(snapshot.robot.room, room("kitchen"));
    assert!(snapshot.people.is_empty());
    let names: Vec<&str> = snapshot.rooms.iter().map(|r| r.name.0.as_str()).collect();
    assert_eq!(names, vec!["kitchen", "hallway", "living_room", "bedroom"]);
    let ids: Vec<Option<RoomId>> = snapshot.rooms.iter().map(|r| r.id).collect();
    assert_eq!(ids, ROOMS.map(|(id, ..)| Some(RoomId(id))));
    let kitchen = &snapshot.rooms[0];
    assert_eq!(kitchen.centroid, p(2.0, 2.0));
    assert_eq!(snapshot.rooms[2].centroid, p(9.0, 3.0));
    assert_eq!(kitchen.doorways.iter().map(|d| (d.place, d.to.0.as_str())).collect::<Vec<_>>(), vec![(PlaceId(2), "hallway")]);
    assert_eq!(kitchen.places.iter().map(|p| p.id).collect::<Vec<_>>(), vec![PlaceId(1)]);
    assert_eq!(kitchen.objects.len(), 4);
    assert_eq!(snapshot.rooms[1].doorways.len(), 3);
    assert_eq!(snapshot.rooms[1].objects.iter().map(|o| o.id).collect::<Vec<_>>(), vec![ObjectId(21)]);
    assert_eq!(snapshot.recent_changes.iter().take(2).map(|c| c.object).collect::<Vec<_>>(), vec![ObjectId(51), ObjectId(21)]);
    assert!(snapshot.stale_sources.is_empty(), "{:?}", snapshot.stale_sources);
    clock.advance(Duration::from_secs(10));
    let later = store.snapshot().await.unwrap();
    assert_eq!(later.stale_sources.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), vec!["robot pose"]);
    assert_eq!(later.stale_sources[0].silent_s, 10.0);
}

#[tokio::test]
async fn an_empty_store_says_what_it_has_not_mapped() {
    let dir = tempfile::tempdir().unwrap();
    let (config, _) = config(dir.path());
    let (store, _writer) = WorldStore::open(config).unwrap();
    assert_eq!(store.stamp().await.unwrap_err(), WorldError::NotMapped("map"));
    assert_eq!(store.snapshot().await.unwrap_err(), WorldError::NotMapped("map"));
    assert_eq!(store.route(Landmark::Robot, Landmark::Place(PlaceId(1))).await.unwrap_err(), WorldError::NotMapped("place graph"));
}

#[tokio::test]
async fn sql_reads_and_is_cut_at_the_row_limit() {
    let House { store, _dir, .. } = house();
    let sql = ReadOnlySql::parse("SELECT label, count(*) AS n FROM object WHERE valid_to_unix_ns IS NULL GROUP BY label ORDER BY label").unwrap();
    let rows = store.query_sql(sql, NonZeroUsize::new(3).unwrap()).await.unwrap();
    assert_eq!(rows.columns, vec!["label", "n"]);
    assert_eq!(rows.rows.len(), 3);
    assert!(rows.truncated);
    assert_eq!(rows.rows[0], vec![SqlValue::Text("armchair".to_owned()), SqlValue::Integer(1)]);
    let views = ReadOnlySql::parse("SELECT jpeg FROM object_view LIMIT 1").unwrap();
    let blob = store.query_sql(views, NonZeroUsize::new(5).unwrap()).await.unwrap();
    assert!(matches!(&blob.rows[0][0], SqlValue::Text(t) if t.ends_with("byte blob>")));
}

#[tokio::test]
async fn sql_that_writes_attaches_or_sets_pragmas_is_refused_by_the_connection() {
    let House { store, _dir, .. } = house();
    let limit = NonZeroUsize::new(10).unwrap();
    for refused in [
        "DELETE FROM object",
        "INSERT INTO room_name VALUES (9, 'attic', 0)",
        "UPDATE goal SET state = 'failed'",
        "DROP TABLE event",
        "ATTACH DATABASE 'other.sqlite' AS other",
        "PRAGMA writable_schema = 1",
        "PRAGMA query_only = 0",
        "SELECT 1; DELETE FROM object",
        "CREATE TEMP TABLE scratch (x)",
    ] {
        let error = store.query_text(refused.to_owned(), limit).await.unwrap_err();
        assert!(matches!(&error, WorldError::BadQuery(_)), "{refused}: {error}");
        let reason = error.to_string();
        assert!(
            ["not authorized", "only one statement", "readonly", "read-only"].iter().any(|r| reason.contains(r)),
            "{refused}: {reason}"
        );
    }
    let objects = store.query_text("SELECT count(*) FROM object".to_owned(), limit).await.unwrap();
    assert_eq!(objects.rows[0][0], SqlValue::Integer(9));
}

#[tokio::test]
async fn sql_that_runs_too_long_is_interrupted() {
    let House { store, _dir, .. } = house();
    let endless = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT max(i) FROM n";
    let error = store.query_text(endless.to_owned(), NonZeroUsize::new(1).unwrap()).await.unwrap_err();
    assert_eq!(error, WorldError::QueryTooSlow { limit_ms: 200 });
    let after = store.query_text("SELECT 1".to_owned(), NonZeroUsize::new(1).unwrap()).await.unwrap();
    assert_eq!(after.rows, vec![vec![SqlValue::Integer(1)]]);
}

#[tokio::test]
async fn rooms_are_renamed_with_an_event_and_bad_names_are_refused() {
    let House { store, writer, _dir, .. } = house();
    writer.name_room(RoomId(3), room("lounge")).unwrap();
    assert_eq!(store.describe_object(ObjectId(12)).await.unwrap().summary.room, room("lounge"));
    let refusals = [
        (RoomId(1), "lounge", Rejection::NameTaken { name: room("lounge"), by: RoomId(3) }),
        (RoomId(1), "room7", Rejection::ReservedName(room("room7"))),
        (RoomId(1), "unmapped", Rejection::ReservedName(room("unmapped"))),
        (RoomId(1), " ", Rejection::EmptyName),
        (RoomId(9), "attic", Rejection::UnknownRoom(RoomId(9))),
    ];
    for (id, name, rejection) in refusals {
        assert_eq!(writer.name_room(id, room(name)), Err(IngestError::Rejected(rejection)));
    }
    let renamed = store
        .query_text("SELECT room_name, previous_room_name FROM event WHERE kind = 'room_renamed' AND room_id = 3 ORDER BY event_id".to_owned(), NonZeroUsize::new(5).unwrap())
        .await
        .unwrap();
    assert_eq!(renamed.rows.last().unwrap(), &vec![SqlValue::Text("lounge".to_owned()), SqlValue::Text("living_room".to_owned())]);
}

#[tokio::test]
async fn updates_that_break_the_object_contract_are_rejected_without_a_trace() {
    let House { store, writer, _dir, .. } = house();
    let table = FURNITURE.iter().find(|f| f.id == 42).unwrap();
    let ghost = Furniture { id: 77, label: "ghost", at: (1.0, 1.0), size: (0.2, 0.2, 0.2), resting_on_m: 0.0, caption: "" };
    let before = writer.stats().rejected;
    writer.apply_object_update(update(&ghost, ObjectChange::Moved, ghost.at, NOW)).unwrap();
    writer.apply_object_update(update(table, ObjectChange::Appeared, table.at, NOW)).unwrap();
    let mut not_a_jpeg = update(table, ObjectChange::Updated, table.at, NOW);
    not_a_jpeg.crop = Some(JpegImage(b"GIF89a".to_vec()));
    writer.apply_object_update(not_a_jpeg).unwrap();
    writer.flush().unwrap();
    assert_eq!(writer.stats().rejected - before, 3);
    assert_eq!(store.describe_object(ObjectId(77)).await.unwrap_err(), WorldError::UnknownObject(ObjectId(77)));
    assert_eq!(store.last_seen(EntityRef::Object(ObjectId(42))).await.unwrap().at, ago(40));
}

#[tokio::test]
async fn disappearing_and_merging_close_the_object_and_its_relations() {
    let House { store, writer, _dir, .. } = house();
    let mug = FURNITURE.iter().find(|f| f.id == 51).unwrap();
    let chair = FURNITURE.iter().find(|f| f.id == 43).unwrap();
    writer.apply_object_update(update(mug, ObjectChange::Disappeared, mug.at, ago(10))).unwrap();
    writer.apply_object_update(update(chair, ObjectChange::MergedInto(ObjectId(44)), chair.at, ago(5))).unwrap();
    writer.flush().unwrap();
    assert_eq!(store.check_target(GoalTarget::Object(ObjectId(51))).await, Err(WorldError::UnknownObject(ObjectId(51))));
    let gone = store.changes_since(ago(60), None).await.unwrap();
    assert_eq!(gone.iter().map(|c| (c.object, c.kind.clone())).collect::<Vec<_>>(), vec![(ObjectId(51), ChangeKind::Disappeared)]);
    assert_eq!(gone[0].earliest, ago(300));
    let table = store.describe_object(ObjectId(42)).await.unwrap();
    assert!(table.relations.iter().all(|r| ![ObjectId(51), ObjectId(43)].contains(&r.other)), "{:?}", table.relations);
    let still_described = store.describe_object(ObjectId(51)).await.unwrap();
    assert_eq!(still_described.history[0].until, Some(ago(10)));
}

#[tokio::test]
async fn goals_and_their_status_are_recorded_with_events() {
    let House { store, writer, _dir, .. } = house();
    let goal = NavGoal {
        goal_id: GoalId(17),
        target: GoalTarget::Object(ObjectId(12)),
        standoff_m: NonNegative::new("standoff_m", 0.5).unwrap(),
        max_speed_m_s: NonNegative::new("max_speed_m_s", 0.3).unwrap(),
        deadline: UnixNanos(NOW.0 + 60_000_000_000),
        map_version: VERSION,
    };
    let issued = IssuedGoal { goal, agent: AgentName("spatial-0".to_owned()), model: ModelName("claude-test".to_owned()), issued: NOW };
    writer.record_goal(issued.clone()).unwrap();
    writer.record_goal(issued).unwrap();
    writer.record_goal_status(GoalStatus { goal_id: GoalId(17), state: GoalState::Blocked(GoalReason::NoPath), updated: NOW }).unwrap();
    writer.record_goal_status(GoalStatus { goal_id: GoalId(18), state: GoalState::Active, updated: NOW }).unwrap();
    writer.flush().unwrap();
    assert_eq!(writer.stats().rejected, 2, "the repeated goal and the status of an unknown goal");
    let limit = NonZeroUsize::new(5).unwrap();
    let goal = store.query_text("SELECT agent, model, kind, target_id, state, reason FROM goal".to_owned(), limit).await.unwrap();
    assert_eq!(
        goal.rows,
        vec![vec![
            SqlValue::Text("spatial-0".to_owned()),
            SqlValue::Text("claude-test".to_owned()),
            SqlValue::Text("object".to_owned()),
            SqlValue::Integer(12),
            SqlValue::Text("blocked".to_owned()),
            SqlValue::Text("no_path".to_owned()),
        ]]
    );
    let events = store.query_text("SELECT kind FROM event WHERE goal_id = 17 ORDER BY event_id".to_owned(), limit).await.unwrap();
    assert_eq!(events.rows, vec![vec![SqlValue::Text("goal_sent".to_owned())], vec![SqlValue::Text("goal_status".to_owned())]]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_writer_batches_while_readers_keep_answering() {
    let House { store, writer, _dir, .. } = house();
    let before = writer.stats();
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let store = store.clone();
            tokio::spawn(async move {
                let mut answers = 0;
                for _ in 0..100 {
                    store.find(FindQuery { text: Some("chair".to_owned()), ..FindQuery::default() }).await.unwrap();
                    store.snapshot().await.unwrap();
                    answers += 1;
                }
                answers
            })
        })
        .collect();
    let poses = 3000;
    for i in 0..poses {
        writer
            .record_robot_pose(RobotPoseSample {
                captured: UnixNanos(NOW.0 + i * 1_000_000),
                map_version: VERSION,
                pose: MapPose { x_m: 3.2, y_m: 2.5, yaw_rad: 0.0 },
            })
            .unwrap();
    }
    writer.flush().unwrap();
    for reader in readers {
        assert_eq!(reader.await.unwrap(), 100);
    }
    let after = writer.stats();
    let commands = after.commands - before.commands;
    let batches = after.batches - before.batches;
    assert_eq!(commands, poses as u64 + 1);
    assert!(batches * 10 < commands, "{batches} batches for {commands} commands");
    let count = store.query_text("SELECT count(*) FROM robot_pose".to_owned(), NonZeroUsize::new(1).unwrap()).await.unwrap();
    assert_eq!(count.rows[0][0], SqlValue::Integer(poses + 1));
}

#[tokio::test]
async fn reopening_keeps_the_map_names_and_history() {
    let dir = tempfile::tempdir().unwrap();
    {
        let (config, _) = config(dir.path());
        let (store, writer) = WorldStore::open(config).unwrap();
        seed(&writer);
        drop(store);
        writer.close().unwrap();
        assert!(matches!(writer.flush(), Err(IngestError::WriterStopped(_))));
    }
    let (config, _) = config(dir.path());
    let (store, _writer) = WorldStore::open(config).unwrap();
    let route = store.route(Landmark::Robot, Landmark::Object(ObjectId(12))).await.unwrap();
    assert_eq!(route.rooms, vec![room("kitchen"), room("hallway"), room("living_room")]);
    let clearance = store.clearance_between(Landmark::Place(PlaceId(3)), Landmark::Place(PlaceId(7))).await.unwrap();
    assert!(!clearance.passable);
    assert_eq!(store.stamp().await.unwrap().map_version, VERSION);
}

#[tokio::test]
async fn a_backup_is_a_whole_readable_copy() {
    let House { store, _dir: dir, .. } = house();
    let target = dir.path().join("backup.sqlite");
    store.back_up_to(&target).unwrap();
    let copy = rusqlite::Connection::open(&target).unwrap();
    let objects: i64 = copy.query_row("SELECT count(*) FROM object", [], |r| r.get(0)).unwrap();
    assert_eq!(objects, 9);
}

#[tokio::test]
async fn backups_run_on_their_interval() {
    let dir = tempfile::tempdir().unwrap();
    let (mut config, _) = config(dir.path());
    let target = dir.path().join("periodic.sqlite");
    config.backup = Some(world_store::BackupConfig { path: target.clone(), every: Duration::from_millis(50) });
    let (_store, writer) = WorldStore::open(config).unwrap();
    seed(&writer);
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while !target.exists() && std::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(target.exists());
}

#[test]
fn a_corrupt_database_fails_its_integrity_check_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("world.sqlite");
    {
        let (config, _) = config(dir.path());
        let (store, writer) = WorldStore::open(config).unwrap();
        seed(&writer);
        drop(store);
        writer.close().unwrap();
    }
    let mut bytes = std::fs::read(&path).unwrap();
    assert!(bytes.len() > 8 * 4096, "{} bytes", bytes.len());
    for byte in &mut bytes[3 * 4096..6 * 4096] {
        *byte = 0x55;
    }
    std::fs::write(&path, bytes).unwrap();
    let (config, _) = config(dir.path());
    match WorldStore::open(config) {
        Err(OpenError::Integrity { problems, .. }) => assert!(!problems.is_empty()),
        Err(OpenError::Sql(error)) => assert!(error.to_string().contains("malformed"), "{error}"),
        Err(other) => panic!("expected an integrity failure, got {other}"),
        Ok(_) => panic!("a corrupt database opened"),
    }
}
