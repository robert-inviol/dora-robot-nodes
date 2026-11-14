//! The space_map messages of lib/dora-rig-messages/dora_rig_messages/space_map.py, as single-row Arrow structs.
//! Field names, types and nullability must match that module exactly; the tests read its golden fixtures.
//!
//! Coordinates are the map frame: z up, gravity aligned, metres.

use std::fmt;
use std::sync::Arc;

use dora_node_api::arrow::array::{
    Array, ArrayRef, AsArray, Float32Array, Int64Array, ListArray, StringArray, StructArray, UInt16Array, UInt32Array,
    UInt64Array,
};
use dora_node_api::arrow::buffer::OffsetBuffer;
use dora_node_api::arrow::datatypes::{DataType, Field, Fields, Float32Type, Int64Type, UInt16Type, UInt32Type, UInt64Type};
use glam::Vec2;

use crate::navigation::{MapVersion, ObjectId, PlaceId, UnixNanos};
pub use crate::navigation::RoomId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DoorwayId(pub u64);

impl fmt::Display for DoorwayId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "d{}", self.0)
    }
}

/// The height band above the floor that a free-space cell describes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightBand {
    pub min_m: f32,
    pub max_m: f32,
}

/// What the free-space grid says about one cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clearance {
    Occupied,
    Unknown,
    /// Free, this far from the nearest occupied cell.
    Free { mm: u16 },
}

impl Clearance {
    pub const OCCUPIED_MM: u16 = 0;
    pub const UNKNOWN_MM: u16 = u16::MAX;
    /// The largest distance a free cell can carry; anything further is reported as this.
    pub const MAX_FREE_MM: u16 = u16::MAX - 1;

    pub fn from_wire(mm: u16) -> Clearance {
        match mm {
            Clearance::OCCUPIED_MM => Clearance::Occupied,
            Clearance::UNKNOWN_MM => Clearance::Unknown,
            mm => Clearance::Free { mm },
        }
    }

    pub fn to_wire(self) -> u16 {
        match self {
            Clearance::Occupied => Clearance::OCCUPIED_MM,
            Clearance::Unknown => Clearance::UNKNOWN_MM,
            Clearance::Free { mm } => mm.clamp(1, Clearance::MAX_FREE_MM),
        }
    }
}

/// Which floor cells the robot can occupy, sliced through a height band above the floor.
#[derive(Clone, Debug, PartialEq)]
pub struct FreeSpace {
    pub map_version: MapVersion,
    pub built: UnixNanos,
    /// The outer corner of cell (row 0, column 0); columns run along x, rows along y.
    pub origin: Vec2,
    pub resolution_m: f32,
    pub width: u32,
    pub height: u32,
    pub band: HeightBand,
    /// Row major from the origin, in the wire encoding of [`Clearance`].
    pub clearance_mm: Vec<u16>,
}

impl FreeSpace {
    pub fn cell(&self, column: u32, row: u32) -> Clearance {
        Clearance::from_wire(self.clearance_mm[(row * self.width + column) as usize])
    }

    /// The cell holding a map point, if the grid covers it.
    pub fn cell_at(&self, point: Vec2) -> Option<(u32, u32)> {
        let cell = ((point - self.origin) / self.resolution_m).floor();
        (cell.x >= 0.0 && cell.y >= 0.0 && cell.x < self.width as f32 && cell.y < self.height as f32)
            .then_some((cell.x as u32, cell.y as u32))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TopologyRoom {
    pub room_id: RoomId,
    pub polygon: Vec<Vec2>,
    /// Heights in the map frame.
    pub floor_m: f32,
    /// None where no ceiling was seen over the room.
    pub ceiling_m: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Doorway {
    pub doorway_id: DoorwayId,
    pub room_a: RoomId,
    pub room_b: RoomId,
    pub position: Vec2,
    pub width_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Place {
    pub place_id: PlaceId,
    pub room_id: RoomId,
    pub position: Vec2,
    /// Distance to the nearest cell that is not known free.
    pub clearance_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaceEdge {
    pub place_a: PlaceId,
    pub place_b: PlaceId,
    pub length_m: f32,
    pub min_clearance_m: f32,
}

/// Rooms cut at their doorways, and a graph of places along the middle of the free space.
#[derive(Clone, Debug, PartialEq)]
pub struct Topology {
    pub map_version: MapVersion,
    pub built: UnixNanos,
    pub rooms: Vec<TopologyRoom>,
    pub doorways: Vec<Doorway>,
    pub places: Vec<Place>,
    pub edges: Vec<PlaceEdge>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KeepOutZone {
    pub object_id: ObjectId,
    pub label: String,
    pub polygon: Vec<Vec2>,
    /// How far beyond the polygon the robot keeps out.
    pub margin_m: f32,
}

/// Every zone the robot must not enter; each message replaces the last.
#[derive(Clone, Debug, PartialEq)]
pub struct KeepOut {
    pub map_version: MapVersion,
    pub zones: Vec<KeepOutZone>,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SpaceMapError {
    #[error("expected the {message} wire type {expected:?}, received {received:?}")]
    WrongType { message: &'static str, expected: DataType, received: DataType },
    #[error("expected one {message} row, received {rows}")]
    NotOneRow { message: &'static str, rows: usize },
    #[error("{field} is null")]
    Null { field: &'static str },
    #[error("free space has {cells} cells for {width}x{height}")]
    CellsDoNotFillGrid { cells: usize, width: u32, height: u32 },
    #[error("a polygon has {0} points, expected at least 3")]
    DegeneratePolygon(usize),
}

fn item(data_type: DataType) -> Arc<Field> {
    Arc::new(Field::new("item", data_type, false))
}

fn items(data_type: DataType) -> DataType {
    DataType::List(item(data_type))
}

fn field(name: &str, data_type: DataType) -> Field {
    Field::new(name, data_type, true)
}

fn point_fields() -> Fields {
    Fields::from(vec![field("x_m", DataType::Float32), field("y_m", DataType::Float32)])
}

pub fn free_space_fields() -> Fields {
    Fields::from(vec![
        field("map_version", DataType::UInt64),
        field("built_unix_ns", DataType::Int64),
        field("origin_x_m", DataType::Float32),
        field("origin_y_m", DataType::Float32),
        field("resolution_m", DataType::Float32),
        field("width", DataType::UInt32),
        field("height", DataType::UInt32),
        field("band_min_m", DataType::Float32),
        field("band_max_m", DataType::Float32),
        field("clearance_mm", items(DataType::UInt16)),
    ])
}

fn room_fields() -> Fields {
    Fields::from(vec![
        field("room_id", DataType::UInt64),
        field("polygon", items(DataType::Struct(point_fields()))),
        field("floor_m", DataType::Float32),
        field("ceiling_m", DataType::Float32),
    ])
}

fn doorway_fields() -> Fields {
    Fields::from(vec![
        field("doorway_id", DataType::UInt64),
        field("room_a", DataType::UInt64),
        field("room_b", DataType::UInt64),
        field("x_m", DataType::Float32),
        field("y_m", DataType::Float32),
        field("width_m", DataType::Float32),
    ])
}

fn place_fields() -> Fields {
    Fields::from(vec![
        field("place_id", DataType::UInt64),
        field("room_id", DataType::UInt64),
        field("x_m", DataType::Float32),
        field("y_m", DataType::Float32),
        field("clearance_m", DataType::Float32),
    ])
}

fn edge_fields() -> Fields {
    Fields::from(vec![
        field("place_a", DataType::UInt64),
        field("place_b", DataType::UInt64),
        field("length_m", DataType::Float32),
        field("min_clearance_m", DataType::Float32),
    ])
}

pub fn topology_fields() -> Fields {
    Fields::from(vec![
        field("map_version", DataType::UInt64),
        field("built_unix_ns", DataType::Int64),
        field("rooms", items(DataType::Struct(room_fields()))),
        field("doorways", items(DataType::Struct(doorway_fields()))),
        field("places", items(DataType::Struct(place_fields()))),
        field("edges", items(DataType::Struct(edge_fields()))),
    ])
}

fn zone_fields() -> Fields {
    Fields::from(vec![
        field("object_id", DataType::UInt64),
        field("label", DataType::Utf8),
        field("polygon", items(DataType::Struct(point_fields()))),
        field("margin_m", DataType::Float32),
    ])
}

pub fn keep_out_fields() -> Fields {
    Fields::from(vec![field("map_version", DataType::UInt64), field("zones", items(DataType::Struct(zone_fields())))])
}

// Encoding.

fn list_of(values: ArrayRef, lengths: impl IntoIterator<Item = usize>) -> ArrayRef {
    let item = item(values.data_type().clone());
    Arc::new(ListArray::new(item, OffsetBuffer::from_lengths(lengths), values, None))
}

fn one_list(values: ArrayRef) -> ArrayRef {
    let length = values.len();
    list_of(values, [length])
}

fn floats(values: impl IntoIterator<Item = f32>) -> ArrayRef {
    Arc::new(values.into_iter().collect::<Float32Array>())
}

fn ids(values: impl IntoIterator<Item = u64>) -> ArrayRef {
    Arc::new(values.into_iter().collect::<UInt64Array>())
}

fn structs(fields: Fields, columns: Vec<ArrayRef>) -> ArrayRef {
    Arc::new(StructArray::new(fields, columns, None))
}

/// One polygon per row of the enclosing struct.
fn polygons<'a>(polygons: impl Iterator<Item = &'a [Vec2]> + Clone) -> ArrayRef {
    let points = polygons.clone().flatten();
    let point_array = structs(point_fields(), vec![floats(points.clone().map(|p| p.x)), floats(points.map(|p| p.y))]);
    list_of(point_array, polygons.map(<[Vec2]>::len))
}

pub fn encode_free_space(free_space: &FreeSpace) -> StructArray {
    let columns: Vec<ArrayRef> = vec![
        ids([free_space.map_version.0]),
        Arc::new(Int64Array::from(vec![free_space.built.0])),
        floats([free_space.origin.x]),
        floats([free_space.origin.y]),
        floats([free_space.resolution_m]),
        Arc::new(UInt32Array::from(vec![free_space.width])),
        Arc::new(UInt32Array::from(vec![free_space.height])),
        floats([free_space.band.min_m]),
        floats([free_space.band.max_m]),
        one_list(Arc::new(UInt16Array::from(free_space.clearance_mm.clone()))),
    ];
    StructArray::new(free_space_fields(), columns, None)
}

pub fn encode_topology(topology: &Topology) -> StructArray {
    let rooms = &topology.rooms;
    let room_columns: Vec<ArrayRef> = vec![
        ids(rooms.iter().map(|r| r.room_id.0)),
        polygons(rooms.iter().map(|r| &r.polygon[..])),
        floats(rooms.iter().map(|r| r.floor_m)),
        Arc::new(rooms.iter().map(|r| r.ceiling_m).collect::<Float32Array>()),
    ];
    let doorways = &topology.doorways;
    let doorway_columns: Vec<ArrayRef> = vec![
        ids(doorways.iter().map(|d| d.doorway_id.0)),
        ids(doorways.iter().map(|d| d.room_a.0)),
        ids(doorways.iter().map(|d| d.room_b.0)),
        floats(doorways.iter().map(|d| d.position.x)),
        floats(doorways.iter().map(|d| d.position.y)),
        floats(doorways.iter().map(|d| d.width_m)),
    ];
    let places = &topology.places;
    let place_columns: Vec<ArrayRef> = vec![
        ids(places.iter().map(|p| p.place_id.0)),
        ids(places.iter().map(|p| p.room_id.0)),
        floats(places.iter().map(|p| p.position.x)),
        floats(places.iter().map(|p| p.position.y)),
        floats(places.iter().map(|p| p.clearance_m)),
    ];
    let edges = &topology.edges;
    let edge_columns: Vec<ArrayRef> = vec![
        ids(edges.iter().map(|e| e.place_a.0)),
        ids(edges.iter().map(|e| e.place_b.0)),
        floats(edges.iter().map(|e| e.length_m)),
        floats(edges.iter().map(|e| e.min_clearance_m)),
    ];
    let columns: Vec<ArrayRef> = vec![
        ids([topology.map_version.0]),
        Arc::new(Int64Array::from(vec![topology.built.0])),
        one_list(structs(room_fields(), room_columns)),
        one_list(structs(doorway_fields(), doorway_columns)),
        one_list(structs(place_fields(), place_columns)),
        one_list(structs(edge_fields(), edge_columns)),
    ];
    StructArray::new(topology_fields(), columns, None)
}

pub fn encode_keep_out(keep_out: &KeepOut) -> StructArray {
    let zones = &keep_out.zones;
    let zone_columns: Vec<ArrayRef> = vec![
        ids(zones.iter().map(|z| z.object_id.0)),
        Arc::new(StringArray::from_iter_values(zones.iter().map(|z| &z.label))),
        polygons(zones.iter().map(|z| &z.polygon[..])),
        floats(zones.iter().map(|z| z.margin_m)),
    ];
    let columns: Vec<ArrayRef> = vec![ids([keep_out.map_version.0]), one_list(structs(zone_fields(), zone_columns))];
    StructArray::new(keep_out_fields(), columns, None)
}

// Decoding.

fn checked<'a>(array: &'a dyn Array, fields: Fields, message: &'static str) -> Result<&'a StructArray, SpaceMapError> {
    let expected = DataType::Struct(fields);
    if array.data_type() != &expected {
        return Err(SpaceMapError::WrongType { message, expected, received: array.data_type().clone() });
    }
    if array.len() != 1 {
        return Err(SpaceMapError::NotOneRow { message, rows: array.len() });
    }
    Ok(array.as_struct())
}

/// Typed access to the rows of a struct array whose type has been checked.
struct Rows<'a>(&'a StructArray);

impl<'a> Rows<'a> {
    fn column(&self, name: &str) -> &'a ArrayRef {
        self.0.column_by_name(name).expect("checked wire type")
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn u64(&self, name: &'static str, row: usize) -> Result<u64, SpaceMapError> {
        let values = self.column(name).as_primitive::<UInt64Type>();
        values.is_valid(row).then(|| values.value(row)).ok_or(SpaceMapError::Null { field: name })
    }

    fn i64(&self, name: &'static str, row: usize) -> Result<i64, SpaceMapError> {
        let values = self.column(name).as_primitive::<Int64Type>();
        values.is_valid(row).then(|| values.value(row)).ok_or(SpaceMapError::Null { field: name })
    }

    fn u32(&self, name: &'static str, row: usize) -> Result<u32, SpaceMapError> {
        let values = self.column(name).as_primitive::<UInt32Type>();
        values.is_valid(row).then(|| values.value(row)).ok_or(SpaceMapError::Null { field: name })
    }

    fn optional_f32(&self, name: &str, row: usize) -> Option<f32> {
        let values = self.column(name).as_primitive::<Float32Type>();
        values.is_valid(row).then(|| values.value(row))
    }

    fn f32(&self, name: &'static str, row: usize) -> Result<f32, SpaceMapError> {
        self.optional_f32(name, row).ok_or(SpaceMapError::Null { field: name })
    }

    fn str(&self, name: &'static str, row: usize) -> Result<&'a str, SpaceMapError> {
        let values = self.column(name).as_string::<i32>();
        values.is_valid(row).then(|| values.value(row)).ok_or(SpaceMapError::Null { field: name })
    }

    /// The struct rows of the list in `name` at `row`.
    fn list(&self, name: &'static str, row: usize) -> Result<StructArray, SpaceMapError> {
        let list = self.column(name).as_list::<i32>();
        if list.is_null(row) {
            return Err(SpaceMapError::Null { field: name });
        }
        Ok(list.value(row).as_struct().clone())
    }

    fn point(&self, row: usize) -> Result<Vec2, SpaceMapError> {
        Ok(Vec2::new(self.f32("x_m", row)?, self.f32("y_m", row)?))
    }

    fn polygon(&self, row: usize) -> Result<Vec<Vec2>, SpaceMapError> {
        let points = self.list("polygon", row)?;
        let points = Rows(&points);
        let polygon = (0..points.len()).map(|i| points.point(i)).collect::<Result<Vec<_>, _>>()?;
        if polygon.len() < 3 {
            return Err(SpaceMapError::DegeneratePolygon(polygon.len()));
        }
        Ok(polygon)
    }
}

fn each<T>(rows: &StructArray, decode: impl Fn(&Rows, usize) -> Result<T, SpaceMapError>) -> Result<Vec<T>, SpaceMapError> {
    let rows = Rows(rows);
    (0..rows.len()).map(|i| decode(&rows, i)).collect()
}

pub fn decode_free_space(array: &dyn Array) -> Result<FreeSpace, SpaceMapError> {
    let row = Rows(checked(array, free_space_fields(), "free space")?);
    let (width, height) = (row.u32("width", 0)?, row.u32("height", 0)?);
    let list = row.column("clearance_mm").as_list::<i32>();
    if list.is_null(0) {
        return Err(SpaceMapError::Null { field: "clearance_mm" });
    }
    let clearance_mm = list.value(0).as_primitive::<UInt16Type>().values().to_vec();
    if clearance_mm.len() != width as usize * height as usize {
        return Err(SpaceMapError::CellsDoNotFillGrid { cells: clearance_mm.len(), width, height });
    }
    Ok(FreeSpace {
        map_version: MapVersion(row.u64("map_version", 0)?),
        built: UnixNanos(row.i64("built_unix_ns", 0)?),
        origin: Vec2::new(row.f32("origin_x_m", 0)?, row.f32("origin_y_m", 0)?),
        resolution_m: row.f32("resolution_m", 0)?,
        width,
        height,
        band: HeightBand { min_m: row.f32("band_min_m", 0)?, max_m: row.f32("band_max_m", 0)? },
        clearance_mm,
    })
}

pub fn decode_topology(array: &dyn Array) -> Result<Topology, SpaceMapError> {
    let row = Rows(checked(array, topology_fields(), "topology")?);
    Ok(Topology {
        map_version: MapVersion(row.u64("map_version", 0)?),
        built: UnixNanos(row.i64("built_unix_ns", 0)?),
        rooms: each(&row.list("rooms", 0)?, |r, i| {
            Ok(TopologyRoom {
                room_id: RoomId(r.u64("room_id", i)?),
                polygon: r.polygon(i)?,
                floor_m: r.f32("floor_m", i)?,
                ceiling_m: r.optional_f32("ceiling_m", i),
            })
        })?,
        doorways: each(&row.list("doorways", 0)?, |r, i| {
            Ok(Doorway {
                doorway_id: DoorwayId(r.u64("doorway_id", i)?),
                room_a: RoomId(r.u64("room_a", i)?),
                room_b: RoomId(r.u64("room_b", i)?),
                position: r.point(i)?,
                width_m: r.f32("width_m", i)?,
            })
        })?,
        places: each(&row.list("places", 0)?, |r, i| {
            Ok(Place {
                place_id: PlaceId(r.u64("place_id", i)?),
                room_id: RoomId(r.u64("room_id", i)?),
                position: r.point(i)?,
                clearance_m: r.f32("clearance_m", i)?,
            })
        })?,
        edges: each(&row.list("edges", 0)?, |r, i| {
            Ok(PlaceEdge {
                place_a: PlaceId(r.u64("place_a", i)?),
                place_b: PlaceId(r.u64("place_b", i)?),
                length_m: r.f32("length_m", i)?,
                min_clearance_m: r.f32("min_clearance_m", i)?,
            })
        })?,
    })
}

pub fn decode_keep_out(array: &dyn Array) -> Result<KeepOut, SpaceMapError> {
    let row = Rows(checked(array, keep_out_fields(), "keep out")?);
    Ok(KeepOut {
        map_version: MapVersion(row.u64("map_version", 0)?),
        zones: each(&row.list("zones", 0)?, |r, i| {
            Ok(KeepOutZone {
                object_id: ObjectId(r.u64("object_id", i)?),
                label: r.str("label", i)?.to_owned(),
                polygon: r.polygon(i)?,
                margin_m: r.f32("margin_m", i)?,
            })
        })?,
    })
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use dora_node_api::arrow::ipc::reader::FileReader;

    use super::*;

    fn fixture(name: &str) -> StructArray {
        let path = format!("{}/../../../lib/dora-rig-messages/dora_rig_messages/fixtures/{name}.arrow", env!("CARGO_MANIFEST_DIR"));
        let mut reader = FileReader::try_new(File::open(&path).unwrap_or_else(|e| panic!("{path}: {e}")), None).unwrap();
        StructArray::from(reader.next().unwrap().unwrap())
    }

    fn square(x: f32, y: f32, side: f32) -> Vec<Vec2> {
        vec![Vec2::new(x, y), Vec2::new(x + side, y), Vec2::new(x + side, y + side), Vec2::new(x, y + side)]
    }

    #[test]
    fn the_free_space_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("free_space");
        let free_space = decode_free_space(&golden).unwrap();
        assert_eq!(
            free_space,
            FreeSpace {
                map_version: MapVersion(412),
                built: UnixNanos(1_762_560_002_000_000_000),
                origin: Vec2::new(-1.25, 0.5),
                resolution_m: 0.0625,
                width: 3,
                height: 2,
                band: HeightBand { min_m: 0.0625, max_m: 0.625 },
                clearance_mm: vec![0, 50, 100, 65535, 70, 0],
            }
        );
        assert_eq!(free_space.cell(0, 1), Clearance::Unknown);
        assert_eq!(free_space.cell(2, 0), Clearance::Free { mm: 100 });
        assert_eq!(free_space.cell(2, 1), Clearance::Occupied);
        assert_eq!(free_space.cell_at(Vec2::new(-1.0, 0.6)), None);
        assert_eq!(free_space.cell_at(Vec2::new(-1.1, 0.6)), Some((2, 1)));
        assert_eq!(encode_free_space(&free_space), golden);
    }

    #[test]
    fn the_topology_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("topology");
        let topology = decode_topology(&golden).unwrap();
        assert_eq!(topology.map_version, MapVersion(412));
        assert_eq!(
            topology.rooms,
            vec![
                TopologyRoom { room_id: RoomId(3), polygon: square(0.0, 0.0, 4.0), floor_m: -0.875, ceiling_m: Some(1.5) },
                TopologyRoom { room_id: RoomId(5), polygon: square(4.0, 0.0, 3.0), floor_m: -0.875, ceiling_m: None },
            ]
        );
        assert_eq!(
            topology.doorways,
            vec![Doorway {
                doorway_id: DoorwayId(1),
                room_a: RoomId(3),
                room_b: RoomId(5),
                position: Vec2::new(4.0, 1.5),
                width_m: 0.75
            }]
        );
        assert_eq!(topology.places.len(), 3);
        assert_eq!(topology.places[1].clearance_m, 0.375);
        assert_eq!(
            topology.edges[0],
            PlaceEdge { place_a: PlaceId(10), place_b: PlaceId(11), length_m: 2.0625, min_clearance_m: 0.375 }
        );
        assert_eq!(encode_topology(&topology), golden);
    }

    #[test]
    fn the_keep_out_fixtures_decode_and_encode_back_to_themselves() {
        let golden = fixture("keep_out");
        let keep_out = decode_keep_out(&golden).unwrap();
        assert_eq!(
            keep_out.zones,
            vec![KeepOutZone {
                object_id: ObjectId(42),
                label: "glass door".to_owned(),
                polygon: square(4.0, 1.0, 1.0),
                margin_m: 0.25
            }]
        );
        assert_eq!(encode_keep_out(&keep_out), golden);

        let empty = fixture("keep_out_empty");
        let keep_out = decode_keep_out(&empty).unwrap();
        assert_eq!(keep_out, KeepOut { map_version: MapVersion(413), zones: vec![] });
        assert_eq!(encode_keep_out(&keep_out), empty);
    }

    #[test]
    fn free_space_whose_cells_do_not_fill_the_grid_is_rejected() {
        let mut free_space = decode_free_space(&fixture("free_space")).unwrap();
        free_space.clearance_mm.pop();
        assert_eq!(
            decode_free_space(&encode_free_space(&free_space)),
            Err(SpaceMapError::CellsDoNotFillGrid { cells: 5, width: 3, height: 2 })
        );
    }

    #[test]
    fn a_zone_polygon_of_two_points_is_rejected() {
        let mut keep_out = decode_keep_out(&fixture("keep_out")).unwrap();
        keep_out.zones[0].polygon.truncate(2);
        assert_eq!(decode_keep_out(&encode_keep_out(&keep_out)), Err(SpaceMapError::DegeneratePolygon(2)));
    }

    #[test]
    fn a_topology_of_another_wire_type_is_rejected() {
        let error = decode_topology(&fixture("keep_out")).unwrap_err();
        assert!(matches!(error, SpaceMapError::WrongType { message: "topology", .. }), "{error}");
    }

    #[test]
    fn clearance_round_trips_through_its_wire_encoding() {
        for clearance in [Clearance::Occupied, Clearance::Unknown, Clearance::Free { mm: 1 }, Clearance::Free { mm: 65534 }] {
            assert_eq!(Clearance::from_wire(clearance.to_wire()), clearance);
        }
    }
}
