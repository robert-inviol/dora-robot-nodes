//! The space_map messages of lib/dora-rig-messages/dora_rig_messages/space_map.py, as single-row Arrow structs.
//! Field names, types and nullability must match that module exactly; the tests read its golden fixtures.
//!
//! Coordinates are the map frame: z up, gravity aligned, metres.

use std::fmt;
use std::sync::Arc;

use dora_node_api::arrow::array::{
    Array, ArrayRef, AsArray, Float32Array, Int32Array, Int64Array, ListArray, StringArray, StructArray, UInt16Array,
    UInt32Array, UInt64Array,
};
use dora_node_api::arrow::buffer::OffsetBuffer;
use dora_node_api::arrow::datatypes::{
    DataType, Field, Fields, Float32Type, Int32Type, Int64Type, UInt16Type, UInt32Type, UInt64Type,
};
use glam::{IVec3, Vec2, Vec3};

use crate::navigation::{MapVersion, ObjectId, PlaceId, UnixNanos};
pub use crate::navigation::RoomId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DoorwayId(pub u64);

/// The number of a mesh_blocks message within one run of the dense map: 0 first, one up per message.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MeshSequence(pub u64);

impl MeshSequence {
    pub fn next(self) -> MeshSequence {
        MeshSequence(self.0 + 1)
    }
}

impl fmt::Display for MeshSequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// What a mesh_blocks message holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MeshBlocksKind {
    /// The blocks whose surface changed since the last message, each with its whole new surface; a mesh block without
    /// triangles says its block has no surface now.
    Changes,
    /// Every block that has surface, each with its whole surface: it replaces every block the receiver holds.
    Snapshot,
}

impl MeshBlocksKind {
    pub fn wire_name(self) -> &'static str {
        match self {
            MeshBlocksKind::Changes => "changes",
            MeshBlocksKind::Snapshot => "snapshot",
        }
    }

    fn from_wire_name(name: &str) -> Result<MeshBlocksKind, SpaceMapError> {
        match name {
            "changes" => Ok(MeshBlocksKind::Changes),
            "snapshot" => Ok(MeshBlocksKind::Snapshot),
            other => Err(SpaceMapError::UnknownMeshBlocksKind(other.to_owned())),
        }
    }
}

impl fmt::Display for DoorwayId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "d{}", self.0)
    }
}

/// The height band above the floor that a occupancy grid cell describes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightBand {
    pub min_m: f32,
    pub max_m: f32,
}

/// What the occupancy grid says about one cell.
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
pub struct OccupancyGrid {
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

impl OccupancyGrid {
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

/// The dense map's surface inside one block; a mesh block without triangles says the block has no surface now.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshBlock {
    /// Block (x, y, z) spans `[x, x + 1) * block_voxels * voxel_m` along x, and likewise along y and z.
    pub block: IVec3,
    pub positions_m: Vec<Vec3>,
    /// Vertex indices, anticlockwise seen from the free side of the surface.
    pub triangles: Vec<[u32; 3]>,
}

/// The dense map's surface, block by block: the blocks whose surface changed since the last message, or, in a
/// snapshot, every block that has surface, replacing every block the receiver holds.
///
/// The messages of one run of the dense map share `run_started` and number from `sequence` 0 up by one, snapshots
/// included, so a receiver that sees a gap in the sequence knows it missed blocks.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshBlocks {
    pub map_version: MapVersion,
    pub built: UnixNanos,
    /// When the dense map that sent the message started: a later value is a new run whose blocks replace every
    /// block of the last.
    pub run_started: UnixNanos,
    pub sequence: MeshSequence,
    pub kind: MeshBlocksKind,
    pub voxel_m: f32,
    pub block_voxels: u32,
    pub blocks: Vec<MeshBlock>,
}

/// A receiver of mesh_blocks asking the dense map to send its whole surface as the next message, a snapshot,
/// because it missed a message or joined the run late.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshSnapshotRequest {
    pub requested: UnixNanos,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SpaceMapError {
    #[error("expected the {message} wire type {expected:?}, received {received:?}")]
    WrongType { message: &'static str, expected: DataType, received: DataType },
    #[error("expected one {message} row, received {rows}")]
    NotOneRow { message: &'static str, rows: usize },
    #[error("{field} is null")]
    Null { field: &'static str },
    #[error("occupancy grid has {cells} cells for {width}x{height}")]
    CellsDoNotFillGrid { cells: usize, width: u32, height: u32 },
    #[error("a polygon has {0} points, expected at least 3")]
    DegeneratePolygon(usize),
    #[error("{field} holds {count} values, not a multiple of 3")]
    NotTriples { field: &'static str, count: usize },
    #[error("block {block:?} has a triangle corner beyond its {vertices} vertices")]
    CornerBeyondVertices { block: [i32; 3], vertices: usize },
    #[error("mesh blocks kind {0:?} is neither changes nor snapshot")]
    UnknownMeshBlocksKind(String),
    #[error("a mesh snapshot holds only blocks with surface; block {block:?} has none")]
    SnapshotBlockWithoutSurface { block: [i32; 3] },
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

pub fn occupancy_grid_fields() -> Fields {
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

fn mesh_block_fields() -> Fields {
    Fields::from(vec![
        field("block_x", DataType::Int32),
        field("block_y", DataType::Int32),
        field("block_z", DataType::Int32),
        field("positions_m", items(DataType::Float32)),
        field("triangles", items(DataType::UInt32)),
    ])
}

pub fn mesh_blocks_fields() -> Fields {
    Fields::from(vec![
        field("map_version", DataType::UInt64),
        field("built_unix_ns", DataType::Int64),
        field("run_started_unix_ns", DataType::Int64),
        field("sequence", DataType::UInt64),
        field("kind", DataType::Utf8),
        field("voxel_m", DataType::Float32),
        field("block_voxels", DataType::UInt32),
        field("blocks", items(DataType::Struct(mesh_block_fields()))),
    ])
}

pub fn mesh_snapshot_request_fields() -> Fields {
    Fields::from(vec![field("requested_unix_ns", DataType::Int64)])
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

pub fn encode_occupancy_grid(occupancy_grid: &OccupancyGrid) -> StructArray {
    let columns: Vec<ArrayRef> = vec![
        ids([occupancy_grid.map_version.0]),
        Arc::new(Int64Array::from(vec![occupancy_grid.built.0])),
        floats([occupancy_grid.origin.x]),
        floats([occupancy_grid.origin.y]),
        floats([occupancy_grid.resolution_m]),
        Arc::new(UInt32Array::from(vec![occupancy_grid.width])),
        Arc::new(UInt32Array::from(vec![occupancy_grid.height])),
        floats([occupancy_grid.band.min_m]),
        floats([occupancy_grid.band.max_m]),
        one_list(Arc::new(UInt16Array::from(occupancy_grid.clearance_mm.clone()))),
    ];
    StructArray::new(occupancy_grid_fields(), columns, None)
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

pub fn encode_mesh_blocks(surface: &MeshBlocks) -> StructArray {
    let mesh_blocks = &surface.blocks;
    let block = |axis: fn(IVec3) -> i32| -> ArrayRef { Arc::new(mesh_blocks.iter().map(|p| axis(p.block)).collect::<Int32Array>()) };
    let positions = floats(mesh_blocks.iter().flat_map(|p| p.positions_m.iter().flat_map(|v| v.to_array())));
    let triangles: ArrayRef = Arc::new(mesh_blocks.iter().flat_map(|p| p.triangles.iter().flatten().copied()).collect::<UInt32Array>());
    let patch_columns: Vec<ArrayRef> = vec![
        block(|b| b.x),
        block(|b| b.y),
        block(|b| b.z),
        list_of(positions, mesh_blocks.iter().map(|p| 3 * p.positions_m.len())),
        list_of(triangles, mesh_blocks.iter().map(|p| 3 * p.triangles.len())),
    ];
    let columns: Vec<ArrayRef> = vec![
        ids([surface.map_version.0]),
        Arc::new(Int64Array::from(vec![surface.built.0])),
        Arc::new(Int64Array::from(vec![surface.run_started.0])),
        ids([surface.sequence.0]),
        Arc::new(StringArray::from(vec![surface.kind.wire_name()])),
        floats([surface.voxel_m]),
        Arc::new(UInt32Array::from(vec![surface.block_voxels])),
        one_list(structs(mesh_block_fields(), patch_columns)),
    ];
    StructArray::new(mesh_blocks_fields(), columns, None)
}

pub fn encode_mesh_snapshot_request(request: &MeshSnapshotRequest) -> StructArray {
    let columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![request.requested.0]))];
    StructArray::new(mesh_snapshot_request_fields(), columns, None)
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

    fn i32(&self, name: &'static str, row: usize) -> Result<i32, SpaceMapError> {
        let values = self.column(name).as_primitive::<Int32Type>();
        values.is_valid(row).then(|| values.value(row)).ok_or(SpaceMapError::Null { field: name })
    }

    /// The values of the list of numbers in `name` at `row`, in threes.
    fn triples<T: dora_node_api::arrow::datatypes::ArrowPrimitiveType>(
        &self,
        name: &'static str,
        row: usize,
    ) -> Result<Vec<[T::Native; 3]>, SpaceMapError> {
        let list = self.column(name).as_list::<i32>();
        if list.is_null(row) {
            return Err(SpaceMapError::Null { field: name });
        }
        let values = list.value(row);
        let values = values.as_primitive::<T>().values();
        if values.len() % 3 != 0 {
            return Err(SpaceMapError::NotTriples { field: name, count: values.len() });
        }
        Ok(values.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
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

pub fn decode_occupancy_grid(array: &dyn Array) -> Result<OccupancyGrid, SpaceMapError> {
    let row = Rows(checked(array, occupancy_grid_fields(), "occupancy grid")?);
    let (width, height) = (row.u32("width", 0)?, row.u32("height", 0)?);
    let list = row.column("clearance_mm").as_list::<i32>();
    if list.is_null(0) {
        return Err(SpaceMapError::Null { field: "clearance_mm" });
    }
    let clearance_mm = list.value(0).as_primitive::<UInt16Type>().values().to_vec();
    if clearance_mm.len() != width as usize * height as usize {
        return Err(SpaceMapError::CellsDoNotFillGrid { cells: clearance_mm.len(), width, height });
    }
    Ok(OccupancyGrid {
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

pub fn decode_mesh_blocks(array: &dyn Array) -> Result<MeshBlocks, SpaceMapError> {
    let row = Rows(checked(array, mesh_blocks_fields(), "mesh blocks")?);
    let kind = MeshBlocksKind::from_wire_name(row.str("kind", 0)?)?;
    let surface = MeshBlocks {
        map_version: MapVersion(row.u64("map_version", 0)?),
        built: UnixNanos(row.i64("built_unix_ns", 0)?),
        run_started: UnixNanos(row.i64("run_started_unix_ns", 0)?),
        sequence: MeshSequence(row.u64("sequence", 0)?),
        kind,
        voxel_m: row.f32("voxel_m", 0)?,
        block_voxels: row.u32("block_voxels", 0)?,
        blocks: each(&row.list("blocks", 0)?, |r, i| {
            let block = IVec3::new(r.i32("block_x", i)?, r.i32("block_y", i)?, r.i32("block_z", i)?);
            let positions_m: Vec<Vec3> = r.triples::<Float32Type>("positions_m", i)?.into_iter().map(Vec3::from_array).collect();
            let triangles = r.triples::<UInt32Type>("triangles", i)?;
            if triangles.iter().flatten().any(|&corner| corner as usize >= positions_m.len()) {
                return Err(SpaceMapError::CornerBeyondVertices { block: block.to_array(), vertices: positions_m.len() });
            }
            Ok(MeshBlock { block, positions_m, triangles })
        })?,
    };
    if kind == MeshBlocksKind::Snapshot {
        if let Some(empty) = surface.blocks.iter().find(|mesh_block| mesh_block.triangles.is_empty()) {
            return Err(SpaceMapError::SnapshotBlockWithoutSurface { block: empty.block.to_array() });
        }
    }
    Ok(surface)
}

pub fn decode_mesh_snapshot_request(array: &dyn Array) -> Result<MeshSnapshotRequest, SpaceMapError> {
    let row = Rows(checked(array, mesh_snapshot_request_fields(), "mesh snapshot request")?);
    Ok(MeshSnapshotRequest { requested: UnixNanos(row.i64("requested_unix_ns", 0)?) })
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
    fn the_occupancy_grid_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("occupancy_grid");
        let occupancy_grid = decode_occupancy_grid(&golden).unwrap();
        assert_eq!(
            occupancy_grid,
            OccupancyGrid {
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
        assert_eq!(occupancy_grid.cell(0, 1), Clearance::Unknown);
        assert_eq!(occupancy_grid.cell(2, 0), Clearance::Free { mm: 100 });
        assert_eq!(occupancy_grid.cell(2, 1), Clearance::Occupied);
        assert_eq!(occupancy_grid.cell_at(Vec2::new(-1.0, 0.6)), None);
        assert_eq!(occupancy_grid.cell_at(Vec2::new(-1.1, 0.6)), Some((2, 1)));
        assert_eq!(encode_occupancy_grid(&occupancy_grid), golden);
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
    fn occupancy_grid_whose_cells_do_not_fill_the_grid_is_rejected() {
        let mut occupancy_grid = decode_occupancy_grid(&fixture("occupancy_grid")).unwrap();
        occupancy_grid.clearance_mm.pop();
        assert_eq!(
            decode_occupancy_grid(&encode_occupancy_grid(&occupancy_grid)),
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
    fn the_mesh_blocks_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("mesh_blocks");
        let surface = decode_mesh_blocks(&golden).unwrap();
        assert_eq!((surface.map_version, surface.voxel_m, surface.block_voxels), (MapVersion(412), 0.0625, 8));
        assert_eq!((surface.run_started, surface.sequence), (UnixNanos(1_762_559_990_000_000_000), MeshSequence(17)));
        assert_eq!(surface.kind, MeshBlocksKind::Changes);
        let floor = &surface.blocks[0];
        assert_eq!(floor.block, IVec3::new(0, 1, -1));
        assert_eq!(floor.positions_m[2], Vec3::new(0.25, 0.75, -0.125));
        assert_eq!(floor.triangles, vec![[0, 1, 2], [0, 2, 3]]);
        assert_eq!(surface.blocks[1], MeshBlock { block: IVec3::new(2, 0, 0), positions_m: vec![], triangles: vec![] });
        assert_eq!(encode_mesh_blocks(&surface), golden);
    }

    #[test]
    fn the_mesh_blocks_snapshot_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("mesh_blocks_snapshot");
        let snapshot = decode_mesh_blocks(&golden).unwrap();
        assert_eq!((snapshot.kind, snapshot.sequence, snapshot.map_version), (MeshBlocksKind::Snapshot, MeshSequence(18), MapVersion(413)));
        assert_eq!(snapshot.blocks.iter().map(|p| p.block).collect::<Vec<_>>(), vec![IVec3::new(0, 1, -1)]);
        assert_eq!(encode_mesh_blocks(&snapshot), golden);
    }

    #[test]
    fn a_snapshot_with_a_block_without_surface_is_rejected() {
        let mut snapshot = decode_mesh_blocks(&fixture("mesh_blocks_snapshot")).unwrap();
        snapshot.blocks.push(MeshBlock { block: IVec3::new(2, 0, 0), positions_m: vec![], triangles: vec![] });
        assert_eq!(
            decode_mesh_blocks(&encode_mesh_blocks(&snapshot)),
            Err(SpaceMapError::SnapshotBlockWithoutSurface { block: [2, 0, 0] })
        );
    }

    #[test]
    fn the_mesh_snapshot_request_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("mesh_snapshot_request");
        let request = decode_mesh_snapshot_request(&golden).unwrap();
        assert_eq!(request, MeshSnapshotRequest { requested: UnixNanos(1_762_560_003_500_000_000) });
        assert_eq!(encode_mesh_snapshot_request(&request), golden);
    }

    #[test]
    fn a_triangle_beyond_its_patch_vertices_is_rejected() {
        let mut surface = decode_mesh_blocks(&fixture("mesh_blocks")).unwrap();
        surface.blocks[0].triangles.push([0, 1, 4]);
        assert_eq!(
            decode_mesh_blocks(&encode_mesh_blocks(&surface)),
            Err(SpaceMapError::CornerBeyondVertices { block: [0, 1, -1], vertices: 4 })
        );
    }

    #[test]
    fn clearance_round_trips_through_its_wire_encoding() {
        for clearance in [Clearance::Occupied, Clearance::Unknown, Clearance::Free { mm: 1 }, Clearance::Free { mm: 65534 }] {
            assert_eq!(Clearance::from_wire(clearance.to_wire()), clearance);
        }
    }
}
