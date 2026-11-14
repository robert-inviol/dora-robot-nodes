//! What the store is told, in domain types: object updates, the map's topology and free space, robot poses, goals
//! and room names. All positions are in the map frame, metres.

use std::fmt;

use perception_wire::navigation::{MapPose, MapVersion, NavGoal, ObjectId, PlaceId, UnixNanos};
use world_model::{Point, RoomName, Size};

pub use perception_wire::localization::FrameId;
pub use perception_wire::space_map::{DoorwayId, RoomId};

#[derive(Clone, Debug, PartialEq)]
pub struct RoomShape {
    pub id: RoomId,
    /// The floor outline, counter-clockwise or clockwise.
    pub polygon: Vec<Point>,
    pub floor_m: f32,
    pub ceiling_m: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoorwaySpan {
    pub id: DoorwayId,
    pub room_a: RoomId,
    pub room_b: RoomId,
    pub centre: Point,
    pub width_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaceNode {
    pub id: PlaceId,
    pub room: RoomId,
    pub position: Point,
    pub clearance_m: f32,
}

/// A drivable link between two places.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaceEdge {
    pub a: PlaceId,
    pub b: PlaceId,
    pub length_m: f32,
    pub min_clearance_m: f32,
}

/// The rooms, doorways and places graph of one map version.
#[derive(Clone, Debug, PartialEq)]
pub struct Topology {
    pub map_version: MapVersion,
    pub built: UnixNanos,
    pub rooms: Vec<RoomShape>,
    pub doorways: Vec<DoorwaySpan>,
    pub places: Vec<PlaceNode>,
    pub edges: Vec<PlaceEdge>,
}

/// One cell of the free-space grid: millimetres to the nearest occupied cell, with two reserved values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClearanceMm(pub u16);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cell {
    Occupied,
    Unknown,
    Free { clearance_m: f32 },
}

impl ClearanceMm {
    pub const OCCUPIED: Self = Self(0);
    pub const UNKNOWN: Self = Self(u16::MAX);

    pub fn cell(self) -> Cell {
        match self {
            Self::OCCUPIED => Cell::Occupied,
            Self::UNKNOWN => Cell::Unknown,
            Self(mm) => Cell::Free { clearance_m: f32::from(mm) / 1000.0 },
        }
    }
}

/// The height band above the floor that the free-space grid considers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightBand {
    pub min_m: f32,
    pub max_m: f32,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum InvalidFreeSpace {
    #[error("a {width} by {height} grid needs {expected} cells, received {received}")]
    WrongCellCount { width: u32, height: u32, expected: usize, received: usize },
    #[error("the resolution must be a positive number of metres, received {0}")]
    BadResolution(f32),
}

/// The 2D clearance grid of one map version, row-major from `origin` (the corner of cell 0), rows along y.
#[derive(Clone, Debug, PartialEq)]
pub struct FreeSpace {
    map_version: MapVersion,
    built: UnixNanos,
    origin: Point,
    resolution_m: f32,
    width: u32,
    height: u32,
    band: HeightBand,
    clearance: Vec<ClearanceMm>,
}

#[allow(clippy::too_many_arguments)]
impl FreeSpace {
    pub fn new(
        map_version: MapVersion,
        built: UnixNanos,
        origin: Point,
        resolution_m: f32,
        width: u32,
        height: u32,
        band: HeightBand,
        clearance: Vec<ClearanceMm>,
    ) -> Result<Self, InvalidFreeSpace> {
        if !(resolution_m.is_finite() && resolution_m > 0.0) {
            return Err(InvalidFreeSpace::BadResolution(resolution_m));
        }
        let expected = width as usize * height as usize;
        if clearance.len() != expected {
            return Err(InvalidFreeSpace::WrongCellCount { width, height, expected, received: clearance.len() });
        }
        Ok(Self { map_version, built, origin, resolution_m, width, height, band, clearance })
    }

    pub fn map_version(&self) -> MapVersion {
        self.map_version
    }

    pub fn built(&self) -> UnixNanos {
        self.built
    }

    pub fn origin(&self) -> Point {
        self.origin
    }

    pub fn resolution_m(&self) -> f32 {
        self.resolution_m
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn band(&self) -> HeightBand {
        self.band
    }

    pub fn clearance(&self) -> &[ClearanceMm] {
        &self.clearance
    }

    /// The far corner of the grid.
    pub fn extent(&self) -> Point {
        Point {
            x_m: self.origin.x_m + self.width as f32 * self.resolution_m,
            y_m: self.origin.y_m + self.height as f32 * self.resolution_m,
        }
    }

    /// The cell under a point, or None outside the grid.
    pub fn at(&self, p: Point) -> Option<Cell> {
        let column = ((p.x_m - self.origin.x_m) / self.resolution_m).floor();
        let row = ((p.y_m - self.origin.y_m) / self.resolution_m).floor();
        if column < 0.0 || row < 0.0 || column >= self.width as f32 || row >= self.height as f32 {
            return None;
        }
        Some(self.clearance[row as usize * self.width as usize + column as usize].cell())
    }

    pub fn cell_centre(&self, column: u32, row: u32) -> Point {
        Point {
            x_m: self.origin.x_m + (column as f32 + 0.5) * self.resolution_m,
            y_m: self.origin.y_m + (row as f32 + 0.5) * self.resolution_m,
        }
    }
}

/// What an object update reports about the object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectChange {
    Appeared,
    /// Seen again where it stands; its geometry may be refined.
    Updated,
    /// Seen standing somewhere else.
    Moved,
    Disappeared,
    /// Found to be the same object as another, which carries on.
    MergedInto(ObjectId),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Position3 {
    pub x_m: f32,
    pub y_m: f32,
    pub z_m: f32,
}

impl Position3 {
    pub fn on_floor(self) -> Point {
        Point { x_m: self.x_m, y_m: self.y_m }
    }
}

/// The four corners of an object's footprint on the floor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Footprint(pub [Point; 4]);

#[derive(Clone, PartialEq, Eq)]
pub struct JpegImage(pub Vec<u8>);

impl fmt::Debug for JpegImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "JpegImage({} bytes)", self.0.len())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObjectUpdate {
    pub change: ObjectChange,
    pub object_id: ObjectId,
    pub label: String,
    pub score: f32,
    pub confidence: f32,
    pub seen_count: u32,
    pub frame_id: FrameId,
    pub captured: UnixNanos,
    pub map_version: MapVersion,
    pub centre: Position3,
    pub footprint: Footprint,
    pub size: Size,
    /// Height of the object's top in the map frame.
    pub top_m: f32,
    /// A crop of the object from the keyframe.
    pub crop: Option<JpegImage>,
}

/// A sentence describing an object, matched by `find`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Caption(pub String);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobotPoseSample {
    pub captured: UnixNanos,
    pub map_version: MapVersion,
    pub pose: MapPose,
}

/// The agent slot that sent a goal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentName(pub String);

/// The model the agent ran when it sent a goal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelName(pub String);

#[derive(Clone, Debug, PartialEq)]
pub struct IssuedGoal {
    pub goal: NavGoal,
    pub agent: AgentName,
    pub model: ModelName,
    pub issued: UnixNanos,
}

/// A room of the current topology as the room namer sees it: its given name, if any, and the label of every object
/// standing in it now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoomLabels {
    pub room: RoomId,
    pub name: Option<RoomName>,
    pub labels: Vec<String>,
}

/// The live inputs whose silence makes the store's answers stale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum InputSource {
    RobotPose,
    Objects,
    Topology,
    FreeSpace,
}

impl InputSource {
    pub const ALL: [InputSource; 4] = [InputSource::RobotPose, InputSource::Objects, InputSource::Topology, InputSource::FreeSpace];

    pub fn name(self) -> &'static str {
        match self {
            InputSource::RobotPose => "robot pose",
            InputSource::Objects => "object map",
            InputSource::Topology => "topology",
            InputSource::FreeSpace => "free space",
        }
    }
}
