//! The perception messages of lib/dora-rig-messages/dora_rig_messages/perception.py, as single-row Arrow structs.
//! Field names, types and nullability must match that module exactly; the tests read its golden fixtures.

use std::sync::Arc;

use depth_perception::{CameraIntrinsics, ObjectFootprint, Plane, Room, RoomError};
use dora_node_api::arrow::array::{
    Array, ArrayRef, AsArray, BinaryArray, BooleanArray, Float32Array, Int16Array, Int64Array, ListArray, StringArray,
    StructArray, UInt16Array, UInt32Array, UInt64Array,
};
use dora_node_api::arrow::buffer::{OffsetBuffer, ScalarBuffer};
use dora_node_api::arrow::datatypes::{
    ArrowPrimitiveType, DataType, Field, Fields, Float32Type, Int16Type, Int64Type, UInt16Type, UInt64Type,
};
use eyre::{bail, eyre, Context, Result};
use glam::Vec3;

pub const NO_SEGMENT: i16 = -1;

pub struct Keyframe {
    pub frame_id: u64,
    pub captured_unix_ns: i64,
    pub width: usize,
    pub height: usize,
    pub intrinsics: CameraIntrinsics,
    /// The accelerometer at rest, in camera coordinates: the reaction to gravity.
    pub gravity_reaction: Vec3,
    /// Row major, aligned to the colour image, 0 where the sensor has no reading.
    pub depth_mm: ScalarBuffer<u16>,
    pub color_jpeg: Vec<u8>,
}

impl Keyframe {
    pub fn depth_m(&self) -> Vec<f32> {
        self.depth_mm.iter().map(|&mm| mm as f32 * 0.001).collect()
    }
}

/// Every accelerometer and gyro sample since the previous keyframe, oldest first. Camera timestamps are on the
/// camera's own clock, shared by all its sensors, in nanoseconds (not unix times); axes are the IMU's, which on the
/// D455 are the colour camera's: x right, y down, z forward.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Imu {
    /// The keyframe these samples lead up to.
    pub frame_id: u64,
    pub keyframe_camera_ns: i64,
    pub accel_camera_ns: Vec<i64>,
    pub accel_m_s2: Vec<Vec3>,
    pub gyro_camera_ns: Vec<i64>,
    pub gyro_rad_s: Vec<Vec3>,
}

pub struct Segments {
    pub frame_id: u64,
    pub width: usize,
    pub height: usize,
    pub labels: Vec<String>,
    pub scores: Vec<f32>,
    /// Per pixel, the index into labels or NO_SEGMENT; where masks overlap the smaller one wins.
    pub segment_ids: ScalarBuffer<i16>,
}

pub struct SceneObject<'a> {
    pub label: &'a str,
    /// Index into the labels of the segments message the object came from.
    pub segment_index: u16,
    pub score: f32,
    pub footprint: &'a ObjectFootprint,
}

pub struct Scene<'a> {
    pub frame_id: u64,
    pub captured_unix_ns: i64,
    pub camera_forward: Vec3,
    pub camera_up: Vec3,
    pub planes: &'a [Plane],
    pub room: &'a Result<Room, RoomError>,
    pub objects: &'a [SceneObject<'a>],
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

fn keyframe_fields() -> Fields {
    Fields::from(vec![
        field("frame_id", DataType::UInt64),
        field("captured_unix_ns", DataType::Int64),
        field("width", DataType::UInt16),
        field("height", DataType::UInt16),
        field("fx", DataType::Float32),
        field("fy", DataType::Float32),
        field("ppx", DataType::Float32),
        field("ppy", DataType::Float32),
        field("gravity_reaction", items(DataType::Float32)),
        field("depth_mm", items(DataType::UInt16)),
        field("color_jpeg", DataType::Binary),
    ])
}

fn imu_fields() -> Fields {
    Fields::from(vec![
        field("frame_id", DataType::UInt64),
        field("keyframe_camera_ns", DataType::Int64),
        field("accel_camera_ns", items(DataType::Int64)),
        field("accel_m_s2", items(DataType::Float32)),
        field("gyro_camera_ns", items(DataType::Int64)),
        field("gyro_rad_s", items(DataType::Float32)),
    ])
}

fn segments_fields() -> Fields {
    Fields::from(vec![
        field("frame_id", DataType::UInt64),
        field("width", DataType::UInt16),
        field("height", DataType::UInt16),
        field("labels", items(DataType::Utf8)),
        field("scores", items(DataType::Float32)),
        field("segment_ids", items(DataType::Int16)),
    ])
}

fn plane_fields() -> Fields {
    Fields::from(vec![
        field("kind", DataType::Utf8),
        field("normal", items(DataType::Float32)),
        field("offset_m", DataType::Float32),
        field("area_m2", DataType::Float32),
        field("outline_m", items(DataType::Float32)),
    ])
}

fn room_fields() -> Fields {
    Fields::from(vec![
        field("floor_m", DataType::Float32),
        field("ceiling_m", DataType::Float32),
        field("footprint_m", items(DataType::Float32)),
        field("edge_seen", items(DataType::Boolean)),
    ])
}

fn object_fields() -> Fields {
    Fields::from(vec![
        field("label", DataType::Utf8),
        field("segment_index", DataType::UInt16),
        field("score", DataType::Float32),
        field("point_count", DataType::UInt32),
        field("distance_m", DataType::Float32),
        field("footprint_long_m", DataType::Float32),
        field("footprint_short_m", DataType::Float32),
        field("height_m", DataType::Float32),
        field("top_m", DataType::Float32),
        field("corners_m", items(DataType::Float32)),
    ])
}

fn scene_fields() -> Fields {
    Fields::from(vec![
        field("frame_id", DataType::UInt64),
        field("captured_unix_ns", DataType::Int64),
        field("camera_forward", items(DataType::Float32)),
        field("camera_up", items(DataType::Float32)),
        field("planes", DataType::List(item(DataType::Struct(plane_fields())))),
        field("room", DataType::Struct(room_fields())),
        field("room_problem", DataType::Utf8),
        field("objects", DataType::List(item(DataType::Struct(object_fields())))),
    ])
}

/// One list per row: each row's values are `values[offsets[r]..offsets[r+1]]`.
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

fn checked<'a>(array: &'a dyn Array, fields: &Fields, name: &str) -> Result<&'a StructArray> {
    let expected = DataType::Struct(fields.clone());
    if array.data_type() != &expected {
        bail!("expected the {name} wire type {expected:?}, received {:?}", array.data_type());
    }
    if array.len() != 1 {
        bail!("expected one {name} row, received {}", array.len());
    }
    Ok(array.as_struct())
}

fn scalar<T: ArrowPrimitiveType>(row: &StructArray, name: &str) -> T::Native {
    row.column_by_name(name).expect("checked wire type").as_primitive::<T>().value(0)
}

fn list_values<T: ArrowPrimitiveType>(row: &StructArray, name: &str) -> ScalarBuffer<T::Native> {
    let list = row.column_by_name(name).expect("checked wire type").as_list::<i32>();
    let range = list.value_offsets()[0] as usize..list.value_offsets()[1] as usize;
    list.values().as_primitive::<T>().values().slice(range.start, range.len())
}

pub fn decode_keyframe(array: &dyn Array) -> Result<Keyframe> {
    let row = checked(array, &keyframe_fields(), "keyframe")?;
    let (width, height) = (scalar::<UInt16Type>(row, "width") as usize, scalar::<UInt16Type>(row, "height") as usize);
    let depth_mm = list_values::<UInt16Type>(row, "depth_mm");
    if depth_mm.len() != width * height {
        bail!("keyframe depth has {} values for {width}x{height}", depth_mm.len());
    }
    let gravity = list_values::<Float32Type>(row, "gravity_reaction");
    let [gx, gy, gz] = gravity[..] else {
        bail!("keyframe gravity reaction has {} values, not 3", gravity.len());
    };
    Ok(Keyframe {
        frame_id: scalar::<UInt64Type>(row, "frame_id"),
        captured_unix_ns: scalar::<Int64Type>(row, "captured_unix_ns"),
        width,
        height,
        intrinsics: CameraIntrinsics {
            fx: scalar::<Float32Type>(row, "fx"),
            fy: scalar::<Float32Type>(row, "fy"),
            ppx: scalar::<Float32Type>(row, "ppx"),
            ppy: scalar::<Float32Type>(row, "ppy"),
        },
        gravity_reaction: Vec3::new(gx, gy, gz),
        depth_mm,
        color_jpeg: row.column_by_name("color_jpeg").expect("checked wire type").as_binary::<i32>().value(0).to_vec(),
    })
}

pub fn encode_keyframe(keyframe: &Keyframe) -> StructArray {
    let k = keyframe.intrinsics;
    let g = keyframe.gravity_reaction;
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(vec![keyframe.frame_id])),
        Arc::new(Int64Array::from(vec![keyframe.captured_unix_ns])),
        Arc::new(UInt16Array::from(vec![keyframe.width as u16])),
        Arc::new(UInt16Array::from(vec![keyframe.height as u16])),
        floats([k.fx]),
        floats([k.fy]),
        floats([k.ppx]),
        floats([k.ppy]),
        one_list(floats([g.x, g.y, g.z])),
        one_list(Arc::new(UInt16Array::new(keyframe.depth_mm.clone(), None))),
        Arc::new(BinaryArray::from_vec(vec![&keyframe.color_jpeg[..]])),
    ];
    StructArray::new(keyframe_fields(), columns, None)
}

fn triples(row: &StructArray, name: &str, count: usize) -> Result<Vec<Vec3>> {
    let flat = list_values::<Float32Type>(row, name);
    if flat.len() != 3 * count {
        bail!("imu {name} has {} values for {count} timestamps", flat.len());
    }
    Ok(flat.chunks_exact(3).map(Vec3::from_slice).collect())
}

pub fn decode_imu(array: &dyn Array) -> Result<Imu> {
    let row = checked(array, &imu_fields(), "imu")?;
    let accel_camera_ns = list_values::<Int64Type>(row, "accel_camera_ns").to_vec();
    let gyro_camera_ns = list_values::<Int64Type>(row, "gyro_camera_ns").to_vec();
    Ok(Imu {
        frame_id: scalar::<UInt64Type>(row, "frame_id"),
        keyframe_camera_ns: scalar::<Int64Type>(row, "keyframe_camera_ns"),
        accel_m_s2: triples(row, "accel_m_s2", accel_camera_ns.len())?,
        gyro_rad_s: triples(row, "gyro_rad_s", gyro_camera_ns.len())?,
        accel_camera_ns,
        gyro_camera_ns,
    })
}

pub fn encode_imu(imu: &Imu) -> StructArray {
    let flat = |samples: &[Vec3]| floats(samples.iter().flat_map(|s| s.to_array()));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(vec![imu.frame_id])),
        Arc::new(Int64Array::from(vec![imu.keyframe_camera_ns])),
        one_list(Arc::new(Int64Array::from(imu.accel_camera_ns.clone()))),
        one_list(flat(&imu.accel_m_s2)),
        one_list(Arc::new(Int64Array::from(imu.gyro_camera_ns.clone()))),
        one_list(flat(&imu.gyro_rad_s)),
    ];
    StructArray::new(imu_fields(), columns, None)
}

pub fn decode_segments(array: &dyn Array) -> Result<Segments> {
    let row = checked(array, &segments_fields(), "segments")?;
    let (width, height) = (scalar::<UInt16Type>(row, "width") as usize, scalar::<UInt16Type>(row, "height") as usize);
    let labels_list = row.column_by_name("labels").expect("checked wire type").as_list::<i32>();
    let labels_array = labels_list.value(0);
    let labels: Vec<String> = labels_array.as_string::<i32>().iter().map(|l| l.unwrap_or_default().to_owned()).collect();
    let scores = list_values::<Float32Type>(row, "scores").to_vec();
    let segment_ids = list_values::<Int16Type>(row, "segment_ids");
    if labels.len() != scores.len() {
        bail!("segments have {} labels for {} scores", labels.len(), scores.len());
    }
    if segment_ids.len() != width * height {
        bail!("segment ids have {} values for {width}x{height}", segment_ids.len());
    }
    if let Some(bad) = segment_ids.iter().find(|&&id| id != NO_SEGMENT && (id < 0 || id as usize >= labels.len())) {
        bail!("segment id {bad} is outside the {} labels", labels.len());
    }
    Ok(Segments { frame_id: scalar::<UInt64Type>(row, "frame_id"), width, height, labels, scores, segment_ids })
}

pub fn encode_segments(segments: &Segments) -> StructArray {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(vec![segments.frame_id])),
        Arc::new(UInt16Array::from(vec![segments.width as u16])),
        Arc::new(UInt16Array::from(vec![segments.height as u16])),
        one_list(Arc::new(StringArray::from_iter_values(&segments.labels))),
        one_list(floats(segments.scores.iter().copied())),
        one_list(Arc::new(Int16Array::new(segments.segment_ids.clone(), None))),
    ];
    StructArray::new(segments_fields(), columns, None)
}

fn planes_column(planes: &[Plane]) -> ArrayRef {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(planes.iter().map(|p| p.kind.name()))),
        list_of(floats(planes.iter().flat_map(|p| p.normal.to_array())), planes.iter().map(|_| 3)),
        floats(planes.iter().map(|p| p.offset_m)),
        floats(planes.iter().map(|p| p.area_m2)),
        list_of(
            floats(planes.iter().flat_map(|p| p.outline_m.iter().flat_map(|c| c.to_array()))),
            planes.iter().map(|p| 3 * p.outline_m.len()),
        ),
    ];
    one_list(Arc::new(StructArray::new(plane_fields(), columns, None)))
}

fn room_columns(room: &Result<Room, RoomError>) -> (ArrayRef, ArrayRef) {
    let (floor, ceiling, footprint, edge_seen, problem) = match room {
        Ok(room) => (
            room.floor_m,
            room.ceiling_m,
            room.footprint_m.iter().flat_map(|c| c.to_array()).collect::<Vec<_>>(),
            room.edge_walls.iter().map(|w| w.is_some()).collect::<Vec<_>>(),
            None,
        ),
        Err(error) => (0.0, 0.0, Vec::new(), Vec::new(), Some(error.to_string())),
    };
    let columns: Vec<ArrayRef> = vec![
        floats([floor]),
        floats([ceiling]),
        one_list(floats(footprint)),
        one_list(Arc::new(BooleanArray::from(edge_seen))),
    ];
    let validity = Some(vec![problem.is_none()].into());
    (Arc::new(StructArray::new(room_fields(), columns, validity)), Arc::new(StringArray::from(vec![problem])))
}

fn objects_column(objects: &[SceneObject]) -> ArrayRef {
    let f = |get: fn(&ObjectFootprint) -> f32| floats(objects.iter().map(|o| get(o.footprint)));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(objects.iter().map(|o| o.label))),
        Arc::new(UInt16Array::from_iter_values(objects.iter().map(|o| o.segment_index))),
        floats(objects.iter().map(|o| o.score)),
        Arc::new(UInt32Array::from_iter_values(objects.iter().map(|o| o.footprint.point_count as u32))),
        f(|p| p.distance_m),
        f(|p| p.footprint_m.0),
        f(|p| p.footprint_m.1),
        f(|p| p.height_m),
        f(|p| p.top_above_camera_m),
        list_of(floats(objects.iter().flat_map(|o| o.footprint.corners_m.iter().flat_map(|c| c.to_array()))), objects.iter().map(|_| 8)),
    ];
    one_list(Arc::new(StructArray::new(object_fields(), columns, None)))
}

pub fn encode_scene(scene: &Scene) -> StructArray {
    let (room, room_problem) = room_columns(scene.room);
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(vec![scene.frame_id])),
        Arc::new(Int64Array::from(vec![scene.captured_unix_ns])),
        one_list(floats(scene.camera_forward.to_array())),
        one_list(floats(scene.camera_up.to_array())),
        planes_column(scene.planes),
        room,
        room_problem,
        objects_column(scene.objects),
    ];
    StructArray::new(scene_fields(), columns, None)
}

/// The data of a dora input, which arrives as an Arrow array of any type.
pub fn as_struct_row<'a>(data: &'a dyn Array, name: &str) -> Result<&'a dyn Array> {
    if !matches!(data.data_type(), DataType::Struct(_)) {
        return Err(eyre!("{name} input is not a struct message: {:?}", data.data_type()));
    }
    Ok(data)
}

/// Context for an error about one message.
pub trait MessageContext<T> {
    fn for_message(self, name: &str) -> Result<T>;
}

impl<T> MessageContext<T> for Result<T> {
    fn for_message(self, name: &str) -> Result<T> {
        self.wrap_err_with(|| format!("decoding a {name} message"))
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use depth_perception::SurfaceKind;
    use dora_node_api::arrow::ipc::reader::FileReader;
    use glam::Vec2;

    use super::*;

    fn fixture(name: &str) -> StructArray {
        let path = format!("{}/../../../lib/dora-rig-messages/dora_rig_messages/fixtures/{name}.arrow", env!("CARGO_MANIFEST_DIR"));
        let mut reader = FileReader::try_new(File::open(&path).unwrap_or_else(|e| panic!("{path}: {e}")), None).unwrap();
        StructArray::from(reader.next().unwrap().unwrap())
    }

    #[test]
    fn the_keyframe_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("keyframe");
        let keyframe = decode_keyframe(&golden).unwrap();
        assert_eq!((keyframe.frame_id, keyframe.width, keyframe.height), (42, 3, 2));
        assert_eq!(&keyframe.depth_mm[..], &[0, 1200, 1210, 2450, 65535, 400]);
        assert_eq!(keyframe.gravity_reaction, Vec3::new(-0.0625, -9.75, -0.375));
        assert_eq!(keyframe.color_jpeg, vec![0xFF, 0xD8, 0xFF, 0xD9]);
        assert_eq!(encode_keyframe(&keyframe), golden);
    }

    #[test]
    fn the_segments_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("segments");
        let segments = decode_segments(&golden).unwrap();
        assert_eq!(segments.labels, vec!["sofa", "cushion"]);
        assert_eq!(&segments.segment_ids[..], &[-1, 0, 0, 1, 0, -1]);
        assert_eq!(encode_segments(&segments), golden);
    }

    #[test]
    fn a_scene_encodes_exactly_as_the_golden_fixture() {
        let planes = vec![
            Plane {
                kind: SurfaceKind::Floor,
                normal: Vec3::Z,
                offset_m: 0.875,
                area_m2: 5.25,
                outline_m: vec![Vec3::new(0.0, 1.0, -0.875), Vec3::new(1.0, 1.0, -0.875), Vec3::new(0.5, 2.0, -0.875)],
            },
            Plane {
                kind: SurfaceKind::Wall,
                normal: Vec3::NEG_Y,
                offset_m: 4.75,
                area_m2: 3.5,
                outline_m: vec![Vec3::new(-1.0, 4.75, 0.0), Vec3::new(1.0, 4.75, 0.0), Vec3::new(0.0, 4.75, 1.5)],
            },
        ];
        let room = Ok(Room {
            floor_m: -0.875,
            ceiling_m: 1.5,
            footprint_m: vec![Vec2::new(-1.5, 0.25), Vec2::new(2.0, 0.25), Vec2::new(2.0, 4.75), Vec2::new(-1.5, 4.75)],
            edge_walls: vec![None, Some(1), Some(1), Some(1)],
        });
        let sofa = ObjectFootprint {
            point_count: 23580,
            distance_m: 1.25,
            footprint_m: (1.75, 0.5),
            height_m: 0.5,
            top_above_camera_m: -0.375,
            corners_m: [Vec2::new(0.0, 1.0), Vec2::new(1.75, 1.0), Vec2::new(1.75, 1.5), Vec2::new(0.0, 1.5)],
        };
        let objects = [SceneObject { label: "sofa", segment_index: 0, score: 0.875, footprint: &sofa }];
        let scene = Scene {
            frame_id: 42,
            captured_unix_ns: 1_762_560_000_123_456_789,
            camera_forward: Vec3::Y,
            camera_up: Vec3::Z,
            planes: &planes,
            room: &room,
            objects: &objects,
        };
        assert_eq!(encode_scene(&scene), fixture("scene"));
    }

    #[test]
    fn a_scene_without_a_room_carries_the_reason_as_the_golden_fixture() {
        let room = Err(RoomError::NoCeiling);
        let scene = Scene {
            frame_id: 43,
            captured_unix_ns: 1_762_560_001_000_000_000,
            camera_forward: Vec3::Y,
            camera_up: Vec3::Z,
            planes: &[],
            room: &room,
            objects: &[],
        };
        assert_eq!(encode_scene(&scene), fixture("scene_without_room"));
    }

    #[test]
    fn the_imu_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("imu");
        let imu = decode_imu(&golden).unwrap();
        assert_eq!((imu.frame_id, imu.keyframe_camera_ns), (42, 1_762_560_000_100_000_000));
        assert_eq!(imu.accel_camera_ns, vec![1_762_560_000_090_000_000, 1_762_560_000_095_000_000]);
        assert_eq!(imu.accel_m_s2[1], Vec3::new(0.125, -9.8125, -0.25));
        assert_eq!(imu.gyro_rad_s.len(), 3);
        assert_eq!(encode_imu(&imu), golden);
    }

    #[test]
    fn imu_samples_out_of_step_with_their_timestamps_are_rejected() {
        let mut imu = decode_imu(&fixture("imu")).unwrap();
        imu.gyro_camera_ns.pop();
        let error = decode_imu(&encode_imu(&imu)).err().unwrap();
        assert!(error.to_string().contains("9 values for 2 timestamps"), "{error}");
    }

    #[test]
    fn a_keyframe_whose_depth_does_not_fill_the_image_is_rejected() {
        let golden = fixture("keyframe");
        let mut keyframe = decode_keyframe(&golden).unwrap();
        keyframe.width = 4;
        let error = decode_keyframe(&encode_keyframe(&keyframe)).err().unwrap();
        assert!(error.to_string().contains("6 values for 4x2"), "{error}");
    }
}
