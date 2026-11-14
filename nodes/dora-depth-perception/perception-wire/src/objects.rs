//! The object messages of lib/dora-rig-messages/dora_rig_messages/objects.py. Coordinates are in the map frame:
//! z up, gravity aligned, metres.

use std::sync::Arc;

use dora_node_api::arrow::array::{
    Array, ArrayRef, AsArray, BinaryArray, Int64Array, StringArray, StructArray, UInt32Array, UInt64Array,
};
use dora_node_api::arrow::datatypes::{DataType, Fields, Float32Type, Int64Type, UInt32Type, UInt64Type};
use eyre::{bail, Result};
use glam::{Vec2, Vec3};

use super::{checked, field, floats, item, list_of, scalar};
use crate::localization::FrameId;
pub use crate::navigation::ObjectId;
use crate::navigation::{MapVersion, UnixNanos};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectUpdateKind {
    /// Confirmed after enough sightings; first report of this id.
    Appeared,
    /// Seen again; position, size, label or crop may have changed.
    Updated,
    /// Gone from where it was and confirmed somewhere else; the id is kept.
    Moved,
    /// The camera saw clearly past where it should be, often enough to say it has gone.
    Disappeared,
    /// Found to be the same object as `merged_into`, which keeps the identity from now on.
    Merged,
}

impl ObjectUpdateKind {
    pub fn name(self) -> &'static str {
        match self {
            ObjectUpdateKind::Appeared => "appeared",
            ObjectUpdateKind::Updated => "updated",
            ObjectUpdateKind::Moved => "moved",
            ObjectUpdateKind::Disappeared => "disappeared",
            ObjectUpdateKind::Merged => "merged",
        }
    }

    pub fn from_name(name: &str) -> Result<ObjectUpdateKind> {
        Ok(match name {
            "appeared" => ObjectUpdateKind::Appeared,
            "updated" => ObjectUpdateKind::Updated,
            "moved" => ObjectUpdateKind::Moved,
            "disappeared" => ObjectUpdateKind::Disappeared,
            "merged" => ObjectUpdateKind::Merged,
            other => bail!("unknown object update kind {other:?}"),
        })
    }
}

/// An object's size and place: the minimum-area floor-plan rectangle and the height band above it.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectExtent {
    pub centre_m: Vec3,
    /// Corners of the footprint rectangle, in order.
    pub footprint_m: [Vec2; 4],
    pub long_m: f32,
    pub short_m: f32,
    pub height_m: f32,
    /// Height of the object's top in the map frame.
    pub top_m: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObjectUpdate {
    pub kind: ObjectUpdateKind,
    pub object_id: ObjectId,
    /// The surviving object; present exactly when the kind is merged.
    pub merged_into: Option<ObjectId>,
    /// The label of the best-scoring detection.
    pub label: String,
    /// The best segmentation score among the object's detections.
    pub score: f32,
    /// Sightings over sightings plus misses.
    pub confidence: f32,
    pub seen_count: u32,
    /// The keyframe whose evidence caused this update.
    pub frame_id: FrameId,
    pub captured: UnixNanos,
    pub map_version: MapVersion,
    pub extent: ObjectExtent,
    /// The best view of the object so far, when it is new to the receiver.
    pub crop_jpeg: Option<Vec<u8>>,
}

fn map_point_fields() -> Fields {
    Fields::from(vec![field("x_m", DataType::Float32), field("y_m", DataType::Float32)])
}

fn object_update_fields() -> Fields {
    Fields::from(vec![
        field("kind", DataType::Utf8),
        field("object_id", DataType::UInt64),
        field("merged_into", DataType::UInt64),
        field("label", DataType::Utf8),
        field("score", DataType::Float32),
        field("confidence", DataType::Float32),
        field("seen_count", DataType::UInt32),
        field("frame_id", DataType::UInt64),
        field("captured_unix_ns", DataType::Int64),
        field("map_version", DataType::UInt64),
        field("centre_x_m", DataType::Float32),
        field("centre_y_m", DataType::Float32),
        field("centre_z_m", DataType::Float32),
        field("footprint", DataType::List(item(DataType::Struct(map_point_fields())))),
        field("long_m", DataType::Float32),
        field("short_m", DataType::Float32),
        field("height_m", DataType::Float32),
        field("top_m", DataType::Float32),
        field("crop_jpeg", DataType::Binary),
    ])
}

fn column<'a>(row: &'a StructArray, name: &str) -> &'a ArrayRef {
    row.column_by_name(name).expect("checked wire type")
}

fn footprint_column(corners: &[Vec2; 4]) -> ArrayRef {
    let columns: Vec<ArrayRef> = vec![floats(corners.iter().map(|c| c.x)), floats(corners.iter().map(|c| c.y))];
    list_of(Arc::new(StructArray::new(map_point_fields(), columns, None)), [corners.len()])
}

fn decode_footprint(row: &StructArray) -> Result<[Vec2; 4]> {
    let corners = column(row, "footprint").as_list::<i32>().value(0);
    let corners = corners.as_struct();
    let coordinate = |name: &str| corners.column_by_name(name).expect("checked wire type").as_primitive::<Float32Type>().clone();
    let (xs, ys) = (coordinate("x_m"), coordinate("y_m"));
    let points: Vec<Vec2> = xs.values().iter().zip(ys.values().iter()).map(|(&x, &y)| Vec2::new(x, y)).collect();
    match <[Vec2; 4]>::try_from(points) {
        Ok(footprint) => Ok(footprint),
        Err(points) => bail!("an object footprint has {} corners, not 4", points.len()),
    }
}

pub fn decode_object_update(array: &dyn Array) -> Result<ObjectUpdate> {
    let row = checked(array, &object_update_fields(), "object_update")?;
    let kind = ObjectUpdateKind::from_name(column(row, "kind").as_string::<i32>().value(0))?;
    let merged_into_column = column(row, "merged_into").as_primitive::<UInt64Type>();
    let merged_into = merged_into_column.is_valid(0).then(|| ObjectId(merged_into_column.value(0)));
    if (kind == ObjectUpdateKind::Merged) != merged_into.is_some() {
        bail!("a {} object update {} merged_into", kind.name(), if merged_into.is_some() { "takes no" } else { "needs" });
    }
    let crop_column = column(row, "crop_jpeg").as_binary::<i32>();
    let float = |name: &str| scalar::<Float32Type>(row, name);
    Ok(ObjectUpdate {
        kind,
        object_id: ObjectId(scalar::<UInt64Type>(row, "object_id")),
        merged_into,
        label: column(row, "label").as_string::<i32>().value(0).to_owned(),
        score: float("score"),
        confidence: float("confidence"),
        seen_count: scalar::<UInt32Type>(row, "seen_count"),
        frame_id: FrameId(scalar::<UInt64Type>(row, "frame_id")),
        captured: UnixNanos(scalar::<Int64Type>(row, "captured_unix_ns")),
        map_version: MapVersion(scalar::<UInt64Type>(row, "map_version")),
        extent: ObjectExtent {
            centre_m: Vec3::new(float("centre_x_m"), float("centre_y_m"), float("centre_z_m")),
            footprint_m: decode_footprint(row)?,
            long_m: float("long_m"),
            short_m: float("short_m"),
            height_m: float("height_m"),
            top_m: float("top_m"),
        },
        crop_jpeg: crop_column.is_valid(0).then(|| crop_column.value(0).to_vec()),
    })
}

pub fn encode_object_update(update: &ObjectUpdate) -> StructArray {
    let extent = &update.extent;
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec![update.kind.name()])),
        Arc::new(UInt64Array::from(vec![update.object_id.0])),
        Arc::new(UInt64Array::from(vec![update.merged_into.map(|id| id.0)])),
        Arc::new(StringArray::from(vec![update.label.as_str()])),
        floats([update.score]),
        floats([update.confidence]),
        Arc::new(UInt32Array::from(vec![update.seen_count])),
        Arc::new(UInt64Array::from(vec![update.frame_id.0])),
        Arc::new(Int64Array::from(vec![update.captured.0])),
        Arc::new(UInt64Array::from(vec![update.map_version.0])),
        floats([extent.centre_m.x]),
        floats([extent.centre_m.y]),
        floats([extent.centre_m.z]),
        footprint_column(&extent.footprint_m),
        floats([extent.long_m]),
        floats([extent.short_m]),
        floats([extent.height_m]),
        floats([extent.top_m]),
        Arc::new(BinaryArray::from_opt_vec(vec![update.crop_jpeg.as_deref()])),
    ];
    StructArray::new(object_update_fields(), columns, None)
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

    #[test]
    fn the_appeared_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("object_update_appeared");
        let update = decode_object_update(&golden).unwrap();
        assert_eq!((update.kind, update.object_id, update.merged_into), (ObjectUpdateKind::Appeared, ObjectId(17), None));
        assert_eq!((update.label.as_str(), update.score, update.seen_count), ("sofa", 0.875, 3));
        assert_eq!(update.extent.centre_m, Vec3::new(1.875, 2.25, 0.25));
        assert_eq!(update.extent.footprint_m[1], Vec2::new(2.75, 2.0));
        assert_eq!(update.crop_jpeg, Some(vec![0xFF, 0xD8, 0xFF, 0xD9]));
        assert_eq!(encode_object_update(&update), golden);
    }

    #[test]
    fn the_merged_fixture_decodes_and_encodes_back_to_itself() {
        let golden = fixture("object_update_merged");
        let update = decode_object_update(&golden).unwrap();
        assert_eq!((update.kind, update.merged_into, update.map_version), (ObjectUpdateKind::Merged, Some(ObjectId(17)), MapVersion(413)));
        assert_eq!(update.crop_jpeg, None);
        assert_eq!(encode_object_update(&update), golden);
    }

    #[test]
    fn a_merge_without_the_surviving_object_is_rejected() {
        let mut update = decode_object_update(&fixture("object_update_merged")).unwrap();
        update.merged_into = None;
        let error = decode_object_update(&encode_object_update(&update)).err().unwrap();
        assert!(error.to_string().contains("merged object update needs merged_into"), "{error}");
    }

    #[test]
    fn a_footprint_without_four_corners_is_rejected() {
        let golden = fixture("object_update_appeared");
        let three_corners = list_of(
            Arc::new(StructArray::new(map_point_fields(), vec![floats([0.0, 1.0, 1.0]), floats([0.0, 0.0, 1.0])], None)),
            [3],
        );
        let (fields, mut columns, nulls) = golden.into_parts();
        let at = fields.iter().position(|f| f.name() == "footprint").unwrap();
        columns[at] = three_corners;
        let error = decode_object_update(&StructArray::new(fields, columns, nulls)).err().unwrap();
        assert!(error.to_string().contains("3 corners, not 4"), "{error}");
    }
}
