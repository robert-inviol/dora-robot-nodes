//! Conversions from the decoded perception-wire messages to the store's ingest types.

use perception_wire::objects::{self, ObjectUpdateKind};
use perception_wire::space_map;
use perception_wire::navigation::ObjectId;
use world_model::{Point, Size};

use crate::types::{
    ClearanceMm, DoorwaySpan, Footprint, FreeSpace, HeightBand, InvalidFreeSpace, JpegImage, ObjectChange, ObjectUpdate,
    PlaceEdge, PlaceNode, Position3, RoomShape, Topology,
};

fn point(v: glam::Vec2) -> Point {
    Point { x_m: v.x, y_m: v.y }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BadObjectUpdate {
    #[error("merged update for {0} names no object it merged into")]
    MergedIntoNothing(ObjectId),
    #[error("{kind} update for {object} names an object it merged into")]
    MergedIntoOnNonMerge { kind: &'static str, object: ObjectId },
}

impl TryFrom<objects::ObjectUpdate> for ObjectUpdate {
    type Error = BadObjectUpdate;

    fn try_from(wire: objects::ObjectUpdate) -> Result<Self, BadObjectUpdate> {
        let change = match (wire.kind, wire.merged_into) {
            (ObjectUpdateKind::Merged, Some(into)) => ObjectChange::MergedInto(into),
            (ObjectUpdateKind::Merged, None) => return Err(BadObjectUpdate::MergedIntoNothing(wire.object_id)),
            (kind, Some(_)) => return Err(BadObjectUpdate::MergedIntoOnNonMerge { kind: kind.name(), object: wire.object_id }),
            (ObjectUpdateKind::Appeared, None) => ObjectChange::Appeared,
            (ObjectUpdateKind::Updated, None) => ObjectChange::Updated,
            (ObjectUpdateKind::Moved, None) => ObjectChange::Moved,
            (ObjectUpdateKind::Disappeared, None) => ObjectChange::Disappeared,
        };
        let extent = wire.extent;
        Ok(ObjectUpdate {
            change,
            object_id: wire.object_id,
            label: wire.label,
            score: wire.score,
            confidence: wire.confidence,
            seen_count: wire.seen_count,
            frame_id: wire.frame_id,
            captured: wire.captured,
            map_version: wire.map_version,
            centre: Position3 { x_m: extent.centre_m.x, y_m: extent.centre_m.y, z_m: extent.centre_m.z },
            footprint: Footprint(extent.footprint_m.map(point)),
            size: Size { long_m: extent.long_m, short_m: extent.short_m, height_m: extent.height_m },
            top_m: extent.top_m,
            crop: wire.crop_jpeg.map(JpegImage),
        })
    }
}

impl TryFrom<space_map::FreeSpace> for FreeSpace {
    type Error = InvalidFreeSpace;

    fn try_from(wire: space_map::FreeSpace) -> Result<Self, InvalidFreeSpace> {
        FreeSpace::new(
            wire.map_version,
            wire.built,
            point(wire.origin),
            wire.resolution_m,
            wire.width,
            wire.height,
            HeightBand { min_m: wire.band.min_m, max_m: wire.band.max_m },
            wire.clearance_mm.into_iter().map(ClearanceMm).collect(),
        )
    }
}

impl From<space_map::Topology> for Topology {
    fn from(wire: space_map::Topology) -> Self {
        Topology {
            map_version: wire.map_version,
            built: wire.built,
            rooms: wire
                .rooms
                .into_iter()
                .map(|r| RoomShape {
                    id: r.room_id,
                    polygon: r.polygon.into_iter().map(point).collect(),
                    floor_m: r.floor_m,
                    ceiling_m: r.ceiling_m,
                })
                .collect(),
            doorways: wire
                .doorways
                .into_iter()
                .map(|d| DoorwaySpan { id: d.doorway_id, room_a: d.room_a, room_b: d.room_b, centre: point(d.position), width_m: d.width_m })
                .collect(),
            places: wire
                .places
                .into_iter()
                .map(|p| PlaceNode { id: p.place_id, room: p.room_id, position: point(p.position), clearance_m: p.clearance_m })
                .collect(),
            edges: wire
                .edges
                .into_iter()
                .map(|e| PlaceEdge { a: e.place_a, b: e.place_b, length_m: e.length_m, min_clearance_m: e.min_clearance_m })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};
    use perception_wire::localization::FrameId;
    use perception_wire::navigation::{MapVersion, UnixNanos};

    use super::*;

    fn wire_update(kind: ObjectUpdateKind, merged_into: Option<ObjectId>) -> objects::ObjectUpdate {
        objects::ObjectUpdate {
            kind,
            object_id: ObjectId(4),
            merged_into,
            label: "chair".to_owned(),
            score: 0.9,
            confidence: 0.7,
            seen_count: 3,
            frame_id: FrameId(11),
            captured: UnixNanos(5),
            map_version: MapVersion(2),
            extent: objects::ObjectExtent {
                centre_m: Vec3::new(1.0, 2.0, 0.45),
                footprint_m: [Vec2::new(0.75, 1.75), Vec2::new(1.25, 1.75), Vec2::new(1.25, 2.25), Vec2::new(0.75, 2.25)],
                long_m: 0.5,
                short_m: 0.5,
                height_m: 0.9,
                top_m: 0.9,
            },
            crop_jpeg: None,
        }
    }

    #[test]
    fn a_merge_carries_the_object_it_merged_into() {
        let update = ObjectUpdate::try_from(wire_update(ObjectUpdateKind::Merged, Some(ObjectId(3)))).unwrap();
        assert_eq!(update.change, ObjectChange::MergedInto(ObjectId(3)));
        assert_eq!(update.footprint.0[2], Point { x_m: 1.25, y_m: 2.25 });
        assert_eq!(
            ObjectUpdate::try_from(wire_update(ObjectUpdateKind::Merged, None)),
            Err(BadObjectUpdate::MergedIntoNothing(ObjectId(4)))
        );
        assert!(ObjectUpdate::try_from(wire_update(ObjectUpdateKind::Moved, Some(ObjectId(3)))).is_err());
    }

    #[test]
    fn a_free_space_grid_with_the_wrong_cell_count_is_refused() {
        let wire = space_map::FreeSpace {
            map_version: MapVersion(1),
            built: UnixNanos(0),
            origin: Vec2::ZERO,
            resolution_m: 0.05,
            width: 3,
            height: 2,
            band: space_map::HeightBand { min_m: 0.05, max_m: 1.2 },
            clearance_mm: vec![0; 5],
        };
        assert!(matches!(FreeSpace::try_from(wire), Err(InvalidFreeSpace::WrongCellCount { expected: 6, received: 5, .. })));
    }
}
