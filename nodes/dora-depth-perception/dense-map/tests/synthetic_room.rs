//! The TSDF of depth-perception's synthetic living room, seen from a few places in it.

use dense_map::settings::{FloorSettings, TsdfSettings};
use dense_map::synthetic::keyframe_at;
use dense_map::{levels, slice_field, BlockTsdf, Cell, DistanceField, PosedDepth, Slice, SurfacePoint};
use depth_perception::synthetic::SyntheticRoom;
use glam::{IVec2, Vec2};
use perception_wire::navigation::MapVersion;

const VOXEL_M: f32 = 0.05;

fn living_room_field() -> (SyntheticRoom, BlockTsdf) {
    let room = SyntheticRoom::living_room();
    let mut field = BlockTsdf::new(TsdfSettings::default());
    let mut frame_id = 0;
    for position in [Vec2::new(0.3, 0.5), Vec2::new(1.2, 3.2), Vec2::new(-0.8, 2.0)] {
        for quarter in 0..4 {
            frame_id += 1;
            let (keyframe, pose) = keyframe_at(&room, frame_id, position, quarter as f32 * std::f32::consts::FRAC_PI_2, MapVersion(1));
            field.integrate(&PosedDepth::new(&keyframe, &pose));
        }
    }
    (room, field)
}

/// Where a surface facing along one axis may lie: the room's wall facing that way, or the box's face.
fn faces(room: &SyntheticRoom, axis: usize, positive: bool) -> [f32; 2] {
    let (walls, solid) = (room.room, room.objects[0]);
    if positive { [walls.min[axis], solid.max[axis]] } else { [walls.max[axis], solid.min[axis]] }
}

/// Silhouette edges leave a few stray points behind objects; nearly all level and upright surface points lie on a
/// face, and every wall, the floor and the ceiling are covered.
#[test]
fn walls_floor_and_ceiling_are_surfaces_within_a_voxel_of_where_they_are() {
    let (room, field) = living_room_field();
    let surface = field.surface();
    let mut on_each_wall = [[0usize; 2]; 3];
    let (mut aligned, mut on_a_face) = (0, 0);
    for SurfacePoint { position, normal } in &surface {
        for axis in 0..3 {
            for (side, positive) in [(0, true), (1, false)] {
                let along = if positive { normal[axis] } else { -normal[axis] };
                if along < 0.95 {
                    continue;
                }
                aligned += 1;
                let [wall, solid] = faces(&room, axis, positive);
                let off_wall = (position[axis] - wall).abs();
                let off_solid = (position[axis] - solid).abs();
                if off_wall.min(off_solid) < 0.5 * VOXEL_M {
                    on_a_face += 1;
                }
                if off_wall < 0.5 * VOXEL_M {
                    on_each_wall[axis][side] += 1;
                }
            }
        }
    }
    assert!(on_a_face as f32 > 0.98 * aligned as f32, "{on_a_face} of {aligned} on a face");
    for (axis, sides) in on_each_wall.iter().enumerate() {
        for count in sides {
            assert!(*count > 200, "axis {axis}: {on_each_wall:?}");
        }
    }
}

#[test]
fn the_floor_is_found_at_its_height() {
    let (room, field) = living_room_field();
    let floor = levels::find_floor(&field.surface(), VOXEL_M, &FloorSettings::default()).expect("a floor");
    assert!((floor - room.room.min.z).abs() < 0.5 * VOXEL_M, "{floor}");
}

#[test]
fn the_box_is_occupied_and_the_open_floor_free_in_the_slice() {
    let (room, field) = living_room_field();
    let floor = room.room.min.z;
    let grid = slice_field(&field, Slice { min_z_m: floor + 0.05, max_z_m: floor + 0.6, occupied_below_m: 0.025 }).unwrap();
    let at = |x: f32, y: f32| grid.cells[grid.geometry.index(grid.geometry.cell_at(Vec2::new(x, y))).unwrap()];
    assert_eq!(at(0.3, 2.2), Cell::Occupied, "inside the box");
    for (x, y) in [(1.2, 1.5), (-0.8, 3.5), (1.5, 0.0), (1.5, 3.5)] {
        assert_eq!(at(x, y), Cell::Free, "open floor at {x}, {y}");
    }
    assert_eq!(at(2.2, 1.0), Cell::Unknown, "behind the wall beyond the truncation");
    let free = grid.cells.iter().filter(|c| **c == Cell::Free).count() as f32 * VOXEL_M * VOXEL_M;
    let floor_area = (room.room.max.x - room.room.min.x) * (room.room.max.y - room.room.min.y);
    assert!(free > 0.6 * floor_area && free < floor_area, "{free} m2 free of {floor_area}");
    assert!(grid.geometry.index(IVec2::ZERO).is_some());
}
