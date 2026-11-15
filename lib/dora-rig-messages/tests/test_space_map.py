import numpy as np
import pyarrow as pa
import pytest

from dora_rig_messages.fixtures import EXAMPLES, read_fixture
from dora_rig_messages.navigation import MapVersion, ObjectId, PlaceId
from dora_rig_messages.space_map import (
    OCCUPANCY_GRID_TYPE, UNKNOWN_MM, Doorway, DoorwayId, OccupancyGrid, KeepOut, KeepOutZone, Place, Point, RoomId,
    MeshBlock, MeshBlocks, MeshBlocksKind, MeshSequence, MeshSnapshotRequest, Topology, TopologyRoom,
)

SPACE_MAP_FIXTURES = [
    "occupancy_grid", "topology", "keep_out", "keep_out_empty", "mesh_blocks", "mesh_blocks_snapshot",
    "mesh_snapshot_request",
]
SQUARE = (Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0))


@pytest.mark.parametrize("name", SPACE_MAP_FIXTURES)
def test_a_space_map_fixture_is_one_of_the_examples(name):
    assert name in EXAMPLES


def test_occupancy_grid_reads_back_its_grid_row_major():
    occupancy_grid = OccupancyGrid.from_arrow(read_fixture("occupancy_grid"))

    assert (occupancy_grid.width, occupancy_grid.height) == (3, 2)
    assert occupancy_grid.clearance_mm[1, 0] == UNKNOWN_MM
    assert occupancy_grid.clearance_mm[0, 2] == 100


def test_a_topology_room_without_a_ceiling_reads_back_as_none():
    topology = Topology.from_arrow(read_fixture("topology"))

    assert [room.ceiling_m for room in topology.rooms] == [1.5, None]


def test_occupancy_grid_whose_cells_do_not_fill_the_grid_is_rejected():
    short = pa.array(
        [
            {
                "map_version": 1, "built_unix_ns": 0, "origin_x_m": 0.0, "origin_y_m": 0.0, "resolution_m": 0.05,
                "width": 2, "height": 2, "band_min_m": 0.05, "band_max_m": 0.6, "clearance_mm": [0, 1, 2],
            }
        ],
        type=OCCUPANCY_GRID_TYPE,
    )

    with pytest.raises(ValueError, match="3 cells for 2x2"):
        OccupancyGrid.from_arrow(short)


def test_an_empty_height_band_is_rejected():
    with pytest.raises(ValueError, match="is empty"):
        OccupancyGrid(MapVersion(1), 0, 0.0, 0.0, 0.05, 0.6, 0.05, np.zeros((1, 1), np.uint16))


def test_a_doorway_into_a_room_the_topology_does_not_have_is_rejected():
    room = TopologyRoom(RoomId(1), SQUARE, floor_m=0.0, ceiling_m=None)

    with pytest.raises(ValueError, match="doorway 4 joins a room"):
        Topology(MapVersion(1), 0, (room,), (Doorway(DoorwayId(4), RoomId(1), RoomId(2), 0.0, 0.0, 0.8),), (), ())


def test_a_place_outside_every_room_is_rejected():
    with pytest.raises(ValueError, match="place 7 is in room 9"):
        Topology(MapVersion(1), 0, (), (), (Place(PlaceId(7), RoomId(9), 0.0, 0.0, 0.5),), ())


def test_a_keep_out_zone_with_a_negative_margin_is_rejected():
    with pytest.raises(ValueError, match="margin_m is -0.1"):
        KeepOut(MapVersion(1), (KeepOutZone(ObjectId(1), "mirror", SQUARE, margin_m=-0.1),))


def test_a_room_polygon_of_two_points_is_rejected():
    with pytest.raises(ValueError, match="polygon of 2 points"):
        TopologyRoom(RoomId(1), SQUARE[:2], floor_m=0.0, ceiling_m=None)


def test_mesh_blocks_read_back_with_an_emptied_block():
    mesh_blocks = MeshBlocks.from_arrow(read_fixture("mesh_blocks"))

    assert mesh_blocks == EXAMPLES["mesh_blocks"]
    assert [len(p.triangles) // 3 for p in mesh_blocks.blocks] == [2, 0]
    assert (mesh_blocks.run_started_unix_ns, mesh_blocks.sequence) == (1_762_559_990_000_000_000, MeshSequence(17))


def test_a_triangle_beyond_the_vertices_is_rejected():
    with pytest.raises(ValueError, match="beyond its 1 vertices"):
        MeshBlock((0, 0, 0), (0.0, 0.0, 0.0), (0, 0, 1))


def test_two_patches_of_one_block_are_rejected():
    empty = MeshBlock((1, 2, 3), (), ())
    with pytest.raises(ValueError, match="more than one mesh block"):
        MeshBlocks(MapVersion(1), 0, 0, MeshSequence(0), MeshBlocksKind.CHANGES, 0.05, 8, (empty, empty))


def test_a_snapshot_reads_back_as_a_snapshot_in_the_runs_sequence():
    snapshot = MeshBlocks.from_arrow(read_fixture("mesh_blocks_snapshot"))
    changes = MeshBlocks.from_arrow(read_fixture("mesh_blocks"))

    assert snapshot.kind is MeshBlocksKind.SNAPSHOT
    assert (snapshot.run_started_unix_ns, snapshot.sequence) == (changes.run_started_unix_ns, changes.sequence + 1)


def test_a_snapshot_with_a_block_without_surface_is_rejected():
    with pytest.raises(ValueError, match="block \\(2, 0, 0\\) has none"):
        MeshBlocks(MapVersion(1), 0, 0, MeshSequence(0), MeshBlocksKind.SNAPSHOT, 0.05, 8, (MeshBlock((2, 0, 0), (), ()),))


def test_a_mesh_snapshot_request_reads_back():
    request = MeshSnapshotRequest.from_arrow(read_fixture("mesh_snapshot_request"))

    assert request == MeshSnapshotRequest(requested_unix_ns=1_762_560_003_500_000_000)
