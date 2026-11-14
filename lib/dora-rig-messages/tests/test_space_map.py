import numpy as np
import pyarrow as pa
import pytest

from dora_rig_messages.fixtures import EXAMPLES, read_fixture
from dora_rig_messages.navigation import MapVersion, ObjectId, PlaceId
from dora_rig_messages.space_map import (
    FREE_SPACE_TYPE, UNKNOWN_MM, Doorway, DoorwayId, FreeSpace, KeepOut, KeepOutZone, Place, Point, RoomId, Topology,
    TopologyRoom,
)

SPACE_MAP_FIXTURES = ["free_space", "topology", "keep_out", "keep_out_empty"]
SQUARE = (Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0))


@pytest.mark.parametrize("name", SPACE_MAP_FIXTURES)
def test_a_space_map_fixture_is_one_of_the_examples(name):
    assert name in EXAMPLES


def test_free_space_reads_back_its_grid_row_major():
    free_space = FreeSpace.from_arrow(read_fixture("free_space"))

    assert (free_space.width, free_space.height) == (3, 2)
    assert free_space.clearance_mm[1, 0] == UNKNOWN_MM
    assert free_space.clearance_mm[0, 2] == 100


def test_a_topology_room_without_a_ceiling_reads_back_as_none():
    topology = Topology.from_arrow(read_fixture("topology"))

    assert [room.ceiling_m for room in topology.rooms] == [1.5, None]


def test_free_space_whose_cells_do_not_fill_the_grid_is_rejected():
    short = pa.array(
        [
            {
                "map_version": 1, "built_unix_ns": 0, "origin_x_m": 0.0, "origin_y_m": 0.0, "resolution_m": 0.05,
                "width": 2, "height": 2, "band_min_m": 0.05, "band_max_m": 0.6, "clearance_mm": [0, 1, 2],
            }
        ],
        type=FREE_SPACE_TYPE,
    )

    with pytest.raises(ValueError, match="3 cells for 2x2"):
        FreeSpace.from_arrow(short)


def test_an_empty_height_band_is_rejected():
    with pytest.raises(ValueError, match="is empty"):
        FreeSpace(MapVersion(1), 0, 0.0, 0.0, 0.05, 0.6, 0.05, np.zeros((1, 1), np.uint16))


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
