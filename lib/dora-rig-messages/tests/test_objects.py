import pytest

from dora_rig_messages.fixtures import read_fixture
from dora_rig_messages.navigation import MapVersion, ObjectId
from dora_rig_messages.objects import FIXTURE_EXAMPLES, MapPoint, ObjectUpdate, ObjectUpdateKind

APPEARED = FIXTURE_EXAMPLES["object_update_appeared"]


def _with(**changes) -> ObjectUpdate:
    fields = {name: getattr(APPEARED, name) for name in APPEARED.__dataclass_fields__}
    return ObjectUpdate(**{**fields, **changes})


@pytest.mark.parametrize("name", FIXTURE_EXAMPLES)
def test_an_object_update_round_trips_through_its_golden_fixture(name):
    assert ObjectUpdate.from_arrow(read_fixture(name)) == FIXTURE_EXAMPLES[name]


def test_a_merged_update_names_the_object_it_merged_into():
    merged = FIXTURE_EXAMPLES["object_update_merged"]

    assert merged.kind is ObjectUpdateKind.MERGED and merged.merged_into == ObjectId(17)
    assert merged.crop_jpeg is None


def test_a_merged_update_without_merged_into_is_rejected():
    with pytest.raises(ValueError, match="merged object update needs merged_into"):
        _with(kind=ObjectUpdateKind.MERGED, merged_into=None)


def test_an_update_of_another_kind_naming_merged_into_is_rejected():
    with pytest.raises(ValueError, match="moved object update takes no merged_into"):
        _with(kind=ObjectUpdateKind.MOVED, merged_into=ObjectId(3))


def test_a_footprint_without_four_corners_is_rejected():
    with pytest.raises(ValueError, match="3 corners, not 4"):
        _with(footprint=(MapPoint(0.0, 0.0), MapPoint(1.0, 0.0), MapPoint(1.0, 1.0)))


def test_a_confidence_above_one_is_rejected():
    with pytest.raises(ValueError, match="confidence is 1.5"):
        _with(confidence=1.5)


def test_map_version_travels_with_the_update():
    assert ObjectUpdate.from_arrow(APPEARED.to_arrow()).map_version == MapVersion(412)
