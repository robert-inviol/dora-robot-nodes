import pyarrow as pa
import pytest

from dora_rig_messages.fixtures import read_fixture
from dora_rig_messages.localization import (
    FIXTURE_EXAMPLES, KEYFRAME_POSE_TYPE, CameraNanos, FrameId, KeyframePose, MapToOdom, NotLocalized, NotSelected,
    Selected, Tracking,
)
from dora_rig_messages.navigation import MapVersion


@pytest.mark.parametrize("name", FIXTURE_EXAMPLES)
def test_a_localization_message_is_written_exactly_as_its_golden_fixture(name):
    assert FIXTURE_EXAMPLES[name].to_arrow().equals(read_fixture(name))


@pytest.mark.parametrize("name", FIXTURE_EXAMPLES)
def test_a_localization_fixture_is_read_back_as_its_message(name):
    example = FIXTURE_EXAMPLES[name]

    assert type(example).from_arrow(read_fixture(name)) == example


def _pose(**changes) -> KeyframePose:
    fields = dict(
        frame_id=FrameId(1),
        captured_unix_ns=10,
        map_version=MapVersion(0),
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        yaw_rad=0.0,
        tracking=Tracking.TRACKING,
        selection=Selected(),
        residual_m=0.01,
        inlier_fraction=0.5,
    )
    fields.update(changes)
    return KeyframePose(**fields)


def test_an_inlier_fraction_above_one_is_rejected():
    with pytest.raises(ValueError, match="inlier_fraction"):
        _pose(inlier_fraction=1.5)


def test_a_pose_that_is_not_finite_is_rejected():
    with pytest.raises(ValueError, match="x_m"):
        _pose(x_m=float("nan"))


def test_an_unknown_tracking_state_is_rejected():
    row = {**_pose().to_arrow()[0].as_py(), "tracking": "drifting"}

    with pytest.raises(ValueError, match="drifting"):
        KeyframePose.from_arrow(pa.array([row], type=KEYFRAME_POSE_TYPE))


def test_a_not_selected_keyframe_pose_names_its_reference_keyframe():
    pose = _pose(selection=NotSelected(FrameId(7)), residual_m=0.015625)

    row = pose.to_arrow()[0].as_py()

    assert (row["selection"], row["reference_frame_id"]) == ("not_selected", 7)
    assert KeyframePose.from_arrow(pose.to_arrow()) == pose


@pytest.mark.parametrize(
    ("selection", "reference_frame_id", "message"),
    [
        ("not_selected", None, "carries no reference_frame_id"),
        ("selected", 7, "carries a reference_frame_id"),
        ("not_localized", 7, "carries a reference_frame_id"),
        ("chosen", None, "chosen"),
    ],
)
def test_a_selection_and_reference_that_disagree_are_rejected(selection, reference_frame_id, message):
    row = {**_pose().to_arrow()[0].as_py(), "selection": selection, "reference_frame_id": reference_frame_id}

    with pytest.raises(ValueError, match=message):
        KeyframePose.from_arrow(pa.array([row], type=KEYFRAME_POSE_TYPE))


def test_exactly_the_lost_keyframe_poses_are_not_localized():
    with pytest.raises(ValueError, match="exactly the lost ones"):
        _pose(tracking=Tracking.LOST, selection=Selected())
    with pytest.raises(ValueError, match="exactly the lost ones"):
        _pose(tracking=Tracking.TRACKING, selection=NotLocalized())
    with pytest.raises(ValueError, match="a restarted keyframe is in the map"):
        _pose(tracking=Tracking.RESTARTED, selection=NotSelected(FrameId(3)))


def test_a_negative_sigma_is_rejected():
    with pytest.raises(ValueError, match="sigma_yaw_rad"):
        MapToOdom(MapVersion(1), CameraNanos(10), CameraNanos(0), 0.0, 0.0, 0.0, 0.0, 0.01, 0.01, -0.01)


def test_a_keyframe_pose_is_not_read_as_a_map_to_odom():
    with pytest.raises(ValueError, match="expected wire type"):
        MapToOdom.from_arrow(read_fixture("keyframe_pose_tracking"))


def test_a_correction_valid_from_before_its_odom_frame_started_is_rejected():
    with pytest.raises(ValueError, match="before its odom frame started"):
        MapToOdom(MapVersion(1), CameraNanos(5), CameraNanos(10), 0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01)
