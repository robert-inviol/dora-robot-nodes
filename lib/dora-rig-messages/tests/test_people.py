import pyarrow as pa
import pytest

from dora_rig_messages.fixtures import EXAMPLES, read_fixture
from dora_rig_messages.localization import CameraNanos
from dora_rig_messages.people import PEOPLE_TRACKS_TYPE, CameraFrame, OdomFrame, PeopleTracks


def _row_of(name: str) -> dict:
    return EXAMPLES[name].to_arrow().to_pylist()[0]


def test_odom_frame_tracks_name_their_odom_frame():
    tracks = PeopleTracks.from_arrow(read_fixture("people_tracks_in_odom"))

    assert tracks.position_frame == OdomFrame(CameraNanos(1_762_559_990_000_000_000))


def test_camera_frame_tracks_name_no_odom_frame():
    tracks = PeopleTracks.from_arrow(read_fixture("people_tracks"))

    assert tracks.position_frame == CameraFrame()
    assert _row_of("people_tracks")["odom_started_camera_ns"] is None


def test_odom_frame_tracks_without_their_odom_frame_are_rejected():
    row = _row_of("people_tracks_in_odom") | {"odom_started_camera_ns": None}
    array = pa.array([row], type=PEOPLE_TRACKS_TYPE)

    with pytest.raises(ValueError, match="carry no odom_started_camera_ns"):
        PeopleTracks.from_arrow(array)


def test_camera_frame_tracks_with_an_odom_frame_are_rejected():
    row = _row_of("people_tracks") | {"odom_started_camera_ns": 5}
    array = pa.array([row], type=PEOPLE_TRACKS_TYPE)

    with pytest.raises(ValueError, match="camera frame carry an odom_started_camera_ns"):
        PeopleTracks.from_arrow(array)
