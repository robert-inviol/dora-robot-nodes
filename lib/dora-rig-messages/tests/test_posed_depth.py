import numpy as np
import pytest

from dora_rig_messages.camera import StreamIntrinsics
from dora_rig_messages.fixtures import read_fixture
from dora_rig_messages.localization import CameraNanos, FrameId
from dora_rig_messages.posed_depth import FIXTURE_EXAMPLES, PosedDepthFrame


def test_the_posed_depth_frame_fixture_reads_back_its_image_row_major():
    frame = PosedDepthFrame.from_arrow(read_fixture("posed_depth_frame"))

    assert frame == FIXTURE_EXAMPLES["posed_depth_frame"]
    assert frame.depth_mm[1, 0] == 2450
    assert frame.odom_started_camera_ns == CameraNanos(1_762_559_990_000_000_000)


def test_a_posed_depth_frame_whose_depth_does_not_fit_its_intrinsics_is_rejected():
    with pytest.raises(ValueError, match="not the intrinsics"):
        PosedDepthFrame(
            FrameId(1), CameraNanos(10), CameraNanos(0), StreamIntrinsics(4, 2, 1.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), np.zeros((2, 3), np.uint16),
        )


def test_a_posed_depth_frame_from_before_its_odom_frame_started_is_rejected():
    with pytest.raises(ValueError, match="before its odom frame started"):
        PosedDepthFrame(
            FrameId(1), CameraNanos(5), CameraNanos(10), StreamIntrinsics(1, 1, 1.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), np.zeros((1, 1), np.uint16),
        )
