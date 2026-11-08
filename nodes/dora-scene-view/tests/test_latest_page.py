import numpy as np

from dora_rig_messages.perception import CameraIntrinsics, Keyframe, Scene, Segments
from dora_scene_view.node import LatestPage

JPEG_1X1 = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432ffc0000b080001000101011100ffc4001f0000010501010101010100000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffda0008010100003f00fbd3ffd9"
)


def keyframe(frame_id: int) -> Keyframe:
    return Keyframe(frame_id, 0, CameraIntrinsics(1.0, 1.0, 0.5, 0.5), (0.0, -9.8, 0.0), np.full((1, 1), 1000, np.uint16), JPEG_1X1)


def test_the_page_draws_the_frame_its_segments_belong_to_not_the_newest_keyframe():
    page = LatestPage()
    for frame_id in range(4):
        page.add_keyframe(keyframe(frame_id))
    page.add_segments(Segments(1, ("sofa",), (0.9,), np.zeros((1, 1), np.int16)))

    assert '"name": "frame 1"' in page.html()


def test_before_any_segments_the_newest_keyframe_is_drawn():
    page = LatestPage()
    page.add_keyframe(keyframe(0))
    page.add_keyframe(keyframe(1))

    assert '"name": "frame 1"' in page.html()


def test_a_structure_scene_does_not_replace_a_scene_with_objects():
    page = LatestPage()
    page.add_keyframe(keyframe(0))
    full = Scene(0, 0, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (), None, "no floor", ())
    page.add_scene(full, has_objects=True)
    version = page.version()
    page.add_scene(full, has_objects=False)

    assert page.version() == version
