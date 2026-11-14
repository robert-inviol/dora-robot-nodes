from dora_hailo_perception.poses import HailoPerson, ModelInputFrame, person_poses
from dora_rig_messages.people import Keypoint, PersonPose, PersonPoses, PixelBox

# 1280x720 colour halved onto a 640 square: 140 rows of padding above and below.
FRAME = ModelInputFrame(
    frame_id=7, captured_camera_ns=123, side=640, source_width=1280, source_height=720, shrink=2, pad_x=0, pad_y=140
)


def test_model_input_metadata_reads_into_a_frame():
    metadata = {
        "encoding": "rgb8", "width": 640, "height": 640, "frame_id": 7, "camera_ns": 123,
        "source_width": 1280, "source_height": 720, "shrink": 2, "pad_x": 0, "pad_y": 140,
    }
    assert ModelInputFrame.from_metadata(metadata) == FRAME


def test_a_box_and_its_keypoints_map_back_to_colour_pixels():
    corner = (0.0, 0.0, 0.25)
    middle = (0.5, 0.5, 0.75)
    person = HailoPerson(
        track_id=4, score=0.875, xmin=0.25, ymin=140 / 640, width=0.5, height=360 / 640,
        keypoints=(corner, middle) + (corner,) * 15,
    )

    poses = person_poses(FRAME, [person])

    assert poses == PersonPoses(
        frame_id=7, captured_camera_ns=123, width=1280, height=720,
        people=(
            PersonPose(
                4, 0.875, PixelBox(320.0, 0.0, 640.0, 720.0),
                (Keypoint(320.0, 0.0, 0.25), Keypoint(640.0, 360.0, 0.75)) + (Keypoint(320.0, 0.0, 0.25),) * 15,
            ),
        ),
    )


def test_an_untracked_person_has_no_track_id_and_an_empty_frame_has_nobody():
    person = HailoPerson(None, 0.5, 0.0, 0.25, 0.25, 0.25, ((0.5, 0.5, 0.5),) * 17)
    assert person_poses(FRAME, [person]).people[0].track_id is None
    assert person_poses(FRAME, []).people == ()
