from dora_person_follower.people import Person, people_in
from dora_rig_messages.status import TrackId


def _row(**overrides):
    row = {
        "frame_id": 1,
        "timestamp_ms": 33.0,
        "track_id": 7,
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.9,
        "x": 0.2,
        "y": 0.1,
        "w": 0.4,
        "h": 0.5,
    }
    return row | overrides


def test_a_person_is_placed_by_box_centre_and_sized_by_box_height():
    assert people_in([_row()], min_confidence=0.5) == [
        Person(track_id=TrackId(7), centre_x=0.4, frame_fill=0.5, area=0.2)
    ]


def test_other_classes_are_not_people():
    assert people_in([_row(class_name="dog")], min_confidence=0.5) == []


def test_a_detection_below_the_confidence_floor_is_ignored():
    assert people_in([_row(confidence=0.49)], min_confidence=0.5) == []


def test_a_detection_at_the_confidence_floor_counts():
    assert len(people_in([_row(confidence=0.5)], min_confidence=0.5)) == 1


def test_a_detection_the_tracker_has_not_numbered_cannot_be_followed():
    assert people_in([_row(track_id=-1)], min_confidence=0.5) == []


def test_an_empty_frame_has_no_people():
    assert people_in([], min_confidence=0.5) == []
