import pytest

from dora_hailo_perception.in_flight import FrameLost, FramesInFlight, InferenceStalled
from dora_hailo_perception.poses import ModelInputFrame


def frame(frame_id: int) -> ModelInputFrame:
    return ModelInputFrame(
        frame_id=frame_id, captured_camera_ns=frame_id, side=640, source_width=1280, source_height=720,
        shrink=2, pad_x=0, pad_y=140,
    )


def test_frames_come_back_in_the_order_given_until_none_are_left():
    in_flight = FramesInFlight(capacity=3)
    in_flight.give(10, frame(1), now_s=0.0)
    in_flight.give(20, frame(2), now_s=0.1)

    assert in_flight.take(10) == frame(1)
    assert in_flight.take(20) == frame(2)
    assert len(in_flight) == 0


def test_a_frame_returned_ahead_of_an_older_one_means_the_older_one_was_lost():
    in_flight = FramesInFlight(capacity=3)
    in_flight.give(10, frame(1), now_s=0.0)
    in_flight.give(20, frame(2), now_s=0.1)

    with pytest.raises(FrameLost, match="pts 20 while pts 10"):
        in_flight.take(20)


def test_a_frame_returned_with_nothing_in_flight_was_never_given():
    with pytest.raises(FrameLost, match="no frame in flight"):
        FramesInFlight(capacity=1).take(10)


def test_a_full_flight_has_no_room_and_refuses_another_frame():
    in_flight = FramesInFlight(capacity=1)
    in_flight.give(10, frame(1), now_s=0.0)

    assert not in_flight.has_room()
    with pytest.raises(OverflowError):
        in_flight.give(20, frame(2), now_s=0.1)


def test_frames_have_to_be_given_in_rising_pts():
    in_flight = FramesInFlight(capacity=3)
    in_flight.give(20, frame(2), now_s=0.0)
    in_flight.take(20)

    with pytest.raises(ValueError, match="does not follow"):
        in_flight.give(20, frame(2), now_s=0.1)


def test_inference_has_stalled_when_the_oldest_frame_waits_past_the_limit():
    in_flight = FramesInFlight(capacity=3)
    in_flight.give(10, frame(1), now_s=1.0)
    in_flight.give(20, frame(2), now_s=5.0)

    in_flight.raise_if_stalled(now_s=6.0, stall_s=5.0)
    with pytest.raises(InferenceStalled, match="frame 1 has waited 5.5 s"):
        in_flight.raise_if_stalled(now_s=6.5, stall_s=5.0)


def test_nothing_in_flight_never_stalls():
    FramesInFlight(capacity=3).raise_if_stalled(now_s=1e9, stall_s=5.0)


def test_a_returned_frame_stops_counting_towards_a_stall():
    in_flight = FramesInFlight(capacity=3)
    in_flight.give(10, frame(1), now_s=0.0)
    in_flight.give(20, frame(2), now_s=4.0)
    in_flight.take(10)

    in_flight.raise_if_stalled(now_s=8.0, stall_s=5.0)
