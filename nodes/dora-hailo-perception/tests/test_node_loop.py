import pytest

from dora_hailo_perception.node_loop import EventKind, MainLoopBackstop, classify

FRAMES = "model_input"


def test_a_frame_on_the_frame_input_is_a_frame():
    assert classify({"type": "INPUT", "id": FRAMES}, FRAMES) is EventKind.FRAME


def test_a_timed_out_next_is_a_poll():
    event = {"type": "ERROR", "error": "Timeout event stream error: Receiver timed out"}
    assert classify(event, FRAMES) is EventKind.POLL


@pytest.mark.parametrize(
    "event",
    [None, {"type": "STOP", "id": None}, {"type": "INPUT_CLOSED", "id": FRAMES}],
    ids=["stream ended", "stop", "frame input closed"],
)
def test_the_node_ends_when_the_stream_ends_stops_or_closes_its_frame_input(event):
    assert classify(event, FRAMES) is EventKind.END


def test_any_other_error_from_dora_fails_the_node():
    with pytest.raises(RuntimeError, match="daemon went away"):
        classify({"type": "ERROR", "error": "daemon went away"}, FRAMES)


def test_an_input_the_node_does_not_take_fails_the_node():
    with pytest.raises(ValueError, match="INPUT event for tick"):
        classify({"type": "INPUT", "id": "tick"}, FRAMES)


class Timer:
    def __init__(self):
        self.armed: list[tuple[float, bool]] = []
        self.disarmed = 0

    def arm(self, timeout_s: float, exit: bool) -> None:
        self.armed.append((timeout_s, exit))

    def disarm(self) -> None:
        self.disarmed += 1


def test_the_backstop_arms_an_exiting_timer_and_rearms_it_at_most_once_a_period():
    timer = Timer()
    backstop = MainLoopBackstop(timeout_s=7.0, rearm_every_s=1.0, arm=timer.arm, disarm=timer.disarm)

    for now_s in (0.0, 0.2, 0.9, 1.0, 1.5, 2.1):
        backstop.keep_alive(now_s)

    assert timer.armed == [(7.0, True)] * 3


def test_a_disarmed_backstop_arms_again_on_the_next_keep_alive():
    timer = Timer()
    backstop = MainLoopBackstop(timeout_s=7.0, rearm_every_s=1.0, arm=timer.arm, disarm=timer.disarm)
    backstop.keep_alive(0.0)
    backstop.disarm()
    backstop.keep_alive(0.1)

    assert (len(timer.armed), timer.disarmed) == (2, 1)


def test_the_backstop_timeout_has_to_outlast_its_rearm_period():
    with pytest.raises(ValueError):
        MainLoopBackstop(timeout_s=1.0, rearm_every_s=1.0)
