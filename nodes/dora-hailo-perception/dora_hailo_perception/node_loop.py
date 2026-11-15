"""What the pose node's main loop makes of each dora event, and the backstop that ends the process when the loop
itself stops coming round."""

import enum
import faulthandler
from collections.abc import Callable

# dora reports a `next(timeout=...)` that ran out as an ERROR event with this text; `None` means the stream ended.
DORA_TIMEOUT_ERROR_PREFIX = "Timeout event stream error"


class EventKind(enum.Enum):
    FRAME = "frame"
    """A frame on the node's frame input."""
    POLL = "poll"
    """`next` ran out of time with nothing to report."""
    END = "end"
    """STOP, the frame input closed, or the event stream ended: the node has nothing more to do."""


def classify(event: dict | None, frame_input: str) -> EventKind:
    if event is None:
        return EventKind.END
    kind = event["type"]
    if kind == "STOP":
        return EventKind.END
    if kind == "INPUT_CLOSED" and event["id"] == frame_input:
        return EventKind.END
    if kind == "INPUT" and event["id"] == frame_input:
        return EventKind.FRAME
    if kind == "ERROR":
        if event["error"].startswith(DORA_TIMEOUT_ERROR_PREFIX):
            return EventKind.POLL
        raise RuntimeError(f"the dora event stream failed: {event['error']}")
    raise ValueError(f"the node takes only {frame_input}, received a {kind} event for {event.get('id')}")


class MainLoopBackstop:
    """Ends the process when the main loop has not come round for `timeout_s`.

    The timer is faulthandler's: it runs on a native thread that needs neither the GIL nor any lock the main loop
    can be stuck behind, and when it fires it writes every thread's Python stack to stderr ("Timeout (h:mm:ss)!")
    and exits with status 1. Re-arming starts a new timer thread, so the loop re-arms it at most once every
    `rearm_every_s`.
    """

    def __init__(
        self,
        timeout_s: float,
        rearm_every_s: float,
        arm: Callable[..., None] = faulthandler.dump_traceback_later,
        disarm: Callable[[], None] = faulthandler.cancel_dump_traceback_later,
    ):
        if timeout_s <= rearm_every_s:
            raise ValueError(f"the backstop's timeout ({timeout_s} s) has to be longer than its re-arm period ({rearm_every_s} s)")
        self.timeout_s = timeout_s
        self.rearm_every_s = rearm_every_s
        self._arm = arm
        self._disarm = disarm
        self._armed_at_s: float | None = None

    def keep_alive(self, now_s: float) -> None:
        if self._armed_at_s is not None and now_s - self._armed_at_s < self.rearm_every_s:
            return
        self._arm(self.timeout_s, exit=True)
        self._armed_at_s = now_s

    def disarm(self) -> None:
        self._disarm()
        self._armed_at_s = None
