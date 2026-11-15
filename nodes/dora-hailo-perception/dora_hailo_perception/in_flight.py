"""The frames the pose node has given the Hailo pipeline and is waiting on.

The pipeline returns every frame it is given, in the order given: nothing in it drops a frame. A frame that comes
back out of turn means the pipeline lost the frames before it, and a frame that has waited longer than the stall
limit means inference has stopped. Both are failures, never something to wait out.
"""

from collections import deque
from dataclasses import dataclass

from .poses import ModelInputFrame


class FrameLost(RuntimeError):
    """The pipeline returned a frame other than the oldest one it was given."""


class InferenceStalled(RuntimeError):
    """The oldest frame given to the pipeline has waited longer than the stall limit for its result."""


@dataclass(frozen=True)
class GivenFrame:
    pts: int
    frame: ModelInputFrame
    given_at_s: float


class FramesInFlight:
    """At most `capacity` frames given to the pipeline and not yet returned, oldest first."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError(f"at least one frame has to fit in flight, asked for {capacity}")
        self.capacity = capacity
        self._frames: deque[GivenFrame] = deque()
        self._last_pts: int | None = None

    def __len__(self) -> int:
        return len(self._frames)

    def has_room(self) -> bool:
        return len(self._frames) < self.capacity

    def give(self, pts: int, frame: ModelInputFrame, now_s: float) -> None:
        if not self.has_room():
            raise OverflowError(f"{self.capacity} frames are already in flight")
        if self._last_pts is not None and pts <= self._last_pts:
            raise ValueError(f"frame pts {pts} does not follow the last one given, {self._last_pts}")
        self._frames.append(GivenFrame(pts, frame, now_s))
        self._last_pts = pts

    def take(self, pts: int) -> ModelInputFrame:
        """The frame the pipeline has just returned, which has to be the oldest one in flight."""
        if not self._frames:
            raise FrameLost(f"the pipeline returned pts {pts} with no frame in flight")
        oldest = self._frames[0]
        if pts != oldest.pts:
            raise FrameLost(f"the pipeline returned pts {pts} while pts {oldest.pts} was the oldest in flight")
        self._frames.popleft()
        return oldest.frame

    def raise_if_stalled(self, now_s: float, stall_s: float) -> None:
        if not self._frames:
            return
        oldest = self._frames[0]
        waited_s = now_s - oldest.given_at_s
        if waited_s > stall_s:
            raise InferenceStalled(
                f"frame {oldest.frame.frame_id} has waited {waited_s:.1f} s for its result, longer than the "
                f"{stall_s:.1f} s limit, with {len(self._frames)} frames in flight"
            )
