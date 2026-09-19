"""Follows one person: picks the target each frame, drives towards it, and stops when it cannot see."""

from typing import Any, Mapping, Sequence

from .people import TrackId, people_in
from .pursuit import PursuitGains, pursue
from .tank import Tank
from .target import TargetLock


class Pilot:
    def __init__(
        self,
        lock: TargetLock,
        gains: PursuitGains,
        tank: Tank,
        min_confidence: float,
        stale_detections_after_s: float,
    ):
        self._lock = lock
        self._gains = gains
        self._tank = tank
        self._min_confidence = min_confidence
        self._stale_detections_after_s = stale_detections_after_s
        self._last_detections_s = float("-inf")

    def on_detections(self, rows: Sequence[Mapping[str, Any]], now_s: float) -> str | None:
        """Act on one frame of detections. Returns an announcement when the followed person changes."""
        self._last_detections_s = now_s
        previously_locked = self._lock.locked_id
        target = self._lock.update(people_in(rows, self._min_confidence), now_s)
        if target is None:
            self._tank.stop()
        else:
            self._tank.drive(pursue(target, self._gains))
        return _announcement(previously_locked, self._lock.locked_id)

    def on_tick(self, now_s: float) -> None:
        """Stop the tank when the camera pipeline has gone quiet."""
        if now_s - self._last_detections_s > self._stale_detections_after_s:
            self._tank.stop()


def _announcement(previously_locked: TrackId | None, locked: TrackId | None) -> str | None:
    if locked == previously_locked:
        return None
    if locked is None:
        return f"Lost person {previously_locked}"
    return f"Following person {locked}"
