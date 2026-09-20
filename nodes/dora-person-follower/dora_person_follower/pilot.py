"""Decides the drive demand: the operator's stick in manual mode, towards one person in follow mode."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from dora_rig_messages.drive import STOPPED, DriveDemand, PowerCap
from dora_rig_messages.operator import DriveMode, OperatorCommand
from dora_rig_messages.status import PilotStatus, TrackId

from .people import people_in
from .pursuit import PursuitGains, pursue
from .target import TargetLock


@dataclass(frozen=True)
class StaleAfter:
    """How long each input may stay silent before the demand drops to stopped."""

    detections_s: float
    operator_s: float


class Pilot:
    """Asks for movement only while the operator's commands keep arriving, whichever mode it is in."""

    def __init__(
        self,
        lock: TargetLock,
        gains: PursuitGains,
        manual_max_power: PowerCap,
        min_confidence: float,
        stale_after: StaleAfter,
    ):
        self._lock = lock
        self._gains = gains
        self._manual_max_power = manual_max_power
        self._min_confidence = min_confidence
        self._stale_after = stale_after
        self._operator = OperatorCommand.idle()
        self._operator_heard_s = float("-inf")
        self._detections_heard_s = float("-inf")
        self._demand = STOPPED

    @property
    def demand(self) -> DriveDemand:
        return self._demand

    @property
    def status(self) -> PilotStatus:
        return PilotStatus(self._operator.mode, self._lock.locked_id)

    def on_operator(self, command: OperatorCommand, now_s: float) -> None:
        self._operator = command
        self._operator_heard_s = now_s
        if command.mode is DriveMode.MANUAL:
            self._lock.release()
            self._demand = DriveDemand.limited(
                command.throttle, command.steer, self._manual_max_power
            )

    def on_detections(self, rows: Sequence[Mapping[str, Any]], now_s: float) -> str | None:
        """Act on one frame of detections. Returns an announcement when the followed person changes."""
        self._detections_heard_s = now_s
        if not self._is_following(now_s):
            return None
        previously_locked = self._lock.locked_id
        target = self._lock.update(people_in(rows, self._min_confidence), now_s)
        self._demand = STOPPED if target is None else pursue(target, self._gains)
        return _announcement(previously_locked, self._lock.locked_id)

    def on_tick(self, now_s: float) -> None:
        """Drop the demand to stopped when the input it came from has gone quiet."""
        if self._is_stale(self._operator_heard_s, self._stale_after.operator_s, now_s):
            self._demand = STOPPED
        elif self._operator.mode is DriveMode.FOLLOW and self._is_stale(
            self._detections_heard_s, self._stale_after.detections_s, now_s
        ):
            self._demand = STOPPED

    def _is_following(self, now_s: float) -> bool:
        return self._operator.mode is DriveMode.FOLLOW and not self._is_stale(
            self._operator_heard_s, self._stale_after.operator_s, now_s
        )

    @staticmethod
    def _is_stale(heard_s: float, stale_after_s: float, now_s: float) -> bool:
        return now_s - heard_s > stale_after_s


def _announcement(previously_locked: TrackId | None, locked: TrackId | None) -> str | None:
    if locked == previously_locked:
        return None
    if locked is None:
        return f"Lost person {previously_locked}"
    return f"Following person {locked}"
