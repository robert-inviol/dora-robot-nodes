"""Choosing one person to follow and staying with them."""

from enum import Enum
from typing import Callable, Sequence

from .people import Person, TrackId

FRAME_CENTRE_X = 0.5


class SelectionRule(Enum):
    LARGEST = "largest"
    MOST_CENTRAL = "most_central"
    FIRST_SEEN = "first_seen"


# Each rule ranks a person; the lowest rank is chosen.
_RANK: dict[SelectionRule, Callable[[Person], float]] = {
    SelectionRule.LARGEST: lambda person: -person.area,
    SelectionRule.MOST_CENTRAL: lambda person: abs(person.centre_x - FRAME_CENTRE_X),
    # hailotracker numbers tracks in the order they appear.
    SelectionRule.FIRST_SEEN: lambda person: person.track_id,
}


class TargetLock:
    """Holds on to one track id so a second person walking past does not steal the follow."""

    def __init__(self, rule: SelectionRule, release_after_s: float):
        self._rank = _RANK[rule]
        self._release_after_s = release_after_s
        self._locked_id: TrackId | None = None
        self._last_seen_s = 0.0

    @property
    def locked_id(self) -> TrackId | None:
        return self._locked_id

    def update(self, people: Sequence[Person], now_s: float) -> Person | None:
        """Return the person to follow in this frame, or None when there is nobody to follow."""
        if self._locked_id is not None:
            locked = next((p for p in people if p.track_id == self._locked_id), None)
            if locked is not None:
                self._last_seen_s = now_s
                return locked
            if now_s - self._last_seen_s < self._release_after_s:
                return None
            self._locked_id = None

        if not people:
            return None
        target = min(people, key=self._rank)
        self._locked_id = target.track_id
        self._last_seen_s = now_s
        return target
