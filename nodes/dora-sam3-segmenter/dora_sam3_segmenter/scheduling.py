"""Which object types to search for in each keyframe."""


class ConceptRota:
    """Types found in the last keyframe are searched for every time; the others take turns."""

    def __init__(self, idle_per_keyframe: int) -> None:
        self.idle_per_keyframe = idle_per_keyframe
        self._known: list[str] = []
        self._recently_found: set[str] = set()
        self._next_idle = 0

    def add(self, concepts: tuple[str, ...] | list[str]) -> list[str]:
        """Adds newly named types and returns them, so they can be searched for straight away."""
        named = [c for c in dict.fromkeys(concepts) if c not in self._known]
        self._known.extend(named)
        self._recently_found.update(named)
        return named

    def finished(self, found: set[str]) -> None:
        """Records which types the keyframe just finished actually contained."""
        self._recently_found = found & set(self._known)

    def for_next_keyframe(self) -> list[str]:
        idle = [c for c in self._known if c not in self._recently_found]
        turn = []
        if idle:
            count = min(self.idle_per_keyframe, len(idle))
            start = self._next_idle % len(idle)
            turn = [idle[(start + k) % len(idle)] for k in range(count)]
            self._next_idle = start + count
        return [c for c in self._known if c in self._recently_found] + turn
