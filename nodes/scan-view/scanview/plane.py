"""The scan plane as the map page gets it: coarse in bearing, steadied over the last few revolutions."""

import threading
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Any, Sequence

from messages.scan import RangeReading

FULL_TURN_DEGREES = 360.0
MILLIMETRES_PER_METRE = 1000


@dataclass(frozen=True)
class PlaneSnapshot:
    """What the map page draws. Bin 0 starts at bearing 0 and the bins run clockwise."""

    live: bool
    spin_rev_per_s: float | None
    # None where the lidar does not steadily see anything.
    ranges_m: tuple[float | None, ...]

    def to_message(self) -> dict[str, Any]:
        return {
            "live": self.live,
            "spin_rev_per_s": None if self.spin_rev_per_s is None else round(self.spin_rev_per_s, 2),
            "bin_deg": FULL_TURN_DEGREES / len(self.ranges_m),
            "ranges_mm": [
                None if range_m is None else round(range_m * MILLIMETRES_PER_METRE) for range_m in self.ranges_m
            ],
        }


class PointPlane:
    """Filled by the dora loop and read by the web server thread, so every access takes the guard."""

    def __init__(self, bin_count: int, window_revolutions: int, min_hits: int, stale_after_s: float):
        if not 1 <= min_hits <= window_revolutions:
            raise ValueError(f"min_hits is {min_hits}, expected 1 to {window_revolutions} (the window)")
        self._bin_count = bin_count
        self._min_hits = min_hits
        self._stale_after_s = stale_after_s
        self._guard = threading.Lock()
        self._window: deque[list[float | None]] = deque(maxlen=window_revolutions)
        self._spin_rev_per_s: float | None = None
        self._latest_revolution_s = float("-inf")

    def add_revolution(self, readings: Sequence[RangeReading], spin_rev_per_s: float, now_s: float) -> None:
        nearest: list[float | None] = [None] * self._bin_count
        for reading in readings:
            if reading.range_m is None:
                continue
            bin_index = int(reading.bearing_deg % FULL_TURN_DEGREES / FULL_TURN_DEGREES * self._bin_count)
            held = nearest[bin_index]
            # The nearest return of a bin wins, so a chair leg in front of a wall stays on the map.
            nearest[bin_index] = reading.range_m if held is None else min(held, reading.range_m)
        with self._guard:
            self._window.append(nearest)
            self._spin_rev_per_s = spin_rev_per_s
            self._latest_revolution_s = now_s

    def snapshot(self, now_s: float) -> PlaneSnapshot:
        with self._guard:
            if now_s - self._latest_revolution_s > self._stale_after_s:
                return PlaneSnapshot(live=False, spin_rev_per_s=None, ranges_m=(None,) * self._bin_count)
            window = list(self._window)
            spin_rev_per_s = self._spin_rev_per_s
        return PlaneSnapshot(
            live=True,
            spin_rev_per_s=spin_rev_per_s,
            ranges_m=tuple(self._steady_range(hits) for hits in zip(*window)),
        )

    def _steady_range(self, bin_over_window: Sequence[float | None]) -> float | None:
        hits = [range_m for range_m in bin_over_window if range_m is not None]
        # A return that shows up in only a few revolutions is flicker, not a surface.
        return median(hits) if len(hits) >= self._min_hits else None
