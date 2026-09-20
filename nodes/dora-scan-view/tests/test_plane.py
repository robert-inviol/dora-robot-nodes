import pytest

from dora_rig_messages.scan import RangeReading
from dora_scan_view.plane import PointPlane

BIN_COUNT = 360
WINDOW_REVOLUTIONS = 4
MIN_HITS = 2
STALE_AFTER_S = 1.0
SPIN_REV_PER_S = 8.0


def _plane() -> PointPlane:
    return PointPlane(BIN_COUNT, WINDOW_REVOLUTIONS, MIN_HITS, STALE_AFTER_S)


def _revolution(*readings: tuple[float, float | None]) -> list[RangeReading]:
    return [RangeReading(bearing_deg, range_m) for bearing_deg, range_m in readings]


def _add_revolutions(plane: PointPlane, *revolutions: list[RangeReading]) -> None:
    for revolution in revolutions:
        plane.add_revolution(revolution, SPIN_REV_PER_S, now_s=0.0)


def test_a_surface_seen_every_revolution_is_on_the_map_at_its_range():
    plane = _plane()

    _add_revolutions(plane, *[_revolution((90.4, 2.0))] * WINDOW_REVOLUTIONS)

    snapshot = plane.snapshot(now_s=0.0)
    assert snapshot.ranges_m[90] == 2.0
    assert sum(range_m is not None for range_m in snapshot.ranges_m) == 1


def test_a_return_seen_in_too_few_revolutions_is_left_off_as_flicker():
    plane = _plane()

    _add_revolutions(plane, _revolution((10.0, 3.0)), _revolution(), _revolution())

    assert plane.snapshot(now_s=0.0).ranges_m[10] is None


def test_jitter_between_revolutions_settles_on_the_median():
    plane = _plane()

    _add_revolutions(plane, _revolution((45.0, 1.00)), _revolution((45.0, 1.04)), _revolution((45.0, 5.00)))

    assert plane.snapshot(now_s=0.0).ranges_m[45] == 1.04


def test_the_nearest_return_of_a_bin_is_the_one_kept():
    plane = _plane()

    _add_revolutions(plane, *[_revolution((200.1, 4.0), (200.7, 0.6))] * MIN_HITS)

    assert plane.snapshot(now_s=0.0).ranges_m[200] == 0.6


def test_a_direction_with_no_return_stays_empty():
    plane = _plane()

    _add_revolutions(plane, *[_revolution((30.0, None))] * WINDOW_REVOLUTIONS)

    assert plane.snapshot(now_s=0.0).ranges_m[30] is None


def test_something_that_has_gone_drops_off_once_it_leaves_the_window():
    plane = _plane()
    _add_revolutions(plane, *[_revolution((120.0, 1.5))] * WINDOW_REVOLUTIONS)

    _add_revolutions(plane, *[_revolution()] * (WINDOW_REVOLUTIONS - MIN_HITS + 1))

    assert plane.snapshot(now_s=0.0).ranges_m[120] is None


def test_the_last_bin_ends_at_a_full_turn():
    plane = _plane()

    _add_revolutions(plane, *[_revolution((359.9, 1.0), (360.0, 2.0))] * MIN_HITS)

    snapshot = plane.snapshot(now_s=0.0)
    assert snapshot.ranges_m[359] == 1.0
    assert snapshot.ranges_m[0] == 2.0


def test_a_plane_that_has_stopped_getting_revolutions_is_not_live():
    plane = _plane()
    _add_revolutions(plane, *[_revolution((90.0, 2.0))] * WINDOW_REVOLUTIONS)

    assert plane.snapshot(now_s=STALE_AFTER_S).live
    stale = plane.snapshot(now_s=STALE_AFTER_S + 0.01)
    assert not stale.live
    assert stale.spin_rev_per_s is None
    assert all(range_m is None for range_m in stale.ranges_m)


def test_a_plane_that_never_got_a_revolution_is_not_live():
    assert not _plane().snapshot(now_s=0.0).live


def test_the_message_for_the_page_carries_whole_millimetres_per_bin():
    plane = PointPlane(bin_count=4, window_revolutions=1, min_hits=1, stale_after_s=STALE_AFTER_S)
    plane.add_revolution(_revolution((0.0, 1.23456), (180.0, 0.5)), SPIN_REV_PER_S, now_s=0.0)

    assert plane.snapshot(now_s=0.0).to_message() == {
        "live": True,
        "spin_rev_per_s": SPIN_REV_PER_S,
        "bin_deg": 90.0,
        "ranges_mm": [1235, None, 500, None],
    }


def test_the_message_rounds_the_spin_rate_to_what_the_page_shows():
    plane = PointPlane(bin_count=4, window_revolutions=1, min_hits=1, stale_after_s=STALE_AFTER_S)
    plane.add_revolution(_revolution(), spin_rev_per_s=7.849999904632568, now_s=0.0)

    assert plane.snapshot(now_s=0.0).to_message()["spin_rev_per_s"] == 7.85


def test_a_plane_cannot_ask_for_more_hits_than_its_window_holds():
    with pytest.raises(ValueError, match="min_hits is 5"):
        PointPlane(BIN_COUNT, window_revolutions=4, min_hits=5, stale_after_s=STALE_AFTER_S)
