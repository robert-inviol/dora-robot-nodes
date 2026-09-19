from scanview.scan import RangeReading, readings_in


def test_rows_become_readings_and_a_miss_has_no_range():
    rows = [{"bearing_deg": 0.0, "range_m": 3.17}, {"bearing_deg": 0.59, "range_m": None}]

    assert readings_in(rows) == [RangeReading(0.0, 3.17), RangeReading(0.59, None)]
