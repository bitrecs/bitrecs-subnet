import pytz
import pytest
from datetime import datetime
from bitrecs.commerce.events import get_current_ecommerce_event

def test_specific_example_dates():
    cases = [
        (datetime(2025, 2, 10, tzinfo=pytz.UTC), "Valentine's Day"),
        (datetime(2025, 5, 10, tzinfo=pytz.UTC), "Mother's Day"),
        (datetime(2025, 11, 28, tzinfo=pytz.UTC), "Black Friday"),
        (datetime(2025, 12, 30, tzinfo=pytz.UTC), "New Year's Sales"),
        (datetime(2025, 4, 15, tzinfo=pytz.UTC), None),
    ]
    for dt, expected in cases:
        assert get_current_ecommerce_event(dt) == expected

def test_naive_datetime_is_interpreted_as_utc():    
    naive = datetime(2025, 2, 10) # no tzinfo
    assert get_current_ecommerce_event(naive) == "Valentine's Day"

def test_current_utc_date_returns_str_or_none():    
    evt = get_current_ecommerce_event()
    assert evt is None or isinstance(evt, str)

def test_new_year_span_includes_jan_2():    
    dt = datetime(2025, 1, 2, tzinfo=pytz.UTC)
    assert get_current_ecommerce_event(dt) == "New Year's Sales"

if __name__ == "__main__":    
    raise SystemExit(pytest.main([__file__]))