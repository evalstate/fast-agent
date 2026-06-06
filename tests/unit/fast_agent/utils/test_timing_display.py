from fast_agent.utils.timing_display import format_duration_ms, format_rate_per_second


def test_format_duration_ms_uses_dash_for_missing_values() -> None:
    assert format_duration_ms(None) == "-"


def test_format_duration_ms_uses_dash_for_invalid_values() -> None:
    assert format_duration_ms(-1) == "-"
    assert format_duration_ms(float("nan")) == "-"
    assert format_duration_ms(float("inf")) == "-"
    assert format_duration_ms(True) == "-"
    assert format_duration_ms("123") == "-"


def test_format_duration_ms_uses_milliseconds_below_one_second() -> None:
    assert format_duration_ms(120.4) == "120ms"


def test_format_duration_ms_uses_seconds_when_rounded_milliseconds_reach_one_second() -> None:
    assert format_duration_ms(999.6) == "1.0s"


def test_format_duration_ms_uses_seconds_at_one_second_or_more() -> None:
    assert format_duration_ms(1150) == "1.1s"


def test_format_rate_per_second_uses_dash_for_missing_values() -> None:
    assert format_rate_per_second(None) == "-"


def test_format_rate_per_second_uses_dash_for_invalid_values() -> None:
    assert format_rate_per_second(-0.1) == "-"
    assert format_rate_per_second(float("nan")) == "-"
    assert format_rate_per_second(float("inf")) == "-"
    assert format_rate_per_second(True) == "-"
    assert format_rate_per_second("123") == "-"


def test_format_rate_per_second_uses_one_decimal_place() -> None:
    assert format_rate_per_second(26.96) == "27.0"
