from typing import cast

import pytest

from fast_agent.ui.context_usage_display import (
    ContextUsageAccumulator,
    format_compact_context_usage_percent,
    normalize_context_usage_percent,
    resolve_context_usage_percent,
)


class UsageAccumulator:
    context_window_size = 200
    current_context_tokens = 50


def test_format_compact_context_usage_percent_uses_stable_widths() -> None:
    assert format_compact_context_usage_percent(0) == "0.00%"
    assert format_compact_context_usage_percent(9.999) == "9.99%"
    assert format_compact_context_usage_percent(42.42) == "42.4%"
    assert format_compact_context_usage_percent(100) == "100%+"


def test_format_compact_context_usage_percent_omits_missing_and_non_finite_values() -> None:
    assert format_compact_context_usage_percent(None) is None
    assert format_compact_context_usage_percent(float("nan")) is None
    assert format_compact_context_usage_percent(float("inf")) is None


def test_normalize_context_usage_percent_rejects_bool_and_non_finite_values() -> None:
    assert normalize_context_usage_percent(42) == 42.0
    assert normalize_context_usage_percent(True) is None
    assert normalize_context_usage_percent(float("nan")) is None


def test_resolve_context_usage_percent_uses_accumulator_when_needed() -> None:
    assert resolve_context_usage_percent(
        context_pct=None,
        usage_accumulator=UsageAccumulator(),
    ) == 25


def test_resolve_context_usage_percent_prefers_positive_accumulator_window_over_fallback() -> None:
    class TinyWindowUsageAccumulator:
        context_window_size = 0.5
        current_context_tokens = 0.25

    assert (
        resolve_context_usage_percent(
            context_pct=None,
            usage_accumulator=TinyWindowUsageAccumulator(),
            fallback_window_size=100,
        )
        == 50
    )


def test_resolve_context_usage_percent_prefers_explicit_context_pct() -> None:
    assert (
        resolve_context_usage_percent(
            context_pct=75,
            usage_accumulator=UsageAccumulator(),
        )
        == 75
    )


def test_resolve_context_usage_percent_rejects_boolean_accumulator_values() -> None:
    class BooleanUsageAccumulator:
        context_window_size = True
        current_context_tokens = True

    assert (
        resolve_context_usage_percent(
            context_pct=None,
            usage_accumulator=BooleanUsageAccumulator(),
            fallback_window_size=True,
        )
        is None
    )


def test_resolve_context_usage_percent_rejects_non_finite_accumulator_values() -> None:
    class NonFiniteUsageAccumulator:
        context_window_size = float("inf")
        current_context_tokens = 50

    assert (
        resolve_context_usage_percent(
            context_pct=None,
            usage_accumulator=NonFiniteUsageAccumulator(),
        )
        is None
    )


def test_resolve_context_usage_percent_rejects_negative_token_count() -> None:
    class NegativeUsageAccumulator:
        context_window_size = 200
        current_context_tokens = -1

    assert (
        resolve_context_usage_percent(
            context_pct=None,
            usage_accumulator=NegativeUsageAccumulator(),
        )
        is None
    )


def test_resolve_context_usage_percent_tolerates_missing_accumulator_fields() -> None:
    class MissingUsageAccumulator:
        context_window_size = 200

    assert (
        resolve_context_usage_percent(
            context_pct=None,
            usage_accumulator=cast("ContextUsageAccumulator", MissingUsageAccumulator()),
        )
        is None
    )


def test_resolve_context_usage_percent_does_not_mask_property_failures() -> None:
    class BrokenUsageAccumulator:
        context_window_size = 200

        @property
        def current_context_tokens(self) -> int:
            raise RuntimeError("usage tracking failed")

    with pytest.raises(RuntimeError, match="usage tracking failed"):
        resolve_context_usage_percent(
            context_pct=None,
            usage_accumulator=BrokenUsageAccumulator(),
        )
