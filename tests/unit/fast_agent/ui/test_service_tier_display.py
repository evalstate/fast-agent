from fast_agent.ui.reasoning_effort_display import AUTO_COLOR
from fast_agent.ui.service_tier_display import (
    SERVICE_TIER_DISABLED_COLOR,
    SERVICE_TIER_FAST_COLOR,
    SERVICE_TIER_FLEX_COLOR,
    SERVICE_TIER_GLYPH,
    cycle_service_tier,
    render_service_tier_indicator,
)


def test_render_service_tier_indicator_hidden_when_unsupported() -> None:
    assert render_service_tier_indicator(supported=False, service_tier=None) is None


def test_render_service_tier_indicator_dim_when_disabled() -> None:
    indicator = render_service_tier_indicator(supported=True, service_tier=None)

    assert indicator == f"<style bg='{SERVICE_TIER_DISABLED_COLOR}'>{SERVICE_TIER_GLYPH}</style>"


def test_render_service_tier_indicator_red_when_fast_enabled() -> None:
    indicator = render_service_tier_indicator(supported=True, service_tier="fast")

    assert indicator == f"<style bg='{SERVICE_TIER_FAST_COLOR}'>{SERVICE_TIER_GLYPH}</style>"


def test_render_service_tier_indicator_blue_for_flex_tier() -> None:
    indicator = render_service_tier_indicator(supported=True, service_tier="flex")

    assert indicator == f"<style bg='{SERVICE_TIER_FLEX_COLOR}'>{SERVICE_TIER_GLYPH}</style>"


def test_service_tier_flex_color_matches_auto_reasoning() -> None:
    assert SERVICE_TIER_FLEX_COLOR == AUTO_COLOR


def test_cycle_service_tier_rotates_through_all_states() -> None:
    assert cycle_service_tier(None) == "fast"
    assert cycle_service_tier("fast") == "flex"
    assert cycle_service_tier("flex") is None


def test_cycle_service_tier_omits_flex_when_not_supported() -> None:
    assert cycle_service_tier(None, allowed_tiers=("fast",)) == "fast"
    assert cycle_service_tier("fast", allowed_tiers=("fast",)) is None
