"""Service-tier indicator rendering for the TUI toolbar."""

from __future__ import annotations

from fast_agent.commands.model_capabilities import SERVICE_TIER_VALUES, ServiceTierValue
from fast_agent.ui.binary_indicator import render_supported_glyph_indicator
from fast_agent.ui.reasoning_effort_display import AUTO_COLOR
from fast_agent.utils.collections import cycle_next, unique_preserve_order

ServiceTier = ServiceTierValue | None

SERVICE_TIER_GLYPH = "»"
SERVICE_TIER_FAST_COLOR = "ansired"
SERVICE_TIER_FLEX_COLOR = AUTO_COLOR
SERVICE_TIER_DISABLED_COLOR = "ansibrightblack"
DEFAULT_SERVICE_TIERS = SERVICE_TIER_VALUES
SUPPORTED_SERVICE_TIERS = frozenset(SERVICE_TIER_VALUES)
SERVICE_TIER_COLOR_BY_VALUE: dict[ServiceTier, str] = {
    "fast": SERVICE_TIER_FAST_COLOR,
    "flex": SERVICE_TIER_FLEX_COLOR,
    None: SERVICE_TIER_DISABLED_COLOR,
}


def _normalize_allowed_tiers(
    allowed_tiers: tuple[ServiceTierValue, ...] | None,
) -> tuple[ServiceTierValue, ...]:
    if allowed_tiers is None:
        return DEFAULT_SERVICE_TIERS

    return tuple(
        tier
        for tier in unique_preserve_order(allowed_tiers)
        if tier in SUPPORTED_SERVICE_TIERS
    )


def cycle_service_tier(
    service_tier: ServiceTier,
    *,
    allowed_tiers: tuple[ServiceTierValue, ...] | None = None,
) -> ServiceTier:
    normalized_allowed_tiers = _normalize_allowed_tiers(allowed_tiers)
    if not normalized_allowed_tiers:
        return None

    cycle_order: tuple[ServiceTier, ...] = (*normalized_allowed_tiers, None)
    return cycle_next(service_tier, cycle_order)


def render_service_tier_indicator(
    *,
    supported: bool,
    service_tier: ServiceTier,
) -> str | None:
    color = SERVICE_TIER_COLOR_BY_VALUE.get(service_tier, SERVICE_TIER_DISABLED_COLOR)
    return render_supported_glyph_indicator(
        supported=supported,
        glyph=SERVICE_TIER_GLYPH,
        color=color,
    )
