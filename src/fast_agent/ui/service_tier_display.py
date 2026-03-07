"""Service-tier indicator rendering for the TUI toolbar."""

from __future__ import annotations

from typing import Literal

from fast_agent.ui.reasoning_effort_display import AUTO_COLOR

ServiceTier = Literal["fast", "flex"] | None

SERVICE_TIER_GLYPH = "»"
SERVICE_TIER_FAST_COLOR = "ansired"
SERVICE_TIER_FLEX_COLOR = AUTO_COLOR
SERVICE_TIER_DISABLED_COLOR = "ansibrightblack"


def _normalize_allowed_tiers(
    allowed_tiers: tuple[Literal["fast", "flex"], ...] | None,
) -> tuple[Literal["fast", "flex"], ...]:
    if not allowed_tiers:
        return ("fast", "flex")
    return tuple(tier for tier in allowed_tiers if tier in {"fast", "flex"})


def cycle_service_tier(
    service_tier: ServiceTier,
    *,
    allowed_tiers: tuple[Literal["fast", "flex"], ...] | None = None,
) -> ServiceTier:
    normalized_allowed_tiers = _normalize_allowed_tiers(allowed_tiers)
    if not normalized_allowed_tiers:
        return None

    if service_tier is None:
        return normalized_allowed_tiers[0]

    try:
        current_index = normalized_allowed_tiers.index(service_tier)
    except ValueError:
        return normalized_allowed_tiers[0]

    next_index = current_index + 1
    if next_index >= len(normalized_allowed_tiers):
        return None
    return normalized_allowed_tiers[next_index]


def render_service_tier_indicator(
    *,
    supported: bool,
    service_tier: ServiceTier,
) -> str | None:
    if not supported:
        return None

    if service_tier == "fast":
        color = SERVICE_TIER_FAST_COLOR
    elif service_tier == "flex":
        color = SERVICE_TIER_FLEX_COLOR
    else:
        color = SERVICE_TIER_DISABLED_COLOR
    return f"<style bg='{color}'>{SERVICE_TIER_GLYPH}</style>"
