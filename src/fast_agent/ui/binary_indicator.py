"""Shared rendering for supported/enabled toolbar indicators."""

from __future__ import annotations

from html import escape as escape_html

TOOLBAR_BINARY_ENABLED_COLOR = "ansigreen"
TOOLBAR_BINARY_DISABLED_COLOR = "ansibrightblack"


def render_glyph_indicator(*, glyph: str, color: str) -> str:
    return f"<style bg='{escape_html(color, quote=True)}'>{escape_html(glyph, quote=False)}</style>"


def render_supported_glyph_indicator(
    *,
    supported: bool,
    glyph: str,
    color: str,
) -> str | None:
    return render_binary_indicator(
        supported=supported,
        enabled=True,
        glyph=glyph,
        enabled_color=color,
        disabled_color=color,
    )


def render_binary_indicator(
    *,
    supported: bool,
    enabled: bool,
    glyph: str,
    enabled_color: str,
    disabled_color: str,
) -> str | None:
    if not supported:
        return None

    color = enabled_color if enabled else disabled_color
    return render_glyph_indicator(glyph=glyph, color=color)


def render_toolbar_binary_indicator(
    *,
    supported: bool,
    enabled: bool,
    glyph: str,
) -> str | None:
    return render_binary_indicator(
        supported=supported,
        enabled=enabled,
        glyph=glyph,
        enabled_color=TOOLBAR_BINARY_ENABLED_COLOR,
        disabled_color=TOOLBAR_BINARY_DISABLED_COLOR,
    )
