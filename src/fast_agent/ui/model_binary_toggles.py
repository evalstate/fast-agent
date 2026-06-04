"""Shared descriptors for binary model tool toggles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from fast_agent.commands.model_capabilities import (
    resolve_web_fetch_enabled,
    resolve_web_fetch_supported,
    resolve_web_search_enabled,
    resolve_web_search_supported,
    set_web_fetch_enabled,
    set_web_search_enabled,
)
from fast_agent.ui.binary_indicator import render_toolbar_binary_indicator

if TYPE_CHECKING:
    from collections.abc import Callable

    from fast_agent.interfaces import FastAgentLLMProtocol

ModelBinaryToggleId = Literal["web_search", "web_fetch"]


@dataclass(frozen=True, slots=True)
class ModelBinaryToggle:
    toggle_id: ModelBinaryToggleId
    shortcut_key: str
    label: str
    glyph: str
    resolve_supported: "Callable[[FastAgentLLMProtocol | None], bool]"
    resolve_enabled: "Callable[[FastAgentLLMProtocol | None], bool]"
    set_enabled: "Callable[[FastAgentLLMProtocol, bool | None], None]"


WEB_SEARCH_TOGGLE = ModelBinaryToggle(
    toggle_id="web_search",
    shortcut_key="F8",
    label="Web search",
    glyph="⊕",
    resolve_supported=resolve_web_search_supported,
    resolve_enabled=resolve_web_search_enabled,
    set_enabled=set_web_search_enabled,
)
WEB_FETCH_TOGGLE = ModelBinaryToggle(
    toggle_id="web_fetch",
    shortcut_key="F9",
    label="Web fetch",
    glyph=" ⇣",
    resolve_supported=resolve_web_fetch_supported,
    resolve_enabled=resolve_web_fetch_enabled,
    set_enabled=set_web_fetch_enabled,
)
MODEL_BINARY_TOGGLES: tuple[ModelBinaryToggle, ...] = (
    WEB_SEARCH_TOGGLE,
    WEB_FETCH_TOGGLE,
)


def cycle_model_binary_toggle(
    llm: "FastAgentLLMProtocol | None", toggle: ModelBinaryToggle
) -> None:
    if llm is None or not toggle.resolve_supported(llm):
        return

    try:
        toggle.set_enabled(llm, not toggle.resolve_enabled(llm))
    except ValueError:
        return


def render_model_binary_indicator(
    toggle: ModelBinaryToggle,
    *,
    supported: bool,
    enabled: bool,
) -> str | None:
    return render_toolbar_binary_indicator(
        supported=supported,
        enabled=enabled,
        glyph=toggle.glyph,
    )
