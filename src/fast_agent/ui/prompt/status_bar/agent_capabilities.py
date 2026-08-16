"""Agent capability indicator rendering for the TUI toolbar."""

from __future__ import annotations

from fast_agent.core.agent_capabilities import AgentCapabilityMode
from fast_agent.ui.binary_indicator import (
    TOOLBAR_BINARY_DISABLED_COLOR,
    TOOLBAR_BINARY_ENABLED_COLOR,
    render_glyph_indicator,
)

SUBAGENT_GLYPH = "↳"
HARNESS_GLYPH = "⌘"


def render_agent_capability_indicator(mode: AgentCapabilityMode) -> str:
    subagents = mode in {AgentCapabilityMode.DELEGATE, AgentCapabilityMode.ORCHESTRATE}
    harness = mode in {AgentCapabilityMode.HARNESS_ONLY, AgentCapabilityMode.ORCHESTRATE}
    subagent_indicator = render_glyph_indicator(
        glyph=SUBAGENT_GLYPH,
        color=TOOLBAR_BINARY_ENABLED_COLOR if subagents else TOOLBAR_BINARY_DISABLED_COLOR,
    )
    harness_indicator = render_glyph_indicator(
        glyph=f"{HARNESS_GLYPH} ",
        color=TOOLBAR_BINARY_ENABLED_COLOR if harness else TOOLBAR_BINARY_DISABLED_COLOR,
    )
    return f"{subagent_indicator}{harness_indicator}"
