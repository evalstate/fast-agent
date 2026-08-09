"""Runtime presets for fast-agent built-in tools."""

from __future__ import annotations

from enum import StrEnum
from typing import TypeGuard

from fast_agent.agents.subagent_tool import (
    set_subagent_tool_enabled,
    subagent_tool_enabled,
)
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.harness_tools import harness_tools_enabled, set_harness_tools


class AgentCapabilityMode(StrEnum):
    STANDARD = "standard"
    DELEGATE = "delegate"
    ORCHESTRATE = "orchestrate"
    HARNESS_ONLY = "harness_only"


def agent_capability_mode_supported(agent: object) -> TypeGuard[ToolAgent]:
    return (
        isinstance(agent, ToolAgent)
        and not agent.config.tool_only
        and not agent.config.subagent_child
    )


def resolve_agent_capability_mode(agent: object) -> AgentCapabilityMode:
    subagents = subagent_tool_enabled(agent)
    harness = harness_tools_enabled(agent)
    if subagents and harness:
        return AgentCapabilityMode.ORCHESTRATE
    if subagents:
        return AgentCapabilityMode.DELEGATE
    if harness:
        return AgentCapabilityMode.HARNESS_ONLY
    return AgentCapabilityMode.STANDARD


def cycle_agent_capability_mode(agent: object) -> AgentCapabilityMode:
    """Cycle Standard → Delegate → Orchestrate → Harness-only → Standard."""
    if not agent_capability_mode_supported(agent):
        return resolve_agent_capability_mode(agent)

    mode = resolve_agent_capability_mode(agent)
    if mode is AgentCapabilityMode.STANDARD:
        _enable_subagents(agent)
    elif mode is AgentCapabilityMode.DELEGATE:
        set_harness_tools(agent, True)
    elif mode is AgentCapabilityMode.ORCHESTRATE:
        set_subagent_tool_enabled(agent, False)
    else:
        set_harness_tools(agent, False)
    return resolve_agent_capability_mode(agent)


def _enable_subagents(agent: ToolAgent) -> None:
    if set_subagent_tool_enabled(agent, True):
        return
    source = agent.config.subagent_activation_source
    details = (
        f"Subagents are disabled by {source}."
        if source in {"configuration", "cli"}
        else "The built-in subagent tool is unavailable."
    )
    raise AgentConfigError("Unable to enable Delegate mode", details)
