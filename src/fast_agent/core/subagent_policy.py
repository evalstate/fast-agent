"""Run-scoped policy for built-in subagent activation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fast_agent.agents.agent_types import AgentConfig


@dataclass(frozen=True, slots=True)
class SubagentRuntimePolicy:
    enabled: bool | None = None
    model: str | None = None


def apply_subagent_runtime_policy(
    config: AgentConfig,
    policy: SubagentRuntimePolicy,
    *,
    tool_only: bool = False,
) -> None:
    """Apply run-scoped overrides while preserving explicit card disables."""
    if config.tool_only or tool_only:
        config.subagents = False
        config.subagent_activation_source = None
        return

    configuration_disables_subagents = (
        config.subagents is False and config.subagent_activation_source == "configuration"
    )
    if policy.enabled is not None:
        if policy.enabled and configuration_disables_subagents:
            return
        config.subagents = policy.enabled
        config.subagent_activation_source = "cli"
    if policy.model is not None:
        config.subagent_model = policy.model
