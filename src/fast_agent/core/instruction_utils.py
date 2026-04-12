"""Shared helpers for instruction template resolution."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from fast_agent.core.instruction_refresh import (
    McpInstructionCapable,
    build_instruction,
    resolve_instruction_skill_manifests,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.core.instruction_refresh import ConfiguredMcpInstructionCapable

INTERNAL_AGENT_CARD_SENTINEL = "(internal)"


@runtime_checkable
class InstructionTemplateCapable(Protocol):
    instruction_template: str


class InstructionContextAgent(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def instruction(self) -> str: ...

    @property
    def agent_type(self) -> object: ...

    @property
    def config(self) -> "AgentConfig": ...

    def set_instruction(self, instruction: str) -> None: ...


def _normalize_agent_type_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Enum) and isinstance(value.value, str):
        return value.value
    if isinstance(value, str):
        return value
    return str(value)


def _resolve_agent_card_paths(agent: InstructionContextAgent) -> tuple[str, str]:
    source_path = agent.config.source_path
    if source_path is None:
        return INTERNAL_AGENT_CARD_SENTINEL, INTERNAL_AGENT_CARD_SENTINEL

    path = source_path if isinstance(source_path, Path) else Path(str(source_path))
    expanded = path.expanduser()
    try:
        resolved = expanded.resolve()
    except OSError:
        resolved = expanded
    return str(resolved), str(resolved.parent)


def build_agent_instruction_context(
    agent: InstructionContextAgent,
    base_context: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Merge shared template context with per-agent metadata placeholders."""
    context = dict(base_context or {})

    agent_name = agent.name or agent.config.name
    config = agent.config

    agent_type = _normalize_agent_type_value(agent.agent_type)
    if not agent_type:
        agent_type = _normalize_agent_type_value(config.agent_type)

    card_path, card_dir = _resolve_agent_card_paths(agent)

    context["agentName"] = agent_name
    context["agentType"] = agent_type
    context["agentCardPath"] = card_path
    context["agentCardDir"] = card_dir
    return context


def get_instruction_template(agent: InstructionContextAgent) -> str | None:
    if isinstance(agent, InstructionTemplateCapable) and agent.instruction_template:
        return agent.instruction_template

    instruction = agent.config.instruction
    if isinstance(instruction, str) and instruction:
        return instruction

    instruction_value = agent.instruction
    if isinstance(instruction_value, str) and instruction_value:
        return instruction_value
    return None


async def apply_instruction_context(
    agents: Iterable[InstructionContextAgent],
    context_vars: Mapping[str, str],
) -> None:
    for agent in agents:
        template = get_instruction_template(agent)
        if not template:
            continue
        resolved_context = build_agent_instruction_context(agent, context_vars)
        aggregator = None
        skill_manifests = None
        skill_read_tool_name = "read_skill"
        if isinstance(agent, McpInstructionCapable):
            configured_agent = cast("ConfiguredMcpInstructionCapable", agent)
            agent.set_instruction_context(dict(resolved_context))
            aggregator = agent.aggregator
            skill_manifests = resolve_instruction_skill_manifests(
                configured_agent,
                configured_agent.skill_manifests,
            )
            skill_read_tool_name = configured_agent.skill_read_tool_name

        resolved = await build_instruction(
            template,
            aggregator=aggregator,
            skill_manifests=skill_manifests,
            skill_read_tool_name=skill_read_tool_name,
            context=dict(resolved_context),
            source=agent.name,
        )
        if resolved is None or resolved == template:
            continue

        agent.set_instruction(resolved)
