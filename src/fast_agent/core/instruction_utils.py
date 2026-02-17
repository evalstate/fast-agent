"""Shared helpers for instruction template resolution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.core.instruction_refresh import McpInstructionCapable, build_instruction

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def _normalize_agent_type_value(value: object) -> str:
    if value is None:
        return ""
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value
    if isinstance(value, str):
        return value
    return str(value)


def _resolve_agent_card_paths(agent: object) -> tuple[str, str]:
    config = getattr(agent, "config", None)
    source_path = getattr(config, "source_path", None) if config is not None else None
    if source_path is None:
        return "", ""

    path = source_path if isinstance(source_path, Path) else Path(str(source_path))
    expanded = path.expanduser()
    try:
        resolved = expanded.resolve()
    except OSError:
        resolved = expanded
    return str(resolved), str(resolved.parent)


def build_agent_instruction_context(
    agent: object,
    base_context: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Merge shared template context with per-agent metadata placeholders."""
    context = dict(base_context or {})

    config = getattr(agent, "config", None)
    agent_name = getattr(agent, "name", None)
    if not isinstance(agent_name, str) or not agent_name:
        config_name = getattr(config, "name", None) if config is not None else None
        if isinstance(config_name, str) and config_name:
            agent_name = config_name

    agent_type = _normalize_agent_type_value(getattr(agent, "agent_type", None))
    if not agent_type and config is not None:
        agent_type = _normalize_agent_type_value(getattr(config, "agent_type", None))

    card_path, card_dir = _resolve_agent_card_paths(agent)

    context["agentName"] = agent_name if isinstance(agent_name, str) else ""
    context["agentType"] = agent_type
    context["agentCardPath"] = card_path
    context["agentCardDir"] = card_dir
    return context


def get_instruction_template(agent: object) -> str | None:
    template = getattr(agent, "instruction_template", None)
    if isinstance(template, str) and template:
        return template

    config = getattr(agent, "config", None)
    instruction = getattr(config, "instruction", None) if config is not None else None
    if isinstance(instruction, str) and instruction:
        return instruction

    instruction_value = getattr(agent, "instruction", None)
    if isinstance(instruction_value, str) and instruction_value:
        return instruction_value
    return None


async def apply_instruction_context(
    agents: Iterable[object],
    context_vars: Mapping[str, str],
) -> None:
    for agent in agents:
        template = get_instruction_template(agent)
        if not template:
            continue
        resolved_context = build_agent_instruction_context(agent, context_vars)
        aggregator = None
        skill_manifests = None
        has_filesystem_runtime = False
        if isinstance(agent, McpInstructionCapable):
            agent.set_instruction_context(dict(resolved_context))
            aggregator = agent.aggregator
            skill_manifests = agent.skill_manifests
            has_filesystem_runtime = agent.has_filesystem_runtime

        resolved = await build_instruction(
            template,
            aggregator=aggregator,
            skill_manifests=skill_manifests,
            has_filesystem_runtime=has_filesystem_runtime,
            context=dict(resolved_context),
            source=getattr(agent, "name", None),
        )
        if resolved is None or resolved == template:
            continue

        set_instruction = getattr(agent, "set_instruction", None)
        if callable(set_instruction):
            set_instruction(resolved)
