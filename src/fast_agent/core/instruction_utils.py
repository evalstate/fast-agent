"""Shared helpers for instruction template resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.core.prompt_templates import apply_template_variables

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


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


def apply_instruction_context(
    agents: Iterable[object],
    context_vars: Mapping[str, str],
) -> None:
    if not context_vars:
        return

    for agent in agents:
        template = get_instruction_template(agent)
        if not template:
            continue

        resolved = apply_template_variables(template, context_vars)
        if resolved is None or resolved == template:
            continue

        set_instruction = getattr(agent, "set_instruction", None)
        if callable(set_instruction):
            set_instruction(resolved)
