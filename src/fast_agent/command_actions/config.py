"""Config parsing helpers for plugin command actions."""

from __future__ import annotations

from typing import Any

from fast_agent.command_actions.models import PluginCommandActionSpec
from fast_agent.core.exceptions import AgentConfigError


def parse_plugin_command_action_specs(
    raw_commands: Any,
    *,
    source: str,
) -> dict[str, PluginCommandActionSpec] | None:
    if raw_commands is None:
        return None
    if not isinstance(raw_commands, dict):
        raise AgentConfigError(f"'commands' must be a dict in {source}")

    commands: dict[str, PluginCommandActionSpec] = {}
    for raw_name, raw_value in raw_commands.items():
        name = str(raw_name).strip().lstrip("/")
        if not name:
            raise AgentConfigError(f"Command action names must not be empty in {source}")
        if not isinstance(raw_value, dict):
            raise AgentConfigError(f"Command action '{name}' must be a dict in {source}")

        description = raw_value.get("description")
        if not isinstance(description, str) or not description.strip():
            raise AgentConfigError(
                f"Command action '{name}' requires a non-empty 'description' in {source}"
            )

        handler = raw_value.get("handler")
        if not isinstance(handler, str) or not handler.strip():
            raise AgentConfigError(
                f"Command action '{name}' requires a non-empty 'handler' in {source}"
            )

        input_hint = raw_value.get("input_hint")
        if input_hint is not None and not isinstance(input_hint, str):
            raise AgentConfigError(
                f"Command action '{name}' field 'input_hint' must be a string in {source}"
            )

        key = raw_value.get("key")
        if key is not None and not isinstance(key, str):
            raise AgentConfigError(
                f"Command action '{name}' field 'key' must be a string in {source}"
            )

        commands[name] = PluginCommandActionSpec(
            name=name,
            description=description.strip(),
            handler=handler.strip(),
            input_hint=input_hint,
            key=key,
        )

    return commands
