"""Config parsing helpers for plugin command actions."""

from __future__ import annotations

from typing import Any

from fast_agent.command_actions.models import PluginCommandActionSpec
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.utils.text import strip_casefold, strip_to_none


def normalize_plugin_command_name(raw_name: str) -> str:
    return strip_casefold(raw_name).lstrip("/")


def _required_string_field(
    raw_value: dict[str, Any],
    field_name: str,
    *,
    command_name: str,
    source: str,
) -> str:
    value = raw_value.get(field_name)
    if not isinstance(value, str) or (normalized := strip_to_none(value)) is None:
        raise AgentConfigError(
            f"Command action '{command_name}' requires a non-empty '{field_name}' in {source}"
        )
    return normalized


def _optional_string_field(
    raw_value: dict[str, Any],
    field_name: str,
    *,
    command_name: str,
    source: str,
) -> str | None:
    value = raw_value.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise AgentConfigError(
            f"Command action '{command_name}' field '{field_name}' must be a string in {source}"
        )
    return strip_to_none(value)


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
        if not isinstance(raw_name, str):
            raise AgentConfigError(f"Command action names must be strings in {source}")
        name = normalize_plugin_command_name(raw_name)
        if not name:
            raise AgentConfigError(f"Command action names must not be empty in {source}")
        if name in commands:
            raise AgentConfigError(
                f"Duplicate command action '{name}' after normalization in {source}"
            )
        if not isinstance(raw_value, dict):
            raise AgentConfigError(f"Command action '{name}' must be a dict in {source}")

        description = _required_string_field(
            raw_value,
            "description",
            command_name=name,
            source=source,
        )
        handler = _required_string_field(
            raw_value,
            "handler",
            command_name=name,
            source=source,
        )
        input_hint = _optional_string_field(
            raw_value,
            "input_hint",
            command_name=name,
            source=source,
        )
        key = _optional_string_field(
            raw_value,
            "key",
            command_name=name,
            source=source,
        )
        completer = _optional_string_field(
            raw_value,
            "completer",
            command_name=name,
            source=source,
        )

        commands[name] = PluginCommandActionSpec(
            name=name,
            description=description,
            handler=handler,
            input_hint=input_hint,
            key=key,
            completer=completer,
        )

    return commands
