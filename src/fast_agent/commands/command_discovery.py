"""Slash command discovery rendering helpers."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from fast_agent.commands.command_catalog import (
    COMMAND_SPECS,
    CommandActionSpec,
    get_command_action_spec,
    get_command_spec,
)
from fast_agent.commands.session_export_help import (
    SESSION_EXPORT_EXAMPLES,
    build_session_export_action_detail,
)

if TYPE_CHECKING:
    from collections.abc import Collection

SCHEMA_VERSION = "1"


@dataclass(frozen=True, slots=True)
class DiscoveryRequest:
    """Parsed request for /commands rendering."""

    command_name: str | None
    action_name: str | None
    as_json: bool


def parse_commands_discovery_arguments(arguments: str) -> DiscoveryRequest:
    """Parse /commands arguments into a request object."""

    trimmed = arguments.strip()
    if not trimmed:
        return DiscoveryRequest(command_name=None, action_name=None, as_json=False)

    try:
        tokens = shlex.split(trimmed)
    except ValueError as exc:
        raise ValueError(f"Invalid /commands arguments: {exc}") from exc

    command_name: str | None = None
    action_name: str | None = None
    as_json = False

    for token in tokens:
        lowered = token.lower().strip()
        if lowered == "--json":
            as_json = True
            continue
        if lowered.startswith("--"):
            raise ValueError(f"Unknown /commands option: {token}")
        if command_name is None:
            command_name = lowered
            continue
        if action_name is None:
            action_name = lowered
            continue
        raise ValueError("Usage: /commands [<command> [<action>]] [--json]")

    return DiscoveryRequest(command_name=command_name, action_name=action_name, as_json=as_json)


def _action_payload_from_catalog(action: CommandActionSpec) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": action.action,
        "summary": action.help,
        "aliases": list(action.aliases),
    }
    if action.usage:
        payload["usage"] = action.usage
    if action.examples:
        payload["examples"] = list(action.examples)
    if action.arguments:
        payload["arguments"] = [
            {
                "name": item.name,
                "summary": item.summary,
                "value_name": item.value_name,
                "required": item.required,
            }
            for item in action.arguments
        ]
    if action.options:
        payload["options"] = [
            {
                "name": item.name,
                "summary": item.summary,
                "value_name": item.value_name,
                "aliases": list(item.aliases),
            }
            for item in action.options
        ]
    if action.notes:
        payload["notes"] = list(action.notes)
    return payload


def _session_detail_entry() -> dict[str, object]:
    return {
        "name": "session",
        "summary": "List, manage, or export sessions",
        "usage": "/session [list|new|resume|title|fork|delete|pin|export|help] [args]",
        "actions": [
            {"name": "list", "summary": "show recent sessions", "usage": "/session list"},
            {"name": "new", "summary": "create a new session", "usage": "/session new [title]"},
            {
                "name": "resume",
                "summary": "resume a saved session",
                "usage": "/session resume [id|number]",
            },
            {
                "name": "title",
                "summary": "set the current session title",
                "usage": "/session title <text>",
            },
            {"name": "fork", "summary": "fork the current session", "usage": "/session fork [title]"},
            {
                "name": "delete",
                "summary": "delete one or all sessions",
                "usage": "/session delete <id|number|all>",
            },
            {
                "name": "pin",
                "summary": "pin or unpin a session",
                "usage": "/session pin [on|off|id|number]",
            },
            build_session_export_action_detail(),
            {"name": "help", "summary": "show session usage"},
        ],
        "examples": [
            "/session list",
            "/session resume 1",
            *SESSION_EXPORT_EXAMPLES[:2],
        ],
    }


def _discovery_top_level_catalog() -> tuple[dict[str, object], ...]:
    families: list[dict[str, object]] = []
    for spec in COMMAND_SPECS:
        families.append(
            {
                "name": spec.command,
                "summary": spec.summary,
                "usage": spec.usage,
                "actions": [action.action for action in spec.actions],
                "examples": list(spec.examples),
            }
        )

    extras: tuple[dict[str, object], ...] = (
        {
            "name": "commands",
            "summary": "Command map + help",
            "usage": "/commands [<command> [<action>]] [--json]",
            "actions": [],
            "examples": [
                "/commands",
                "/commands skills",
                "/commands skills add",
                "/commands --json",
            ],
        },
        {
            "name": "mcp",
            "summary": "Runtime MCP control",
            "usage": "/mcp [list|connect|disconnect|reconnect|session|help] [args]",
            "actions": [
                {"name": "list", "summary": "show attached servers"},
                {"name": "connect", "summary": "attach runtime server"},
                {"name": "disconnect", "summary": "detach runtime server"},
                {"name": "reconnect", "summary": "restart server session"},
                {"name": "session", "summary": "manage cookie sessions"},
                {"name": "help", "summary": "show mcp usage"},
            ],
            "examples": ["/mcp list", "/mcp connect <target>", "/mcp session list"],
        },
        _session_detail_entry(),
        {
            "name": "tools",
            "summary": "List callable tools",
            "usage": "/tools",
            "actions": [],
            "examples": ["/tools"],
        },
        {
            "name": "prompts",
            "summary": "List prompt templates",
            "usage": "/prompts",
            "actions": [],
            "examples": ["/prompts"],
        },
        {
            "name": "usage",
            "summary": "Token/cost summary",
            "usage": "/usage",
            "actions": [],
            "examples": ["/usage"],
        },
        {
            "name": "system",
            "summary": "Show resolved instruction",
            "usage": "/system",
            "actions": [],
            "examples": ["/system"],
        },
        {
            "name": "markdown",
            "summary": "Show markdown buffer",
            "usage": "/markdown",
            "actions": [],
            "examples": ["/markdown"],
        },
    )

    families.extend(extras)
    families.sort(key=lambda item: str(item["name"]))
    return tuple(families)


def command_discovery_names() -> tuple[str, ...]:
    """Return discoverable command names for /commands."""

    return tuple(str(item["name"]) for item in _discovery_top_level_catalog())


def _build_command_detail(name: str) -> dict[str, object] | None:
    normalized = name.strip().lower()
    spec = get_command_spec(normalized)
    if spec is not None:
        return {
            "name": spec.command,
            "summary": spec.summary,
            "usage": spec.usage,
            "actions": [_action_payload_from_catalog(action) for action in spec.actions],
            "examples": list(spec.examples),
        }

    for entry in _discovery_top_level_catalog():
        if str(entry["name"]) != normalized:
            continue
        detail = dict(entry)
        actions = detail.get("actions")
        if isinstance(actions, list) and actions and isinstance(actions[0], str):
            detail["actions"] = [{"name": str(action), "summary": ""} for action in actions]
        return detail
    return None


def _build_command_action_detail(command_name: str, action_name: str) -> dict[str, object] | None:
    detail = _build_command_detail(command_name)
    if detail is None:
        return None

    normalized_action = action_name.strip().lower()
    actions = detail.get("actions")
    if not isinstance(actions, list):
        return None
    for action in actions:
        if not isinstance(action, dict):
            continue
        action_map = cast("dict[str, object]", action)
        name = action_map.get("name")
        if not isinstance(name, str):
            continue
        if name.lower() == normalized_action:
            return action_map
        aliases = action_map.get("aliases")
        if isinstance(aliases, list) and normalized_action in {
            alias.lower() for alias in aliases if isinstance(alias, str)
        }:
            return action_map

    action_spec = get_command_action_spec(command_name, normalized_action)
    if action_spec is None:
        return None
    return _action_payload_from_catalog(action_spec)


def _render_action_metadata(lines: list[str], action_map: dict[str, object], *, indent: str) -> None:
    usage = action_map.get("usage")
    if usage:
        lines.append(f"{indent}- usage: `{usage}`")

    arguments = action_map.get("arguments")
    if isinstance(arguments, list) and arguments:
        lines.append(f"{indent}- arguments:")
        for argument in arguments:
            if not isinstance(argument, dict):
                continue
            argument_map = cast("dict[str, object]", argument)
            argument_name = str(argument_map.get("name", "")).strip()
            if not argument_name:
                continue
            value_name = argument_map.get("value_name")
            label = f"`{argument_name}`"
            if isinstance(value_name, str) and value_name:
                label = f"`{argument_name}` (`{value_name}`)"
            argument_summary = str(argument_map.get("summary", "")).strip()
            if argument_summary:
                lines.append(f"{indent}  - {label} — {argument_summary}")
            else:
                lines.append(f"{indent}  - {label}")

    options = action_map.get("options")
    if isinstance(options, list) and options:
        lines.append(f"{indent}- options:")
        for option in options:
            if not isinstance(option, dict):
                continue
            option_map = cast("dict[str, object]", option)
            option_name = str(option_map.get("name", "")).strip()
            if not option_name:
                continue
            labels = [f"`{option_name}`"]
            aliases = option_map.get("aliases")
            if isinstance(aliases, list):
                labels.extend(f"`{alias}`" for alias in aliases if isinstance(alias, str) and alias)
            value_name = option_map.get("value_name")
            if isinstance(value_name, str) and value_name:
                labels[0] = f"`{option_name} {value_name}`"
            option_summary = str(option_map.get("summary", "")).strip()
            if option_summary:
                lines.append(f"{indent}  - {', '.join(labels)} — {option_summary}")
            else:
                lines.append(f"{indent}  - {', '.join(labels)}")

    notes = action_map.get("notes")
    if isinstance(notes, list) and notes:
        lines.append(f"{indent}- notes:")
        for note in notes:
            if isinstance(note, str) and note:
                lines.append(f"{indent}  - {note}")

    examples = action_map.get("examples")
    if isinstance(examples, list):
        for example in examples:
            lines.append(f"{indent}- example: `{example}`")


def render_commands_index_markdown(*, command_names: Collection[str] | None = None) -> str:
    """Render markdown for /commands index."""

    allowed = {name.lower() for name in command_names} if command_names is not None else None
    lines = ["# commands", "", "Command map:"]
    for entry in _discovery_top_level_catalog():
        name = str(entry["name"])
        if allowed is not None and name not in allowed:
            continue

        lines.append(f"- `/{name}` — {entry['summary']}")
        actions = entry.get("actions")
        if not isinstance(actions, list) or not actions:
            continue

        action_names: list[str] = []
        for action in actions:
            if isinstance(action, str):
                action_names.append(action)
                continue
            if not isinstance(action, dict):
                continue
            action_map = cast("dict[str, object]", action)
            action_name = action_map.get("name")
            if isinstance(action_name, str) and action_name:
                action_names.append(action_name)
        if action_names:
            lines.append(f"  - {', '.join(action_names)}")

    lines.extend(
        [
            "",
            "Next:",
            "- `/commands <name>` for detailed help",
            "- `/commands <name> <action>` for action-level help",
            "- `/commands --json` for machine-readable map",
        ]
    )
    return "\n".join(lines)


def render_command_detail_markdown(command_name: str, action_name: str | None = None) -> str | None:
    """Render markdown for /commands <name> [<action>]."""

    if action_name is not None:
        action = _build_command_action_detail(command_name, action_name)
        detail = _build_command_detail(command_name)
        if action is None or detail is None:
            return None

        action_heading = str(action.get("name", action_name))
        lines = [
            f"# commands {detail['name']} {action_heading}",
            "",
            str(action.get("summary", "")).strip() or f"`/{detail['name']}` action",
        ]
        usage = action.get("usage")
        if usage:
            lines.extend(["", f"Usage: `{usage}`", f"Usage: {usage}"])
        _render_action_metadata(lines, action, indent="")
        lines.extend(["", f"JSON: `/commands {detail['name']} {action_heading} --json`"])
        return "\n".join(lines)

    detail = _build_command_detail(command_name)
    if detail is None:
        return None

    lines = [
        f"# commands {detail['name']}",
        "",
        str(detail["summary"]),
        "",
        f"Usage: `{detail['usage']}`",
        f"Usage: {detail['usage']}",
    ]
    actions = detail.get("actions")
    if isinstance(actions, list) and actions:
        lines.extend(["", "Actions:"])
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_map = cast("dict[str, object]", action)
            action_name = str(action_map.get("name", "")).strip()
            if not action_name:
                continue
            action_summary = str(action_map.get("summary", "")).strip()
            aliases = action_map.get("aliases")
            alias_text = ""
            if isinstance(aliases, list) and aliases:
                alias_text = f" (aliases: {', '.join(str(alias) for alias in aliases)})"
            if action_summary:
                lines.append(f"- `{action_name}` — {action_summary}{alias_text}")
            else:
                lines.append(f"- `{action_name}`{alias_text}")
            _render_action_metadata(lines, action_map, indent="  ")

    examples = detail.get("examples")
    if isinstance(examples, list) and examples:
        lines.extend(["", "Examples:"])
        for example in examples:
            lines.append(f"- `{example}`")

    lines.extend(
        [
            "",
            f"JSON: `/commands {detail['name']} --json`",
        ]
    )
    return "\n".join(lines)


def render_commands_json(
    *,
    command_name: str | None = None,
    action_name: str | None = None,
    command_names: Collection[str] | None = None,
) -> str:
    """Render JSON payload for /commands outputs."""

    allowed = {name.lower() for name in command_names} if command_names is not None else None

    if command_name is None:
        commands = [
            item
            for item in _discovery_top_level_catalog()
            if allowed is None or str(item["name"]) in allowed
        ]
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "command_index",
                "commands": commands,
            },
            indent=2,
            sort_keys=True,
        )

    detail = _build_command_detail(command_name)
    if detail is None:
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "error",
                "error": f"Unknown command: {command_name}",
                "suggestions": command_discovery_names(),
            },
            indent=2,
            sort_keys=True,
        )

    if allowed is not None and str(detail["name"]) not in allowed:
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "error",
                "error": f"Command '/{detail['name']}' is not available in this context.",
            },
            indent=2,
            sort_keys=True,
        )

    if action_name is None:
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "command_detail",
                "command": detail,
            },
            indent=2,
            sort_keys=True,
        )

    action = _build_command_action_detail(command_name, action_name)
    if action is None:
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "error",
                "error": f"Unknown action '{action_name}' for '/{detail['name']}'.",
            },
            indent=2,
            sort_keys=True,
        )

    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "command_action_detail",
            "command": {"name": detail["name"], "summary": detail["summary"], "usage": detail["usage"]},
            "action": action,
        },
        indent=2,
        sort_keys=True,
    )


def render_direct_command_help(command_name: str, arguments: str | None) -> str | None:
    """Render action-specific help for direct slash commands when requested."""

    trimmed = (arguments or "").strip()
    if not trimmed:
        return None

    try:
        tokens = shlex.split(trimmed)
    except ValueError:
        return None

    if not tokens:
        return None

    first = tokens[0].lower()
    if first in {"help", "--help", "-h"}:
        return render_command_detail_markdown(command_name)

    if len(tokens) >= 2 and tokens[-1].lower() in {"--help", "-h"}:
        return render_command_detail_markdown(command_name, first)

    return None
