"""Slash command discovery rendering helpers."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from fast_agent.commands.command_catalog import COMMAND_SPECS, get_command_spec

if TYPE_CHECKING:
    from collections.abc import Collection

SCHEMA_VERSION = "1"


@dataclass(frozen=True, slots=True)
class DiscoveryRequest:
    """Parsed request for /commands rendering."""

    command_name: str | None
    as_json: bool


def parse_commands_discovery_arguments(arguments: str) -> DiscoveryRequest:
    """Parse /commands arguments into a request object."""

    trimmed = arguments.strip()
    if not trimmed:
        return DiscoveryRequest(command_name=None, as_json=False)

    try:
        tokens = shlex.split(trimmed)
    except ValueError as exc:
        raise ValueError(f"Invalid /commands arguments: {exc}") from exc

    command_name: str | None = None
    as_json = False

    for token in tokens:
        lowered = token.lower().strip()
        if lowered == "--json":
            as_json = True
            continue
        if lowered.startswith("--"):
            raise ValueError(f"Unknown /commands option: {token}")
        if command_name is not None:
            raise ValueError("Usage: /commands [<command>] [--json]")
        command_name = lowered

    return DiscoveryRequest(command_name=command_name, as_json=as_json)


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
            "usage": "/commands [<command>] [--json]",
            "actions": [],
            "examples": ["/commands", "/commands skills", "/commands --json"],
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
        {
            "name": "model",
            "summary": "Request behavior controls",
            "usage": "/model [reasoning|verbosity|web_search|web_fetch|help] <value>",
            "actions": [
                {"name": "reasoning", "summary": "set reasoning effort"},
                {"name": "verbosity", "summary": "set response length"},
                {"name": "web_search", "summary": "toggle web search"},
                {"name": "web_fetch", "summary": "toggle web fetch"},
                {"name": "help", "summary": "show model usage"},
            ],
            "examples": ["/model reasoning high", "/model web_search on"],
        },
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
        {
            "name": "check",
            "summary": "Run check diagnostics",
            "usage": "/check [args]",
            "actions": [],
            "examples": ["/check", "/check --for-model"],
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
        actions: list[dict[str, object]] = []
        for action in spec.actions:
            action_payload: dict[str, object] = {
                "name": action.action,
                "summary": action.help,
                "aliases": list(action.aliases),
            }
            if action.usage:
                action_payload["usage"] = action.usage
            if action.examples:
                action_payload["examples"] = list(action.examples)
            actions.append(action_payload)

        return {
            "name": spec.command,
            "summary": spec.summary,
            "usage": spec.usage,
            "actions": actions,
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


def render_commands_index_markdown(*, command_names: Collection[str] | None = None) -> str:
    """Render markdown for /commands index."""

    allowed = {name.lower() for name in command_names} if command_names is not None else None
    lines = ["# commands", "", "Command map:"]
    for entry in _discovery_top_level_catalog():
        name = str(entry["name"])
        if allowed is not None and name not in allowed:
            continue

        summary = str(entry["summary"])
        lines.append(f"- `/{name}` — {summary}")

        actions = entry.get("actions")
        if not isinstance(actions, list) or not actions:
            continue

        action_names: list[str] = []
        for action in actions:
            if isinstance(action, str):
                action_names.append(action)
                continue
            if isinstance(action, dict):
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
            "- `/commands --json` for machine-readable map",
        ]
    )
    return "\n".join(lines)


def render_command_detail_markdown(command_name: str) -> str | None:
    """Render markdown for /commands <name>."""

    detail = _build_command_detail(command_name)
    if detail is None:
        return None

    lines = [f"# commands {detail['name']}", "", str(detail["summary"]), "", f"Usage: `{detail['usage']}`"]

    actions = detail.get("actions")
    if isinstance(actions, list) and actions:
        lines.extend(["", "Actions:"])
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_map = cast("dict[str, object]", action)
            action_name = str(action_map.get("name", ""))
            action_summary = str(action_map.get("summary", "")).strip()
            aliases = action_map.get("aliases")
            alias_text = ""
            if isinstance(aliases, list) and aliases:
                alias_text = f" (aliases: {', '.join(str(alias) for alias in aliases)})"

            if action_summary:
                lines.append(f"- `{action_name}` — {action_summary}{alias_text}")
            else:
                lines.append(f"- `{action_name}`{alias_text}")

            usage = action_map.get("usage")
            if usage:
                lines.append(f"  - usage: `{usage}`")
            examples = action_map.get("examples")
            if isinstance(examples, list):
                for example in examples:
                    lines.append(f"  - example: `{example}`")

    examples = detail.get("examples")
    if isinstance(examples, list) and examples:
        lines.extend(["", "Examples:"])
        for example in examples:
            lines.append(f"- `{example}`")

    lines.append("")
    lines.append(f"JSON: `/commands {detail['name']} --json`")
    return "\n".join(lines)


def render_commands_json(
    *,
    command_name: str | None = None,
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
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "command_index",
            "commands": commands,
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    detail = _build_command_detail(command_name)
    if detail is None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "error",
            "error": f"Unknown command: {command_name}",
            "suggestions": command_discovery_names(),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    if allowed is not None and str(detail["name"]) not in allowed:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "error",
            "error": f"Command '/{detail['name']}' is not available in this context.",
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "command_detail",
        "command": detail,
    }
    return json.dumps(payload, indent=2, sort_keys=True)
