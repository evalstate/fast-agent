#!/usr/bin/env python3
"""Generate plugin command API documentation from source signatures."""

from __future__ import annotations

import inspect
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUTPUT = ROOT / "docs" / "docs" / "_generated" / "plugin_api.md"


def _load_api() -> tuple[type[Any], type[Any], type[Any], type[Any]]:
    sys.path.insert(0, str(SRC))
    from fast_agent.command_actions.models import (
        PluginCommandActionContext,
        PluginCommandActionResult,
        PluginCommandActionSpec,
    )
    from fast_agent.command_actions.runtime import PluginRuntime

    return (
        PluginCommandActionSpec,
        PluginCommandActionResult,
        PluginCommandActionContext,
        PluginRuntime,
    )


FIELD_DESCRIPTIONS = {
    "PluginCommandActionSpec.name": "Slash command name without the leading `/`.",
    "PluginCommandActionSpec.description": "Human-readable help text for listings and `/plugins` output.",
    "PluginCommandActionSpec.handler": "Python handler reference, for example `./commands.py:run`.",
    "PluginCommandActionSpec.input_hint": "Optional placeholder shown by surfaces that support command input hints.",
    "PluginCommandActionSpec.key": "Optional key binding metadata.",
    "PluginCommandActionResult.message": "Plain text shown in the command output.",
    "PluginCommandActionResult.markdown": "Markdown output rendered by the UI.",
    "PluginCommandActionResult.buffer_prefill": "Draft text inserted into the user's input buffer.",
    "PluginCommandActionResult.switch_agent": "Switch the active TUI agent after the command.",
    "PluginCommandActionResult.refresh_agents": "Refresh agent/card state after the command.",
    "PluginCommandActionContext.command_name": "Slash command name being executed.",
    "PluginCommandActionContext.arguments": "Raw text after the slash command.",
    "PluginCommandActionContext.agent": "Active agent surface exposed to the command.",
    "PluginCommandActionContext.settings": "Resolved fast-agent settings, when available.",
    "PluginCommandActionContext.session_cwd": "Working directory for the interactive session, when available.",
    "PluginCommandActionContext.runtime": "Optional live-runtime capabilities.",
    "PluginCommandActionContext.is_tui": "True when the command is running in the TUI surface.",
    "PluginCommandActionContext.is_acp": "True when the command is running in the ACP surface.",
}

CONTEXT_HELPERS = {
    "agent_name": "Active agent name.",
    "context": "Current agent context, when available.",
    "message_history": "Current agent message history.",
    "agent_registry": "Registered agents, when available.",
    "load_message_history": "Replace the active agent's message history.",
    "get_agent": "Look up another registered agent.",
    "mark_user_adjusted": "Mark a message as user-adjusted in the audit channel.",
}

RUNTIME_DESCRIPTIONS = {
    "attach_mcp_server": "Attach an MCP server to a running MCP-capable agent and refresh instructions.",
    "detach_mcp_server": "Detach an MCP server from a running MCP-capable agent and refresh instructions.",
    "list_attached_mcp_servers": "List MCP servers attached to a running MCP-capable agent.",
    "list_configured_detached_mcp_servers": "List configured MCP servers that are not currently attached.",
}


def _type_name(annotation: object) -> str:
    text = str(annotation)
    text = text.replace("typing.", "")
    text = text.replace("NoneType", "None")
    text = text.replace('"', "")
    text = text.replace("'", "")
    return text


def _field_table(cls: type[object]) -> list[str]:
    lines = [
        "| Field | Type | Description |",
        "|-------|------|-------------|",
    ]
    for field in fields(cls):
        key = f"{cls.__name__}.{field.name}"
        lines.append(
            f"| `{field.name}` | `{_type_name(field.type)}` | "
            f"{FIELD_DESCRIPTIONS.get(key, '')} |"
        )
    return lines


def _signature(member: object) -> str:
    signature = str(inspect.signature(member))
    if signature.startswith("(self, *, "):
        signature = "(*, " + signature.removeprefix("(self, *, ")
    elif signature.startswith("(self, "):
        signature = "(" + signature.removeprefix("(self, ")
    elif signature == "(self)":
        signature = "()"
    signature = signature.replace('"', "")
    signature = signature.replace("'", "")
    return signature


def _context_helper_table() -> list[str]:
    _, _, plugin_context, _ = _load_api()
    lines = [
        "| API | Signature | Description |",
        "|-----|-----------|-------------|",
    ]
    for name, description in CONTEXT_HELPERS.items():
        member = plugin_context.__dict__[name]
        signature = "property"
        if not isinstance(member, property):
            signature = _signature(member)
        lines.append(f"| `ctx.{name}` | `{signature}` | {description} |")
    return lines


def _runtime_table() -> list[str]:
    _, _, _, plugin_runtime = _load_api()
    lines = [
        "| API | Signature | Description |",
        "|-----|-----------|-------------|",
    ]
    for name, description in RUNTIME_DESCRIPTIONS.items():
        member = plugin_runtime.__dict__[name]
        lines.append(f"| `{name}` | `{_signature(member)}` | {description} |")
    return lines


def build_content() -> str:
    plugin_spec, plugin_result, plugin_context, _ = _load_api()
    lines = [
        "<!--",
        "  Generated by docs/generate_plugin_api_docs.py.",
        "  Do not edit this file directly.",
        "-->",
        "",
        "Handlers are async Python callables with this signature:",
        "",
        "```python",
        "async def handler(",
        "    ctx: PluginCommandActionContext,",
        ") -> PluginCommandActionResult | str | None:",
        "    ...",
        "```",
        "",
        "Returning a plain string is shorthand for `PluginCommandActionResult(message=...)`.",
        "Return `None` for no visible output.",
        "",
        "### Manifest Command Fields",
        "",
        *_field_table(plugin_spec),
        "",
        "### Result Fields",
        "",
        *_field_table(plugin_result),
        "",
        "### Context Fields",
        "",
        *_field_table(plugin_context),
        "",
        "### Context Helpers",
        "",
        *_context_helper_table(),
        "",
        "### Runtime API",
        "",
        "Runtime capabilities are optional because not every surface can support live changes.",
        "",
        "```python",
        "if ctx.runtime is not None:",
        "    attached = await ctx.runtime.list_attached_mcp_servers()",
        "```",
        "",
        *_runtime_table(),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build_content(), encoding="utf-8")
    print(f"Generated {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
