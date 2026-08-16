"""Opt-in model tools for inspecting the fast-agent harness."""

from __future__ import annotations

import base64

from mcp_types import BlobResourceContents, ReadResourceResult, TextResourceContents

from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.commands.harness import execute_harness_command
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.internal_resources import read_internal_resource
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.tools.function_tool_loader import build_default_function_tool

SLASH_COMMAND_TOOL_NAME = "slash_command"
GET_RESOURCE_TOOL_NAME = "get_resource"
HARNESS_TOOL_NAMES = frozenset({SLASH_COMMAND_TOOL_NAME, GET_RESOURCE_TOOL_NAME})
HARNESS_TOOL_METADATA = {
    "fast_agent": {
        "builtin": "harness",
        "inherit_to_clone": False,
    }
}


def _is_harness_tool(agent: ToolAgent, name: str) -> bool:
    tool = agent._execution_tools.get(name)
    return tool is not None and tool.meta == HARNESS_TOOL_METADATA


def harness_tools_enabled(agent: object) -> bool:
    return isinstance(agent, ToolAgent) and all(
        _is_harness_tool(agent, name) for name in HARNESS_TOOL_NAMES
    )


def _resource_text(result: ReadResourceResult, *, max_chars: int = 4000) -> str:
    lines: list[str] = []
    for index, content in enumerate(result.contents, start=1):
        if isinstance(content, TextResourceContents):
            lines.extend((f"[{index}] text ({content.mime_type or 'unknown'})", content.text))
        elif isinstance(content, BlobResourceContents):
            try:
                decoded = base64.b64decode(content.blob)
                preview = decoded[:400].decode("utf-8", errors="replace")
            except (ValueError, TypeError):
                preview = "<binary blob>"
            lines.extend(
                (
                    f"[{index}] blob ({content.mime_type or 'unknown'}, {len(content.blob)} b64 chars)",
                    preview,
                )
            )
        elif text := get_text(content):
            lines.extend((f"[{index}] content", text))
    rendered = "\n".join(lines).strip()
    return rendered if len(rendered) <= max_chars else rendered[: max_chars - 1] + "…\n[truncated]"


async def _read_resource(agent: ToolAgent, uri: str, server_name: str | None = None) -> str:
    if uri.startswith("internal://"):
        return read_internal_resource(uri)
    result = await agent.get_resource(resource_uri=uri, namespace=server_name)
    return _resource_text(result)


def set_harness_tools(agent: object, enabled: bool | None = None) -> bool:
    """Enable or disable built-in harness tools on a compatible agent."""
    if not isinstance(agent, ToolAgent):
        return False
    if enabled is None:
        enabled = agent.config.harness_tools

    if not enabled or agent.config.tool_only:
        agent.config.harness_tools = enabled
        for name in HARNESS_TOOL_NAMES:
            if _is_harness_tool(agent, name):
                agent.remove_tool(name)
        return False

    for name in HARNESS_TOOL_NAMES:
        existing = agent._execution_tools.get(name)
        if existing is not None and existing.meta != HARNESS_TOOL_METADATA:
            raise AgentConfigError(f"Tool name '{name}' is reserved by fast-agent")

    agent.config.harness_tools = True
    skill_source_overrides: dict[str, str] = {}

    async def slash_command(command: str) -> str:
        return await execute_harness_command(
            agent,
            command,
            skill_source_overrides=skill_source_overrides,
        )

    async def get_resource(uri: str, server_name: str | None = None) -> str:
        return await _read_resource(agent, uri, server_name)

    if not _is_harness_tool(agent, SLASH_COMMAND_TOOL_NAME):
        agent.add_tool(
            build_default_function_tool(
                slash_command,
                name=SLASH_COMMAND_TOOL_NAME,
                description=(
                    "Execute an allow-listed fast-agent slash command, including model-managed "
                    "MCP servers and skills. "
                    "Use `/commands` for help or `/commands --json` for the machine-readable "
                    "command surface."
                ),
                metadata=HARNESS_TOOL_METADATA,
            )
        )
    if not _is_harness_tool(agent, GET_RESOURCE_TOOL_NAME):
        agent.add_tool(
            build_default_function_tool(
                get_resource,
                name=GET_RESOURCE_TOOL_NAME,
                description=(
                    "Read a bundled `internal://` resource or an attached MCP resource by URI."
                ),
                metadata=HARNESS_TOOL_METADATA,
            )
        )
    return True
