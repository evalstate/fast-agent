"""Shared helpers to summarize tool metadata for rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from fast_agent.commands.model_capabilities import (
    resolve_web_fetch_enabled,
    resolve_web_fetch_supported,
    resolve_web_search_enabled,
    resolve_web_search_supported,
    resolve_x_search_enabled,
    resolve_x_search_supported,
)
from fast_agent.interfaces import (
    AgentBackedToolProvider,
    CardToolProvider,
    LlmCapableProtocol,
    SmartToolingCapable,
)
from fast_agent.mcp.common import is_namespaced_name
from fast_agent.tools.tool_sources import TOOL_SOURCE_LABELS, tool_source

if TYPE_CHECKING:
    from mcp.types import Tool

    from fast_agent.mcp.provider_management import ProviderManagedMCPState


@dataclass(slots=True)
class ToolSummary:
    name: str
    title: str | None
    description: str | None
    args: list[str] | None
    suffix: str | None
    template: str | None
    is_mcp: bool = False


@dataclass(slots=True)
class ProviderToolSummary:
    name: str
    enabled: bool | None
    description: str
    suffix: str = "provider-hosted"


@runtime_checkable
class _ProviderManagedMCPStateCapable(Protocol):
    @property
    def provider_managed_mcp_state(self) -> "ProviderManagedMCPState": ...


def _provider_managed_description(
    *,
    base_description: str,
    allowlist: tuple[str, ...] | None,
) -> str:
    if allowlist is None:
        return f"{base_description}; tools loaded by provider"
    if not allowlist:
        return f"{base_description}; no allowed tools configured"
    return base_description


def _provider_managed_tool_summaries(agent: object) -> list[ProviderToolSummary]:
    llm = agent.llm if isinstance(agent, LlmCapableProtocol) else None
    if llm is None:
        return []

    if not isinstance(llm, _ProviderManagedMCPStateCapable):
        return [
            ProviderToolSummary(
                name="provider_managed_mcp",
                enabled=None,
                description="Provider-managed MCP state is unavailable for this model.",
                suffix="provider-managed MCP",
            )
        ]

    state = llm.provider_managed_mcp_state
    summaries: list[ProviderToolSummary] = []
    for attachment in state.attachments:
        suffix = (
            "provider-managed connector"
            if attachment.connector_id is not None
            else "provider-managed MCP"
        )
        base_description = attachment.server_description
        if base_description is None:
            base_description = (
                f"OpenAI connector {attachment.connector_id}"
                if attachment.connector_id is not None
                else "Provider-managed MCP server"
            )

        allowlist = state.tool_allowlists.get(attachment.server_name)
        description = _provider_managed_description(
            base_description=base_description,
            allowlist=allowlist,
        )
        enabled = allowlist != ()

        if allowlist:
            summaries.extend(
                ProviderToolSummary(
                    name=f"{attachment.server_name}/{tool_name}",
                    enabled=True,
                    description=description,
                    suffix=suffix,
                )
                for tool_name in allowlist
            )
        else:
            summaries.append(
                ProviderToolSummary(
                    name=attachment.server_name,
                    enabled=enabled,
                    description=description,
                    suffix=suffix,
                )
            )

    return summaries


def build_provider_tool_summaries(agent: object) -> list[ProviderToolSummary]:
    llm = agent.llm if isinstance(agent, LlmCapableProtocol) else None
    candidates = (
        (
            "web_search",
            resolve_web_search_supported(llm),
            resolve_web_search_enabled(llm),
            "Provider-hosted web search tool.",
        ),
        (
            "web_fetch",
            resolve_web_fetch_supported(llm),
            resolve_web_fetch_enabled(llm),
            "Provider-hosted web fetch tool.",
        ),
        (
            "x_search",
            resolve_x_search_supported(llm),
            resolve_x_search_enabled(llm),
            "Provider-hosted X search tool.",
        ),
    )
    summaries = [
        ProviderToolSummary(name=name, enabled=enabled, description=description)
        for name, supported, enabled, description in candidates
        if supported and enabled
    ]
    summaries.extend(_provider_managed_tool_summaries(agent))
    return summaries


def _format_tool_args(schema: dict[str, Any] | None) -> list[str] | None:
    if not schema:
        return None

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None

    required = schema.get("required", [])
    if not isinstance(required, list):
        required = []

    arg_list: list[str] = []
    for prop_name in properties:
        arg_list.append(f"{prop_name}*" if prop_name in required else prop_name)

    return arg_list or None


def _tool_meta(tool: "Tool") -> dict[str, Any]:
    """Return MCP tool metadata, working around upstream model access quirks."""
    if tool.meta:
        return tool.meta
    dumped = tool.model_dump().get("meta")
    return dumped if isinstance(dumped, dict) else {}


def _collect_tool_name_sets(agent: object) -> tuple[set[str], set[str], set[str]]:
    card_tool_names = set(agent.card_tool_names) if isinstance(agent, CardToolProvider) else set()
    smart_tool_names = (
        set(agent.smart_tool_names) if isinstance(agent, SmartToolingCapable) else set()
    )
    agent_tool_names = (
        set(agent.agent_backed_tools.keys())
        if isinstance(agent, AgentBackedToolProvider)
        else set()
    )
    return card_tool_names, smart_tool_names, agent_tool_names


def build_tool_summaries(agent: object, tools: list[Tool]) -> list[ToolSummary]:
    card_tool_names, smart_tool_names, agent_tool_names = _collect_tool_name_sets(agent)
    child_agent_tool_names = agent_tool_names

    summaries: list[ToolSummary] = []

    for tool in tools:
        name = tool.name
        title = tool.title
        description = (tool.description or "").strip() or None
        meta = _tool_meta(tool)
        source = tool_source(tool)

        is_mcp = False
        suffix = None
        if name in smart_tool_names:
            suffix = "(Smart)"
        elif name in card_tool_names:
            suffix = "(Card Function)"
        elif name in child_agent_tool_names:
            suffix = "(Subagent)"
        elif source is not None:
            suffix = f"({TOOL_SOURCE_LABELS[source]})"
        elif name not in agent_tool_names and is_namespaced_name(name):
            is_mcp = True
            suffix = "(MCP)"

        if meta.get("openai/skybridgeEnabled"):
            suffix = f"{suffix} (Apps SDK)" if suffix else "(Apps SDK)"
        if meta.get("ui/appEnabled"):
            suffix = f"{suffix} (MCP App)" if suffix else "(MCP App)"

        args = _format_tool_args(tool.inputSchema)
        template = meta.get("ui/appTemplate") or meta.get("openai/skybridgeTemplate")

        summaries.append(
            ToolSummary(
                name=name,
                title=title,
                description=description,
                args=args,
                suffix=suffix,
                template=template,
                is_mcp=is_mcp,
            )
        )

    return sorted(summaries, key=lambda summary: summary.is_mcp)
