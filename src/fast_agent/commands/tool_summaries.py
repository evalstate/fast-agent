"""Shared helpers to summarize tool metadata for rendering."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fast_agent.commands.model_capabilities import (
    resolve_web_fetch_enabled,
    resolve_web_fetch_supported,
    resolve_web_search_enabled,
    resolve_web_search_supported,
    resolve_x_search_enabled,
    resolve_x_search_supported,
)
from fast_agent.commands.summary_utils import JsonObject, json_object, optional_string
from fast_agent.interfaces import (
    AgentBackedToolProvider,
    CardToolProvider,
    FastAgentLLMProtocol,
    LlmCapableProtocol,
    SmartToolingCapable,
)
from fast_agent.mcp.common import is_namespaced_name
from fast_agent.tools.tool_sources import TOOL_SOURCE_LABELS, ToolSource, tool_source
from fast_agent.utils.action_normalization import enabled_disabled_label
from fast_agent.utils.text import strip_to_none

PROVIDER_HOSTED_SUFFIX = "provider-hosted"
PROVIDER_MANAGED_CONNECTOR_SUFFIX = "provider-managed connector"
PROVIDER_MANAGED_MCP_SUFFIX = "provider-managed MCP"

if TYPE_CHECKING:
    from mcp.types import Tool

    from fast_agent.interfaces import FastAgentLLMProtocol
    from fast_agent.mcp.provider_management import (
        ProviderManagedMCPAttachment,
        ProviderManagedMCPState,
    )


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
    suffix: str = PROVIDER_HOSTED_SUFFIX


@dataclass(frozen=True, slots=True)
class _ToolNameSets:
    card: set[str]
    smart: set[str]
    agent_backed: set[str]

    def classification_candidates(self) -> tuple[tuple[set[str], str], ...]:
        return (
            (self.smart, "(Smart)"),
            (self.card, "(Card Function)"),
            (self.agent_backed, "(Subagent)"),
        )


@dataclass(frozen=True, slots=True)
class _ProviderToolDescriptor:
    name: str
    supported: "Callable[[FastAgentLLMProtocol | None], bool]"
    enabled: "Callable[[FastAgentLLMProtocol | None], bool]"
    description: str


PROVIDER_HOSTED_TOOL_DESCRIPTORS: tuple[_ProviderToolDescriptor, ...] = (
    _ProviderToolDescriptor(
        name="web_search",
        supported=resolve_web_search_supported,
        enabled=resolve_web_search_enabled,
        description="Provider-hosted web search tool.",
    ),
    _ProviderToolDescriptor(
        name="web_fetch",
        supported=resolve_web_fetch_supported,
        enabled=resolve_web_fetch_enabled,
        description="Provider-hosted web fetch tool.",
    ),
    _ProviderToolDescriptor(
        name="x_search",
        supported=resolve_x_search_supported,
        enabled=resolve_x_search_enabled,
        description="Provider-hosted X search tool.",
    ),
)


@dataclass(frozen=True, slots=True)
class _ToolClassification:
    suffix: str | None
    is_mcp: bool = False


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


def provider_tool_state_label(enabled: bool | None) -> str:
    if enabled is None:
        return "Unknown"
    return enabled_disabled_label(enabled)


def provider_tool_status_label(summary: ProviderToolSummary) -> str:
    return f"{summary.suffix}, {provider_tool_state_label(summary.enabled)}"


def _provider_managed_summary(
    *,
    name: str,
    enabled: bool,
    description: str,
    suffix: str,
) -> ProviderToolSummary:
    return ProviderToolSummary(
        name=name,
        enabled=enabled,
        description=description,
        suffix=suffix,
    )


def _provider_managed_attachment_summaries(
    attachment: "ProviderManagedMCPAttachment",
    *,
    allowlist: tuple[str, ...] | None,
) -> list[ProviderToolSummary]:
    suffix = (
        PROVIDER_MANAGED_CONNECTOR_SUFFIX
        if attachment.connector_id is not None
        else PROVIDER_MANAGED_MCP_SUFFIX
    )
    base_description = optional_string(attachment.server_description)
    if base_description is None:
        base_description = (
            f"OpenAI connector {attachment.connector_id}"
            if attachment.connector_id is not None
            else "Provider-managed MCP server"
        )

    description = _provider_managed_description(
        base_description=base_description,
        allowlist=allowlist,
    )

    if allowlist:
        return [
            _provider_managed_summary(
                name=f"{attachment.server_name}/{tool_name}",
                enabled=True,
                description=description,
                suffix=suffix,
            )
            for tool_name in allowlist
        ]

    return [
        _provider_managed_summary(
            name=attachment.server_name,
            enabled=allowlist != (),
            description=description,
            suffix=suffix,
        )
    ]


def _agent_llm(agent: object) -> FastAgentLLMProtocol | None:
    if isinstance(agent, LlmCapableProtocol):
        return agent.llm
    return None


def _provider_managed_tool_summaries(agent: object) -> list[ProviderToolSummary]:
    llm = _agent_llm(agent)
    if llm is None:
        return []

    if not isinstance(llm, _ProviderManagedMCPStateCapable):
        return [
            ProviderToolSummary(
                name="provider_managed_mcp",
                enabled=None,
                description="Provider-managed MCP state is unavailable for this model.",
                suffix=PROVIDER_MANAGED_MCP_SUFFIX,
            )
        ]

    state = llm.provider_managed_mcp_state
    summaries: list[ProviderToolSummary] = []
    for attachment in state.attachments:
        summaries.extend(
            _provider_managed_attachment_summaries(
                attachment,
                allowlist=state.tool_allowlists.get(attachment.server_name),
            )
        )

    return summaries


def _provider_hosted_tool_summary(
    descriptor: _ProviderToolDescriptor,
    llm: FastAgentLLMProtocol | None,
) -> ProviderToolSummary | None:
    if not descriptor.supported(llm):
        return None

    enabled = descriptor.enabled(llm)
    if not enabled:
        return None
    return ProviderToolSummary(
        name=descriptor.name,
        enabled=enabled,
        description=descriptor.description,
    )


def build_provider_tool_summaries(agent: object) -> list[ProviderToolSummary]:
    llm = _agent_llm(agent)
    summaries = []
    for descriptor in PROVIDER_HOSTED_TOOL_DESCRIPTORS:
        summary = _provider_hosted_tool_summary(descriptor, llm)
        if summary is not None:
            summaries.append(summary)
    summaries.extend(
        summary for summary in _provider_managed_tool_summaries(agent) if summary.enabled is True
    )
    return summaries


def _string_set(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def _format_tool_args(schema: Mapping[str, object] | None) -> list[str] | None:
    if not schema:
        return None

    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return None

    required = _string_set(schema.get("required"))

    arg_list = [
        f"{prop_name}*" if prop_name in required else prop_name
        for prop_name in properties
        if isinstance(prop_name, str)
    ]

    return arg_list or None


def _tool_meta(tool: "Tool") -> JsonObject:
    """Return MCP tool metadata, working around upstream model access quirks."""
    if tool.meta:
        return json_object(tool.meta)
    return json_object(tool.model_dump().get("meta"))


def _collect_tool_name_sets(agent: object) -> _ToolNameSets:
    card_tool_names = set(agent.card_tool_names) if isinstance(agent, CardToolProvider) else set()
    smart_tool_names = (
        set(agent.smart_tool_names) if isinstance(agent, SmartToolingCapable) else set()
    )
    agent_tool_names = (
        set(agent.agent_backed_tools.keys())
        if isinstance(agent, AgentBackedToolProvider)
        else set()
    )
    return _ToolNameSets(
        card=card_tool_names, smart=smart_tool_names, agent_backed=agent_tool_names
    )


def _classify_tool(
    name: str,
    source: ToolSource | None,
    tool_name_sets: _ToolNameSets,
) -> _ToolClassification:
    for names, suffix in tool_name_sets.classification_candidates():
        if name in names:
            return _ToolClassification(suffix=suffix)
    if source is not None:
        return _ToolClassification(suffix=f"({TOOL_SOURCE_LABELS[source]})")
    if is_namespaced_name(name):
        return _ToolClassification(suffix="(MCP)", is_mcp=True)
    return _ToolClassification(suffix=None)


def _append_tool_suffix(suffix: str | None, label: str) -> str:
    return f"{suffix} {label}" if suffix else label


def _append_app_tool_suffixes(suffix: str | None, meta: Mapping[str, object]) -> str | None:
    if meta.get("openai/skybridgeEnabled"):
        suffix = _append_tool_suffix(suffix, "(Apps SDK)")
    if meta.get("ui/appEnabled"):
        suffix = _append_tool_suffix(suffix, "(MCP App)")
    return suffix


def build_tool_summaries(agent: object, tools: list[Tool]) -> list[ToolSummary]:
    tool_name_sets = _collect_tool_name_sets(agent)

    summaries: list[ToolSummary] = []

    for tool in tools:
        name = tool.name
        title = tool.title
        description = strip_to_none(tool.description)
        meta = _tool_meta(tool)
        source = tool_source(tool)
        classification = _classify_tool(name, source, tool_name_sets)

        suffix = classification.suffix
        suffix = _append_app_tool_suffixes(suffix, meta)

        args = _format_tool_args(tool.inputSchema)
        template = optional_string(meta.get("ui/appTemplate")) or optional_string(
            meta.get("openai/skybridgeTemplate")
        )

        summaries.append(
            ToolSummary(
                name=name,
                title=title,
                description=description,
                args=args,
                suffix=suffix,
                template=template,
                is_mcp=classification.is_mcp,
            )
        )

    return sorted(summaries, key=lambda summary: summary.is_mcp)
