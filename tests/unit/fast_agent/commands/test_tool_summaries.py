"""Tests for tool summary suffix classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.types import Tool

from fast_agent.commands import tool_summaries
from fast_agent.commands.tool_summaries import (
    PROVIDER_HOSTED_SUFFIX,
    PROVIDER_HOSTED_TOOL_DESCRIPTORS,
    PROVIDER_MANAGED_CONNECTOR_SUFFIX,
    PROVIDER_MANAGED_MCP_SUFFIX,
    ProviderToolSummary,
    build_provider_tool_summaries,
    build_tool_summaries,
    provider_tool_state_label,
    provider_tool_status_label,
)
from fast_agent.mcp.provider_management import (
    ProviderManagedMCPAttachment,
    ProviderManagedMCPState,
)
from fast_agent.tools.filesystem_tool_definitions import READ_TEXT_FILE_TOOL_NAME
from fast_agent.tools.tool_sources import (
    ACP_FILESYSTEM_TOOL_SOURCE,
    SHELL_TOOL_SOURCE,
    ToolSource,
    set_tool_source,
)

if TYPE_CHECKING:
    import pytest


def _tool(
    name: str,
    *,
    meta: dict | None = None,
    description: str = "",
    input_schema: dict | None = None,
):
    return Tool(
        name=name,
        title=None,
        description=description,
        _meta=meta or {},
        inputSchema=input_schema or {},
    )


class _AgentStub:
    def __init__(
        self,
        *,
        card_tool_names=(),
        smart_tool_names=(),
        agent_backed_tools: dict[str, object] | None = None,
    ) -> None:
        self._card_tool_names = set(card_tool_names)
        self._smart_tool_names = set(smart_tool_names)
        self._agent_backed_tools = agent_backed_tools or {}

    @property
    def card_tool_names(self) -> set[str]:
        return self._card_tool_names

    @property
    def smart_tool_names(self) -> set[str]:
        return self._smart_tool_names

    @smart_tool_names.setter
    def smart_tool_names(self, value) -> None:
        self._smart_tool_names = set(value)

    @property
    def parallel_smart_tool_calls(self) -> bool:
        return False

    @parallel_smart_tool_calls.setter
    def parallel_smart_tool_calls(self, value: bool) -> None:
        del value

    @property
    def agent_backed_tools(self) -> dict[str, object]:
        return self._agent_backed_tools


class _ProviderToolLlmStub:
    web_fetch_supported = True
    web_fetch_enabled = False
    x_search_supported = False
    x_search_enabled = False

    def __init__(
        self,
        state: ProviderManagedMCPState | None = None,
        *,
        web_search_supported: bool = True,
        web_search_enabled: bool = True,
    ) -> None:
        self._state = state or ProviderManagedMCPState()
        self.web_search_supported = web_search_supported
        self.web_search_enabled = web_search_enabled

    @property
    def provider_managed_mcp_state(self) -> ProviderManagedMCPState:
        return self._state


class _ProviderToolAgentStub:
    def __init__(
        self,
        state: ProviderManagedMCPState | None = None,
        *,
        llm: object | None = None,
    ) -> None:
        self._llm = llm or _ProviderToolLlmStub(state)

    @property
    def llm(self):
        return self._llm


class _ProviderToolLlmWithoutManagedMCPStateStub:
    web_search_supported = True
    web_search_enabled = True
    web_fetch_supported = False
    web_fetch_enabled = False
    x_search_supported = False
    x_search_enabled = False


class _TruthyProviderToolLlmStub:
    web_fetch_supported = False
    web_fetch_enabled = False
    x_search_supported = False
    x_search_enabled = False

    def __init__(self) -> None:
        self._state = ProviderManagedMCPState()

    @property
    def web_search_supported(self):
        return 1

    @property
    def web_search_enabled(self):
        return 1

    @property
    def provider_managed_mcp_state(self) -> ProviderManagedMCPState:
        return self._state


class _ProviderToolAgentWithoutManagedMCPStateStub:
    @property
    def llm(self):
        return _ProviderToolLlmWithoutManagedMCPStateStub()


def _tool_with_source(name: str, source: ToolSource) -> Tool:
    return set_tool_source(_tool(name), source)


def _provider_summary_by_name(
    summaries: list[ProviderToolSummary],
    name: str,
) -> ProviderToolSummary:
    return next(summary for summary in summaries if summary.name == name)


def test_build_tool_summaries_marks_smart_tools() -> None:
    agent = _AgentStub(smart_tool_names={"smart", "smart_with_resource"})

    summaries = build_tool_summaries(agent, [_tool("smart"), _tool("smart_with_resource")])

    assert summaries[0].suffix == "(Smart)"
    assert summaries[1].suffix == "(Smart)"


def test_build_tool_summaries_strips_blank_descriptions() -> None:
    summaries = build_tool_summaries(_AgentStub(), [_tool("demo", description="   ")])

    assert summaries[0].description is None


def test_build_tool_summaries_uses_shell_source_suffix() -> None:
    summaries = build_tool_summaries(
        _AgentStub(), [_tool_with_source(READ_TEXT_FILE_TOOL_NAME, SHELL_TOOL_SOURCE)]
    )

    assert summaries[0].suffix == "(Shell)"


def test_build_tool_summaries_uses_acp_filesystem_source_suffix() -> None:
    summaries = build_tool_summaries(
        _AgentStub(),
        [_tool_with_source(READ_TEXT_FILE_TOOL_NAME, ACP_FILESYSTEM_TOOL_SOURCE)],
    )

    assert summaries[0].suffix == "(ACP Filesystem)"


def test_build_tool_summaries_does_not_label_unstamped_execute_internal() -> None:
    summaries = build_tool_summaries(_AgentStub(), [_tool("execute")])

    assert summaries[0].suffix is None


def test_build_tool_summaries_prefers_smart_suffix_over_source() -> None:
    agent = _AgentStub(smart_tool_names={"smart"})

    summaries = build_tool_summaries(agent, [_tool_with_source("smart", "shell")])

    assert summaries[0].suffix == "(Smart)"


def test_build_tool_summaries_preserves_non_smart_suffixes() -> None:
    agent = _AgentStub(smart_tool_names={"smart"})

    summaries = build_tool_summaries(agent, [_tool("demo__search")])

    assert summaries[0].suffix == "(MCP)"


def test_build_tool_summaries_orders_mcp_tools_last() -> None:
    agent = _AgentStub(
        card_tool_names={"card"},
        smart_tool_names={"smart"},
        agent_backed_tools={"child": object()},
    )

    summaries = build_tool_summaries(
        agent,
        [
            _tool("server__search"),
            _tool("smart"),
            _tool("server__fetch"),
            _tool("card"),
            _tool("child"),
        ],
    )

    assert [summary.name for summary in summaries] == [
        "smart",
        "card",
        "child",
        "server__search",
        "server__fetch",
    ]
    assert [summary.is_mcp for summary in summaries] == [False, False, False, True, True]


def test_build_tool_summaries_marks_smart_skybridge_tools() -> None:
    agent = _AgentStub(smart_tool_names={"smart_with_resource"})

    summaries = build_tool_summaries(
        agent,
        [_tool("smart_with_resource", meta={"openai/skybridgeEnabled": True})],
    )

    assert summaries[0].suffix == "(Smart) (Apps SDK)"


def test_build_tool_summaries_marks_mcp_app_tools() -> None:
    agent = _AgentStub()

    summaries = build_tool_summaries(
        agent,
        [_tool("app_tool", meta={"ui/appEnabled": True, "ui/appTemplate": "ui://app"})],
    )

    assert summaries[0].suffix == "(MCP App)"
    assert summaries[0].template == "ui://app"


def test_build_tool_summaries_ignores_non_string_template_metadata() -> None:
    summaries = build_tool_summaries(
        _AgentStub(),
        [_tool("app_tool", meta={"ui/appEnabled": True, "ui/appTemplate": {"uri": "ui://app"}})],
    )

    assert summaries[0].suffix == "(MCP App)"
    assert summaries[0].template is None


def test_build_tool_summaries_strips_blank_template_metadata() -> None:
    summaries = build_tool_summaries(
        _AgentStub(),
        [
            _tool(
                "app_tool",
                meta={
                    "ui/appEnabled": True,
                    "ui/appTemplate": "   ",
                    "openai/skybridgeTemplate": " ui://fallback ",
                },
            )
        ],
    )

    assert summaries[0].template == "ui://fallback"


def test_build_tool_summaries_filters_non_string_schema_keys() -> None:
    summaries = build_tool_summaries(
        _AgentStub(),
        [
            _tool(
                "mixed_schema",
                input_schema={
                    "properties": {"path": {}, 7: {}, "mode": {}},
                    "required": ["path", 7],
                },
            )
        ],
    )

    assert summaries[0].args == ["path*", "mode"]


def test_build_tool_summaries_keeps_app_badges_additive_with_source_suffix() -> None:
    summaries = build_tool_summaries(
        _AgentStub(),
        [
            set_tool_source(
                _tool("read_text_file", meta={"ui/appEnabled": True}),
                SHELL_TOOL_SOURCE,
            )
        ],
    )

    assert summaries[0].suffix == "(Shell) (MCP App)"


def test_build_provider_tool_summaries_lists_enabled_supported_hosted_tools() -> None:
    summaries = build_provider_tool_summaries(_ProviderToolAgentStub())

    assert [(summary.name, summary.suffix, summary.enabled) for summary in summaries] == [
        ("web_search", PROVIDER_HOSTED_SUFFIX, True),
    ]


def test_build_provider_tool_summaries_reflects_current_hosted_tool_state() -> None:
    llm = _ProviderToolLlmStub(web_search_enabled=False)
    agent = _ProviderToolAgentStub(llm=llm)

    assert build_provider_tool_summaries(agent) == []

    llm.web_search_enabled = True

    assert (
        _provider_summary_by_name(build_provider_tool_summaries(agent), "web_search").enabled
        is True
    )


def test_build_provider_tool_summaries_accepts_truthy_hosted_tool_state() -> None:
    summary = _provider_summary_by_name(
        build_provider_tool_summaries(_ProviderToolAgentStub(llm=_TruthyProviderToolLlmStub())),
        "web_search",
    )

    assert summary.enabled is True


def test_build_provider_tool_summaries_evaluates_hosted_enabled_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled_calls = 0

    def _enabled(_llm: object | None) -> bool:
        nonlocal enabled_calls
        enabled_calls += 1
        return True

    monkeypatch.setattr(
        tool_summaries,
        "PROVIDER_HOSTED_TOOL_DESCRIPTORS",
        (
            tool_summaries._ProviderToolDescriptor(
                name="single_eval",
                supported=lambda _llm: True,
                enabled=_enabled,
                description="Provider-hosted single evaluation tool.",
            ),
        ),
    )

    summaries = build_provider_tool_summaries(_ProviderToolAgentStub())

    assert enabled_calls == 1
    assert summaries[0].name == "single_eval"


def test_provider_hosted_tool_descriptors_are_unique_and_described() -> None:
    names = [descriptor.name for descriptor in PROVIDER_HOSTED_TOOL_DESCRIPTORS]

    assert len(names) == len(set(names))
    assert all(descriptor.description for descriptor in PROVIDER_HOSTED_TOOL_DESCRIPTORS)


def test_provider_tool_state_label_distinguishes_unknown_state() -> None:
    assert provider_tool_state_label(True) == "enabled"
    assert provider_tool_state_label(False) == "disabled"
    assert provider_tool_state_label(None) == "Unknown"


def test_provider_tool_status_label_combines_suffix_and_state() -> None:
    summary = ProviderToolSummary(
        name="provider_managed_mcp",
        enabled=None,
        description="Provider-managed MCP state is unavailable.",
        suffix=PROVIDER_MANAGED_MCP_SUFFIX,
    )

    assert provider_tool_status_label(summary) == f"{PROVIDER_MANAGED_MCP_SUFFIX}, Unknown"


def test_build_provider_tool_summaries_omits_missing_managed_mcp_state() -> None:
    summaries = build_provider_tool_summaries(_ProviderToolAgentWithoutManagedMCPStateStub())

    assert [(summary.name, summary.suffix, summary.enabled) for summary in summaries] == [
        ("web_search", PROVIDER_HOSTED_SUFFIX, True),
    ]


def test_build_provider_tool_summaries_lists_connector_allowlist() -> None:
    state = ProviderManagedMCPState(
        attachments=(
            ProviderManagedMCPAttachment(
                server_name="gmail",
                server_description="Gmail connector",
                connector_id="connector_gmail",
                access_token="token",
            ),
        ),
        tool_allowlists={"gmail": ("search_gmail",)},
    )

    summaries = build_provider_tool_summaries(_ProviderToolAgentStub(state))

    summary = _provider_summary_by_name(summaries, "gmail/search_gmail")
    assert summary.suffix == PROVIDER_MANAGED_CONNECTOR_SUFFIX
    assert summary.enabled is True
    assert summary.description == "Gmail connector"


def test_build_provider_tool_summaries_lists_connector_toolset_without_allowlist() -> None:
    state = ProviderManagedMCPState(
        attachments=(
            ProviderManagedMCPAttachment(
                server_name="gmail",
                server_description="Gmail connector",
                connector_id="connector_gmail",
                access_token="token",
            ),
        ),
    )

    summaries = build_provider_tool_summaries(_ProviderToolAgentStub(state))

    summary = _provider_summary_by_name(summaries, "gmail")
    assert summary.suffix == PROVIDER_MANAGED_CONNECTOR_SUFFIX
    assert summary.enabled is True
    assert summary.description == "Gmail connector; tools loaded by provider"


def test_build_provider_tool_summaries_uses_connector_fallback_for_blank_description() -> None:
    state = ProviderManagedMCPState(
        attachments=(
            ProviderManagedMCPAttachment(
                server_name="gmail",
                server_description="   ",
                connector_id="connector_gmail",
                access_token="token",
            ),
        ),
    )

    summaries = build_provider_tool_summaries(_ProviderToolAgentStub(state))

    summary = _provider_summary_by_name(summaries, "gmail")
    assert summary.description == "OpenAI connector connector_gmail; tools loaded by provider"


def test_build_provider_tool_summaries_lists_url_mcp_allowlist() -> None:
    state = ProviderManagedMCPState(
        attachments=(
            ProviderManagedMCPAttachment(
                server_name="stripe",
                server_description="Stripe tools",
                server_url="https://stripe.example/mcp",
            ),
        ),
        tool_allowlists={"stripe": ("create_payment_link",)},
    )

    summaries = build_provider_tool_summaries(_ProviderToolAgentStub(state))

    summary = _provider_summary_by_name(summaries, "stripe/create_payment_link")
    assert summary.suffix == PROVIDER_MANAGED_MCP_SUFFIX
    assert summary.enabled is True


def test_build_provider_tool_summaries_uses_mcp_fallback_for_blank_description() -> None:
    state = ProviderManagedMCPState(
        attachments=(
            ProviderManagedMCPAttachment(
                server_name="stripe",
                server_description="\t",
                server_url="https://stripe.example/mcp",
            ),
        ),
    )

    summaries = build_provider_tool_summaries(_ProviderToolAgentStub(state))

    summary = _provider_summary_by_name(summaries, "stripe")
    assert summary.description == "Provider-managed MCP server; tools loaded by provider"


def test_build_provider_tool_summaries_omits_empty_allowlist() -> None:
    state = ProviderManagedMCPState(
        attachments=(
            ProviderManagedMCPAttachment(
                server_name="gmail",
                server_description="Gmail connector",
                connector_id="connector_gmail",
                access_token="token",
            ),
        ),
        tool_allowlists={"gmail": ()},
    )

    summaries = build_provider_tool_summaries(_ProviderToolAgentStub(state))

    assert [summary.name for summary in summaries] == ["web_search"]
