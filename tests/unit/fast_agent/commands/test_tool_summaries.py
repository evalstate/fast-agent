"""Tests for tool summary suffix classification."""

from __future__ import annotations

from mcp.types import Tool

from fast_agent.commands.tool_summaries import build_provider_tool_summaries, build_tool_summaries
from fast_agent.mcp.provider_management import (
    ProviderManagedMCPAttachment,
    ProviderManagedMCPState,
)
from fast_agent.tools.tool_sources import ToolSource, set_tool_source


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
    web_search_supported = True
    web_search_enabled = True
    web_fetch_supported = True
    web_fetch_enabled = False
    x_search_supported = False
    x_search_enabled = False

    def __init__(self, state: ProviderManagedMCPState | None = None) -> None:
        self._state = state or ProviderManagedMCPState()

    @property
    def provider_managed_mcp_state(self) -> ProviderManagedMCPState:
        return self._state


class _ProviderToolAgentStub:
    def __init__(self, state: ProviderManagedMCPState | None = None) -> None:
        self._state = state

    @property
    def llm(self):
        return _ProviderToolLlmStub(self._state)


class _ProviderToolLlmWithoutManagedMCPStateStub:
    web_search_supported = True
    web_search_enabled = True
    web_fetch_supported = False
    web_fetch_enabled = False
    x_search_supported = False
    x_search_enabled = False


class _ProviderToolAgentWithoutManagedMCPStateStub:
    @property
    def llm(self):
        return _ProviderToolLlmWithoutManagedMCPStateStub()


def _tool_with_source(name: str, source: ToolSource) -> Tool:
    return set_tool_source(_tool(name), source)


def test_build_tool_summaries_marks_smart_tools() -> None:
    agent = _AgentStub(smart_tool_names={"smart", "smart_with_resource"})

    summaries = build_tool_summaries(agent, [_tool("smart"), _tool("smart_with_resource")])

    assert summaries[0].suffix == "(Smart)"
    assert summaries[1].suffix == "(Smart)"


def test_build_tool_summaries_uses_shell_source_suffix() -> None:
    summaries = build_tool_summaries(_AgentStub(), [_tool_with_source("read_text_file", "shell")])

    assert summaries[0].suffix == "(Shell)"


def test_build_tool_summaries_uses_acp_filesystem_source_suffix() -> None:
    summaries = build_tool_summaries(
        _AgentStub(),
        [_tool_with_source("read_text_file", "acp_filesystem")],
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


def test_build_tool_summaries_keeps_app_badges_additive_with_source_suffix() -> None:
    summaries = build_tool_summaries(
        _AgentStub(),
        [
            set_tool_source(
                _tool("read_text_file", meta={"ui/appEnabled": True}),
                "shell",
            )
        ],
    )

    assert summaries[0].suffix == "(Shell) (MCP App)"


def test_build_provider_tool_summaries_lists_enabled_hosted_tools() -> None:
    summaries = build_provider_tool_summaries(_ProviderToolAgentStub())

    assert [(summary.name, summary.suffix, summary.enabled) for summary in summaries] == [
        ("web_search", "provider-hosted", True),
    ]


def test_build_provider_tool_summaries_marks_missing_managed_mcp_state_unknown() -> None:
    summaries = build_provider_tool_summaries(_ProviderToolAgentWithoutManagedMCPStateStub())

    assert [(summary.name, summary.suffix, summary.enabled) for summary in summaries] == [
        ("web_search", "provider-hosted", True),
        ("provider_managed_mcp", "provider-managed MCP", None),
    ]
    assert summaries[1].description == "Provider-managed MCP state is unavailable for this model."


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

    assert [(summary.name, summary.suffix, summary.enabled) for summary in summaries] == [
        ("web_search", "provider-hosted", True),
        ("gmail/search_gmail", "provider-managed connector", True),
    ]
    assert summaries[1].description == "Gmail connector"


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

    assert summaries[1].name == "gmail"
    assert summaries[1].suffix == "provider-managed connector"
    assert summaries[1].enabled is True
    assert summaries[1].description == "Gmail connector; tools loaded by provider"


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

    assert summaries[1].name == "stripe/create_payment_link"
    assert summaries[1].suffix == "provider-managed MCP"
    assert summaries[1].enabled is True


def test_build_provider_tool_summaries_marks_empty_allowlist_disabled() -> None:
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

    assert summaries[1].name == "gmail"
    assert summaries[1].enabled is False
    assert summaries[1].description == "Gmail connector; no allowed tools configured"
