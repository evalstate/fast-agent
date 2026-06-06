from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from mcp.types import TextContent

from fast_agent.agents.agent_types import AgentType
from fast_agent.commands.renderers import status_markdown as status_renderers
from fast_agent.commands.renderers.status_markdown import render_status_markdown
from fast_agent.commands.status_summaries import (
    DEFAULT_ERROR_SUMMARY_LIMIT,
    ERROR_BLOCK_PREVIEW_LENGTH,
    MODEL_CAPABILITY_LABELS,
    AgentModelSummary,
    ConversationStatsSummary,
    ErrorHandlingSummary,
    ParallelModelSummary,
    PermissionsSummary,
    StatusSummary,
    SystemPromptSummary,
    ToolUsageSummary,
    _count_tokens_with_tiktoken,
    _error_block_summary,
    _instance_card_collision_warnings,
    _positive_milliseconds_to_seconds,
    _tool_usage_breakdown,
    _usage_accumulator_context_line,
    build_conversation_stats_summary,
    build_error_handling_summary,
    build_status_summary,
    build_warning_summary,
)
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


class _MissingModelMetadataLLM:
    model_name = "custom-model"


class _BrokenModelMetadataLLM(_MissingModelMetadataLLM):
    @property
    def model_info(self) -> object:
        raise AttributeError("model_info")

    @property
    def resolved_model(self) -> object:
        raise AttributeError("resolved_model")


class _MalformedHfDisplayInfoLLM(_MissingModelMetadataLLM):
    def get_hf_display_info(self) -> dict[str, object]:
        return {"provider": 123}


def _summary(*, model_source: str | None) -> StatusSummary:
    return StatusSummary(
        fast_agent_version="1.2.3",
        client_info=None,
        model_summary=None,
        parallel_summary=None,
        model_source=model_source,
        conversation_stats=ConversationStatsSummary(
            agent_name="agent",
            turns=0,
            message_count=0,
            user_message_count=0,
            assistant_message_count=0,
            tool_calls=0,
            tool_successes=0,
            tool_errors=0,
            context_usage_line="Context Used: 0%",
        ),
        uptime_seconds=0.0,
        error_report=ErrorHandlingSummary(
            channel_label="Error Channel: fast-agent-error",
            recent_entries=[],
        ),
        warnings=[],
    )


def test_render_status_markdown_includes_model_source_when_present() -> None:
    rendered = render_status_markdown(_summary(model_source="last used model"), heading="status")

    assert "- Model Source: last used model" in rendered


def test_status_markdown_renderers_normalize_markdown_headings() -> None:
    status_rendered = render_status_markdown(_summary(model_source=None), heading="# status")

    system_prompt_rendered = status_renderers.render_system_prompt_markdown(
        SystemPromptSummary(agent_name="agent", system_prompt="hello"),
        heading="# system prompt",
    )
    permissions_rendered = status_renderers.render_permissions_markdown(
        PermissionsSummary(heading="# permissions", message="allowed", path="/tmp/example")
    )

    assert status_rendered.startswith("# status\n")
    assert system_prompt_rendered.startswith("# system prompt\n")
    assert permissions_rendered.startswith("# permissions\n")


def test_status_markdown_renderers_escape_markdown_headings() -> None:
    status_rendered = render_status_markdown(
        _summary(model_source=None),
        heading="status_[draft]*",
    )
    system_prompt_rendered = status_renderers.render_system_prompt_markdown(
        SystemPromptSummary(agent_name="agent", system_prompt="hello"),
        heading="system_[draft]*",
    )
    permissions_rendered = status_renderers.render_permissions_markdown(
        PermissionsSummary(
            heading="permissions_[draft]*",
            message="allowed",
            path="/tmp/example",
        )
    )

    assert status_rendered.startswith("# status\\_\\[draft\\]\\*\n")
    assert system_prompt_rendered.startswith("# system\\_\\[draft\\]\\*\n")
    assert permissions_rendered.startswith("# permissions\\_\\[draft\\]\\*\n")


def test_render_permissions_markdown_handles_backticks_in_path() -> None:
    rendered = status_renderers.render_permissions_markdown(
        PermissionsSummary(
            heading="permissions",
            message="allowed",
            path="/tmp/a`b/auths.md",
        )
    )

    assert "Path: `` /tmp/a`b/auths.md ``" in rendered


def test_render_system_prompt_markdown_escapes_agent_name_but_preserves_prompt() -> None:
    rendered = status_renderers.render_system_prompt_markdown(
        SystemPromptSummary(
            agent_name="agent_[draft]*",
            system_prompt="Use [docs](https://example.test)",
        ),
        heading="system prompt",
    )

    assert "**Agent:** agent\\_\\[draft\\]\\*" in rendered
    assert "Use [docs](https://example.test)" in rendered


def test_render_status_markdown_omits_model_source_when_missing() -> None:
    rendered = render_status_markdown(_summary(model_source=None), heading="status")

    assert "- Model Source:" not in rendered


def test_render_status_markdown_formats_conversation_count_breakdowns() -> None:
    summary = _summary(model_source=None)
    summary.conversation_stats.message_count = 5
    summary.conversation_stats.user_message_count = 2
    summary.conversation_stats.assistant_message_count = 3
    summary.conversation_stats.tool_calls = 4
    summary.conversation_stats.tool_successes = 3
    summary.conversation_stats.tool_errors = 1

    rendered = render_status_markdown(summary, heading="status")

    assert "- Messages: 5 (user: 2, assistant: 3)" in rendered
    assert "- Tool Calls: 4 (successes: 3, errors: 1)" in rendered


def test_status_markdown_context_window_display() -> None:
    assert status_renderers._context_window_display(None) == "unknown"
    assert status_renderers._context_window_display(0) == "unknown"
    assert status_renderers._context_window_display(4096) == "4096 tokens"


def test_build_status_summary_prefers_agent_context_model_source() -> None:
    agent = SimpleNamespace(
        agent_type=AgentType.BASIC,
        name="agent",
        context=SimpleNamespace(config=SimpleNamespace(model_source="last used model")),
        llm=None,
        usage_accumulator=None,
        message_history=[],
    )

    summary = build_status_summary(
        fast_agent_version="1.2.3",
        agent=cast("AgentProtocol", agent),
        client_info=None,
        client_capabilities=None,
        protocol_version=None,
        uptime_seconds=0.0,
        instance=None,
    )

    assert summary.model_source == "last used model"


def test_build_status_summary_omits_blank_model_source() -> None:
    agent = SimpleNamespace(
        agent_type=AgentType.BASIC,
        name="agent",
        context=SimpleNamespace(config=SimpleNamespace(model_source="   ")),
        llm=None,
        usage_accumulator=None,
        message_history=[],
    )

    summary = build_status_summary(
        fast_agent_version="1.2.3",
        agent=cast("AgentProtocol", agent),
        client_info=None,
        client_capabilities=None,
        protocol_version=None,
        uptime_seconds=0.0,
        instance=None,
    )

    assert summary.model_source is None


def test_build_status_summary_normalizes_client_info_fields() -> None:
    summary = build_status_summary(
        fast_agent_version="1.2.3",
        agent=None,
        client_info={"name": 123, "version": " 2.0 ", "title": "   "},
        client_capabilities={
            "fs": {"read": True, 3: "ignored"},
            "terminal": 80,
            "_meta": {"theme": "dark", 4: "ignored"},
        },
        protocol_version="1",
        uptime_seconds=0.0,
        instance=None,
    )

    assert summary.client_info is not None
    assert summary.client_info.name is None
    assert summary.client_info.version == "2.0"
    assert summary.client_info.title is None
    assert summary.client_info.filesystem_caps == {"read": True}
    assert summary.client_info.terminal == "80"
    assert summary.client_info.meta_caps == {"theme": "dark"}


def test_build_status_summary_strips_or_omits_terminal_capability() -> None:
    numeric_summary = build_status_summary(
        fast_agent_version="1.2.3",
        agent=None,
        client_info=None,
        client_capabilities={"terminal": 80},
        protocol_version=None,
        uptime_seconds=0.0,
        instance=None,
    )
    blank_summary = build_status_summary(
        fast_agent_version="1.2.3",
        agent=None,
        client_info=None,
        client_capabilities={"terminal": "   "},
        protocol_version=None,
        uptime_seconds=0.0,
        instance=None,
    )

    assert numeric_summary.client_info is not None
    assert numeric_summary.client_info.terminal == "80"
    assert blank_summary.client_info is not None
    assert blank_summary.client_info.terminal is None


def test_render_status_markdown_formats_client_capability_sections() -> None:
    summary = build_status_summary(
        fast_agent_version="1.2.3",
        agent=None,
        client_info={"name": "editor", "version": "2.0"},
        client_capabilities={
            "fs": {"read": True, "write": False},
            "terminal": "xterm",
            "_meta": {"theme": "dark"},
        },
        protocol_version="1",
        uptime_seconds=0.0,
        instance=None,
    )

    rendered = render_status_markdown(summary, heading="status")

    assert "## Client Information" in rendered
    assert "Client: editor" in rendered
    assert "Filesystem:\n  - read: True\n  - write: False" in rendered
    assert "  - Terminal: xterm" in rendered
    assert "Meta:\n  - theme: dark" in rendered


def test_render_status_markdown_escapes_client_metadata() -> None:
    summary = build_status_summary(
        fast_agent_version="1.2.3",
        agent=None,
        client_info={"name": "zed_beta", "version": "2`dev`", "title": "Editor_[x]"},
        client_capabilities={
            "fs": {"can_edit": True},
            "terminal": "term_1",
            "_meta": {"theme_name": "dark*mode*"},
        },
        protocol_version="1_alpha",
        uptime_seconds=0.0,
        instance=None,
    )

    rendered = render_status_markdown(summary, heading="status")

    assert "Client: Editor\\_\\[x\\] (zed\\_beta)" in rendered
    assert "Client Version: 2\\`dev\\`" in rendered
    assert "ACP Protocol Version: 1\\_alpha" in rendered
    assert "  - can\\_edit: True" in rendered
    assert "  - Terminal: term\\_1" in rendered
    assert "  - theme\\_name: dark\\*mode\\*" in rendered


def test_render_status_markdown_escapes_active_model_values() -> None:
    summary = _summary(model_source="manual_source")
    summary.model_summary = AgentModelSummary(
        agent_name="agent",
        provider="custom_provider",
        provider_display="Custom_Provider",
        model_name="model_one",
        wire_model_name="wire_model",
        context_window=1000,
        capabilities=["Text_Mode", "Vision*Mode*"],
        hf_provider="hf_provider",
    )

    rendered = render_status_markdown(summary, heading="status")

    assert "- Provider: Custom\\_Provider (custom\\_provider) / hf\\_provider" in rendered
    assert "- Model: model\\_one" in rendered
    assert "- Model Source: manual\\_source" in rendered
    assert "- Wire Model: wire\\_model" in rendered
    assert "- Capabilities: Text\\_Mode, Vision\\*Mode\\*" in rendered


def test_render_status_markdown_escapes_runtime_list_values() -> None:
    summary = _summary(model_source=None)
    summary.conversation_stats.agent_name = "agent_one"
    summary.conversation_stats.tool_breakdown = [ToolUsageSummary(name="read_file", count=2)]
    summary.error_report = ErrorHandlingSummary(
        channel_label="Error Channel: fast-agent-error",
        recent_entries=["bad_tool failed"],
    )
    summary.warnings = ["missing_card"]

    rendered = render_status_markdown(summary, heading="status")

    assert "## Conversation Statistics (agent\\_one)" in rendered
    assert "  - read\\_file: 2" in rendered
    assert "- bad\\_tool failed" in rendered
    assert "- missing\\_card" in rendered


def test_render_status_markdown_escapes_parallel_model_values() -> None:
    summary = _summary(model_source=None)
    summary.parallel_summary = ParallelModelSummary(
        fan_out_agents=[
            AgentModelSummary(
                agent_name="fan_out",
                provider="custom_provider",
                provider_display="Custom_Provider",
                model_name="model_one",
                wire_model_name="wire_model",
                context_window=None,
                capabilities=[],
            )
        ],
        fan_in_agent=AgentModelSummary(
            agent_name="fan_in",
            provider="custom_provider",
            provider_display="Custom_Provider",
            model_name="model_two",
            wire_model_name=None,
            context_window=None,
            capabilities=[],
        ),
    )

    rendered = render_status_markdown(summary, heading="status")

    assert "**1. fan\\_out**" in rendered
    assert "  - Provider: Custom\\_Provider" in rendered
    assert "  - Model: model\\_one" in rendered
    assert "  - Wire Model: wire\\_model" in rendered
    assert "### Fan-In Agent: fan\\_in" in rendered
    assert "  - Model: model\\_two" in rendered


def test_build_conversation_stats_summary_uses_empty_fallback_without_agent() -> None:
    summary = build_conversation_stats_summary(None, fallback_agent_name="agent")

    assert summary.agent_name == "agent"
    assert summary.message_count == 0
    assert summary.tool_calls == 0
    assert summary.context_usage_line == "Context Used: 0%"


def test_build_conversation_stats_summary_uses_valid_usage_accumulator_values() -> None:
    agent = SimpleNamespace(
        name="agent",
        usage_accumulator=SimpleNamespace(
            context_window_size=1_000,
            current_context_tokens=250,
            context_usage_percentage=25.0,
        ),
        message_history=[],
        llm=None,
    )

    summary = build_conversation_stats_summary(
        cast("AgentProtocol", agent),
        fallback_agent_name="agent",
    )

    assert summary.context_usage_line == "Context Used: 25.0% (~250 tokens of 1,000)"


def test_usage_accumulator_context_line_clamps_percentage() -> None:
    assert (
        _usage_accumulator_context_line(window=1_000, tokens=1_250, percentage=125.0)
        == "Context Used: 100.0% (~1,250 tokens of 1,000)"
    )
    assert (
        _usage_accumulator_context_line(window=1_000, tokens=250, percentage=-5.0)
        == "Context Used: 0.0% (~250 tokens of 1,000)"
    )


def test_usage_accumulator_context_line_handles_tokens_without_window() -> None:
    assert (
        _usage_accumulator_context_line(window=None, tokens=250, percentage=None)
        == "Context Used: ~250 tokens (window unknown)"
    )
    assert _usage_accumulator_context_line(window=None, tokens=0, percentage=None) is None


def test_tool_usage_breakdown_sorts_by_count_descending() -> None:
    summary = SimpleNamespace(tool_call_map={"search": 1, "read": 3, "write": 2})

    assert _tool_usage_breakdown(cast("Any", summary)) == [
        ToolUsageSummary(name="read", count=3),
        ToolUsageSummary(name="write", count=2),
        ToolUsageSummary(name="search", count=1),
    ]


def test_positive_milliseconds_to_seconds() -> None:
    assert _positive_milliseconds_to_seconds(1500) == 1.5
    assert _positive_milliseconds_to_seconds(0) is None
    assert _positive_milliseconds_to_seconds(-1) is None
    assert _positive_milliseconds_to_seconds(True) is None
    assert _positive_milliseconds_to_seconds(float("inf")) is None
    assert _positive_milliseconds_to_seconds(float("nan")) is None


def test_count_tokens_with_tiktoken_falls_back_for_unknown_model(monkeypatch) -> None:
    fake_tiktoken = SimpleNamespace(
        encoding_for_model=lambda _model_name: (_ for _ in ()).throw(KeyError("unknown")),
        get_encoding=lambda _name: SimpleNamespace(encode=lambda text: text.split()),
    )
    monkeypatch.setitem(sys.modules, "tiktoken", fake_tiktoken)

    assert _count_tokens_with_tiktoken("abcdefgh", "unknown-model") == 2


def test_count_tokens_with_tiktoken_does_not_hide_encoder_runtime_failures(monkeypatch) -> None:
    fake_tiktoken = SimpleNamespace(
        encoding_for_model=lambda _model_name: SimpleNamespace(
            encode=lambda _text: (_ for _ in ()).throw(RuntimeError("encoder failed"))
        ),
        get_encoding=lambda _name: SimpleNamespace(encode=lambda text: text.split()),
    )
    monkeypatch.setitem(sys.modules, "tiktoken", fake_tiktoken)

    with pytest.raises(RuntimeError, match="encoder failed"):
        _count_tokens_with_tiktoken("hello", "known-model")


def test_build_conversation_stats_summary_ignores_malformed_usage_accumulator_values() -> None:
    agent = SimpleNamespace(
        name="agent",
        usage_accumulator=SimpleNamespace(
            context_window_size=True,
            current_context_tokens=True,
            context_usage_percentage=float("nan"),
        ),
        message_history=[],
        llm=None,
    )

    summary = build_conversation_stats_summary(
        cast("AgentProtocol", agent),
        fallback_agent_name="agent",
    )

    assert summary.context_usage_line == "Context Used: 0 chars (~0 tokens est.)"


def test_build_conversation_stats_summary_falls_back_for_malformed_history() -> None:
    agent = SimpleNamespace(
        name="agent",
        usage_accumulator=None,
        message_history=object(),
        llm=None,
    )

    summary = build_conversation_stats_summary(
        cast("AgentProtocol", agent),
        fallback_agent_name="agent",
    )

    assert summary.agent_name == "agent"
    assert summary.message_count == 0
    assert summary.context_usage_line.startswith("Context Used: error (")


def test_build_conversation_stats_summary_does_not_hide_runtime_failures() -> None:
    class _ExplodingUsage:
        @property
        def context_window_size(self) -> int:
            raise RuntimeError("usage failed")

    agent = SimpleNamespace(
        name="agent",
        usage_accumulator=_ExplodingUsage(),
        message_history=[],
        llm=None,
    )

    with pytest.raises(RuntimeError, match="usage failed"):
        build_conversation_stats_summary(
            cast("AgentProtocol", agent),
            fallback_agent_name="agent",
        )


def test_build_status_summary_tolerates_missing_llm_model_metadata() -> None:
    agent = SimpleNamespace(
        agent_type=AgentType.BASIC,
        name="agent",
        context=SimpleNamespace(config=SimpleNamespace(model_source=None)),
        llm=_MissingModelMetadataLLM(),
        usage_accumulator=None,
        message_history=[],
    )

    summary = build_status_summary(
        fast_agent_version="1.2.3",
        agent=cast("AgentProtocol", agent),
        client_info=None,
        client_capabilities=None,
        protocol_version=None,
        uptime_seconds=0.0,
        instance=None,
    )

    assert summary.model_summary is not None
    assert summary.model_summary.model_name == "unknown"
    assert summary.model_summary.provider == "unknown"
    assert summary.conversation_stats.context_usage_line == "Context Used: 0 chars (~0 tokens est.)"


def test_build_status_summary_does_not_hide_broken_llm_model_metadata() -> None:
    agent = SimpleNamespace(
        agent_type=AgentType.BASIC,
        name="agent",
        context=SimpleNamespace(config=SimpleNamespace(model_source=None)),
        llm=_BrokenModelMetadataLLM(),
        usage_accumulator=None,
        message_history=[],
    )

    with pytest.raises(AttributeError, match="resolved_model"):
        build_status_summary(
            fast_agent_version="1.2.3",
            agent=cast("AgentProtocol", agent),
            client_info=None,
            client_capabilities=None,
            protocol_version=None,
            uptime_seconds=0.0,
            instance=None,
        )


def test_build_status_summary_defaults_malformed_hf_provider_display() -> None:
    agent = SimpleNamespace(
        agent_type=AgentType.BASIC,
        name="agent",
        context=SimpleNamespace(config=SimpleNamespace(model_source=None)),
        llm=_MalformedHfDisplayInfoLLM(),
        usage_accumulator=None,
        message_history=[],
    )

    summary = build_status_summary(
        fast_agent_version="1.2.3",
        agent=cast("AgentProtocol", agent),
        client_info=None,
        client_capabilities=None,
        protocol_version=None,
        uptime_seconds=0.0,
        instance=None,
    )

    assert summary.model_summary is not None
    assert summary.model_summary.hf_provider == "auto-routing"


def test_build_error_handling_summary_truncates_unknown_error_blocks() -> None:
    block = SimpleNamespace(payload="x" * 80)
    agent = SimpleNamespace(
        message_history=[
            SimpleNamespace(channels={FAST_AGENT_ERROR_CHANNEL: [block]}),
        ]
    )

    summary = build_error_handling_summary(cast("AgentProtocol", agent))

    assert len(summary.recent_entries) <= DEFAULT_ERROR_SUMMARY_LIMIT
    assert summary.recent_entries == [
        f"{str(block)[:ERROR_BLOCK_PREVIEW_LENGTH]}... ({len(str(block))} characters)"
    ]


def test_error_block_summary_normalizes_text_blocks() -> None:
    block = TextContent(type="text", text=" one\ntwo ")

    assert _error_block_summary(block) == "one two"
    assert _error_block_summary(TextContent(type="text", text=" \n ")) is None


def test_model_capability_labels_match_tdv_order() -> None:
    assert MODEL_CAPABILITY_LABELS == ("Text", "Document", "Vision")


def test_build_warning_summary_normalizes_instance_warning_shapes() -> None:
    instance = SimpleNamespace(
        app=SimpleNamespace(
            card_collision_warnings=(
                warning for warning in [" duplicate ", "", "duplicate", "second"]
            )
        )
    )

    warnings = build_warning_summary(None, instance=cast("Any", instance))

    assert warnings == ["duplicate", "second"]


def test_build_warning_summary_formats_truncated_warning_count() -> None:
    class _Agent:
        skill_registry = None

        @property
        def warnings(self) -> list[str]:
            return ["one", "two", "three"]

    warnings = build_warning_summary(
        cast("AgentProtocol", _Agent()),
        instance=None,
        max_entries=2,
    )

    assert warnings == ["one", "two", "... (1 more warning)"]


def test_instance_card_collision_warnings_normalizes_supported_shapes() -> None:
    iterable_instance = SimpleNamespace(
        app=SimpleNamespace(card_collision_warnings=["one", 2])
    )
    scalar_instance = SimpleNamespace(app=SimpleNamespace(card_collision_warnings="one"))
    empty_instance = SimpleNamespace(app=SimpleNamespace(card_collision_warnings=None))

    assert _instance_card_collision_warnings(cast("Any", iterable_instance)) == ["one", "2"]
    assert _instance_card_collision_warnings(cast("Any", scalar_instance)) == ["one"]
    assert _instance_card_collision_warnings(cast("Any", empty_instance)) == []
