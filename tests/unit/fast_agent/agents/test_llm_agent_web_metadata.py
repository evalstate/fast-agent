from __future__ import annotations

from typing import Any

import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, TextContent
from rich.text import Text

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.llm_decorator import RemovedContentSummary
from fast_agent.constants import ANTHROPIC_CITATIONS_CHANNEL, ANTHROPIC_SERVER_TOOLS_CHANNEL
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import FastAgentUsage, TurnUsage
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui.console_display import ConsoleDisplay


class _CaptureDisplay(ConsoleDisplay):
    def __init__(self) -> None:
        super().__init__(config=None)
        self.calls: list[dict[str, Any]] = []
        self.status_messages: list[Text] = []
        self.mermaid_messages: list[str | Text | PromptMessageExtended] = []

    async def show_assistant_message(
        self,
        message_text: str | Text | PromptMessageExtended,
        bottom_items: list[str] | None = None,
        highlight_index: int | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Text | None = None,
        pre_content=None,
        render_markdown: bool | None = None,
        show_hook_indicator: bool = False,
    ) -> None:
        payload = {
            "message_text": message_text,
            "bottom_items": bottom_items,
            "highlight_index": highlight_index,
            "max_item_length": max_item_length,
            "name": name,
            "model": model,
            "additional_message": additional_message,
            "pre_content": pre_content,
            "render_markdown": render_markdown,
            "show_hook_indicator": show_hook_indicator,
        }
        self.calls.append(payload)

    def show_status_message(self, content: Text) -> None:
        self.status_messages.append(content)

    def show_mermaid_diagrams_from_message_text(
        self,
        message_text: str | Text | PromptMessageExtended,
    ) -> None:
        self.mermaid_messages.append(message_text)


class _UrlCaptureAgent(LlmAgent):
    def __init__(self, name: str) -> None:
        super().__init__(AgentConfig(name))
        self.url_display_names: list[str | None] = []

    def _display_url_elicitations_from_history(self, agent_name: str | None) -> None:
        self.url_display_names.append(agent_name)


class _SummaryHarnessAgent(LlmAgent):
    def __init__(
        self,
        response: PromptMessageExtended,
        summary: RemovedContentSummary | None,
    ) -> None:
        super().__init__(AgentConfig("summary-harness"))
        self._response = response
        self._summary = summary
        self.additional_messages: list[Text | None] = []

    def _should_stream(self) -> bool:
        return False

    async def _generate_with_summary(
        self,
        messages: list[PromptMessageExtended],
        request_params=None,
        tools=None,
    ) -> tuple[PromptMessageExtended, RemovedContentSummary | None]:
        return self._response, self._summary

    async def show_assistant_message(
        self,
        message: PromptMessageExtended,
        bottom_items=None,
        highlight_items=None,
        max_item_length=None,
        name=None,
        model=None,
        additional_message: Text | None = None,
        render_markdown: bool | None = None,
        show_hook_indicator: bool | None = None,
        render_message: bool = True,
    ) -> None:
        self.additional_messages.append(additional_message)


def _web_channels() -> dict[str, list[TextContent]]:
    return {
        ANTHROPIC_SERVER_TOOLS_CHANNEL: [
            TextContent(type="text", text='{"type":"server_tool_use","name":"web_search"}')
        ],
        ANTHROPIC_CITATIONS_CHANNEL: [
            TextContent(
                type="text",
                text=(
                    '{"type":"web_search_result_location",'
                    '"title":"Example",'
                    '"url":"https://example.com/news"}'
                ),
            )
        ],
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_show_assistant_message_hides_web_metadata_for_tool_use() -> None:
    agent = LlmAgent(AgentConfig("web-debug"))
    capture_display = _CaptureDisplay()
    agent.display = capture_display

    tool_use_message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="calling web tool")],
        stop_reason=LlmStopReason.TOOL_USE,
        channels=_web_channels(),
    )

    await agent.show_assistant_message(tool_use_message)

    assert len(capture_display.calls) == 1
    call = capture_display.calls[0]
    assert call.get("bottom_items") is None
    assert call.get("additional_message") is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_show_assistant_message_renders_web_metadata_for_final_turn() -> None:
    agent = LlmAgent(AgentConfig("web-debug"))
    capture_display = _CaptureDisplay()
    agent.display = capture_display

    end_turn_message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="done")],
        stop_reason=LlmStopReason.END_TURN,
        channels=_web_channels(),
    )

    await agent.show_assistant_message(end_turn_message)

    assert len(capture_display.calls) == 1
    call = capture_display.calls[0]
    assert call.get("bottom_items") == ["web_search x1"]
    pre_content = call.get("pre_content")
    assert isinstance(pre_content, Text)
    assert "Sources" in pre_content.plain
    assert "done" not in pre_content.plain

    additional = call.get("additional_message")
    assert isinstance(additional, Text)
    plain = additional.plain
    assert "Sources" not in plain
    assert "Web activity: web_search x1" in plain
    assert call.get("highlight_index") == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_show_assistant_message_suppresses_bottom_metadata_for_shell_tool_use() -> None:
    agent = LlmAgent(AgentConfig("shell-tool-use"))
    capture_display = _CaptureDisplay()
    agent.display = capture_display

    tool_call = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name="bash", arguments={"command": "pwd"}),
    )
    tool_use_message = PromptMessageExtended(
        role="assistant",
        content=[],
        tool_calls={"call_1": tool_call},
        stop_reason=LlmStopReason.TOOL_USE,
    )

    await agent.show_assistant_message(
        tool_use_message,
        bottom_items=["shell", "web"],
        highlight_items="shell",
    )

    assert len(capture_display.calls) == 1
    call = capture_display.calls[0]
    assert call.get("bottom_items") is None
    assert call.get("highlight_index") is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_show_assistant_message_render_message_false_shows_status_and_mermaid() -> None:
    agent = _UrlCaptureAgent("web-debug")
    capture_display = _CaptureDisplay()
    agent.display = capture_display

    end_turn_message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="done")],
        stop_reason=LlmStopReason.END_TURN,
        channels=_web_channels(),
    )

    await agent.show_assistant_message(end_turn_message, render_message=False)

    assert capture_display.calls == []
    assert len(capture_display.status_messages) == 1
    assert "Sources" in capture_display.status_messages[0].plain
    assert "Web activity: web_search x1" in capture_display.status_messages[0].plain
    assert capture_display.mermaid_messages == [end_turn_message]
    assert agent.url_display_names == ["web-debug"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_show_assistant_message_appends_websocket_indicator_to_model() -> None:
    agent = LlmAgent(AgentConfig("websocket-indicator"))
    capture_display = _CaptureDisplay()
    agent.display = capture_display
    llm = ResponsesLLM(provider=Provider.RESPONSES, model="gpt-5.3-codex")
    llm._record_ws_turn_outcome("reused")
    agent._llm = llm

    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="done")],
        stop_reason=LlmStopReason.END_TURN,
    )

    await agent.show_assistant_message(message)

    assert len(capture_display.calls) == 1
    call = capture_display.calls[0]
    assert call.get("model") == "gpt-5.3-codex ↔"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_show_assistant_message_places_websocket_indicator_before_context_percentage() -> None:
    agent = LlmAgent(AgentConfig("websocket-indicator-context"))
    capture_display = _CaptureDisplay()
    agent.display = capture_display
    llm = ResponsesLLM(provider=Provider.RESPONSES, model="gpt-5.3-codex")
    llm._record_ws_turn_outcome("reused")
    llm.usage_accumulator.set_context_window_size(1000)
    llm.usage_accumulator.add_turn(
        TurnUsage.from_fast_agent(
            FastAgentUsage(input_chars=90, output_chars=10, model_type="test"),
            model="gpt-5.3-codex",
        )
    )
    agent._llm = llm

    tool_call = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name="demo-tool", arguments={}),
    )
    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="need tool")],
        tool_calls={"call_1": tool_call},
        stop_reason=LlmStopReason.TOOL_USE,
    )

    await agent.show_assistant_message(message)

    assert len(capture_display.calls) == 1
    call = capture_display.calls[0]
    assert call.get("model") == "gpt-5.3-codex ↔ (10.0%)"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_show_assistant_message_uses_compact_context_format_for_low_usage() -> None:
    agent = LlmAgent(AgentConfig("websocket-indicator-low-context"))
    capture_display = _CaptureDisplay()
    agent.display = capture_display
    llm = ResponsesLLM(provider=Provider.RESPONSES, model="gpt-5.3-codex")
    llm._record_ws_turn_outcome("reused")
    llm.usage_accumulator.set_context_window_size(1000)
    llm.usage_accumulator.add_turn(
        TurnUsage.from_fast_agent(
            FastAgentUsage(input_chars=9, output_chars=1, model_type="test"),
            model="gpt-5.3-codex",
        )
    )
    agent._llm = llm

    tool_call = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name="demo-tool", arguments={}),
    )
    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="need tool")],
        tool_calls={"call_1": tool_call},
        stop_reason=LlmStopReason.TOOL_USE,
    )

    await agent.show_assistant_message(message)

    assert len(capture_display.calls) == 1
    call = capture_display.calls[0]
    assert call.get("model") == "gpt-5.3-codex ↔ (1.00%)"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_impl_hides_summary_for_tool_use_turn() -> None:
    summary = RemovedContentSummary(
        model_name="gpt-test",
        counts={},
        category_mimes={},
        alert_flags=frozenset(),
        message="history trimmed",
    )
    response = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="need tool")],
        stop_reason=LlmStopReason.TOOL_USE,
    )
    agent = _SummaryHarnessAgent(response, summary)

    seed_message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="seed")],
    )
    await agent.generate_impl([seed_message])

    assert agent.additional_messages == [None]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_impl_renders_summary_for_final_turn() -> None:
    summary = RemovedContentSummary(
        model_name="gpt-test",
        counts={},
        category_mimes={},
        alert_flags=frozenset(),
        message="history trimmed",
    )
    response = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="done")],
        stop_reason=LlmStopReason.END_TURN,
    )
    agent = _SummaryHarnessAgent(response, summary)

    seed_message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="seed")],
    )
    await agent.generate_impl([seed_message])

    assert len(agent.additional_messages) == 1
    additional = agent.additional_messages[0]
    assert isinstance(additional, Text)
    assert "history trimmed" in additional.plain
