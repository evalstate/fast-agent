from __future__ import annotations

from typing import Any

import pytest
from mcp.types import TextContent
from rich.text import Text

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.llm_decorator import RemovedContentSummary
from fast_agent.constants import ANTHROPIC_CITATIONS_CHANNEL, ANTHROPIC_SERVER_TOOLS_CHANNEL
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui.console_display import ConsoleDisplay


class _CaptureDisplay(ConsoleDisplay):
    def __init__(self) -> None:
        super().__init__(config=None)
        self.calls: list[dict[str, Any]] = []

    async def show_assistant_message(
        self,
        message_text: str | Text | PromptMessageExtended,
        bottom_items: list[str] | None = None,
        highlight_index: int | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Text | None = None,
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
            "render_markdown": render_markdown,
            "show_hook_indicator": show_hook_indicator,
        }
        self.calls.append(payload)


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

    additional = call.get("additional_message")
    assert isinstance(additional, Text)
    plain = additional.plain
    assert "Sources" in plain
    assert "Web activity: web_search x1" in plain


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
