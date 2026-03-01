from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, cast

import pytest
from mcp.types import TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.constants import REASONING
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason


class _FakeStreamHandle:
    def __init__(self, *, has_scrolled: bool, preserve_result: bool) -> None:
        self._has_scrolled = has_scrolled
        self._preserve_result = preserve_result
        self.preserve_called = False
        self.finalize_calls: list[PromptMessageExtended] = []
        self.close_calls = 0
        self.wait_for_drain_calls = 0

    def update(self, _chunk: str) -> None:
        return

    def update_chunk(self, _chunk) -> None:  # pragma: no cover - not used in these tests
        return

    def finalize(self, message: PromptMessageExtended | str) -> None:
        if isinstance(message, PromptMessageExtended):
            self.finalize_calls.append(message)

    def close(self) -> None:
        self.close_calls += 1

    def handle_tool_event(self, _event_type: str, info: dict[str, Any] | None = None) -> None:
        return

    def has_scrolled(self) -> bool:
        return self._has_scrolled

    def preserve_final_frame(self) -> bool:
        self.preserve_called = True
        return self._preserve_result

    async def wait_for_drain(self) -> None:
        self.wait_for_drain_calls += 1


class _FakeDisplay:
    def __init__(self, handle: _FakeStreamHandle) -> None:
        self._handle = handle

    def resolve_streaming_preferences(self) -> tuple[bool, str]:
        return True, "markdown"

    @contextmanager
    def streaming_assistant_message(self, **_kwargs: object) -> Iterator[_FakeStreamHandle]:
        try:
            yield self._handle
        finally:
            self._handle.close()


class _FakeLLM:
    def __init__(self) -> None:
        self.model_name = "fake-model"
        self.websocket_turn_indicator = None

    def add_stream_listener(self, _listener):
        return lambda: None

    def add_tool_stream_listener(self, _listener):
        return lambda: None


class _StreamingHarnessAgent(LlmAgent):
    def __init__(self, *, handle: _FakeStreamHandle, response: PromptMessageExtended) -> None:
        super().__init__(AgentConfig("stream-handoff"))
        self.display = _FakeDisplay(handle)  # type: ignore[assignment]
        self._llm = cast("Any", _FakeLLM())
        self._response = response
        self.shown_messages: list[dict[str, Any]] = []
        self.url_elicitation_calls: list[str | None] = []

    def _should_stream(self) -> bool:
        return True

    async def _generate_with_summary(
        self,
        messages: list[PromptMessageExtended],
        request_params=None,
        tools=None,
    ) -> tuple[PromptMessageExtended, None]:
        return self._response, None

    async def show_assistant_message(
        self,
        message: PromptMessageExtended,
        bottom_items=None,
        highlight_items=None,
        max_item_length=None,
        name=None,
        model=None,
        additional_message=None,
        render_markdown: bool | None = None,
        show_hook_indicator: bool | None = None,
        render_message: bool = True,
    ) -> None:
        self.shown_messages.append(
            {
                "message": message,
                "additional_message": additional_message,
                "render_markdown": render_markdown,
                "show_hook_indicator": show_hook_indicator,
                "render_message": render_message,
            }
        )

    def _display_url_elicitations_from_history(self, agent_name: str | None) -> None:
        self.url_elicitation_calls.append(agent_name)


def _response_message(text: str = "done") -> PromptMessageExtended:
    return PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        stop_reason=LlmStopReason.END_TURN,
    )


def _seed_message() -> PromptMessageExtended:
    return PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="seed")],
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_impl_preserves_streamed_frame_and_skips_reprint_when_safe() -> None:
    handle = _FakeStreamHandle(has_scrolled=False, preserve_result=True)
    response = _response_message("short streamed response")
    agent = _StreamingHarnessAgent(handle=handle, response=response)

    result = await agent.generate_impl([_seed_message()])

    assert result is response
    assert handle.wait_for_drain_calls == 1
    assert handle.preserve_called is True
    assert handle.finalize_calls == [response]
    assert len(agent.shown_messages) == 1
    assert agent.shown_messages[0]["render_message"] is False
    assert agent.url_elicitation_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_impl_reprints_when_stream_scrolled() -> None:
    handle = _FakeStreamHandle(has_scrolled=True, preserve_result=True)
    response = _response_message("response")
    agent = _StreamingHarnessAgent(handle=handle, response=response)

    result = await agent.generate_impl([_seed_message()])

    assert result is response
    assert handle.wait_for_drain_calls == 1
    assert handle.preserve_called is False
    assert handle.finalize_calls == [response]
    assert len(agent.shown_messages) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_impl_preserves_streamed_frame_with_reasoning_channel() -> None:
    handle = _FakeStreamHandle(has_scrolled=False, preserve_result=True)
    response = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="answer")],
        stop_reason=LlmStopReason.END_TURN,
        channels={REASONING: [TextContent(type="text", text="thought")]} ,
    )
    agent = _StreamingHarnessAgent(handle=handle, response=response)

    result = await agent.generate_impl([_seed_message()])

    assert result is response
    assert handle.wait_for_drain_calls == 1
    assert handle.preserve_called is True
    assert handle.finalize_calls == [response]
    assert len(agent.shown_messages) == 1
    assert agent.shown_messages[0]["render_message"] is False
