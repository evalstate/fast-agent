from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mcp_types import CallToolRequest, CallToolRequestParams

from fast_agent.event_progress import ProgressAction
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.prompt_cache import CACHE_KEEP_WARM_MESSAGE, PromptCacheWarmError
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.llm.usage_tracking import (
    CompletionTokenUsage,
    PromptTokenUsage,
    TurnUsage,
    UsageSchema,
)
from fast_agent.mcp.prompt import Prompt

if TYPE_CHECKING:
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


class _ProviderLLM(PassthroughLLM):
    """Streams a reply and reports usage, as a real provider does."""

    reports_usage = True

    async def _apply_prompt_provider_specific(
        self, multipart_messages, request_params=None, tools=None, is_template=False
    ) -> PromptMessageExtended:
        self.sent = list(multipart_messages)
        self.verb_during_request = self.verb
        self._notify_stream_listeners(StreamChunk(text="OK", is_reasoning=False))
        if self.reports_usage:
            self._finalize_turn_usage(
                TurnUsage(
                    provider=Provider.FAST_AGENT,
                    usage_schema=UsageSchema.OPENAI_CHAT,
                    model="passthrough",
                    prompt=PromptTokenUsage(total=5000, cache_read=4900),
                    completion=CompletionTokenUsage(total=3),
                )
            )
        return Prompt.assistant("OK")


def _pending_tool_call() -> PromptMessageExtended:
    message = Prompt.assistant("calling")
    message.tool_calls = {
        "call_1": CallToolRequest(
            method="tools/call", params=CallToolRequestParams(name="slow", arguments={})
        )
    }
    return message


@pytest.mark.asyncio
async def test_keep_warm_request_is_recorded_but_stays_out_of_the_conversation() -> None:
    llm = _ProviderLLM()
    llm.usage_accumulator.set_context_estimate(1234)
    heard: list[StreamChunk] = []
    llm.add_stream_listener(heard.append)
    conversation = [Prompt.user("hello"), Prompt.assistant("hi")]

    usage = await llm.warm_cache(conversation)

    assert usage.prompt.cache_read == 4900
    assert usage.timing is not None
    assert llm.usage_accumulator.turns == [usage]
    assert llm.usage_accumulator.current_context_tokens == 1234
    assert heard == []
    assert llm.verb_during_request is ProgressAction.WARMING_CACHE
    assert llm.verb is None
    assert [message.first_text() for message in llm.sent] == [
        "hello",
        "hi",
        CACHE_KEEP_WARM_MESSAGE,
    ]
    assert len(conversation) == 2

    llm._notify_stream_listeners(StreamChunk(text="later", is_reasoning=False))
    assert [chunk.text for chunk in heard] == ["later"]


@pytest.mark.asyncio
async def test_listeners_changed_during_a_keep_warm_request_keep_their_registration() -> None:
    heard: list[str] = []

    class _ChangingLLM(_ProviderLLM):
        async def _apply_prompt_provider_specific(
            self, multipart_messages, request_params=None, tools=None, is_template=False
        ) -> PromptMessageExtended:
            remove_first()
            self.add_stream_listener(lambda chunk: heard.append(f"second:{chunk.text}"))
            return await super()._apply_prompt_provider_specific(multipart_messages)

    llm = _ChangingLLM()
    remove_first = llm.add_stream_listener(lambda chunk: heard.append(f"first:{chunk.text}"))

    await llm.warm_cache([Prompt.user("hello"), Prompt.assistant("hi")])
    llm._notify_stream_listeners(StreamChunk(text="later", is_reasoning=False))

    assert heard == ["second:later"]


@pytest.mark.asyncio
async def test_tool_call_reply_to_a_keep_warm_request_is_an_error() -> None:
    class _ToolCallingLLM(_ProviderLLM):
        async def _apply_prompt_provider_specific(
            self, multipart_messages, request_params=None, tools=None, is_template=False
        ) -> PromptMessageExtended:
            await super()._apply_prompt_provider_specific(multipart_messages)
            return _pending_tool_call()

    with pytest.raises(PromptCacheWarmError):
        await _ToolCallingLLM().warm_cache([Prompt.user("hello"), Prompt.assistant("hi")])


@pytest.mark.asyncio
async def test_keep_warm_request_leaves_out_tool_calls_that_are_still_running() -> None:
    llm = _ProviderLLM()

    await llm.warm_cache([Prompt.user("do it"), _pending_tool_call()])

    assert [message.role for message in llm.sent] == ["user", "user"]


@pytest.mark.asyncio
async def test_keep_warm_request_without_reported_usage_is_an_error() -> None:
    llm = _ProviderLLM()
    llm.reports_usage = False

    with pytest.raises(PromptCacheWarmError):
        await llm.warm_cache([Prompt.user("hello")])
