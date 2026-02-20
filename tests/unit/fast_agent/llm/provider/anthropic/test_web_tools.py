from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types.beta import (
    BetaCitationsWebSearchResultLocation,
    BetaCodeExecutionToolResultBlock,
    BetaContainer,
    BetaEncryptedCodeExecutionResultBlock,
)
from mcp.types import TextContent
from pydantic import ValidationError

from fast_agent.config import (
    AnthropicSettings,
    AnthropicUserLocationSettings,
    AnthropicWebFetchSettings,
    AnthropicWebSearchSettings,
    Settings,
)
from fast_agent.constants import (
    ANTHROPIC_ASSISTANT_RAW_CONTENT,
    ANTHROPIC_CITATIONS_CHANNEL,
    ANTHROPIC_CONTAINER_CHANNEL,
    ANTHROPIC_SERVER_TOOLS_CHANNEL,
)
from fast_agent.context import Context
from fast_agent.llm.provider.anthropic.beta_types import (
    Message,
    RawContentBlockDeltaEvent,
    ServerToolUseBlock,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.anthropic.web_tools import serialize_anthropic_block_payload
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason


def test_web_tool_settings_validate_domains_and_limits() -> None:
    with pytest.raises(ValidationError):
        AnthropicWebSearchSettings(enabled=True, allowed_domains=["https://example.com"])

    with pytest.raises(ValidationError):
        AnthropicWebSearchSettings(
            enabled=True,
            allowed_domains=["example.com"],
            blocked_domains=["blocked.com"],
        )

    with pytest.raises(ValidationError):
        AnthropicWebFetchSettings(enabled=True, max_content_tokens=0)


def test_serialize_anthropic_text_payload_strips_parsed_output() -> None:
    payload = serialize_anthropic_block_payload(
        {
            "type": "text",
            "text": "hello",
            "parsed_output": None,
            "citations": None,
        }
    )

    assert payload is not None
    assert payload["type"] == "text"
    assert payload["text"] == "hello"
    assert "parsed_output" not in payload


def test_serialize_anthropic_text_payload_flattens_nested_text_objects() -> None:
    payload = serialize_anthropic_block_payload(
        {
            "type": "text",
            "text": {
                "type": "text",
                "text": "hello",
                "parsed_output": None,
            },
            "parsed_output": None,
        }
    )

    assert payload is not None
    assert payload["text"] == "hello"
    assert "parsed_output" not in payload


class _DummyStreamManager:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


class _FinalMessageValidationFailureStream:
    def __init__(self, error: Exception) -> None:
        self._error = error
        self._emitted = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._emitted:
            raise StopAsyncIteration
        self._emitted = True
        return RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=TextDelta(type="text_delta", text="streamed fallback text"),
        )

    async def get_final_message(self):
        raise self._error


def _create_llm(
    *,
    model: str,
    web_search: AnthropicWebSearchSettings | None = None,
    web_fetch: AnthropicWebFetchSettings | None = None,
) -> AnthropicLLM:
    settings = Settings()
    settings.anthropic = AnthropicSettings(
        api_key="test-key",
        web_search=web_search or AnthropicWebSearchSettings(),
        web_fetch=web_fetch or AnthropicWebFetchSettings(),
    )
    context = Context(config=settings)
    return AnthropicLLM(context=context, model=model, name="test-agent")


def _user_message_param() -> dict[str, object]:
    return {
        "role": "user",
        "content": [{"type": "text", "text": "hello"}],
    }


def _user_message_extended() -> PromptMessageExtended:
    return PromptMessageExtended(role="user", content=[TextContent(type="text", text="hello")])


def test_web_search_enabled_property_reflects_search_or_fetch() -> None:
    llm = _create_llm(
        model="claude-sonnet-4-5",
        web_search=AnthropicWebSearchSettings(enabled=False),
        web_fetch=AnthropicWebFetchSettings(enabled=True),
    )

    assert llm.web_tools_enabled == (False, True)
    assert llm.web_search_enabled is True


@pytest.mark.asyncio
async def test_request_includes_web_tools_and_required_beta_for_46_model() -> None:
    llm = _create_llm(
        model="claude-opus-4-6",
        web_search=AnthropicWebSearchSettings(
            enabled=True,
            max_uses=2,
            allowed_domains=["example.com"],
            user_location=AnthropicUserLocationSettings(
                city="London",
                country="UK",
            ),
        ),
        web_fetch=AnthropicWebFetchSettings(
            enabled=True,
            citations_enabled=True,
            max_content_tokens=2048,
        ),
    )

    final_message = Message(
        id="msg_1",
        type="message",
        role="assistant",
        content=[TextBlock(type="text", text="done")],
        model="claude-opus-4-6",
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    with patch("fast_agent.llm.provider.anthropic.llm_anthropic.AsyncAnthropic") as mock_cls:
        client = MagicMock()
        client.beta.messages.stream.return_value = _DummyStreamManager()
        mock_cls.return_value = client

        with patch.object(llm, "_process_stream", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = (final_message, [], [])
            await llm._anthropic_completion(
                _user_message_param(),
                history=[],
                current_extended=_user_message_extended(),
            )

    request_kwargs = client.beta.messages.stream.call_args.kwargs
    tools = request_kwargs.get("tools", [])
    search_tools = [tool for tool in tools if tool.get("name") == "web_search"]
    fetch_tools = [tool for tool in tools if tool.get("name") == "web_fetch"]

    assert len(search_tools) == 1
    assert search_tools[0]["type"] == "web_search_20260209"
    assert search_tools[0]["allowed_domains"] == ["example.com"]
    assert search_tools[0]["max_uses"] == 2
    assert search_tools[0]["user_location"]["city"] == "London"

    assert len(fetch_tools) == 1
    assert fetch_tools[0]["type"] == "web_fetch_20260209"
    assert fetch_tools[0]["citations"] == {"enabled": True}
    assert fetch_tools[0]["max_content_tokens"] == 2048

    assert "code-execution-web-tools-2026-02-09" in request_kwargs.get("betas", [])


@pytest.mark.asyncio
async def test_request_uses_legacy_web_tool_versions_for_non_46_models() -> None:
    llm = _create_llm(
        model="claude-sonnet-4-5",
        web_search=AnthropicWebSearchSettings(enabled=True),
        web_fetch=AnthropicWebFetchSettings(enabled=True),
    )

    final_message = Message(
        id="msg_2",
        type="message",
        role="assistant",
        content=[TextBlock(type="text", text="done")],
        model="claude-sonnet-4-5",
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    with patch("fast_agent.llm.provider.anthropic.llm_anthropic.AsyncAnthropic") as mock_cls:
        client = MagicMock()
        client.beta.messages.stream.return_value = _DummyStreamManager()
        mock_cls.return_value = client

        with patch.object(llm, "_process_stream", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = (final_message, [], [])
            await llm._anthropic_completion(
                _user_message_param(),
                history=[],
                current_extended=_user_message_extended(),
            )

    request_kwargs = client.beta.messages.stream.call_args.kwargs
    tools = request_kwargs.get("tools", [])
    search_tools = [tool for tool in tools if tool.get("name") == "web_search"]
    fetch_tools = [tool for tool in tools if tool.get("name") == "web_fetch"]

    assert search_tools[0]["type"] == "web_search_20250305"
    assert fetch_tools[0]["type"] == "web_fetch_20250910"
    assert "code-execution-web-tools-2026-02-09" not in request_kwargs.get("betas", [])


@pytest.mark.asyncio
async def test_server_tool_only_tool_use_does_not_create_mcp_tool_calls() -> None:
    llm = _create_llm(model="claude-opus-4-6")

    final_message = Message(
        id="msg_3",
        type="message",
        role="assistant",
        content=[
            ServerToolUseBlock(
                type="server_tool_use",
                id="srv_1",
                name="web_search",
                input={"query": "news"},
            )
        ],
        model="claude-opus-4-6",
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    with patch("fast_agent.llm.provider.anthropic.llm_anthropic.AsyncAnthropic") as mock_cls:
        client = MagicMock()
        client.beta.messages.stream.return_value = _DummyStreamManager()
        mock_cls.return_value = client

        with patch.object(llm, "_process_stream", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = (final_message, [], [])
            result = await llm._anthropic_completion(
                _user_message_param(),
                history=[],
                current_extended=_user_message_extended(),
            )

    assert result.stop_reason == LlmStopReason.END_TURN
    assert result.tool_calls is None
    assert result.channels is not None
    assert ANTHROPIC_SERVER_TOOLS_CHANNEL in result.channels
    assert ANTHROPIC_ASSISTANT_RAW_CONTENT in result.channels


@pytest.mark.asyncio
async def test_code_execution_tool_results_are_preserved_in_server_tool_channel() -> None:
    llm = _create_llm(model="claude-opus-4-6")

    final_message = Message(
        id="msg_3b",
        type="message",
        role="assistant",
        content=[
            ServerToolUseBlock(
                type="server_tool_use",
                id="srv_1",
                name="code_execution",
                input={"code": "print(1)"},
            ),
            BetaCodeExecutionToolResultBlock(
                type="code_execution_tool_result",
                tool_use_id="srv_1",
                content=BetaEncryptedCodeExecutionResultBlock(
                    type="encrypted_code_execution_result",
                    content=[],
                    encrypted_stdout="enc",
                    return_code=0,
                    stderr="",
                ),
            ),
            TextBlock(type="text", text="done"),
        ],
        model="claude-opus-4-6",
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    with patch("fast_agent.llm.provider.anthropic.llm_anthropic.AsyncAnthropic") as mock_cls:
        client = MagicMock()
        client.beta.messages.stream.return_value = _DummyStreamManager()
        mock_cls.return_value = client

        with patch.object(llm, "_process_stream", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = (final_message, [], [])
            result = await llm._anthropic_completion(
                _user_message_param(),
                history=[],
                current_extended=_user_message_extended(),
            )

    assert result.channels is not None
    payloads = result.channels.get(ANTHROPIC_SERVER_TOOLS_CHANNEL, [])
    payload_types = [json.loads(block.text).get("type") for block in payloads]
    assert "server_tool_use" in payload_types
    assert "code_execution_tool_result" in payload_types
    replay_payloads = result.channels.get(ANTHROPIC_ASSISTANT_RAW_CONTENT, [])
    replay_types = [json.loads(block.text).get("type") for block in replay_payloads]
    assert replay_types == ["server_tool_use", "code_execution_tool_result", "text"]


@pytest.mark.asyncio
async def test_tool_use_with_thinking_persists_raw_assistant_content_channel() -> None:
    llm = _create_llm(model="claude-haiku-4-5")

    final_message = Message(
        id="msg_3c",
        type="message",
        role="assistant",
        content=[
            ThinkingBlock(type="thinking", thinking="I should call a tool", signature="sig_123"),
            TextBlock(type="text", text="Let me fetch data."),
            ToolUseBlock(
                type="tool_use",
                id="toolu_123",
                name="mcp__demo",
                input={"q": "status"},
            ),
        ],
        model="claude-haiku-4-5",
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    with patch("fast_agent.llm.provider.anthropic.llm_anthropic.AsyncAnthropic") as mock_cls:
        client = MagicMock()
        client.beta.messages.stream.return_value = _DummyStreamManager()
        mock_cls.return_value = client

        with patch.object(llm, "_process_stream", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = (final_message, ["I should call a tool"], [])
            result = await llm._anthropic_completion(
                _user_message_param(),
                history=[],
                current_extended=_user_message_extended(),
            )

    assert result.stop_reason == LlmStopReason.TOOL_USE
    assert result.channels is not None
    replay_payloads = result.channels.get(ANTHROPIC_ASSISTANT_RAW_CONTENT, [])
    replay_types = [json.loads(block.text).get("type") for block in replay_payloads]
    assert replay_types == ["thinking", "text", "tool_use"]


@pytest.mark.asyncio
async def test_response_container_id_is_persisted_for_followup_requests() -> None:
    llm = _create_llm(model="claude-opus-4-6")

    final_message = Message(
        id="msg_container",
        type="message",
        role="assistant",
        content=[TextBlock(type="text", text="done")],
        model="claude-opus-4-6",
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=20),
        container=BetaContainer(id="cont_123", expires_at=datetime.now(UTC)),
    )

    with patch("fast_agent.llm.provider.anthropic.llm_anthropic.AsyncAnthropic") as mock_cls:
        client = MagicMock()
        client.beta.messages.stream.return_value = _DummyStreamManager()
        mock_cls.return_value = client

        with patch.object(llm, "_process_stream", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = (final_message, [], [])
            result = await llm._anthropic_completion(
                _user_message_param(),
                history=[],
                current_extended=_user_message_extended(),
            )

    assert result.channels is not None
    container_blocks = result.channels.get(ANTHROPIC_CONTAINER_CHANNEL, [])
    assert len(container_blocks) == 1
    assert json.loads(container_blocks[0].text) == {"id": "cont_123"}


@pytest.mark.asyncio
async def test_request_reuses_container_id_from_history_channel() -> None:
    llm = _create_llm(model="claude-opus-4-6")

    final_message = Message(
        id="msg_container_reuse",
        type="message",
        role="assistant",
        content=[TextBlock(type="text", text="done")],
        model="claude-opus-4-6",
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    history = [
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Previous assistant turn")],
            channels={
                ANTHROPIC_CONTAINER_CHANNEL: [
                    TextContent(type="text", text=json.dumps({"id": "cont_history"}))
                ]
            },
        )
    ]

    with patch("fast_agent.llm.provider.anthropic.llm_anthropic.AsyncAnthropic") as mock_cls:
        client = MagicMock()
        client.beta.messages.stream.return_value = _DummyStreamManager()
        mock_cls.return_value = client

        with patch.object(llm, "_process_stream", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = (final_message, [], [])
            await llm._anthropic_completion(
                _user_message_param(),
                history=history,
                current_extended=_user_message_extended(),
            )

    request_kwargs = client.beta.messages.stream.call_args.kwargs
    assert request_kwargs.get("container") == "cont_history"


def test_convert_message_to_message_param_keeps_code_execution_tool_result() -> None:
    message = Message(
        id="msg_3c",
        type="message",
        role="assistant",
        content=[
            ServerToolUseBlock(
                type="server_tool_use",
                id="srv_1",
                name="code_execution",
                input={"code": "print(1)"},
            ),
            BetaCodeExecutionToolResultBlock(
                type="code_execution_tool_result",
                tool_use_id="srv_1",
                content=BetaEncryptedCodeExecutionResultBlock(
                    type="encrypted_code_execution_result",
                    content=[],
                    encrypted_stdout="enc",
                    return_code=0,
                    stderr="",
                ),
            ),
            TextBlock(type="text", text="done"),
        ],
        model="claude-opus-4-6",
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    converted = AnthropicLLM.convert_message_to_message_param(message)
    blocks = converted["content"]
    assert isinstance(blocks, list)
    block_types = [block.get("type") for block in blocks if isinstance(block, dict)]
    assert "server_tool_use" in block_types
    assert "code_execution_tool_result" in block_types


def test_convert_message_to_message_param_skips_text_blocks_with_null_text() -> None:
    message = Message.model_construct(
        id="msg_null_text",
        type="message",
        role="assistant",
        content=[
            TextBlock.model_construct(type="text", text=None),
            TextBlock(type="text", text="safe text"),
        ],
        model="claude-opus-4-6",
        stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=1),
    )

    converted = AnthropicLLM.convert_message_to_message_param(message)
    blocks = converted["content"]
    assert isinstance(blocks, list)
    assert blocks == [{"type": "text", "text": "safe text"}]


@pytest.mark.asyncio
async def test_anthropic_completion_skips_null_text_blocks_in_final_message() -> None:
    llm = _create_llm(model="claude-opus-4-6")

    final_message = Message.model_construct(
        id="msg_null_text_final",
        type="message",
        role="assistant",
        content=[
            TextBlock.model_construct(type="text", text=None),
            TextBlock(type="text", text="safe text"),
        ],
        model="claude-opus-4-6",
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    with patch("fast_agent.llm.provider.anthropic.llm_anthropic.AsyncAnthropic") as mock_cls:
        client = MagicMock()
        client.beta.messages.stream.return_value = _DummyStreamManager()
        mock_cls.return_value = client

        with patch.object(llm, "_process_stream", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = (final_message, [], [])
            result = await llm._anthropic_completion(
                _user_message_param(),
                history=[],
                current_extended=_user_message_extended(),
            )

    assert result.content == [TextContent(type="text", text="safe text")]


@pytest.mark.asyncio
async def test_text_block_citations_are_captured_in_channel() -> None:
    llm = _create_llm(model="claude-opus-4-6")

    final_message = Message(
        id="msg_4",
        type="message",
        role="assistant",
        content=[
            TextBlock(
                type="text",
                text="Here is a source.",
                citations=[
                    BetaCitationsWebSearchResultLocation(
                        type="web_search_result_location",
                        cited_text="source text",
                        encrypted_index="enc",
                        title="Example",
                        url="https://example.com/path",
                    )
                ],
            )
        ],
        model="claude-opus-4-6",
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    with patch("fast_agent.llm.provider.anthropic.llm_anthropic.AsyncAnthropic") as mock_cls:
        client = MagicMock()
        client.beta.messages.stream.return_value = _DummyStreamManager()
        mock_cls.return_value = client

        with patch.object(llm, "_process_stream", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = (final_message, [], [])
            result = await llm._anthropic_completion(
                _user_message_param(),
                history=[],
                current_extended=_user_message_extended(),
            )

    assert result.channels is not None
    assert ANTHROPIC_CITATIONS_CHANNEL in result.channels
    citation_block = result.channels[ANTHROPIC_CITATIONS_CHANNEL][0]
    assert isinstance(citation_block, TextContent)
    payload_text = citation_block.text
    payload = json.loads(payload_text)
    assert payload["url"] == "https://example.com/path"
    assert payload["title"] == "Example"


@pytest.mark.asyncio
async def test_streamed_text_fallback_when_final_message_has_no_text_blocks() -> None:
    llm = _create_llm(model="claude-opus-4-6")

    final_message = Message(
        id="msg_5",
        type="message",
        role="assistant",
        content=[],
        model="claude-opus-4-6",
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    with patch("fast_agent.llm.provider.anthropic.llm_anthropic.AsyncAnthropic") as mock_cls:
        client = MagicMock()
        client.beta.messages.stream.return_value = _DummyStreamManager()
        mock_cls.return_value = client

        with patch.object(llm, "_process_stream", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = (
                final_message,
                [],
                ["Top headlines: ", "One, Two, Three"],
            )
            result = await llm._anthropic_completion(
                _user_message_param(),
                history=[],
                current_extended=_user_message_extended(),
            )

    assert result.content
    first_block = result.content[0]
    assert isinstance(first_block, TextContent)
    assert first_block.text == "Top headlines: One, Two, Three"


@pytest.mark.asyncio
async def test_streamed_text_overrides_mismatched_final_text_blocks() -> None:
    llm = _create_llm(model="claude-opus-4-6")

    final_message = Message(
        id="msg_6",
        type="message",
        role="assistant",
        content=[TextBlock(type="text", text="Short final text")],
        model="claude-opus-4-6",
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    with patch("fast_agent.llm.provider.anthropic.llm_anthropic.AsyncAnthropic") as mock_cls:
        client = MagicMock()
        client.beta.messages.stream.return_value = _DummyStreamManager()
        mock_cls.return_value = client

        with patch.object(llm, "_process_stream", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = (
                final_message,
                [],
                ["Richer streamed text ", "with search results"],
            )
            result = await llm._anthropic_completion(
                _user_message_param(),
                history=[],
                current_extended=_user_message_extended(),
            )

    assert result.content
    first_block = result.content[0]
    assert isinstance(first_block, TextContent)
    assert first_block.text == "Richer streamed text with search results"


@pytest.mark.asyncio
async def test_process_stream_falls_back_when_final_message_has_invalid_text_block() -> None:
    llm = _create_llm(model="claude-opus-4-6")

    try:
        TextBlock.model_validate({"type": "text", "text": None})
    except ValidationError as exc:
        error = exc
    else:
        raise AssertionError("Expected BetaTextBlock validation error")

    stream = _FinalMessageValidationFailureStream(error)
    message, thinking, streamed = await llm._process_stream(cast("Any", stream), "claude-opus-4-6")

    assert message.role == "assistant"
    assert message.content == []
    assert thinking == []
    assert streamed == ["streamed fallback text"]
