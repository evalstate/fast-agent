import json

import pytest
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
)
from pydantic import AnyUrl

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_decorator import LlmDecorator
from fast_agent.constants import (
    FAST_AGENT_ERROR_CHANNEL,
    FAST_AGENT_REMOVED_METADATA_CHANNEL,
)
from fast_agent.interfaces import FastAgentLLMProtocol
from fast_agent.llm.provider_types import Provider
from fast_agent.types import PromptMessageExtended, text_content


class RecordingStubLLM(FastAgentLLMProtocol):
    """Minimal FastAgentLLMProtocol implementation for testing."""

    def __init__(self, model_name: str = "passthrough") -> None:
        self._model_name = model_name
        self._provider = Provider.FAST_AGENT
        self.generated_messages: list[PromptMessageExtended] | None = None
        self._message_history: list[PromptMessageExtended] = []

    #        self.usage_accumulator = None

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def provider(self) -> Provider:
        return self._provider

    async def generate(self, messages, request_params=None, tools=None):
        self.generated_messages = messages
        self._message_history = messages
        return PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="ok")],
        )

    async def structured(self, messages, model, request_params=None):
        await self.generate(messages, request_params)
        return None, PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="ok")],
        )

    async def apply_prompt_template(self, prompt_result, prompt_name: str) -> str:
        return ""

    @property
    def message_history(self):
        return self._message_history

    @property
    def model_info(self):
        from fast_agent.llm.model_info import ModelInfo

        return ModelInfo.from_name(self.model_name, self.provider)


def make_decorator(model_name: str = "passthrough") -> tuple[LlmDecorator, RecordingStubLLM]:
    config = AgentConfig(name="tester", model=model_name)
    decorator = LlmDecorator(config=config)
    stub = RecordingStubLLM(model_name=model_name)
    decorator._llm = stub  # type: ignore[attr-defined]
    return decorator, stub


def _parse_meta_categories(blocks) -> set[str]:
    categories: set[str] = set()
    for block in blocks or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            continue
        if payload.get("type") != "fast-agent-removed":
            continue
        category = payload.get("category")
        if category:
            categories.add(category)
    return categories


@pytest.mark.asyncio
async def test_sanitizes_image_content_for_text_only_model():
    decorator, stub = make_decorator("passthrough")

    text_block = TextContent(type="text", text="Hello")
    image_block = ImageContent(type="image", data="AAA", mimeType="image/png")
    message = PromptMessageExtended(role="user", content=[text_block, image_block])

    _, summary = decorator._sanitize_messages_for_llm([message])
    assert summary is not None
    assert "vision" in summary.message.lower()
    assert "fast-agent-error" in summary.message

    await decorator.generate_impl([message])

    assert stub.generated_messages is not None
    sent_message = stub.generated_messages[0]
    # Content should have 2 text blocks: original + placeholder for removed image
    assert len(sent_message.content) == 2
    assert isinstance(sent_message.content[0], TextContent)
    assert sent_message.content[0].text == "Hello"
    # Placeholder should indicate what was removed
    assert isinstance(sent_message.content[1], TextContent)
    assert "vision" in sent_message.content[1].text.lower()
    assert "removed" in sent_message.content[1].text.lower()
    assert "image/png" in sent_message.content[1].text

    channels = sent_message.channels or {}
    assert FAST_AGENT_ERROR_CHANNEL in channels
    error_entries = channels[FAST_AGENT_ERROR_CHANNEL]
    assert len(error_entries) == 2
    assert isinstance(error_entries[0], TextContent)
    assert "image/png" in error_entries[0].text
    assert "vision" in error_entries[0].text
    assert isinstance(error_entries[1], ImageContent)

    meta_blocks = channels.get(FAST_AGENT_REMOVED_METADATA_CHANNEL, [])
    categories = _parse_meta_categories(meta_blocks)
    assert categories == {"vision"}


@pytest.mark.asyncio
async def test_removes_unsupported_tool_result_content():
    decorator, stub = make_decorator("passthrough")

    resource = BlobResourceContents(
        uri=AnyUrl("file://example.pdf"),
        mimeType="application/pdf",
        blob="AA==",
    )
    embedded = EmbeddedResource(type="resource", resource=resource)
    tool_result = CallToolResult(content=[embedded], isError=False)
    message = PromptMessageExtended(role="user", tool_results={"tool1": tool_result})

    _, summary = decorator._sanitize_messages_for_llm([message])
    assert summary is not None
    assert "document" in summary.message.lower()

    await decorator.generate_impl([message])

    assert stub.generated_messages is not None
    sent_message = stub.generated_messages[0]
    sanitized_result = sent_message.tool_results["tool1"]
    # Should have placeholder text instead of being empty
    assert len(sanitized_result.content) == 1
    assert isinstance(sanitized_result.content[0], TextContent)
    assert "document" in sanitized_result.content[0].text.lower()
    assert "removed" in sanitized_result.content[0].text.lower()
    assert "application/pdf" in sanitized_result.content[0].text

    channels = sent_message.channels or {}
    error_entries = channels[FAST_AGENT_ERROR_CHANNEL]
    assert isinstance(error_entries[0], TextContent)
    assert "tool1" in error_entries[0].text
    assert "application/pdf" in error_entries[0].text
    assert isinstance(error_entries[1], EmbeddedResource)

    meta_blocks = channels.get(FAST_AGENT_REMOVED_METADATA_CHANNEL, [])
    categories = _parse_meta_categories(meta_blocks)
    assert categories == {"document"}


@pytest.mark.asyncio
async def test_metadata_clears_when_supported_content_only():
    decorator, stub = make_decorator("passthrough")

    image_block = ImageContent(type="image", data="AAA", mimeType="image/png")
    first_message = PromptMessageExtended(role="user", content=[image_block])
    await decorator.generate_impl([first_message])

    channels = (stub.generated_messages or [])[0].channels or {}
    assert FAST_AGENT_REMOVED_METADATA_CHANNEL in channels

    second_message = PromptMessageExtended(role="user", content=[text_content("Next turn")])
    await decorator.generate_impl([second_message])

    assert stub.generated_messages is not None
    meta_blocks = (stub.generated_messages[0].channels or {}).get(
        FAST_AGENT_REMOVED_METADATA_CHANNEL, []
    )
    assert not meta_blocks
