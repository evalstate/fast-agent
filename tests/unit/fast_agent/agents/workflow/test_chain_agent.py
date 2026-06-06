from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from mcp.types import ImageContent
from pydantic import BaseModel

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.workflow.chain_agent import ChainAgent
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.types import PromptMessageExtended, RequestParams

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcp import Tool
    from mcp.types import PromptMessage


class StructuredResult(BaseModel):
    value: str


class RecordingAgent(LlmAgent):
    def __init__(self, name: str, response_text: str) -> None:
        super().__init__(AgentConfig(name))
        self.response_text = response_text
        self.generate_inputs: list[list[PromptMessageExtended]] = []
        self.structured_inputs: list[list[PromptMessageExtended]] = []
        self.structured_schema_inputs: list[list[PromptMessageExtended]] = []
        self.structured_schemas: list[dict[str, Any]] = []
        self.initialize_calls = 0

    async def initialize(self) -> None:
        self.initialize_calls += 1
        await super().initialize()

    async def generate(
        self,
        messages: str
        | PromptMessage
        | PromptMessageExtended
        | Sequence[str | PromptMessage | PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        del request_params, tools
        assert isinstance(messages, list)
        message_list = cast("list[PromptMessageExtended]", messages)
        self.generate_inputs.append(message_list)
        return PromptMessageExtended(
            role="assistant",
            content=[text_content(self.response_text)],
        )

    async def structured(
        self,
        messages: str
        | PromptMessage
        | PromptMessageExtended
        | Sequence[str | PromptMessage | PromptMessageExtended],
        model: type[StructuredResult],
        request_params: RequestParams | None = None,
    ) -> tuple[StructuredResult | None, PromptMessageExtended]:
        del request_params
        assert isinstance(messages, list)
        message_list = cast("list[PromptMessageExtended]", messages)
        self.structured_inputs.append(message_list)
        message = PromptMessageExtended(
            role="assistant",
            content=[text_content(self.response_text)],
        )
        return model(value=self.response_text), message

    async def structured_schema(
        self,
        messages: str
        | PromptMessage
        | PromptMessageExtended
        | Sequence[str | PromptMessage | PromptMessageExtended],
        schema: dict[str, Any],
        request_params: RequestParams | None = None,
    ) -> tuple[dict[str, str], PromptMessageExtended]:
        del request_params
        assert isinstance(messages, list)
        message_list = cast("list[PromptMessageExtended]", messages)
        self.structured_schema_inputs.append(message_list)
        self.structured_schemas.append(schema)
        message = PromptMessageExtended(
            role="assistant",
            content=[text_content(self.response_text)],
        )
        return {"value": self.response_text}, message


class FailingGenerateAgent(RecordingAgent):
    async def generate(
        self,
        messages: str
        | PromptMessage
        | PromptMessageExtended
        | Sequence[str | PromptMessage | PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        del messages, request_params, tools
        raise RuntimeError(f"{self.name} failed")


class MultimodalRecordingAgent(RecordingAgent):
    async def generate(
        self,
        messages: str
        | PromptMessage
        | PromptMessageExtended
        | Sequence[str | PromptMessage | PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        del request_params, tools
        assert isinstance(messages, list)
        self.generate_inputs.append(cast("list[PromptMessageExtended]", messages))
        return PromptMessageExtended(
            role="assistant",
            content=[
                text_content(self.response_text),
                ImageContent(type="image", data="aW1hZ2U=", mimeType="image/png"),
            ],
        )


def test_chain_agent_rejects_empty_agent_list() -> None:
    with pytest.raises(AgentConfigError, match="requires at least one agent"):
        ChainAgent(AgentConfig("chain"), [])


@pytest.mark.asyncio
async def test_chain_structured_uses_final_agent_once() -> None:
    first = RecordingAgent("first", "intermediate")
    final = RecordingAgent("final", "structured")
    chain = ChainAgent(AgentConfig("chain"), [first, final])

    result, message = await chain.structured(
        PromptMessageExtended(role="user", content=[text_content("start")]),
        StructuredResult,
    )

    assert result == StructuredResult(value="structured")
    assert message.all_text() == "structured"
    assert len(first.generate_inputs) == 1
    assert final.generate_inputs == []
    assert len(final.structured_inputs) == 1
    final_input = final.structured_inputs[0][0]
    assert final_input.role == "user"
    assert final_input.all_text() == "intermediate"


@pytest.mark.asyncio
async def test_chain_structured_schema_uses_final_agent_once() -> None:
    first = RecordingAgent("first", "intermediate")
    final = RecordingAgent("final", "structured")
    chain = ChainAgent(AgentConfig("chain"), [first, final])
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}

    result, message = await chain.structured_schema(
        PromptMessageExtended(role="user", content=[text_content("start")]),
        schema,
    )

    assert result == {"value": "structured"}
    assert message.all_text() == "structured"
    assert len(first.generate_inputs) == 1
    assert final.generate_inputs == []
    assert final.structured_inputs == []
    assert len(final.structured_schema_inputs) == 1
    assert final.structured_schemas == [schema]
    final_input = final.structured_schema_inputs[0][0]
    assert final_input.role == "user"
    assert final_input.all_text() == "intermediate"


@pytest.mark.asyncio
async def test_chain_initialize_is_idempotent_for_child_agents() -> None:
    first = RecordingAgent("first", "one")
    second = RecordingAgent("second", "two")
    chain = ChainAgent(AgentConfig("chain"), [first, second])

    await chain.initialize()
    await chain.initialize()

    assert first.initialize_calls == 1
    assert second.initialize_calls == 1


@pytest.mark.asyncio
async def test_cumulative_chain_structured_uses_prior_outputs_as_context() -> None:
    first = RecordingAgent("first", "first output")
    second = RecordingAgent("second", "second output")
    final = RecordingAgent("final", "structured")
    chain = ChainAgent(AgentConfig("chain"), [first, second, final], cumulative=True)

    result, _message = await chain.structured(
        PromptMessageExtended(role="user", content=[text_content("start")]),
        StructuredResult,
    )

    assert result == StructuredResult(value="structured")
    assert len(first.generate_inputs) == 1
    assert len(second.generate_inputs) == 1
    assert final.generate_inputs == []
    assert len(final.structured_inputs) == 1
    assert [message.all_text() for message in final.structured_inputs[0]] == [
        "start",
        "first output",
        "second output",
    ]


@pytest.mark.asyncio
async def test_cumulative_chain_structured_schema_uses_prior_outputs_as_context() -> None:
    first = RecordingAgent("first", "first output")
    second = RecordingAgent("second", "second output")
    final = RecordingAgent("final", "structured")
    chain = ChainAgent(AgentConfig("chain"), [first, second, final], cumulative=True)
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}

    result, _message = await chain.structured_schema(
        PromptMessageExtended(role="user", content=[text_content("start")]),
        schema,
    )

    assert result == {"value": "structured"}
    assert len(first.generate_inputs) == 1
    assert len(second.generate_inputs) == 1
    assert final.generate_inputs == []
    assert final.structured_inputs == []
    assert len(final.structured_schema_inputs) == 1
    assert final.structured_schemas == [schema]
    assert [message.all_text() for message in final.structured_schema_inputs[0]] == [
        "start",
        "first output",
        "second output",
    ]


@pytest.mark.asyncio
async def test_cumulative_chain_structured_preserves_prior_response_content_blocks() -> None:
    first = MultimodalRecordingAgent("first", "first output")
    final = RecordingAgent("final", "structured")
    chain = ChainAgent(AgentConfig("chain"), [first, final], cumulative=True)

    result, _message = await chain.structured(
        PromptMessageExtended(role="user", content=[text_content("start")]),
        StructuredResult,
    )

    assert result == StructuredResult(value="structured")
    prior_response = final.structured_inputs[0][1]
    assert prior_response.role == "user"
    assert len(prior_response.content) == 2
    assert prior_response.content[0] == text_content("first output")
    assert isinstance(prior_response.content[1], ImageContent)


@pytest.mark.asyncio
async def test_chain_structured_propagates_child_execution_errors() -> None:
    first = FailingGenerateAgent("first", "ignored")
    final = RecordingAgent("final", "structured")
    chain = ChainAgent(AgentConfig("chain"), [first, final])

    with pytest.raises(RuntimeError, match="first failed"):
        await chain.structured(
            PromptMessageExtended(role="user", content=[text_content("start")]),
            StructuredResult,
        )


@pytest.mark.asyncio
async def test_chain_structured_schema_propagates_child_execution_errors() -> None:
    first = FailingGenerateAgent("first", "ignored")
    final = RecordingAgent("final", "structured")
    chain = ChainAgent(AgentConfig("chain"), [first, final])

    with pytest.raises(RuntimeError, match="first failed"):
        await chain.structured_schema(
            PromptMessageExtended(role="user", content=[text_content("start")]),
            {"type": "object", "properties": {"value": {"type": "string"}}},
        )
