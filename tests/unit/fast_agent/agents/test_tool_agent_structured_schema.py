import pytest
from mcp import CallToolRequest
from mcp.types import CallToolRequestParams, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason


class ToolThenStructuredLlm(PassthroughLLM):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0
        self.tool_counts: list[int] = []
        self.structured_schemas: list[dict | None] = []

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        del multipart_messages, is_template
        self.call_count += 1
        self.tool_counts.append(len(tools or []))
        self.structured_schemas.append(
            request_params.structured_schema if request_params is not None else None
        )

        if self.call_count == 1:
            return PromptMessageExtended(
                role="assistant",
                content=[text_content("use tool")],
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "call_1": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(name="get_value", arguments={}),
                    )
                },
            )

        return Prompt.assistant('{"value":"from-tool"}', stop_reason=LlmStopReason.END_TURN)


class NoToolThenStructuredLlm(PassthroughLLM):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0
        self.tool_counts: list[int] = []
        self.messages: list[list[str]] = []

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        del request_params, is_template
        self.call_count += 1
        self.tool_counts.append(len(tools or []))
        self.messages.append([message.all_text() for message in multipart_messages])

        if self.call_count == 1:
            return Prompt.assistant("no tool needed", stop_reason=LlmStopReason.END_TURN)

        return Prompt.assistant('{"value":"final"}', stop_reason=LlmStopReason.END_TURN)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_agent_structured_schema_uses_tool_runner_generate_path() -> None:
    tool_call_count = 0

    def get_value() -> str:
        nonlocal tool_call_count
        tool_call_count += 1
        return "from-tool"

    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    llm = ToolThenStructuredLlm()
    agent = ToolAgent(AgentConfig("structured"), [get_value])
    agent._llm = llm

    parsed, response = await agent.structured_schema(
        "call the tool, then return JSON",
        schema,
        RequestParams(use_history=False, max_iterations=3),
    )

    assert parsed == {"value": "from-tool"}
    assert response.last_text() == '{"value":"from-tool"}'
    assert tool_call_count == 1
    assert llm.call_count == 2
    assert llm.tool_counts == [1, 1]
    assert llm.structured_schemas == [schema, schema]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deferred_structured_schema_finalizes_when_no_tool_is_called() -> None:
    def get_value() -> str:
        return "unused"

    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    llm = NoToolThenStructuredLlm()
    agent = ToolAgent(AgentConfig("structured"), [get_value])
    agent._llm = llm

    parsed, response = await agent.structured_schema(
        "return JSON, using a tool only if needed",
        schema,
        RequestParams(
            use_history=False,
            max_iterations=3,
            structured_tool_policy="defer",
        ),
    )

    assert parsed == {"value": "final"}
    assert response.last_text() == '{"value":"final"}'
    assert llm.call_count == 2
    assert llm.tool_counts == [1, 0]
    assert any("no tool needed" in text for text in llm.messages[1])
    assert any(
        "Now produce the final answer as structured JSON" in text for text in llm.messages[1]
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deferred_structured_schema_without_tools_uses_single_call() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    llm = NoToolThenStructuredLlm()
    agent = ToolAgent(AgentConfig("structured"), [])
    agent._llm = llm

    parsed, response = await agent.structured_schema(
        "return JSON",
        schema,
        RequestParams(use_history=False, max_iterations=3, structured_tool_policy="defer"),
    )

    assert parsed is None
    assert response.last_text() == "no tool needed"
    assert llm.call_count == 1
    assert llm.tool_counts == [0]
