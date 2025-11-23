import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

from fast_agent.context import Context
from fast_agent.core.prompt import Prompt
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.types import PromptMessageExtended


class CapturingOpenAI(OpenAILLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.captured = None

    async def _openai_completion(self, message, request_params=None, tools=None, cancellation_token=None):
        self.captured = message
        return Prompt.assistant("ok")


def _build_tool_messages():
    tool_call = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name="demo_tool", arguments={"arg": "value"}),
    )
    assistant_tool_call = Prompt.assistant("calling tool", tool_calls={"call_1": tool_call})

    tool_result_msg = PromptMessageExtended(
        role="user",
        content=[TextContent(type="text", text="tool response payload")],
        tool_results={
            "call_1": CallToolResult(
                content=[TextContent(type="text", text="result details")],
            )
        },
    )
    return assistant_tool_call, tool_result_msg


@pytest.mark.asyncio
async def test_apply_prompt_avoids_duplicate_last_message_when_using_history():
    context = Context()
    llm = CapturingOpenAI(context=context)

    assistant_tool_call, tool_result_msg = _build_tool_messages()
    history = [assistant_tool_call, tool_result_msg]

    await llm._apply_prompt_provider_specific(history, None, None)

    assert isinstance(llm.captured, list)
    assert llm.captured[0]["role"] == "assistant"
    # Tool result conversion should follow the assistant tool_calls
    assert any(msg.get("role") == "tool" for msg in llm.captured)


@pytest.mark.asyncio
async def test_apply_prompt_converts_last_message_when_history_disabled():
    context = Context()
    llm = CapturingOpenAI(context=context)

    _, tool_result_msg = _build_tool_messages()

    await llm._apply_prompt_provider_specific(
        [tool_result_msg], RequestParams(use_history=False), None
    )

    assert isinstance(llm.captured, list)
    assert llm.captured  # should send something to completion when history is off
