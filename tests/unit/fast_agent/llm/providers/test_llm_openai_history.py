import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

from fast_agent.context import Context
from fast_agent.core.prompt import Prompt
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.types import PromptMessageExtended


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
async def test_apply_prompt_avoids_duplicate_last_message_when_using_history(monkeypatch):
    context = Context()
    llm = OpenAILLM(context=context)

    assistant_tool_call, tool_result_msg = _build_tool_messages()
    llm._message_history = [assistant_tool_call, tool_result_msg]

    captured = {"message": "unset"}

    async def fake_completion(message, request_params, tools):
        captured["message"] = message
        return Prompt.assistant("ok")

    monkeypatch.setattr(llm, "_openai_completion", fake_completion)

    await llm._apply_prompt_provider_specific([tool_result_msg], None, None)

    assert captured["message"] is None


@pytest.mark.asyncio
async def test_apply_prompt_converts_last_message_when_history_disabled(monkeypatch):
    context = Context()
    llm = OpenAILLM(context=context)

    assistant_tool_call, tool_result_msg = _build_tool_messages()
    llm._message_history = [assistant_tool_call, tool_result_msg]

    captured = {"message": None}

    async def fake_completion(message, request_params, tools):
        captured["message"] = message
        return Prompt.assistant("ok")

    monkeypatch.setattr(llm, "_openai_completion", fake_completion)

    await llm._apply_prompt_provider_specific(
        [tool_result_msg], RequestParams(use_history=False), None
    )

    assert isinstance(captured["message"], list)
    assert any(msg.get("role") == "tool" for msg in captured["message"])
