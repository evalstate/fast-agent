import json
import os
from pathlib import Path

import pytest
import pytest_asyncio
from mcp.types import CallToolRequest, CallToolResult, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.constants import OPENAI_REASONING_ENCRYPTED, REASONING
from fast_agent.core import Core
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.helpers.content_helpers import get_text, text_content
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason


def get_responses_models() -> list[str]:
    if test_models := os.environ.get("TEST_RESPONSES_MODELS"):
        return [model.strip() for model in test_models.split(",") if model.strip()]
    if test_model := os.environ.get("TEST_RESPONSES_MODEL"):
        return [test_model.strip()]
    return ["responses.gpt-5-mini.low"]


RESPONSES_MODELS = get_responses_models()
if not RESPONSES_MODELS:
    pytest.skip(
        "Set TEST_RESPONSES_MODEL or TEST_RESPONSES_MODELS to run Responses e2e tests",
        allow_module_level=True,
    )


_tool_schema = {
    "type": "object",
    "properties": {"city": {"type": "string", "description": "City to check"}},
    "required": ["city"],
}
_tool = Tool(
    name="weather",
    description="Check the weather in a city",
    inputSchema=_tool_schema,
)


@pytest_asyncio.fixture
async def responses_agent(model_name: str) -> LlmAgent:
    test_config = AgentConfig("test")
    config_path = Path(__file__).parent / "fastagent.config.yaml"

    core = Core(settings=config_path)
    await core.initialize()

    agent = LlmAgent(test_config, core.context)
    await agent.attach_llm(ModelFactory.create_factory(model_name))
    return agent


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", RESPONSES_MODELS)
async def test_responses_streaming_summary(responses_agent, model_name):
    agent = responses_agent
    stream_chunks = []

    remove_listener = agent.llm.add_stream_listener(stream_chunks.append)
    try:
        result: "PromptMessageExtended" = await agent.generate(
            "In one sentence, what is the capital of France?",
            request_params=RequestParams(maxTokens=200),
        )
    finally:
        remove_listener()

    assert result.stop_reason is LlmStopReason.END_TURN
    assert result.last_text()

    non_reasoning = "".join(chunk.text for chunk in stream_chunks if not chunk.is_reasoning)
    assert non_reasoning.strip()

    reasoning_mode = ModelDatabase.get_reasoning(agent.llm.default_request_params.model)
    if reasoning_mode == "openai":
        reasoning_chunks = "".join(chunk.text for chunk in stream_chunks if chunk.is_reasoning)
        reasoning_blocks = (result.channels or {}).get(REASONING, [])
        assert reasoning_chunks.strip() or reasoning_blocks

        encrypted_blocks = (result.channels or {}).get(OPENAI_REASONING_ENCRYPTED, [])
        assert encrypted_blocks
        for block in encrypted_blocks:
            payload_text = get_text(block)
            assert payload_text
            payload = json.loads(payload_text)
            assert payload.get("encrypted_content")


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", RESPONSES_MODELS)
async def test_responses_tool_streaming(responses_agent, model_name):
    agent = responses_agent
    tool_events: list[tuple[str, dict]] = []

    def on_tool_event(event_type: str, payload: dict | None = None) -> None:
        tool_events.append((event_type, payload or {}))

    remove_listener = agent.llm.add_tool_stream_listener(on_tool_event)
    try:
        result: "PromptMessageExtended" = await agent.generate(
            "Call the weather tool for Paris.",
            tools=[_tool],
            request_params=RequestParams(maxTokens=200),
        )
    finally:
        remove_listener()

    assert result.stop_reason is LlmStopReason.TOOL_USE
    assert result.tool_calls
    tool_id = next(iter(result.tool_calls.keys()))
    tool_call: CallToolRequest = result.tool_calls[tool_id]
    assert tool_call.params.name == "weather"

    start_events = [payload for event, payload in tool_events if event == "start"]
    stop_events = [payload for event, payload in tool_events if event == "stop"]
    assert start_events
    assert stop_events

    delta_chunks = [
        payload.get("chunk")
        for event, payload in tool_events
        if event == "delta" and payload.get("tool_use_id") == tool_id
    ]
    if delta_chunks:
        delta_text = "".join(chunk for chunk in delta_chunks if chunk)
        if delta_text.strip():
            json.loads(delta_text)


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", RESPONSES_MODELS)
async def test_responses_tool_followup(responses_agent, model_name):
    agent = responses_agent

    tool_prompt = "Call the weather tool for Paris."
    first_response: "PromptMessageExtended" = await agent.generate(
        tool_prompt,
        tools=[_tool],
        request_params=RequestParams(maxTokens=200),
    )
    assert first_response.stop_reason is LlmStopReason.TOOL_USE
    assert first_response.tool_calls

    reasoning_mode = ModelDatabase.get_reasoning(agent.llm.default_request_params.model)
    if reasoning_mode != "openai":
        pytest.skip("Responses reasoning summary passback is only required for OpenAI reasoning.")

    encrypted_blocks = (first_response.channels or {}).get(OPENAI_REASONING_ENCRYPTED, [])
    if not encrypted_blocks:
        pytest.skip("Model did not return encrypted reasoning to pass back.")

    tool_use_id = next(iter(first_response.tool_calls.keys()))
    tool_result = CallToolResult(content=[text_content("It is sunny.")], isError=False)
    tool_message = PromptMessageExtended(
        role="user",
        content=[],
        tool_results={tool_use_id: tool_result},
    )

    followup_prompt = "Use the tool result to answer in one sentence and do not call tools."
    followup_response: "PromptMessageExtended" = await agent.generate(
        [tool_prompt, first_response, tool_message, followup_prompt],
        request_params=RequestParams(maxTokens=200),
    )

    assert followup_response.stop_reason is LlmStopReason.END_TURN
    assert followup_response.last_text()
