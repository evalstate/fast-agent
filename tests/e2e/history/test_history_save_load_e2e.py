import os
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp.types import CallToolResult, TextContent, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.core import Core
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompt_serialization import save_messages
from fast_agent.mcp.prompts.prompt_load import load_history_into_agent
from fast_agent.types.llm_stop_reason import LlmStopReason

TEST_CONFIG_PATH = Path(__file__).resolve().parent.parent / "llm" / "fastagent.config.yaml"
DEFAULT_CREATE_MODELS = ["gpt-5-mini.minimal", "haiku", "gemini25"]
DEFAULT_CHECK_MODELS = ["haiku", "kimigroq", "gpt-5-mini.minimal"]
MAGIC_STRING = "MAGIC-ACCESS-PHRASE-9F1C"
MAGIC_TOOL = Tool(
    name="fetch_magic_string",
    description="Returns the daily passphrase when the assistant must call a tool.",
    inputSchema={
        "type": "object",
        "properties": {
            "purpose": {
                "type": "string",
                "description": "Explain why you need the passphrase. Must always be supplied.",
            }
        },
        "required": ["purpose"],
    },
)


def _parse_model_list(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return default
    parsed = [value.strip() for value in raw.split(",") if value.strip()]
    return parsed or default


CREATE_MODELS = _parse_model_list(
    os.environ.get("FAST_AGENT_HISTORY_CREATE_MODELS"), DEFAULT_CREATE_MODELS
)
CHECK_MODELS = _parse_model_list(
    os.environ.get("FAST_AGENT_HISTORY_CHECK_MODELS"), DEFAULT_CHECK_MODELS
)
MODEL_MATRIX = [(create, check) for create in CREATE_MODELS for check in CHECK_MODELS]


def _sanitize_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(".", "-").replace(" ", "-").lower()


@asynccontextmanager
async def agent_session(model_name: str, label: str):
    core = Core(settings=str(TEST_CONFIG_PATH))
    async with core.run():
        agent = LlmAgent(AgentConfig(label), core.context)
        await agent.attach_llm(ModelFactory.create_factory(model_name))
        yield agent


async def _create_history(agent: LlmAgent) -> None:
    greeting = await agent.generate(
        "The following messages are part of a test of our LLM history functions. Let's start with a quick friendly greeting."
    )
    assert greeting.stop_reason is LlmStopReason.END_TURN

    request = (
        "Call the fetch_magic_string tool to obtain today's secret passphrase. "
        "You must call the tool before you can continue."
    )
    tool_call = await agent.generate(
        request,
        tools=[MAGIC_TOOL],
        request_params=RequestParams(maxTokens=300),
    )
    assert tool_call.stop_reason is LlmStopReason.TOOL_USE
    assert tool_call.tool_calls
    tool_id = next(iter(tool_call.tool_calls.keys()))

    tool_result = CallToolResult(content=[TextContent(type="text", text=MAGIC_STRING)])
    user_tool_message = PromptMessageExtended(
        role="user",
        content=[
            TextContent(
                type="text",
                text="Here is the tool output. Read it carefully and repeat the passphrase verbatim.",
            )
        ],
        tool_results={tool_id: tool_result},
    )
    confirmation = await agent.generate(user_tool_message)
    # confirmation_text = (confirmation.all_text() or "").lower()
    assert LlmStopReason.END_TURN == confirmation.stop_reason
    # assert MAGIC_STRING.lower() in confirmation_text

    wrap_up = await agent.generate(
        "Great. Say something brief about keeping that passphrase safe so I know you stored it."
    )
    assert wrap_up.stop_reason is LlmStopReason.END_TURN


async def _load_and_verify(agent: LlmAgent, history_file: Path) -> None:
    load_history_into_agent(agent, history_file)

    follow_up = await agent.generate(
        "Without inventing anything new, what exact passphrase did fetch_magic_string return earlier?"
    )
    follow_text = (follow_up.all_text() or "").lower()
    assert MAGIC_STRING.lower() in follow_text


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("create_model,check_model", MODEL_MATRIX)
async def test_history_survives_across_models(tmp_path, create_model, check_model):
    history_file = (
        tmp_path
        / f"history_{_sanitize_model_name(create_model)}_{_sanitize_model_name(check_model)}.json"
    )

    async with agent_session(create_model, f"history-create-{create_model}") as creator_agent:
        await _create_history(creator_agent)
        save_messages(creator_agent.llm.message_history, history_file)

    assert history_file.exists()

    async with agent_session(check_model, f"history-load-{check_model}") as checker_agent:
        await _load_and_verify(checker_agent, history_file)
