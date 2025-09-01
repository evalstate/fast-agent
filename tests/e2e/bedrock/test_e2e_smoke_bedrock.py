import sys
from typing import Annotated, List

import pytest
from pydantic import BaseModel, Field

from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.providers.augmented_llm_bedrock import BedrockAugmentedLLM
from mcp_agent.llm.providers.bedrock_utils import all_bedrock_models
from mcp_agent.mcp.helpers.content_helpers import split_thinking_content


@pytest.fixture(scope="module", autouse=True)
def debug_cache_at_end():
    """Print cache state after all tests in this module complete."""
    yield
    sys.stdout.write("\n=== FINAL CACHE STATE (test_e2e_smoke_bedrock.py) ===\n")
    BedrockAugmentedLLM.debug_cache()


def _bedrock_models_for_smoke() -> List[str]:
    """Return the full Bedrock model list for exhaustive smoke reporting."""
    return all_bedrock_models(prefix="")


# ---------------- Structured output smoke tests (Bedrock) ----------------


class FormattedResponse(BaseModel):
    thinking: Annotated[
        str, Field(description="Your reflection on the conversation that is not seen by the user.")
    ]
    message: str


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize("model_name", _bedrock_models_for_smoke())
async def test_bedrock_basic_textual_prompting(fast_agent, model_name):
    """Bedrock-specific smoke test: simple textual generation."""
    fast = fast_agent

    # These models are expected to fail
    if model_name.startswith("amazon.titan-text-") or model_name.startswith("amazon.titan-tg1"):
        pytest.xfail("Titan text inconsistent for this test")
    if model_name.startswith("anthropic.claude-3-haiku-"):
        pytest.xfail("Claude 3 Haiku often too short for this test")

    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        model=f"bedrock.{model_name}",
    )
    async def agent_function():
        async with fast.run() as agent:
            response = await agent.send(
                Prompt.user(
                    "write a short 40-60 word story about cats. No preamble; output only the story."
                )
            )
            response_text = response.strip()
            words = response_text.split()
            assert 30 <= len(words) <= 80

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    _bedrock_models_for_smoke(),
)
async def test_bedrock_open_ai_history_compat(fast_agent, model_name):
    """Bedrock models should maintain provider history with system prompt preserved."""
    fast = fast_agent

    @fast.agent(
        "agent",
        instruction="SYSTEM PROMPT",
        model=f"bedrock.{model_name}",
    )
    async def agent_function():
        async with fast.run() as agent:
            await agent.send("MESSAGE ONE")
            await agent.send("MESSAGE TWO")

            provider_history = agent.agent._llm.history
            multipart_history = agent.agent.message_history

            assert 4 == len(provider_history.get())
            assert 4 == len(multipart_history)

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    _bedrock_models_for_smoke(),
)
async def test_bedrock_multiple_text_blocks_prompting(fast_agent, model_name):
    fast = fast_agent

    # Per-test expected behavior
    if model_name.startswith(("amazon.titan-text-", "amazon.titan-tg1")):
        pytest.xfail("Titan text inconsistent for this test")

    @fast.agent(
        instruction="You are a helpful AI Agent",
        model=f"bedrock.{model_name}",
    )
    async def agent_function():
        async with fast.run() as agent:
            response = await agent.default.generate(
                [Prompt.user("write a 50-80 word story", "about cats - include the word 'cat'")]
            )
            response_text = response.all_text()
            words = response_text.split()
            assert 32 <= len(words) <= 100
            assert "cat" in response_text.lower()

            response = await agent.default.generate(
                [Prompt.user("write a 50-80 word story"), Prompt.user("about cats - include 'cat'")]
            )
            response_text = response.all_text()
            words = response_text.split()
            assert 32 <= len(words) <= 100
            assert "cat" in response_text.lower()

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    _bedrock_models_for_smoke(),
)
async def test_bedrock_basic_tool_calling(fast_agent, model_name):
    """Bedrock tool calling smoke: simple end-to-end tool invocation.

    Uses 'test_server' tools from the existing E2E environment.
    """
    fast = fast_agent

    @fast.agent(
        "weatherforecast",
        instruction=(
            "You are a helpful assistant. Use tools when needed. When you receive a tool result, return the tool result text EXACTLY with no preamble, explanation, or modifications."
        ),
        model=f"bedrock.{model_name}",
        servers=["test_server"],
    )
    async def weather_forecast():
        async with fast.run() as agent:
            # Expected outliers for tool-calling via Bedrock Converse
            if model_name.startswith("ai21."):
                pytest.xfail(
                    "AI21 Jamba models do not reliably invoke tools through Bedrock Converse"
                )
            if model_name in [
                "amazon.titan-text-lite-v1",
                "amazon.titan-text-express-v1",
                "amazon.titan-tg1-large",
            ]:
                pytest.xfail(
                    "Titan text models either do not tool-call or do not return raw tool result"
                )
            if model_name == "mistral.mistral-7b-instruct-v0:2":
                pytest.xfail("Mistral 7B occasionally calls the wrong tool or paraphrases output")
            if model_name == "mistral.mistral-small-2402-v1:0":
                pytest.xfail(
                    "Mistral Small occasionally emits tool call JSON instead of tool result"
                )

            response = await agent.send(
                Prompt.user(
                    "Use the check_weather tool to get London's weather. Do NOT answer from your own knowledge. After the tool returns, respond with ONLY the tool's raw text output, exactly as returned. No preamble, no extra words."
                )
            )
            assert isinstance(response, str)

            # Check for expected response from tool (matches main smoke test server)
            assert "sunny" in response.lower()

    await weather_forecast()


def _bedrock_models_for_structured() -> List[str]:
    """Return Bedrock models suitable for structured-output tests.

    Prefer Nova and Claude 3.x families.
    """
    candidates = all_bedrock_models(prefix="")
    filtered = [
        m for m in candidates if m.startswith("amazon.nova-") or m.startswith("anthropic.claude-3")
    ]
    return filtered


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    _bedrock_models_for_smoke(),
)
async def test_bedrock_structured_output_auto_format(fast_agent, model_name):
    fast = fast_agent

    # Skip families unreliable for structured outputs
    if model_name.startswith(("amazon.nova-micro-", "amazon.titan-", "meta.llama3-", "mistral.")):
        pytest.skip("Skipping structured-output test for this model family")

    @fast.agent(
        "chat",
        instruction="You are a helpful assistant.",
        model=f"bedrock.{model_name}",
    )
    async def create_structured():
        async with fast.run() as agent:
            thinking, response = await agent.chat.structured(
                [Prompt.user("Let's talk about guitars.")],
                model=FormattedResponse,
            )
            assert isinstance(thinking, FormattedResponse)
            _, json_part = split_thinking_content(response.first_text())
            assert FormattedResponse.model_validate_json(json_part)
            assert "guitar" in thinking.message.lower()

    await create_structured()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    _bedrock_models_for_smoke(),
)
async def test_bedrock_structured_output_parses_assistant_message_if_last(fast_agent, model_name):
    fast = fast_agent

    @fast.agent(
        "chat",
        instruction="You are a helpful assistant.",
        model=f"bedrock.{model_name}",
    )
    async def create_structured():
        async with fast.run() as agent:
            thinking, response = await agent.chat.structured(
                [
                    Prompt.user("Let's talk about guitars."),
                    Prompt.assistant(
                        '{"thinking":"The user wants to have a conversation about guitars, which are a broad...","message":"Sure! I love talking about guitars."}'
                    ),
                ],
                model=FormattedResponse,
            )
            assert thinking.thinking.startswith(
                "The user wants to have a conversation about guitars"
            )

    await create_structured()


response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "formatted_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thinking": {
                    "type": "string",
                    "description": "Your reflection on the conversation that is not seen by the user.",
                },
                "message": {
                    "type": "string",
                    "description": "Your message to the user.",
                },
            },
            "required": ["thinking", "message"],
            "additionalProperties": False,
        },
    },
}


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    _bedrock_models_for_smoke(),
)
async def test_bedrock_structured_output_with_response_format_override(fast_agent, model_name):
    fast = fast_agent

    # Skip families unreliable for structured outputs
    if model_name.startswith(("amazon.nova-micro-", "amazon.titan-", "meta.llama3-", "mistral.")):
        pytest.skip("Skipping structured-output test for this model family")

    @fast.agent(
        "chat",
        instruction="You are a helpful assistant.",
        model=f"bedrock.{model_name}",
    )
    async def create_structured():
        async with fast.run() as agent:
            thinking, response = await agent.chat.structured(
                [Prompt.user("Let's talk about guitars.")],
                model=FormattedResponse,
                request_params=RequestParams(response_format=response_format),
            )
            assert thinking is not None
            assert "guitar" in thinking.message.lower()

    await create_structured()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    _bedrock_models_for_smoke(),
)
async def test_bedrock_history_management_with_structured(fast_agent, model_name):
    fast = fast_agent

    # Skip families unreliable for structured outputs
    # if model_name.startswith(("amazon.nova-micro-", "amazon.titan-", "meta.llama3-", "mistral.")):
    #     pytest.skip("Skipping structured-output test for this model family")

    # Known unreliable for structured/history: xfail Titans
    if model_name.startswith("amazon.titan-"):
        pytest.xfail("Titan models are unreliable for structured outputs with history")
    # Known intermittent failure for Mistral 7B on this structured/history task
    if model_name == "mistral.mistral-7b-instruct-v0:2":
        pytest.xfail("Mistral 7B occasionally ignores topic in structured history JSON")

    @fast.agent(
        "chat",
        instruction=(
            "You are a helpful assistant. The user may request structured outputs. When asked to return structured JSON, respond with ONLY valid JSON conforming to the schema, no prose or preambles."
        ),
        model=f"bedrock.{model_name}",
    )
    async def create_structured():
        async with fast.run() as agent:
            await agent.chat.send("good morning")
            thinking, response = await agent.chat.structured(
                [Prompt.user("Let's talk about guitars.")],
                model=FormattedResponse,
            )
            assert "guitar" in thinking.message.lower()

            thinking, response = await agent.chat.structured(
                [Prompt.user("Let's talk about pianos.")],
                model=FormattedResponse,
            )
            assert "piano" in thinking.message.lower()

            response = await agent.chat.send(
                "Based ONLY on this conversation so far, did we talk about space travel? Respond EXACTLY YES or NO in uppercase, with no punctuation or explanations. If not discussed, respond NO."
            )
            assert "no" in response.lower()

            assert 8 == len(agent.chat.message_history)
            assert len(agent.chat._llm.history.get()) > 7

    await create_structured()
