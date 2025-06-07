# integration_tests/mcp_agent/test_agent_with_image.py

from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams


class FormattedResponse(BaseModel):
    thinking: Annotated[
        str, Field(description="Your reflection on the conversation that is not seen by the user.")
    ]
    message: str


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "generic.qwen2.5:latest",
        "generic.llama3.2:latest",
        "deepseek-chat",
        "haiku",
        "sonnet",
        "gpt-4.1",
        "gpt-4.1-mini",
        "o3-mini.low",
        "openrouter.google/gemini-2.0-flash-001",
        "gemini25",
    ],
)
async def test_structured_output_with_automatic_format_for_model(fast_agent, model_name):
    """Test that the agent can generate structured response with response_format_specified."""
    fast = fast_agent

    @fast.agent(
        "chat",
        instruction="You are a helpful assistant.",
        model=model_name,
    )
    async def create_structured():
        async with fast.run() as agent:
            thinking, response = await agent.chat.structured(
                [Prompt.user("Let's talk about guitars.")],
                model=FormattedResponse,
            )
            assert isinstance(thinking, FormattedResponse)
            assert FormattedResponse.model_validate_json(response.first_text())

            assert "guitar" in thinking.message.lower()

    await create_structured()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4.1-mini",
        "gemini25",
    ],
)
async def test_structured_output_parses_assistant_message_if_last(fast_agent, model_name):
    """Test that the agent can generate structured response with response_format_specified."""
    fast = fast_agent

    @fast.agent(
        "chat",
        instruction="You are a helpful assistant.",
        model=model_name,
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
    [
        "generic.llama3.2:latest",
        # "haiku", -- anthropic do not support structured outputs this way
        "gpt-4.1-mini",
        "openrouter.google/gemini-2.0-flash-001",
        "gemini2",
        "gemini25",
    ],
)
async def test_structured_output_with_response_format_overriden(fast_agent, model_name):
    """Test that the agent can generate structured response with response_format_specified."""
    fast = fast_agent

    @fast.agent(
        "chat",
        instruction="You are a helpful assistant.",
        model=model_name,
    )

    # you can specify a response format string, but this is not preferred
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
    [
        "gpt-4.1-mini",
        "haiku",
        "gemini2",
        "gemini25",
    ],
)
async def test_history_management_with_structured(fast_agent, model_name):
    """Test that the agent can generate structured response with response_format_specified."""
    fast = fast_agent

    @fast.agent(
        "chat",
        instruction="You are a helpful assistant. The user may request structured outputs, follow their instructions",
        model=model_name,
    )
    async def create_structured():
        async with fast.run() as agent:
            await agent.chat.send("good morning")
            thinking, response = await agent.chat.structured(
                [
                    Prompt.user("Let's talk about guitars."),
                ],
                model=FormattedResponse,
            )
            assert "guitar" in thinking.message.lower()

            thinking, response = await agent.chat.structured(
                [
                    Prompt.user("Let's talk about pianos."),
                ],
                model=FormattedResponse,
            )
            assert "piano" in thinking.message.lower()

            response = await agent.chat.send(
                "did we talk about space travel? respond only with YES or NO - no other formatting"
            )
            assert "no" in response.lower()

            assert 8 == len(agent.chat.message_history)
            assert len(agent.chat._llm.history.get()) > 7

    await create_structured()
