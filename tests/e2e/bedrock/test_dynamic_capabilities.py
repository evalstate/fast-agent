import sys
from typing import List

import pytest

from fast_agent.llm.provider.bedrock.bedrock_utils import all_bedrock_models
from fast_agent.llm.provider.bedrock.llm_bedrock import BedrockLLM
from mcp_agent.core.prompt import Prompt


@pytest.fixture(scope="module", autouse=True)
def debug_cache_at_end():
    """Print cache state after all tests in this module complete."""
    yield
    sys.stdout.write("\n=== FINAL CACHE STATE (test_dynamic_capabilities.py) ===\n")
    BedrockLLM.debug_cache()


def _bedrock_models_for_capability_tests() -> List[str]:
    """Return Bedrock models if AWS is configured, otherwise return empty list."""
    try:
        return all_bedrock_models(prefix="")
    except RuntimeError:
        # AWS not configured - return empty list so tests are skipped
        return []


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    _bedrock_models_for_capability_tests()
    or [pytest.param("dummy", marks=pytest.mark.skip("AWS not configured"))],
)
async def test_bedrock_system_prompt_fallback(fast_agent, model_name):
    """Test system prompt fallback: models that don't support system param should inject into first user message."""

    # Mark specific models that don't properly handle system prompts
    if model_name in ["amazon.titan-text-lite-v1"]:
        pytest.xfail("This Titan model doesn't properly process injected system prompts")

    fast = fast_agent

    @fast.agent(
        "system_test",
        instruction="You are a helpful assistant. Always mention the word 'SYSTEM_WORKING' in your response.",
        model=f"bedrock.{model_name}",
    )
    async def system_test():
        async with fast.run() as agent:
            response = await agent.send(Prompt.user("Say hello"))
            # Verify system prompt was applied (either via system param or injection)
            assert "SYSTEM_WORKING" in response.upper()

    await system_test()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    _bedrock_models_for_capability_tests()
    or [pytest.param("dummy", marks=pytest.mark.skip("AWS not configured"))],
)
async def test_bedrock_tool_schema_fallback(fast_agent, model_name):
    """Test tool schema fallback: should try different formats (anthropic/nova/system_prompt) until one works."""

    # Mark Titan text models as expected to fail tool calling
    if model_name.startswith(("amazon.titan-text", "amazon.titan-tg1")):
        pytest.xfail("Titan text models don't support native tool calling")

    # Mark models with tool calling issues
    if model_name.startswith("mistral."):
        pytest.xfail("Mistral models tend to paraphrase tool responses")
    if model_name.startswith("ai21."):
        pytest.xfail("AI21 models may describe tool calls instead of actually calling them")

    fast = fast_agent

    @fast.agent(
        "tool_test",
        instruction="You are a helpful assistant. When you call a tool and receive a result, you MUST include the exact tool response text in your final answer without any modification, paraphrasing, or additional commentary.",
        model=f"bedrock.{model_name}",
        servers=["test_server"],
    )
    async def tool_test():
        async with fast.run() as agent:
            response = await agent.send(Prompt.user("What's the weather in Paris? Use tools."))
            assert isinstance(response, str)

            # Tool should have been called successfully - check for unique response from MCP server
            assert "sunny" in response.lower() and "paris" in response.lower()

    await tool_test()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    _bedrock_models_for_capability_tests()
    or [pytest.param("dummy", marks=pytest.mark.skip("AWS not configured"))],
)
async def test_bedrock_streaming_with_tools_fallback(fast_agent, model_name):
    """Test streaming fallback: should detect if streaming+tools fails and fallback to non-streaming."""

    # Mark Titan text models as expected to fail tool calling
    if model_name.startswith(("amazon.titan-text", "amazon.titan-tg1")):
        pytest.xfail("Titan text models don't support native tool calling")

    # Mark AI21 models that may describe instead of call tools
    if model_name.startswith("ai21."):
        pytest.xfail("AI21 models may describe tool calls instead of actually calling them")

    fast = fast_agent

    @fast.agent(
        "streaming_test",
        instruction="You are a helpful assistant. When you receive tool results, include the exact response text without paraphrasing or modifying it.",
        model=f"bedrock.{model_name}",
        servers=["test_server"],
    )
    async def streaming_test():
        async with fast.run() as agent:
            # Test streaming with tools - should fallback to non-streaming if needed
            response = await agent.send(Prompt.user("What's the weather in Tokyo? Use tools."))
            assert isinstance(response, str)
            # Check for unique response from MCP server
            assert "sunny" in response.lower() and "tokyo" in response.lower()

    await streaming_test()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    _bedrock_models_for_capability_tests()
    or [pytest.param("dummy", marks=pytest.mark.skip("AWS not configured"))],
)
async def test_bedrock_tool_name_policy_fallback(fast_agent, model_name):
    """Test tool name policy: should detect hyphenated tool names and apply underscore conversion."""

    # Mark Titan text models as expected to fail tool calling
    if model_name.startswith(("amazon.titan-text", "amazon.titan-tg1")):
        pytest.xfail("Titan text models don't support native tool calling")

    # Mark specific models that have tool response issues
    if model_name in ["mistral.mistral-7b-instruct-v0:2"] or model_name.startswith("ai21."):
        pytest.xfail("These models paraphrase tool responses instead of including exact content")

    fast = fast_agent

    @fast.agent(
        "name_policy_test",
        instruction="You are a helpful assistant. When you call a tool and receive a result, you MUST include the exact tool response text in your final answer without any modification, paraphrasing, or additional commentary.",
        model=f"bedrock.{model_name}",
        servers=["test_server"],  # This has hyphenated tool names
    )
    async def name_policy_test():
        async with fast.run() as agent:
            # Single tool call to avoid model confusion from repeated calls
            response = await agent.send(Prompt.user("What's the weather in London? Use tools."))
            assert "sunny" in response.lower() and "london" in response.lower()

    await name_policy_test()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    _bedrock_models_for_capability_tests()
    or [pytest.param("dummy", marks=pytest.mark.skip("AWS not configured"))],
)
async def test_bedrock_structured_output_strategy_fallback(fast_agent, model_name):
    """Test structured output strategy: should try different strategies and retry on failures."""

    # Mark models known to have issues with structured output
    if model_name.startswith(
        ("amazon.titan-text", "amazon.titan-tg1", "cohere.", "mistral.", "amazon.nova-")
    ):
        pytest.xfail("These models have unreliable structured output support")

    fast = fast_agent

    @fast.agent(
        "structured_test",
        instruction="You are a helpful assistant that returns structured data.",
        model=f"bedrock.{model_name}",
    )
    async def structured_test():
        async with fast.run() as agent:
            from pydantic import BaseModel

            class WeatherResponse(BaseModel):
                city: str
                condition: str
                temperature: int

            # Use the correct structured() API
            response, _ = await agent.structured_test.structured(
                [Prompt.user("What's the weather in Miami? Return structured data.")],
                model=WeatherResponse,
            )

            assert isinstance(response, WeatherResponse)
            assert response.city.lower() == "miami"
            assert response.condition
            assert isinstance(response.temperature, int)

    await structured_test()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    _bedrock_models_for_capability_tests()
    or [pytest.param("dummy", marks=pytest.mark.skip("AWS not configured"))],
)
async def test_bedrock_force_non_streaming_structured(fast_agent, model_name):
    """Test force non-streaming for structured output: should force non-streaming on first structured call."""

    # Mark models known to have issues with structured output
    if model_name.startswith(
        ("amazon.titan-text", "amazon.titan-tg1", "cohere.", "mistral.", "amazon.nova-")
    ):
        pytest.xfail("These models have unreliable structured output support")

    fast = fast_agent

    @fast.agent(
        "force_non_streaming_test",
        instruction="You are a helpful assistant.",
        model=f"bedrock.{model_name}",
    )
    async def force_non_streaming_test():
        async with fast.run() as agent:
            from pydantic import BaseModel

            class SimpleResponse(BaseModel):
                message: str
                count: int

            # This should trigger _force_non_streaming_once behavior
            response, _ = await agent.force_non_streaming_test.structured(
                [Prompt.user("Say hello and count to 3.")], model=SimpleResponse
            )

            assert isinstance(response, SimpleResponse)
            assert response.message
            assert response.count == 3

    await force_non_streaming_test()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    _bedrock_models_for_capability_tests()
    or [pytest.param("dummy", marks=pytest.mark.skip("AWS not configured"))],
)
async def test_bedrock_reasoning_fallback(fast_agent, model_name):
    """Test reasoning fallback: models that don't support reasoning should fallback gracefully."""

    # Mark models that don't include exact numbers in math responses
    if model_name.startswith(("mistral.", "amazon.titan-")):
        pytest.xfail(
            "These models provide detailed explanations but may not include the exact final number"
        )

    fast = fast_agent

    @fast.agent(
        "reasoning_test",
        instruction="You are a helpful assistant. When solving math problems, always end your response with 'Final answer: [NUMBER]' where NUMBER is the exact integer result.",
        model=f"bedrock.{model_name}.medium",  # Try medium reasoning effort
    )
    async def reasoning_test():
        async with fast.run() as agent:
            # Ask for a simple reasoning task
            response = await agent.reasoning_test.send(
                "What is 15 + 27? Show your work but make sure to end with 'Final answer: 42'"
            )

            # Should get a response regardless of reasoning support
            assert response
            assert (
                "42" in response or "final answer: 42" in response.lower()
            )  # The answer should be 42
            return response

    await reasoning_test()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    _bedrock_models_for_capability_tests()
    or [pytest.param("dummy", marks=pytest.mark.skip("AWS not configured"))],
)
async def test_bedrock_temperature_parameter(fast_agent, model_name):
    """Test temperature parameter: should be applied when reasoning is not enabled."""

    # Mark Titan models as expected to have response formatting issues
    if model_name.startswith("amazon.titan-"):
        pytest.xfail(
            "Titan models have response formatting limitations and may truncate or refuse exact phrases"
        )

    fast = fast_agent

    @fast.agent(
        "temperature_test",
        instruction="You are a helpful assistant. Always respond with the exact text requested without paraphrasing, shortening, or modifying it.",
        model=f"bedrock.{model_name}",  # No reasoning effort = temperature should work
    )
    async def temperature_test():
        async with fast.run() as agent:
            from fast_agent.llm.request_params import RequestParams

            # Test with low temperature (should be more deterministic)
            response = await agent.temperature_test.send(
                "Say exactly: 'Temperature test successful'",
                request_params=RequestParams(temperature=0.0),
            )

            # Should get a response with the temperature parameter applied
            assert response
            assert "temperature test successful" in response.lower()
            return response

    await temperature_test()
