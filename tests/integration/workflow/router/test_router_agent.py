from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from fast_agent.llm.internal.passthrough import FIXED_RESPONSE_INDICATOR
from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.prompts.prompt_load import load_prompt_multipart

if TYPE_CHECKING:
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


@pytest.mark.integration
@pytest.mark.asyncio
async def test_router_functionality(fast_agent):
    """Check that the router routes"""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(name="target1")
    @fast.agent(name="target2")
    @fast.router(name="router", agents=["target1", "target2"], model="playback")
    async def agent_function():
        async with fast.run() as agent:
            await agent.target1.send(f"{FIXED_RESPONSE_INDICATOR} target1-result")
            await agent.target2.send(f"{FIXED_RESPONSE_INDICATOR} target2-result")
            router_setup: list[PromptMessageExtended] = load_prompt_multipart(
                Path("router_script.txt")
            )
            setup: PromptMessageExtended = await agent.router._llm.generate(router_setup)
            assert "LOADED" in setup.first_text()
            result: str = await agent.router.send("some routing")
            assert "target1-result" in result

            result: str = await agent.router.send("more routing")
            assert "target2-result" in result

    await agent_function()


class WeatherData(BaseModel):
    """Sample model for structured output testing."""

    location: str
    temperature: float
    conditions: str


@pytest.mark.integration
@pytest.mark.asyncio
async def test_router_structured_output(fast_agent):
    """Test router can handle structured output from agents."""
    # Use the FastAgent instance
    fast = fast_agent

    # Define test agents and router
    @fast.agent(name="structured_agent", model="passthrough")
    @fast.router(name="router", agents=["structured_agent"], model="passthrough")
    async def agent_function():
        async with fast.run() as agent:
            # Set up the passthrough LLM with JSON response
            json_response = (
                """{"location": "New York", "temperature": 72.5, "conditions": "Sunny"}"""
            )
            await agent.structured_agent.send(f"{FIXED_RESPONSE_INDICATOR} {json_response}")

            # Set up router to route to structured_agent
            routing_response = """{"agent": "structured_agent", "confidence": "high", "reasoning": "Weather request"}"""
            await agent.router._llm.generate(
                [Prompt.user(f"{FIXED_RESPONSE_INDICATOR} {routing_response}")]
            )

            # Send request through router with proper PromptMessageExtended list
            result, _ = await agent.router.structured(
                [Prompt.user("What's the weather in New York?")], WeatherData
            )

            # Verify structured result
            assert result is not None
            assert result.location == "New York"
            assert result.temperature == 72.5
            assert result.conditions == "Sunny"

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_router_invalid_agent_selection(fast_agent):
    """Test router handles invalid agent selection gracefully."""
    # Use the FastAgent instance
    fast = fast_agent

    # Define test agents and router - need two agents to bypass single-agent optimization
    @fast.agent(name="available_agent", model="playback")
    @fast.agent(name="another_agent", model="playback")
    @fast.router(name="router", agents=["available_agent", "another_agent"], model="passthrough")
    async def agent_function():
        async with fast.run() as agent:
            # Set up router to route to non-existent agent
            routing_response = """{"agent": "nonexistent_agent", "confidence": "high", "reasoning": "Test request"}"""
            await agent.router._llm.generate(
                [Prompt.user(f"{FIXED_RESPONSE_INDICATOR} {routing_response}")]
            )

            # Send request through router
            result = await agent.router.send("This should fail with a clear error")

            # Verify error message
            assert (
                "A response was received, but the agent nonexistent_agent was not known to the Router"
                in result
            )

    await agent_function()
