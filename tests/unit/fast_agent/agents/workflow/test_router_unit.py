"""
Unit tests for the router agent, covering models and core functionality.
"""

import pytest

from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.workflow.router_agent import RouterAgent, RoutingResponse
from mcp_agent.agents.agent import Agent
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.core.exceptions import AgentConfigError
from mcp_agent.core.prompt import Prompt
from mcp_agent.llm.augmented_llm_passthrough import (
    FIXED_RESPONSE_INDICATOR,
    PassthroughLLM,
)

# Model tests


def test_routing_response_model():
    """Test the RoutingResponse model validation."""
    # Valid creation
    response = RoutingResponse(
        agent="test_agent", confidence="high", reasoning="This is the best agent for the job"
    )
    assert response.agent == "test_agent"
    assert response.confidence == "high"
    assert response.reasoning == "This is the best agent for the job"

    # Optional field
    response = RoutingResponse(agent="test_agent", confidence="medium")
    assert response.agent == "test_agent"
    assert response.confidence == "medium"
    assert response.reasoning is None


@pytest.mark.asyncio
async def test_disallows_empty_agents():
    """Test that RouterAgent raises AgentConfigError when no agents are provided."""
    # Attempt to create a router with no agents
    with pytest.raises(AgentConfigError):
        RouterAgent(config=AgentConfig(name="test_router"), agents=[])


@pytest.mark.asyncio
async def test_invalid_llm_response():
    """Test router handles invalid LLM responses gracefully."""
    # Create simple agents
    agent1 = LlmAgent(
        config=AgentConfig(name="agent1", instruction="Test agent 1"),
    )
    agent2 = LlmAgent(
        config=AgentConfig(name="agent2", instruction="Test agent 2"),
    )

    # Create router with agents
    router = RouterAgent(config=AgentConfig(name="router"), agents=[agent1, agent2])

    # Replace LLM with passthrough LLM returning invalid JSON
    router._llm = PassthroughLLM()

    # Set the fixed response to invalid JSON that can't be parsed as RoutingResponse
    await router._llm.generate([Prompt.user(f"{FIXED_RESPONSE_INDICATOR} invalid json")])

    # Verify router generates appropriate error message
    response = await router.generate([Prompt.user("test request")])
    assert "No routing response received from LLM" in response.all_text()


@pytest.mark.asyncio
async def test_single_agent_shortcircuit():
    """Test router short-circuits when only one agent is available."""
    # Create a single agent
    agent = Agent(AgentConfig(name="only_agent", instruction="The only available agent"))

    # Create router with a single agent
    router = RouterAgent(config=AgentConfig(name="test_router"), agents=[agent])
    await router.initialize()

    # Test routing directly returns the single agent without LLM call
    response, _ = await router._route_request("some request")

    # Verify result
    assert response
    assert response.agent == "only_agent"
    assert response.confidence == "high"
