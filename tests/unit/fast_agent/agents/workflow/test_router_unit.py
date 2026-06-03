"""
Unit tests for the router agent, covering models and core functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent.agents import McpAgent
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.workflow.router_agent import RouterAgent, RoutingResponse
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.prompt import Prompt
from fast_agent.llm.internal.passthrough import (
    FIXED_RESPONSE_INDICATOR,
    PassthroughLLM,
)
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.types import PromptMessageExtended, RequestParams

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcp.types import PromptMessage


class RecordingSchemaAgent(LlmAgent):
    def __init__(self, name: str) -> None:
        super().__init__(AgentConfig(name=name, instruction="records schema calls"))
        self.structured_schema_inputs: list[list[PromptMessageExtended]] = []
        self.structured_schemas: list[dict[str, Any]] = []
        self.structured_schema_params: list[RequestParams | None] = []

    async def structured_schema(
        self,
        messages: str
        | PromptMessage
        | PromptMessageExtended
        | Sequence[str | PromptMessage | PromptMessageExtended],
        schema: dict[str, Any],
        request_params: RequestParams | None = None,
    ) -> tuple[dict[str, str], PromptMessageExtended]:
        assert isinstance(messages, list)
        self.structured_schema_inputs.append(cast("list[PromptMessageExtended]", messages))
        self.structured_schemas.append(schema)
        self.structured_schema_params.append(request_params)
        message = PromptMessageExtended(role="assistant", content=[text_content("structured")])
        return {"value": self.name}, message

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
    agent = McpAgent(AgentConfig(name="only_agent", instruction="The only available agent"))

    # Create router with a single agent
    router = RouterAgent(config=AgentConfig(name="test_router"), agents=[agent])
    await router.initialize()

    # Test routing directly returns the single agent without LLM call
    response, _ = await router._route_request(Prompt.user("some request"))

    # Verify result
    assert response
    assert response.agent == "only_agent"
    assert response.confidence == "high"


@pytest.mark.asyncio
async def test_structured_schema_routes_to_selected_agent() -> None:
    agent = RecordingSchemaAgent("only_agent")
    router = RouterAgent(config=AgentConfig(name="test_router"), agents=[agent])
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}

    result, message = await router.structured_schema(
        Prompt.user("some request"),
        schema,
        RequestParams(use_history=False, emit_loop_progress=False),
    )

    assert result == {"value": "only_agent"}
    assert message.all_text() == "structured"
    assert len(agent.structured_schema_inputs) == 1
    assert agent.structured_schema_inputs[0][0].all_text() == "some request"
    assert agent.structured_schemas == [schema]
    assert agent.structured_schema_params == [
        RequestParams(use_history=False, emit_loop_progress=False)
    ]
