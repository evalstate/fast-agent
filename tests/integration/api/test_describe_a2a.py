from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from a2a_types.types import AgentCard, AgentSkill

    from mcp_agent.agents.agent import Agent


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_agent_card_and_tools(fast_agent):
    fast = fast_agent

    @fast.agent(name="test", instruction="here are you instructions", servers=["card_test"])
    async def agent_function():
        async with fast.run() as app:
            # Simulate some agent operations
            agent: Agent = app["test"]
            card: AgentCard = await agent.agent_card()

            assert "test" == card.name
            # TODO -- migrate AgentConfig to include "description" - "instruction" is OK for the moment...
            assert "here are you instructions" == card.description
            assert 3 == len(card.skills)

            skill: AgentSkill = card.skills[0]
            assert "card_test-check_weather" == skill.id
            assert "check_weather" == skill.name
            assert "Returns the weather for a specified location."
            assert skill.tags
            assert "tool" == skill.tags[0]

    await agent_function()
