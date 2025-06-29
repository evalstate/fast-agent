"""
Test the prompt listing and selection functionality directly.
"""

import pytest

from mcp_agent.core.interactive_prompt import InteractivePrompt


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_agent_prompt_listing(fast_agent):
    """Test the _get_all_prompts function with None as server name."""
    fast = fast_agent

    @fast.agent(name="agent1", servers=["prompts"])
    @fast.agent(name="agent2", servers=["prompts2"])
    @fast.agent(name="agent3")
    async def agent_function():
        async with fast.run() as agent:
            # Create instance of InteractivePrompt
            prompt_ui = InteractivePrompt()

            # Test listing prompts for each agent separately
            # Agent1 should have prompts from "prompts" server (playback.md -> playback)
            agent1_prompts = await prompt_ui._get_all_prompts(agent, "agent1")
            assert len(agent1_prompts) == 1
            assert agent1_prompts[0]["server"] == "prompts"
            assert agent1_prompts[0]["name"] == "playback"
            assert agent1_prompts[0]["description"] == "[USER] user1 assistant1 user2"
            assert agent1_prompts[0]["arg_count"] == 0

            # Agent2 should have prompts from "prompts2" server (prompt.txt -> prompt)
            agent2_prompts = await prompt_ui._get_all_prompts(agent, "agent2")
            assert len(agent2_prompts) == 1
            assert agent2_prompts[0]["server"] == "prompts2"
            assert agent2_prompts[0]["name"] == "prompt"
            assert agent2_prompts[0]["description"] == "this is from the prompt file"
            assert agent2_prompts[0]["arg_count"] == 0

            # Agent3 should have no prompts (no servers configured)
            agent3_prompts = await prompt_ui._get_all_prompts(agent, "agent3")
            assert len(agent3_prompts) == 0

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_apply_prompt_with_namespaced_name(fast_agent):
    """Test applying a prompt using its namespaced name directly."""
    fast = fast_agent

    @fast.agent(name="test", servers=["prompts"], model="passthrough")
    async def agent_function():
        async with fast.run() as agent:
            prompts = await agent.test.list_prompts(server_name=None)

            # Verify we have prompts from the "prompts" server
            assert "prompts" in prompts
            assert len(prompts["prompts"]) > 0

            # Get name of first prompt to test with
            prompt_name = prompts["prompts"][0].name

            # Create properly namespaced name using the same separator as mcp_aggregator
            from mcp_agent.mcp.mcp_aggregator import SEP

            namespaced_name = f"prompts{SEP}{prompt_name}"

            # Apply the prompt directly
            response = await agent.test.apply_prompt(namespaced_name)

            # Verify the prompt was applied
            assert response, "No response from apply_prompt"
            assert len(agent.test._llm.message_history) > 0

    await agent_function()
