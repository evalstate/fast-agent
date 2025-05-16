"""
Test the prompt listing and selection functionality directly.
"""

import pytest

from mcp_agent.core.interactive_prompt import InteractivePrompt


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_all_prompts_with_none_server(fast_agent):
    """Test the _get_all_prompts function with None as server name."""
    fast = fast_agent

    @fast.agent(name="test", servers=["prompts"])
    async def agent_function():
        async with fast.run() as agent:
            # Create instance of InteractivePrompt
            prompt_ui = InteractivePrompt()

            # Get the list_prompts function from the agent
            list_prompts_func = agent.test.list_prompts

            # Call _get_all_prompts directly
            all_prompts = await prompt_ui._get_all_prompts(list_prompts_func)

            # Verify we got results
            assert len(all_prompts) > 0

            # Verify each prompt has the correct format
            for prompt in all_prompts:
                assert "server" in prompt
                assert "name" in prompt
                assert "namespaced_name" in prompt
                assert prompt["server"] == "prompts"  # From our test config

                # Check namespace format
                assert prompt["namespaced_name"] == f"prompts-{prompt['name']}"

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
