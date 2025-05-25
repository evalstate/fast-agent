import pytest

from mcp_agent.core.exceptions import AgentConfigError
from mcp_agent.core.prompt import Prompt


@pytest.mark.integration
@pytest.mark.asyncio
async def test_disallows_empty_sequence(fast_agent):
    fast = fast_agent

    # Define the agent
    with pytest.raises(AgentConfigError):

        @fast.chain(name="chain", sequence=[], cumulative=True)
        async def agent_function():
            async with fast.run():
                assert True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_simple_chain(fast_agent):
    """Test a simple chain in non-cumulative mode (default)"""
    fast = fast_agent

    # Define the agent
    @fast.agent(name="begin")
    @fast.agent(name="step1")
    @fast.agent(name="finish")
    @fast.chain(name="chain", sequence=["begin", "step1", "finish"])
    async def agent_function():
        async with fast.run() as agent:
            await agent.begin.apply_prompt_messages([Prompt.assistant("begin")])
            await agent.step1.apply_prompt_messages([Prompt.assistant("step1")])
            await agent.finish.apply_prompt_messages([Prompt.assistant("finish")])

            result = await agent.chain.send("foo")
            assert "finish" == result

            assert "EXHAUSTED" in await agent.begin.send("extra")
            assert "EXHAUSTED" in await agent.step1.send("extra")
            assert "EXHAUSTED" in await agent.finish.send("extra")

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cumulative_chain(fast_agent):
    """Test cumulative chain mode with XML tags for request and responses"""
    fast = fast_agent

    @fast.agent(name="begin")
    @fast.agent(name="step1")
    @fast.agent(name="finish")
    @fast.chain(name="chain", sequence=["begin", "step1", "finish"], cumulative=True)
    async def agent_function():
        async with fast.run() as agent:
            await agent.begin.apply_prompt_messages([Prompt.assistant("begin-response")])
            await agent.step1.apply_prompt_messages([Prompt.assistant("step1-response")])
            await agent.finish.apply_prompt_messages([Prompt.assistant("finish-response")])

            initial_prompt = "initial-prompt"
            result = await agent.chain.send(initial_prompt)

            # Check for original request tag
            assert f"<fastagent:request>{initial_prompt}</fastagent:request>" in result

            # Check for agent response tags
            assert "<fastagent:response agent='begin'>begin-response</fastagent:response>" in result
            assert "<fastagent:response agent='step1'>step1-response</fastagent:response>" in result
            assert (
                "<fastagent:response agent='finish'>finish-response</fastagent:response>" in result
            )

            # Verify correct number of tags
            assert result.count("<fastagent:request>") == 1
            assert result.count("<fastagent:response") == 3

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chain_functionality(fast_agent):
    """Test that a chain correctly connects agents together"""
    # The goal of this test is to verify that a chain properly
    # connects agents together and passes messages through them

    fast = fast_agent

    # Use passthrough agents for simplicity and predictability
    # Create two separate chains: one normal and one cumulative
    @fast.agent(name="echo1", model="passthrough")
    @fast.agent(name="echo2", model="passthrough")
    @fast.agent(name="echo3", model="passthrough")
    @fast.chain(name="echo_chain", sequence=["echo1", "echo2", "echo3"])
    @fast.chain(name="cumulative_chain", sequence=["echo1", "echo2", "echo3"], cumulative=True)
    async def agent_function():
        async with fast.run() as agent:
            input_message = "test message"
            result = await agent.echo_chain.send(input_message)

            assert input_message in result

            cumulative_input = "cumulative message"
            cumulative_result = await agent.cumulative_chain.send(cumulative_input)

            # Verify both format and content
            assert "<fastagent:request>" in cumulative_result
            assert "<fastagent:response agent='echo1'>" in cumulative_result
            assert "<fastagent:response agent='echo2'>" in cumulative_result
            assert "<fastagent:response agent='echo3'>" in cumulative_result
            assert cumulative_input in cumulative_result

    await agent_function()
