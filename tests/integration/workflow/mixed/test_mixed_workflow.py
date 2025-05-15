import pytest

from mcp_agent.core.prompt import Prompt


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chaining_routers(fast_agent):
    """Check that the router routes"""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    @fast.agent(name="target1")
    @fast.agent(name="target2")
    @fast.agent(name="target3")
    @fast.router(name="router1", agents=["target1", "target2"])
    @fast.chain(name="chain", sequence=["router1", "target3"], cumulative=True)
    async def agent_function():
        async with fast.run() as agent:
            await agent.router1._llm.generate(
                [
                    Prompt.user(
                        """***FIXED_RESPONSE 
                        {"agent": "target2",
                        "confidence": "high",
                        "reasoning": "Test Request"}"""
                    )
                ]
            )
            result = await agent.chain.send("github.com/varaarul")
            assert "github.com/varaarul" in result
            assert "target3" in result

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_router_selects_parallel(fast_agent):
    """Check that the router routes"""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    @fast.agent(name="target1")
    @fast.agent(name="target2")
    @fast.agent(name="target3")
    @fast.parallel(name="parallel", fan_out=["target2", "target3"])
    @fast.router(name="router1", agents=["target1", "parallel"])
    async def agent_function():
        async with fast.run() as agent:
            await agent.router1._llm.generate(
                [
                    Prompt.user(
                        """***FIXED_RESPONSE 
                        {"agent": "parallel",
                        "confidence": "high",
                        "reasoning": "Test Request"}"""
                    )
                ]
            )
            result = await agent.router1.send("github.com/varaarul")
            assert "github.com/varaarul" in result

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chain_in_eval_optimizer(fast_agent):
    """Check that generator can be a chain"""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    @fast.agent(name="chain1")
    @fast.agent(name="chain2")
    @fast.chain(name="chain", sequence=["chain1", "chain2"])
    @fast.agent(name="check", instruction="You are an evaluator. Rate responses as EXCELLENT.")
    @fast.evaluator_optimizer(name="eval_opt", generator="chain", evaluator="check")
    async def agent_function():
        async with fast.run() as agent:
            # Mock the evaluation response to be EXCELLENT to avoid multiple iterations
            await agent.check._llm.generate(
                [
                    Prompt.user(
                        """***FIXED_RESPONSE 
                        {
                          "rating": "EXCELLENT",
                          "feedback": "Perfect response",
                          "needs_improvement": false,
                          "focus_areas": []
                        }"""
                    )
                ]
            )
            # Test that the chain works as a generator in eval_opt
            result = await agent.eval_opt.send("Test message")
            # We should get a response from the eval_opt workflow
            assert result is not None and len(result) > 0

    await agent_function()
