"""
Integration tests for the MAKER workflow agent.

MAKER: Massively decomposed Agentic processes with K-voting Error Reduction.

Based on the paper "Solving a Million-Step LLM Task with Zero Errors"
(arXiv:2511.09030). Tests verify the first-to-ahead-by-k voting mechanism
and red-flag filtering for statistical error correction.
"""

import pytest

from fast_agent.core.prompt import Prompt
from fast_agent.llm.internal.passthrough import FIXED_RESPONSE_INDICATOR


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basic_voting_consensus(fast_agent):
    """Test that identical responses achieve immediate k-margin consensus."""
    fast = fast_agent

    @fast.agent(name="worker", model="passthrough")
    @fast.maker(name="voter", worker="worker", k=2)
    async def agent_function():
        async with fast.run() as agent:
            # Prime worker to return consistent responses
            consistent_response = f"{FIXED_RESPONSE_INDICATOR}42"
            for _ in range(3):
                await agent.worker._llm.generate([Prompt.user(consistent_response)])

            result = await agent.voter.send("What is the answer?")

            assert "42" in result
            assert agent.voter.last_result is not None
            assert agent.voter.last_result.converged is True
            assert agent.voter.last_result.margin >= 2

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_first_to_ahead_by_k(fast_agent):
    """Test that voting requires k-margin to declare winner."""
    fast = fast_agent

    @fast.agent(name="worker", model="passthrough")
    @fast.maker(name="voter", worker="worker", k=3, max_samples=10)
    async def agent_function():
        async with fast.run() as agent:
            # Prime responses: A gets 4 votes, B gets 1 vote
            # A should win with margin of 3
            responses = [
                f"{FIXED_RESPONSE_INDICATOR}A",
                f"{FIXED_RESPONSE_INDICATOR}A",
                f"{FIXED_RESPONSE_INDICATOR}B",
                f"{FIXED_RESPONSE_INDICATOR}A",
                f"{FIXED_RESPONSE_INDICATOR}A",
            ]
            for resp in responses:
                await agent.worker._llm.generate([Prompt.user(resp)])

            result = await agent.voter.send("Choose A or B")

            assert "A" in result
            assert agent.voter.last_result is not None
            assert agent.voter.last_result.converged is True
            # A has 4 votes, B has 1, margin = 3
            assert agent.voter.last_result.margin >= 3

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_max_samples_fallback(fast_agent):
    """Test that plurality is used when max_samples reached without k-margin."""
    fast = fast_agent

    @fast.agent(name="worker", model="passthrough")
    @fast.maker(name="voter", worker="worker", k=10, max_samples=10)
    async def agent_function():
        async with fast.run() as agent:
            # With passthrough, all responses will be the same (last primed)
            # Set k=10 so we need 10 identical samples to converge
            # But we'll only get 10 samples total, so margin will be 10 (all same)
            # This tests that we get a result even at the limit
            await agent.worker._llm.generate(
                [Prompt.user(f"{FIXED_RESPONSE_INDICATOR}consistent")]
            )

            result = await agent.voter.send("Choose")

            # Should get the consistent response
            assert "consistent" in result
            assert agent.voter.last_result is not None
            assert agent.voter.last_result.total_samples <= 10

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_red_flag_config(fast_agent):
    """Test that red_flag_max_length configuration is accepted."""
    fast = fast_agent

    @fast.agent(name="worker", model="passthrough")
    @fast.maker(name="voter", worker="worker", k=2, red_flag_max_length=100)
    async def agent_function():
        async with fast.run() as agent:
            # Prime a short response that won't be red-flagged
            short_response = f"{FIXED_RESPONSE_INDICATOR}ok"
            await agent.worker._llm.generate([Prompt.user(short_response)])

            result = await agent.voter.send("Question")

            # Should succeed with short response
            assert "ok" in result
            assert agent.voter.last_result is not None
            assert agent.voter.last_result.discarded_samples == 0

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_match_strategy_normalized(fast_agent):
    """Test that normalized matching ignores whitespace and case."""
    fast = fast_agent

    @fast.agent(name="worker", model="passthrough")
    @fast.maker(name="voter", worker="worker", k=2, match_strategy="normalized")
    async def agent_function():
        async with fast.run() as agent:
            # These should all be treated as the same response
            responses = [
                f"{FIXED_RESPONSE_INDICATOR}Hello World",
                f"{FIXED_RESPONSE_INDICATOR}hello world",
                f"{FIXED_RESPONSE_INDICATOR}  HELLO   WORLD  ",
            ]
            for resp in responses:
                await agent.worker._llm.generate([Prompt.user(resp)])

            await agent.voter.send("Greet")

            # All 3 should count as same vote, achieving k=2 margin immediately
            assert agent.voter.last_result is not None
            assert agent.voter.last_result.converged is True
            # With normalized matching, all 3 responses are identical
            assert len(agent.voter.last_result.votes) == 1

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_match_strategy_structured(fast_agent):
    """Test that structured matching compares JSON structurally."""
    fast = fast_agent

    @fast.agent(name="worker", model="passthrough")
    @fast.maker(name="voter", worker="worker", k=2, match_strategy="structured")
    async def agent_function():
        async with fast.run() as agent:
            # These JSON objects are structurally identical despite key order
            responses = [
                f'{FIXED_RESPONSE_INDICATOR}{{"a": 1, "b": 2}}',
                f'{FIXED_RESPONSE_INDICATOR}{{"b": 2, "a": 1}}',
                f'{FIXED_RESPONSE_INDICATOR}{{"a": 1, "b": 2}}',
            ]
            for resp in responses:
                await agent.worker._llm.generate([Prompt.user(resp)])

            await agent.voter.send("Return JSON")

            assert agent.voter.last_result is not None
            assert agent.voter.last_result.converged is True
            # With structured matching, all 3 responses are identical
            assert len(agent.voter.last_result.votes) == 1

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_voting_result_tracking(fast_agent):
    """Test that voting results are properly tracked and accessible."""
    fast = fast_agent

    @fast.agent(name="worker", model="passthrough")
    @fast.maker(name="voter", worker="worker", k=2, max_samples=10)
    async def agent_function():
        async with fast.run() as agent:
            # Prime varied responses - alpha appears twice consecutively
            # so it will win with k=2 margin after 2 samples
            responses = [
                f"{FIXED_RESPONSE_INDICATOR}alpha",
                f"{FIXED_RESPONSE_INDICATOR}alpha",
            ]
            for resp in responses:
                await agent.worker._llm.generate([Prompt.user(resp)])

            await agent.voter.send("Choose")

            # Verify result tracking
            result = agent.voter.last_result
            assert result is not None
            assert "alpha" in result.winner
            assert result.total_samples >= 2
            assert result.votes.get("alpha", 0) >= 2
            assert result.margin >= 2
            assert result.converged is True

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_single_response_wins_immediately(fast_agent):
    """Test that with k=1, a single unique response wins immediately."""
    fast = fast_agent

    @fast.agent(name="worker", model="passthrough")
    @fast.maker(name="voter", worker="worker", k=1)
    async def agent_function():
        async with fast.run() as agent:
            # Single response should win with k=1
            await agent.worker._llm.generate(
                [Prompt.user(f"{FIXED_RESPONSE_INDICATOR}winner")]
            )

            result = await agent.voter.send("Quick test")

            assert "winner" in result
            assert agent.voter.last_result is not None
            assert agent.voter.last_result.total_samples == 1
            assert agent.voter.last_result.converged is True

    await agent_function()
