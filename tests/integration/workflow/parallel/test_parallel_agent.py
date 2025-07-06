import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parallel_run(fast_agent):
    """Single user message."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(name="fan_out_1")
    @fast.agent(name="fan_out_2")
    @fast.agent(name="fan_in")
    @fast.parallel(name="parallel", fan_out=["fan_out_1", "fan_out_2"], fan_in="fan_in")
    async def agent_function():
        async with fast.run() as agent:
            expected: str = """The following request was sent to the agents:

<fastagent:request>
foo
</fastagent:request>

<fastagent:response agent="fan_out_1">
foo
</fastagent:response>

<fastagent:response agent="fan_out_2">
foo
</fastagent:response>"""
            assert expected == await agent.parallel.send("foo")

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parallel_default_fan_in(fast_agent):
    """Single user message."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(name="fan_out_1")
    @fast.agent(name="fan_out_2")
    @fast.parallel(name="parallel", fan_out=["fan_out_1", "fan_out_2"])
    async def agent_function():
        async with fast.run() as agent:
            expected: str = """The following request was sent to the agents:

<fastagent:request>
foo
</fastagent:request>

<fastagent:response agent="fan_out_1">
foo
</fastagent:response>

<fastagent:response agent="fan_out_2">
foo
</fastagent:response>"""
            # in this case the behaviour is the same as the previous test - but the fan-in passthrough was created automatically
            assert expected == await agent.parallel.send("foo")

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parallel_include_request_false(fast_agent):
    """Test that include_request=False prevents original request from being sent to fan_in agent."""
    fast = fast_agent

    @fast.agent(name="fan_out_1")
    @fast.agent(name="fan_out_2")
    @fast.agent(name="fan_in")
    @fast.parallel(
        name="parallel_no_request",
        fan_out=["fan_out_1", "fan_out_2"],
        fan_in="fan_in",
        include_request=False
    )
    async def agent_function():
        async with fast.run() as agent:
            # When include_request=False, the fan_in should only receive the responses,
            # not the original request
            expected: str = """<fastagent:response agent="fan_out_1">
foo
</fastagent:response>

<fastagent:response agent="fan_out_2">
foo
</fastagent:response>"""
            result = await agent.parallel_no_request.send("foo")
            assert expected == result

    await agent_function()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_parallel_include_request_true(fast_agent):
    """Test that include_request=True includes original request in fan_in agent input."""
    fast = fast_agent

    @fast.agent(name="fan_out_1")
    @fast.agent(name="fan_out_2")
    @fast.agent(name="fan_in")
    @fast.parallel(
        name="parallel_with_request",
        fan_out=["fan_out_1", "fan_out_2"],
        fan_in="fan_in",
        include_request=True
    )
    async def agent_function():
        async with fast.run() as agent:
            # When include_request=True, the fan_in should receive both the original request
            # and the responses from fan_out agents
            expected: str = """The following request was sent to the agents:

<fastagent:request>
foo
</fastagent:request>

<fastagent:response agent="fan_out_1">
foo
</fastagent:response>

<fastagent:response agent="fan_out_2">
foo
</fastagent:response>"""
            result = await agent.parallel_with_request.send("foo")
            assert expected == result

    await agent_function()
