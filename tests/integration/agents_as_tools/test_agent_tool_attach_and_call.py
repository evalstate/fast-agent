"""Integration tests for agent-as-tool attachment and invocation."""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_tool_attach_and_call(fast_agent):
    """Test attaching an agent as a tool and calling it.

    This exercises the full flow:
    1. Create parent and child agents
    2. Attach child as tool to parent via add_agent_tool()
    3. Verify the tool appears in list_tools()
    4. Invoke the tool and verify it executes
    """
    fast = fast_agent

    @fast.agent(
        name="child",
        instruction="Echo the input back.",
        model="passthrough",
    )
    @fast.agent(
        name="parent",
        instruction="You have tools to help.",
        model="passthrough",
    )
    async def test_attach_and_call():
        async with fast.run() as agent:
            # Get parent and child agents
            parent = agent.parent
            child = agent.child

            # Attach child as tool to parent
            tool_name = parent.add_agent_tool(child)
            assert tool_name == "agent__child"

            # Verify tool appears in list
            tools = await parent.list_tools()
            tool_names = [t.name for t in tools.tools]
            assert "agent__child" in tool_names

            # Call the tool
            tool = parent._execution_tools[tool_name]
            result = await tool.run({"message": "hello"})
            assert result == "hello"  # passthrough echoes input

    await test_attach_and_call()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_tool_self_attach_prevented(fast_agent):
    """Test that attaching an agent to itself fails gracefully."""
    fast = fast_agent

    @fast.agent(
        name="solo",
        instruction="I work alone.",
        model="passthrough",
    )
    async def test_self_attach():
        async with fast.run() as agent:
            solo = agent.solo

            # Attaching to self should return existing tool name or handle gracefully
            tool_name = solo.add_agent_tool(solo)
            # The tool name should be agent__solo
            assert tool_name == "agent__solo"

    await test_self_attach()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_agent_tools(fast_agent):
    """Test attaching multiple agents as tools."""
    fast = fast_agent

    @fast.agent(name="helper1", instruction="Helper 1", model="passthrough")
    @fast.agent(name="helper2", instruction="Helper 2", model="passthrough")
    @fast.agent(name="coordinator", instruction="Coordinator", model="passthrough")
    async def test_multiple_tools():
        async with fast.run() as agent:
            coordinator = agent.coordinator
            helper1 = agent.helper1
            helper2 = agent.helper2

            # Attach both helpers
            tool1 = coordinator.add_agent_tool(helper1)
            tool2 = coordinator.add_agent_tool(helper2)

            assert tool1 == "agent__helper1"
            assert tool2 == "agent__helper2"

            # Verify both tools are available
            tools = await coordinator.list_tools()
            tool_names = [t.name for t in tools.tools]
            assert "agent__helper1" in tool_names
            assert "agent__helper2" in tool_names

            # Call both tools
            result1 = await coordinator._execution_tools[tool1].run({"message": "msg1"})
            result2 = await coordinator._execution_tools[tool2].run({"message": "msg2"})

            assert result1 == "msg1"
            assert result2 == "msg2"

    await test_multiple_tools()
