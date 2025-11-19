#!/usr/bin/env python3
"""
E2E tests for MCP filtering functionality.
Tests tool, resource, and prompt filtering across different agent types.
"""

import pytest

from fast_agent.agents import McpAgent
from fast_agent.mcp.common import SEP


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_tool_filtering_basic_agent(fast_agent):
    """Test tool filtering with basic agent - no filtering vs with filtering"""
    fast = fast_agent

    # Test 1: Agent without filtering - should have all tools
    @fast.agent(
        name="agent_no_filter",
        instruction="Agent without tool filtering",
        model="passthrough",
        servers=["filtering_test_server"],
    )
    async def agent_no_filter():
        async with fast.run() as agent_app:
            tools = await agent_app.agent_no_filter.list_tools()
            tool_names = [tool.name for tool in tools.tools]

            # Should have all 7 tools
            expected_tools = {
                f"filtering_test_server{SEP}math_add",
                f"filtering_test_server{SEP}math_subtract",
                f"filtering_test_server{SEP}math_multiply",
                f"filtering_test_server{SEP}string_upper",
                f"filtering_test_server{SEP}string_lower",
                f"filtering_test_server{SEP}utility_ping",
                f"filtering_test_server{SEP}utility_status",
            }
            actual_tools = set(tool_names)
            assert actual_tools == expected_tools, f"Expected {expected_tools}, got {actual_tools}"

    # Test 2: Agent with filtering - should have only filtered tools
    @fast.agent(
        name="agent_with_filter",
        instruction="Agent with tool filtering",
        model="passthrough",
        servers=["filtering_test_server"],
        tools={
            "filtering_test_server": ["math_*", "string_upper"]
        },  # Only math tools and string_upper
    )
    async def agent_with_filter():
        async with fast.run() as agent_app:
            tools = await agent_app.agent_with_filter.list_tools()
            tool_names = [tool.name for tool in tools.tools]

            # Should have only math tools + string_upper
            expected_tools = {
                f"filtering_test_server{SEP}math_add",
                f"filtering_test_server{SEP}math_subtract",
                f"filtering_test_server{SEP}math_multiply",
                f"filtering_test_server{SEP}string_upper",
            }
            actual_tools = set(tool_names)
            assert actual_tools == expected_tools, f"Expected {expected_tools}, got {actual_tools}"

            # Should NOT have these tools
            excluded_tools = {
                f"filtering_test_server{SEP}string_lower",
                f"filtering_test_server{SEP}utility_ping",
                f"filtering_test_server{SEP}utility_status",
            }
            for tool_name in excluded_tools:
                assert tool_name not in tool_names, (
                    f"Tool {tool_name} should have been filtered out"
                )

    await agent_no_filter()
    await agent_with_filter()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_resource_filtering_returns_all_without_filters(fast_agent):
    """Agents with no resource filters should expose every resource."""
    fast = fast_agent

    @fast.agent(
        name="resource_no_filter_agent",
        instruction="Agent without resource filtering",
        model="passthrough",
        servers=["filtering_test_server"],
    )
    async def resource_no_filter_agent():
        async with fast.run() as agent_app:
            resources = await agent_app.resource_no_filter_agent.list_resources()
            actual = set(resources["filtering_test_server"])
            expected = {
                "resource://math/constants",
                "resource://math/formulas",
                "resource://string/examples",
                "resource://utility/info",
            }
            assert actual == expected, f"Expected {expected}, got {actual}"

    await resource_no_filter_agent()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_resource_filtering_applies_patterns(fast_agent):
    """Agents honour configured resource filters for a server."""
    fast = fast_agent

    @fast.agent(
        name="resource_filtered_agent",
        instruction="Agent with resource filtering",
        model="passthrough",
        servers=["filtering_test_server"],
        resources={"filtering_test_server": ["resource://math/*", "resource://string/examples"]},
    )
    async def resource_filtered_agent():
        async with fast.run() as agent_app:
            resources = await agent_app.resource_filtered_agent.list_resources()
            actual = set(resources.get("filtering_test_server", []))
            expected = {
                "resource://math/constants",
                "resource://math/formulas",
                "resource://string/examples",
            }
            assert actual == expected, f"Expected {expected}, got {actual}"
            assert "resource://utility/info" not in actual

    await resource_filtered_agent()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_resource_filtering_ignores_irrelevant_filters(fast_agent):
    """Filters on other namespaces must not affect the target server."""
    fast = fast_agent

    @fast.agent(
        name="resource_irrelevant_filter_agent",
        instruction="Agent with irrelevant resource filters",
        model="passthrough",
        servers=["filtering_test_server"],
        resources={"other_server": ["resource://*"]},
    )
    async def resource_irrelevant_filter_agent():
        async with fast.run() as agent_app:
            resources = await agent_app.resource_irrelevant_filter_agent.list_resources()
            actual = set(resources["filtering_test_server"])
            expected = {
                "resource://math/constants",
                "resource://math/formulas",
                "resource://string/examples",
                "resource://utility/info",
            }
            assert actual == expected, f"Expected {expected}, got {actual}"

    await resource_irrelevant_filter_agent()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_prompt_filtering_returns_all_without_filters(fast_agent):
    """Agents with no prompt filters should expose every prompt."""
    fast = fast_agent

    @fast.agent(
        name="prompt_no_filter_agent",
        instruction="Agent without prompt filtering",
        model="passthrough",
        servers=["filtering_test_server"],
    )
    async def prompt_no_filter_agent():
        async with fast.run() as agent_app:
            prompts = await agent_app.prompt_no_filter_agent.list_prompts()
            actual = {prompt.name for prompt in prompts["filtering_test_server"]}
            expected = {"math_helper", "math_teacher", "string_processor", "utility_assistant"}
            assert actual == expected, f"Expected {expected}, got {actual}"

    await prompt_no_filter_agent()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_prompt_filtering_applies_patterns(fast_agent):
    """Agents honour configured prompt filters for a server."""
    fast = fast_agent

    @fast.agent(
        name="prompt_filtered_agent",
        instruction="Agent with prompt filtering",
        model="passthrough",
        servers=["filtering_test_server"],
        prompts={"filtering_test_server": ["math_*", "utility_assistant"]},
    )
    async def prompt_filtered_agent():
        async with fast.run() as agent_app:
            prompts = await agent_app.prompt_filtered_agent.list_prompts()
            actual = {prompt.name for prompt in prompts.get("filtering_test_server", [])}
            expected = {"math_helper", "math_teacher", "utility_assistant"}
            assert actual == expected, f"Expected {expected}, got {actual}"
            assert "string_processor" not in actual

    await prompt_filtered_agent()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_prompt_filtering_ignores_irrelevant_filters(fast_agent):
    """Filters on other namespaces must not affect prompts for the target server."""
    fast = fast_agent

    @fast.agent(
        name="prompt_irrelevant_filter_agent",
        instruction="Agent with irrelevant prompt filters",
        model="passthrough",
        servers=["filtering_test_server"],
        prompts={"some_other_server": ["*"]},
    )
    async def prompt_irrelevant_filter_agent():
        async with fast.run() as agent_app:
            prompts = await agent_app.prompt_irrelevant_filter_agent.list_prompts()
            actual = {prompt.name for prompt in prompts["filtering_test_server"]}
            expected = {"math_helper", "math_teacher", "string_processor", "utility_assistant"}
            assert actual == expected, f"Expected {expected}, got {actual}"

    await prompt_irrelevant_filter_agent()


@pytest.mark.integration
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_tool_filtering_custom_agent(fast_agent):
    """Test tool filtering with custom agent"""
    fast = fast_agent

    # Custom agent with filtering
    @fast.custom(
        McpAgent,
        name="custom_string_agent",
        instruction="Custom agent with tool filtering",
        model="passthrough",
        servers=["filtering_test_server"],
        tools={"filtering_test_server": ["string_*"]},  # Only string tools
    )
    async def custom_string_agent():
        async with fast.run() as agent_app:
            tools = await agent_app.custom_string_agent.list_tools()
            tool_names = [tool.name for tool in tools.tools]

            # Should have only string tools
            expected_tools = {
                f"filtering_test_server{SEP}string_upper",
                f"filtering_test_server{SEP}string_lower",
            }
            actual_tools = set(tool_names)
            assert actual_tools == expected_tools, f"Expected {expected_tools}, got {actual_tools}"

            # Should NOT have math or utility tools
            excluded_tools = {
                f"filtering_test_server{SEP}math_add",
                f"filtering_test_server{SEP}math_subtract",
                f"filtering_test_server{SEP}math_multiply",
                f"filtering_test_server{SEP}utility_ping",
                f"filtering_test_server{SEP}utility_status",
            }
            for tool_name in excluded_tools:
                assert tool_name not in tool_names, (
                    f"Tool {tool_name} should have been filtered out"
                )

    await custom_string_agent()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_combined_filtering(fast_agent):
    """Test combined tool, resource, and prompt filtering"""
    fast = fast_agent

    @fast.agent(
        name="agent_combined_filter",
        instruction="Agent with combined filtering",
        model="passthrough",
        servers=["filtering_test_server"],
        tools={"filtering_test_server": ["math_*"]},
        resources={"filtering_test_server": ["resource://math/*"]},
        prompts={"filtering_test_server": ["math_*"]},
    )
    async def agent_combined_filter():
        async with fast.run() as agent_app:
            # Test tools
            tools = await agent_app.agent_combined_filter.list_tools()
            tool_names = [tool.name for tool in tools.tools]
            expected_tools = {
                f"filtering_test_server{SEP}math_add",
                f"filtering_test_server{SEP}math_subtract",
                f"filtering_test_server{SEP}math_multiply",
            }
            actual_tools = set(tool_names)
            assert actual_tools == expected_tools, (
                f"Tools - Expected {expected_tools}, got {actual_tools}"
            )

            # Test resources
            resources = await agent_app.agent_combined_filter.list_resources()
            resource_uris = resources.get(
                "filtering_test_server", []
            )  # Get list or empty list if server not present
            expected_resources = {"resource://math/constants", "resource://math/formulas"}
            actual_resources = set(resource_uris)
            assert actual_resources == expected_resources, (
                f"Resources - Expected {expected_resources}, got {actual_resources}"
            )

            # Test prompts
            prompts = await agent_app.agent_combined_filter.list_prompts()
            prompt_list = prompts.get(
                "filtering_test_server", []
            )  # Get list or empty list if server not present
            prompt_names = [prompt.name for prompt in prompt_list]
            expected_prompts = {"math_helper", "math_teacher"}
            actual_prompts = set(prompt_names)
            assert actual_prompts == expected_prompts, (
                f"Prompts - Expected {expected_prompts}, got {actual_prompts}"
            )

    await agent_combined_filter()
