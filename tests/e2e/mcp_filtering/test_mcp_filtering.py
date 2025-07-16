#!/usr/bin/env python3
"""
E2E tests for MCP filtering functionality.
Tests tool, resource, and prompt filtering across different agent types.
"""

import pytest

from mcp_agent.agents.agent import Agent


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
                "filtering_test_server-math_add",
                "filtering_test_server-math_subtract", 
                "filtering_test_server-math_multiply",
                "filtering_test_server-string_upper",
                "filtering_test_server-string_lower",
                "filtering_test_server-utility_ping",
                "filtering_test_server-utility_status"
            }
            actual_tools = set(tool_names)
            assert actual_tools == expected_tools, f"Expected {expected_tools}, got {actual_tools}"

    # Test 2: Agent with filtering - should have only filtered tools
    @fast.agent(
        name="agent_with_filter",
        instruction="Agent with tool filtering",
        model="passthrough",
        servers=["filtering_test_server"],
        tools={"filtering_test_server": ["math_*", "string_upper"]},  # Only math tools and string_upper
    )
    async def agent_with_filter():
        async with fast.run() as agent_app:
            tools = await agent_app.agent_with_filter.list_tools()
            tool_names = [tool.name for tool in tools.tools]
            
            # Should have only math tools + string_upper
            expected_tools = {
                "filtering_test_server-math_add",
                "filtering_test_server-math_subtract",
                "filtering_test_server-math_multiply",
                "filtering_test_server-string_upper"
            }
            actual_tools = set(tool_names)
            assert actual_tools == expected_tools, f"Expected {expected_tools}, got {actual_tools}"
            
            # Should NOT have these tools
            excluded_tools = {
                "filtering_test_server-string_lower",
                "filtering_test_server-utility_ping", 
                "filtering_test_server-utility_status"
            }
            for tool_name in excluded_tools:
                assert tool_name not in tool_names, f"Tool {tool_name} should have been filtered out"

    await agent_no_filter()
    await agent_with_filter()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_resource_filtering_basic_agent(fast_agent):
    """Test resource filtering with basic agent - no filtering vs with filtering"""
    fast = fast_agent

    # Test 1: Agent without filtering - should have all resources
    @fast.agent(
        name="agent_no_filter",
        instruction="Agent without resource filtering",
        model="passthrough",
        servers=["filtering_test_server"],
    )
    async def agent_no_filter():
        async with fast.run() as agent_app:
            resources = await agent_app.agent_no_filter.list_resources()
            resource_uris = resources["filtering_test_server"]  # Already a list of URI strings
            
            # Should have all 4 resources
            expected_resources = {
                "resource://math/constants",
                "resource://math/formulas",
                "resource://string/examples", 
                "resource://utility/info"
            }
            actual_resources = set(resource_uris)
            assert actual_resources == expected_resources, f"Expected {expected_resources}, got {actual_resources}"

    # Test 2: Agent with filtering - should have only filtered resources
    @fast.agent(
        name="agent_with_filter",
        instruction="Agent with resource filtering",
        model="passthrough",
        servers=["filtering_test_server"],
        resources={"filtering_test_server": ["resource://math/*", "resource://string/examples"]},
    )
    async def agent_with_filter():
        async with fast.run() as agent_app:
            resources = await agent_app.agent_with_filter.list_resources()
            resource_uris = resources.get("filtering_test_server", [])  # Get list or empty list if server not present
            
            # Should have only math resources + string examples
            expected_resources = {
                "resource://math/constants",
                "resource://math/formulas",
                "resource://string/examples"
            }
            actual_resources = set(resource_uris)
            assert actual_resources == expected_resources, f"Expected {expected_resources}, got {actual_resources}"
            
            # Should NOT have utility resource
            excluded_resources = {"resource://utility/info"}
            for resource_uri in excluded_resources:
                assert resource_uri not in resource_uris, f"Resource {resource_uri} should have been filtered out"

    await agent_no_filter()
    await agent_with_filter()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_prompt_filtering_basic_agent(fast_agent):
    """Test prompt filtering with basic agent - no filtering vs with filtering"""
    fast = fast_agent

    # Test 1: Agent without filtering - should have all prompts
    @fast.agent(
        name="agent_no_filter",
        instruction="Agent without prompt filtering",
        model="passthrough",
        servers=["filtering_test_server"],
    )
    async def agent_no_filter():
        async with fast.run() as agent_app:
            prompts = await agent_app.agent_no_filter.list_prompts()
            prompt_names = [prompt.name for prompt in prompts["filtering_test_server"]]
            
            # Should have all 4 prompts
            expected_prompts = {
                "math_helper",
                "math_teacher",
                "string_processor",
                "utility_assistant"
            }
            actual_prompts = set(prompt_names)
            assert actual_prompts == expected_prompts, f"Expected {expected_prompts}, got {actual_prompts}"

    # Test 2: Agent with filtering - should have only filtered prompts
    @fast.agent(
        name="agent_with_filter",
        instruction="Agent with prompt filtering",
        model="passthrough",
        servers=["filtering_test_server"],
        prompts={"filtering_test_server": ["math_*", "utility_assistant"]},
    )
    async def agent_with_filter():
        async with fast.run() as agent_app:
            prompts = await agent_app.agent_with_filter.list_prompts()
            prompt_list = prompts.get("filtering_test_server", [])  # Get list or empty list if server not present
            prompt_names = [prompt.name for prompt in prompt_list]
            
            # Should have only math prompts + utility_assistant
            expected_prompts = {
                "math_helper",
                "math_teacher",
                "utility_assistant"
            }
            actual_prompts = set(prompt_names)
            assert actual_prompts == expected_prompts, f"Expected {expected_prompts}, got {actual_prompts}"
            
            # Should NOT have string_processor
            excluded_prompts = {"string_processor"}
            for prompt_name in excluded_prompts:
                assert prompt_name not in prompt_names, f"Prompt {prompt_name} should have been filtered out"

    await agent_no_filter()
    await agent_with_filter()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_tool_filtering_router_agent(fast_agent):
    """Test tool filtering with router agent"""
    fast = fast_agent

    # Create a worker agent for the router
    @fast.agent(
        name="math_worker",
        instruction="Math worker agent",
        model="passthrough",
        servers=["filtering_test_server"],
    )
    async def math_worker():
        pass

    # Router agent with filtering
    @fast.router(
        name="math_router",
        agents=["math_worker"],
        servers=["filtering_test_server"],
        tools={"filtering_test_server": ["math_*"]},  # Only math tools
        instruction="Router with tool filtering",
        model="passthrough",
    )
    async def math_router():
        async with fast.run() as agent_app:
            tools = await agent_app.math_router.list_tools()
            tool_names = [tool.name for tool in tools.tools]
            
            # Should have only math tools
            expected_tools = {
                "filtering_test_server-math_add",
                "filtering_test_server-math_subtract",
                "filtering_test_server-math_multiply"
            }
            actual_tools = set(tool_names)
            assert actual_tools == expected_tools, f"Expected {expected_tools}, got {actual_tools}"
            
            # Should NOT have string or utility tools
            excluded_tools = {
                "filtering_test_server-string_upper",
                "filtering_test_server-string_lower",
                "filtering_test_server-utility_ping",
                "filtering_test_server-utility_status"
            }
            for tool_name in excluded_tools:
                assert tool_name not in tool_names, f"Tool {tool_name} should have been filtered out"

    await math_router()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_tool_filtering_custom_agent(fast_agent):
    """Test tool filtering with custom agent"""
    fast = fast_agent

    # Custom agent with filtering
    @fast.custom(
        Agent,
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
                "filtering_test_server-string_upper",
                "filtering_test_server-string_lower"
            }
            actual_tools = set(tool_names)
            assert actual_tools == expected_tools, f"Expected {expected_tools}, got {actual_tools}"
            
            # Should NOT have math or utility tools
            excluded_tools = {
                "filtering_test_server-math_add",
                "filtering_test_server-math_subtract",
                "filtering_test_server-math_multiply",
                "filtering_test_server-utility_ping",
                "filtering_test_server-utility_status"
            }
            for tool_name in excluded_tools:
                assert tool_name not in tool_names, f"Tool {tool_name} should have been filtered out"

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
                "filtering_test_server-math_add",
                "filtering_test_server-math_subtract",
                "filtering_test_server-math_multiply"
            }
            actual_tools = set(tool_names)
            assert actual_tools == expected_tools, f"Tools - Expected {expected_tools}, got {actual_tools}"
            
            # Test resources
            resources = await agent_app.agent_combined_filter.list_resources()
            resource_uris = resources.get("filtering_test_server", [])  # Get list or empty list if server not present
            expected_resources = {
                "resource://math/constants",
                "resource://math/formulas"
            }
            actual_resources = set(resource_uris)
            assert actual_resources == expected_resources, f"Resources - Expected {expected_resources}, got {actual_resources}"
            
            # Test prompts
            prompts = await agent_app.agent_combined_filter.list_prompts()
            prompt_list = prompts.get("filtering_test_server", [])  # Get list or empty list if server not present
            prompt_names = [prompt.name for prompt in prompt_list]
            expected_prompts = {
                "math_helper",
                "math_teacher"
            }
            actual_prompts = set(prompt_names)
            assert actual_prompts == expected_prompts, f"Prompts - Expected {expected_prompts}, got {actual_prompts}"

    await agent_combined_filter() 