#!/usr/bin/env python3
"""
Integration tests for MCP Server instructions feature.
Tests that server instructions are properly captured, formatted, and injected into agent system prompts.
"""

import re

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_server_instructions_template_substitution(fast_agent):
    """Test that {{serverInstructions}} template variable is properly substituted"""
    fast = fast_agent

    # Agent with template variables in instruction
    @fast.agent(
        name="agent_with_template",
        instruction="""You are a helpful AI assistant.

## Server Instructions
{{serverInstructions}}

Please follow the server-specific instructions above.""",
        model="passthrough",
        servers=["instructions_enabled"],
    )
    async def agent_with_template():
        async with fast.run() as agent_app:
            agent = agent_app.agent_with_template

            # Check that template variables were replaced
            assert "{{serverInstructions}}" not in agent.instruction, \
                "{{serverInstructions}} template was not replaced"

            # Check that server instructions are present in XML format
            assert "<mcp-server" in agent.instruction, \
                "Server instructions not formatted as XML"
            assert 'name="instructions_enabled"' in agent.instruction, \
                "Server name not in XML format"

            # Check that the actual instructions content is present
            assert "calculation and text manipulation tools" in agent.instruction, \
                "Server instructions content not found"

            # Check that tools are listed
            assert "<tools>" in agent.instruction, \
                "Tools section not found in XML"
            assert "instructions_enabled-calculate_sum" in agent.instruction, \
                "Tool names not properly prefixed in instructions"


            return agent.instruction

    result = await agent_with_template()
    print("Agent instruction after template substitution:")
    print(result)
    return result


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_server_instructions_include_flag(fast_agent):
    """Test that include_instructions config flag works correctly"""
    fast = fast_agent

    # Agent with instructions_disabled server (include_instructions: false)
    @fast.agent(
        name="agent_disabled",
        instruction="Base instruction. {{serverInstructions}}",
        model="passthrough",
        servers=["instructions_disabled"],
    )
    async def agent_disabled():
        async with fast.run() as agent_app:
            agent = agent_app.agent_disabled

            # Template should be replaced with empty string
            assert "{{serverInstructions}}" not in agent.instruction, \
                "Template variable not replaced"

            # Should NOT contain the server instructions
            assert "calculation and text manipulation tools" not in agent.instruction, \
                "Server instructions found when include_instructions is false"

            # Should not have XML server blocks
            assert "<mcp-server" not in agent.instruction, \
                "Server XML found when include_instructions is false"

            return agent.instruction

    result = await agent_disabled()
    print("Agent instruction with disabled instructions:")
    print(result)
    assert result.strip() == "Base instruction.", \
        f"Expected 'Base instruction.' but got '{result.strip()}'"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_multiple_servers_instructions(fast_agent):
    """Test handling of multiple servers with different instruction configurations"""
    fast = fast_agent

    @fast.agent(
        name="agent_multiple",
        instruction="System prompt\n{{serverInstructions}}",
        model="passthrough",
        servers=["instructions_enabled", "no_instructions"],
    )
    async def agent_multiple():
        async with fast.run() as agent_app:
            agent = agent_app.agent_multiple

            # Should have instructions from instructions_enabled
            assert "<mcp-server" in agent.instruction
            assert 'name="instructions_enabled"' in agent.instruction
            assert "calculation and text manipulation tools" in agent.instruction

            # Should NOT have entry for no_instructions server (it has no instructions)
            assert 'name="no_instructions"' not in agent.instruction

            # Count server blocks - should only be 1 (instructions_enabled)
            server_blocks = agent.instruction.count("<mcp-server")
            assert server_blocks == 1, f"Expected 1 server block, found {server_blocks}"

            return agent.instruction

    result = await agent_multiple()
    print("Agent instruction with multiple servers:")
    print(result)
    return result


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_no_template_no_substitution(fast_agent):
    """Test that agents without template variables are unaffected"""
    fast = fast_agent

    original_instruction = "I am a simple agent with no templates"

    @fast.agent(
        name="agent_no_template",
        instruction=original_instruction,
        model="passthrough",
        servers=["instructions_enabled"],
    )
    async def agent_no_template():
        async with fast.run() as agent_app:
            agent = agent_app.agent_no_template

            # Instruction should remain unchanged
            assert agent.instruction == original_instruction, \
                f"Instruction changed when it shouldn't. Expected '{original_instruction}', got '{agent.instruction}'"

            # Should not have any XML or instructions added
            assert "<mcp-server" not in agent.instruction
            assert "calculation and text manipulation tools" not in agent.instruction

            return agent.instruction

    result = await agent_no_template()
    print("Agent instruction without templates:")
    print(result)
    assert result == original_instruction


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_tools_list_in_instructions(fast_agent):
    """Test that the tools list is properly formatted in server instructions"""
    fast = fast_agent

    @fast.agent(
        name="agent_tools_check",
        instruction="{{serverInstructions}}",
        model="passthrough",
        servers=["instructions_enabled"],
    )
    async def agent_tools_check():
        async with fast.run() as agent_app:
            agent = agent_app.agent_tools_check

            # Extract the tools section
            tools_match = re.search(r'<tools>(.*?)</tools>', agent.instruction, re.DOTALL)
            assert tools_match, "Tools section not found"

            tools_content = tools_match.group(1)

            # Check that all expected tools are listed
            expected_tools = [
                "instructions_enabled-calculate_sum",
                "instructions_enabled-calculate_product",
                "instructions_enabled-calculate_divide",
                "instructions_enabled-text_reverse",
                "instructions_enabled-text_uppercase",
                "instructions_enabled-text_count"
            ]

            for tool in expected_tools:
                assert tool in tools_content, f"Tool {tool} not found in tools list"

            # Check tools are comma-separated
            assert ", " in tools_content, "Tools not properly comma-separated"

            return tools_content

    result = await agent_tools_check()
    print("Tools list from instructions:")
    print(result)
    return result


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
async def test_empty_server_instructions(fast_agent):
    """Test handling of server without instructions"""
    fast = fast_agent

    @fast.agent(
        name="agent_no_server_instructions",
        instruction="Prompt: {{serverInstructions}}",
        model="passthrough",
        servers=["no_instructions"],
    )
    async def agent_no_server_instructions():
        async with fast.run() as agent_app:
            agent = agent_app.agent_no_server_instructions

            # Template should be replaced
            assert "{{serverInstructions}}" not in agent.instruction

            # Should NOT have server block (no instructions means no XML block)
            assert 'name="no_instructions"' not in agent.instruction
            assert "<mcp-server" not in agent.instruction

            # The instruction should just be the base prompt
            assert agent.instruction.strip() == "Prompt:", \
                f"Expected 'Prompt:' but got '{agent.instruction.strip()}'"

            return agent.instruction

    result = await agent_no_server_instructions()
    print("Agent instruction with server without instructions:")
    print(result)
    return result