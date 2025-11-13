"""Unit tests for filesystem runtime integration with McpAgent."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp import CallToolRequest
from mcp.types import CallToolRequestParams, CallToolResult, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.core.prompt import Prompt
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.types.llm_stop_reason import LlmStopReason


class MockFilesystemRuntime:
    """Mock filesystem runtime for testing."""

    def __init__(self):
        self.read_tool = Tool(
            name="read_text_file",
            description="Read a text file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        )
        self.write_tool = Tool(
            name="write_text_file",
            description="Write a text file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        )
        self.tools = [self.read_tool, self.write_tool]

    async def read_text_file(self, arguments):
        return CallToolResult(
            content=[text_content(f"Content of {arguments['path']}")],
            isError=False,
        )

    async def write_text_file(self, arguments):
        return CallToolResult(
            content=[text_content(f"Wrote to {arguments['path']}")],
            isError=False,
        )

    def metadata(self):
        return {
            "type": "mock_filesystem",
            "tools": ["read_text_file", "write_text_file"],
        }


@pytest.mark.asyncio
async def test_filesystem_runtime_tools_listed():
    """Test that filesystem runtime tools are included in list_tools()."""
    config = AgentConfig(name="test-agent", servers=[])

    async with McpAgent(config=config, connection_persistence=False) as agent:
        # Inject mock filesystem runtime
        fs_runtime = MockFilesystemRuntime()
        agent.set_filesystem_runtime(fs_runtime)

        # List tools
        result = await agent.list_tools()

        # Verify filesystem tools are included
        tool_names = [tool.name for tool in result.tools]
        assert "read_text_file" in tool_names
        assert "write_text_file" in tool_names


@pytest.mark.asyncio
async def test_filesystem_runtime_tool_call():
    """Test that filesystem runtime tools can be called via call_tool()."""
    config = AgentConfig(name="test-agent", servers=[])

    async with McpAgent(config=config, connection_persistence=False) as agent:
        # Inject mock filesystem runtime
        fs_runtime = MockFilesystemRuntime()
        agent.set_filesystem_runtime(fs_runtime)

        # Call read_text_file
        result = await agent.call_tool(
            "read_text_file",
            {"path": "/test/file.txt"}
        )

        assert result.isError is False
        assert len(result.content) > 0
        assert "Content of /test/file.txt" in result.content[0].text

        # Call write_text_file
        result = await agent.call_tool(
            "write_text_file",
            {"path": "/test/output.txt", "content": "test"}
        )

        assert result.isError is False
        assert "Wrote to /test/output.txt" in result.content[0].text


@pytest.mark.asyncio
async def test_filesystem_runtime_tools_available_in_run_tools():
    """Test that filesystem tools are recognized as available in run_tools()."""
    config = AgentConfig(name="test-agent", servers=[])

    async with McpAgent(config=config, connection_persistence=False) as agent:
        # Inject mock filesystem runtime
        fs_runtime = MockFilesystemRuntime()
        agent.set_filesystem_runtime(fs_runtime)

        # Create a prompt message with tool calls
        tool_calls = {
            "call_1": CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name="read_text_file",
                    arguments={"path": "/test/file.txt"}
                )
            ),
            "call_2": CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name="write_text_file",
                    arguments={"path": "/test/output.txt", "content": "test"}
                )
            ),
        }

        tool_call_request = Prompt.assistant(
            "Using filesystem tools",
            stop_reason=LlmStopReason.TOOL_USE,
            tool_calls=tool_calls,
        )

        # Run tools - this should NOT produce "Tool is not available" errors
        result = await agent.run_tools(tool_call_request)

        # Verify tools were executed successfully
        assert result.role == "user"
        # Check that we don't have error channel content
        if result.channels:
            from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
            assert FAST_AGENT_ERROR_CHANNEL not in result.channels


@pytest.mark.asyncio
async def test_external_runtime_tools_available_in_run_tools():
    """Test that external runtime tools (like terminal) are recognized as available."""
    config = AgentConfig(name="test-agent", servers=[])

    async with McpAgent(config=config, connection_persistence=False) as agent:
        # Create mock external runtime (like ACPTerminalRuntime)
        external_runtime = MagicMock()
        external_runtime.tool = Tool(
            name="execute",
            description="Execute a command",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        )
        external_runtime.execute = AsyncMock(return_value=CallToolResult(
            content=[text_content("Command executed")],
            isError=False,
        ))
        external_runtime.metadata = MagicMock(return_value={"type": "mock_external"})

        agent.set_external_runtime(external_runtime)

        # Create a prompt message with tool call
        tool_calls = {
            "call_1": CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name="execute",
                    arguments={"command": "echo test"}
                )
            ),
        }

        tool_call_request = Prompt.assistant(
            "Using external runtime tool",
            stop_reason=LlmStopReason.TOOL_USE,
            tool_calls=tool_calls,
        )

        # Run tools - this should NOT produce "Tool is not available" errors
        result = await agent.run_tools(tool_call_request)

        # Verify tool was executed successfully
        assert result.role == "user"
        # Check that we don't have error channel content
        if result.channels:
            from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
            assert FAST_AGENT_ERROR_CHANNEL not in result.channels

        # Verify the mock was called
        external_runtime.execute.assert_called_once()
