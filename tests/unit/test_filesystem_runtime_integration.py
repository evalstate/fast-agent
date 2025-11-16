"""Unit tests for filesystem runtime integration with McpAgent."""

import tempfile
from pathlib import Path

import pytest
from mcp import CallToolRequest
from mcp.types import CallToolRequestParams, CallToolResult, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.core.prompt import Prompt
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.types.llm_stop_reason import LlmStopReason


class NullDisplay:
    """Simple no-op display for testing in headless environments."""

    def show_tool_call(self, *args, **kwargs):
        """No-op tool call display."""
        pass

    def show_tool_result(self, *args, **kwargs):
        """No-op tool result display."""
        pass

    def __getattr__(self, name):
        """Return no-op function for any other method."""
        return lambda *args, **kwargs: None


class SimpleFilesystemRuntime:
    """Simple filesystem runtime for testing that actually reads/writes files."""

    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
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

    async def read_text_file(self, arguments, tool_use_id=None):
        try:
            path = arguments["path"]
            file_path = Path(path)
            content = file_path.read_text()
            return CallToolResult(
                content=[text_content(content)],
                isError=False,
            )
        except Exception as e:
            return CallToolResult(
                content=[text_content(f"Error reading file: {e}")],
                isError=True,
            )

    async def write_text_file(self, arguments, tool_use_id=None):
        try:
            path = arguments["path"]
            content = arguments["content"]
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            return CallToolResult(
                content=[text_content(f"Successfully wrote {len(content)} characters to {path}")],
                isError=False,
            )
        except Exception as e:
            return CallToolResult(
                content=[text_content(f"Error writing file: {e}")],
                isError=True,
            )

    def metadata(self):
        return {
            "type": "simple_filesystem",
            "tools": ["read_text_file", "write_text_file"],
        }


@pytest.mark.asyncio
async def test_filesystem_runtime_tools_listed():
    """Test that filesystem runtime tools are included in list_tools()."""
    config = AgentConfig(name="test-agent", servers=[])

    with tempfile.TemporaryDirectory() as temp_dir:
        async with McpAgent(config=config, connection_persistence=False) as agent:
            # Inject real filesystem runtime
            fs_runtime = SimpleFilesystemRuntime(Path(temp_dir))
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

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a test file
        test_file = temp_path / "test_file.txt"
        test_content = "Hello from test file!"
        test_file.write_text(test_content)

        async with McpAgent(config=config, connection_persistence=False) as agent:
            # Inject real filesystem runtime
            fs_runtime = SimpleFilesystemRuntime(temp_path)
            agent.set_filesystem_runtime(fs_runtime)

            # Call read_text_file
            result = await agent.call_tool(
                "read_text_file",
                {"path": str(test_file)}
            )

            assert result.isError is False
            assert len(result.content) > 0
            assert test_content in result.content[0].text

            # Call write_text_file
            output_file = temp_path / "output.txt"
            write_content = "Written by test"
            result = await agent.call_tool(
                "write_text_file",
                {"path": str(output_file), "content": write_content}
            )

            assert result.isError is False
            assert "Successfully wrote" in result.content[0].text

            # Verify the file was actually written
            assert output_file.exists()
            assert output_file.read_text() == write_content


@pytest.mark.asyncio
async def test_filesystem_runtime_tools_available_in_run_tools():
    """Test that filesystem tools are recognized as available in run_tools()."""
    config = AgentConfig(name="test-agent", servers=[])

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a test file
        test_file = temp_path / "test_file.txt"
        test_content = "Content for run_tools test"
        test_file.write_text(test_content)

        async with McpAgent(config=config, connection_persistence=False) as agent:
            # Inject real filesystem runtime
            fs_runtime = SimpleFilesystemRuntime(temp_path)
            agent.set_filesystem_runtime(fs_runtime)

            # Use NullDisplay to avoid Rich console issues in CI/headless environments
            # This is not mocking the test behavior, just preventing display output
            agent.display = NullDisplay()

            # Create a prompt message with tool calls
            output_file = temp_path / "output.txt"
            tool_calls = {
                "call_1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="read_text_file",
                        arguments={"path": str(test_file)}
                    )
                ),
                "call_2": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="write_text_file",
                        arguments={"path": str(output_file), "content": "test output"}
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

            # Verify the file was actually written
            assert output_file.exists()
            assert output_file.read_text() == "test output"
