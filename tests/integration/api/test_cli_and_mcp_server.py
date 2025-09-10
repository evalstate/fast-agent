import os
import subprocess
from typing import TYPE_CHECKING

import pytest

from fast_agent.mcp.helpers.content_helpers import get_text

if TYPE_CHECKING:
    from mcp import GetPromptResult


@pytest.mark.integration
def test_agent_message_cli():
    """Test sending a message via command line to a FastAgent program."""
    # Get the path to the test_agent.py file (in the same directory as this test)
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_agent_path = os.path.join(test_dir, "integration_agent.py")

    # Test message
    test_message = "Hello from command line test"

    # Run the test agent with the --agent and --message flags
    result = subprocess.run(
        [
            "uv",
            "run",
            test_agent_path,
            "--agent",
            "test",
            "--message",
            test_message,
            #  "--quiet",  # Suppress progress display, etc. for cleaner output
        ],
        capture_output=True,
        text=True,
        cwd=test_dir,  # Run in the test directory to use its config
    )

    # Check that the command succeeded
    assert result.returncode == 0, f"Command failed with output: {result.stderr}"

    command_output = result.stdout
    # With the passthrough model, the output should contain the input message
    assert test_message in command_output, "Test message not found in agent response"
    # this is from show_user_output
    assert "▎▶ test" in command_output, "show chat messages included in output"


@pytest.mark.integration
def test_agent_message_prompt_file():
    """Test sending a message via command line to a FastAgent program."""
    # Get the path to the test_agent.py file (in the same directory as this test)
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_agent_path = os.path.join(test_dir, "integration_agent.py")

    # Run the test agent with the --agent and --message flags
    result = subprocess.run(
        ["uv", "run", test_agent_path, "--agent", "test", "--prompt-file", "prompt.txt"],
        capture_output=True,
        text=True,
        cwd=test_dir,  # Run in the test directory to use its config
    )

    # Check that the command succeeded
    assert result.returncode == 0, f"Command failed with output: {result.stderr}"

    command_output = result.stdout
    # With the passthrough model, the output should contain the input message
    assert "this is from the prompt file" in command_output, (
        "Test message not found in agent response"
    )
    # this is from show_user_output
    assert "▎▶ test" in command_output, "show chat messages included in output"


@pytest.mark.integration
def test_agent_message_cli_quiet_flag():
    """Test sending a message via command line to a FastAgent program."""
    # Get the path to the test_agent.py file (in the same directory as this test)
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_agent_path = os.path.join(test_dir, "integration_agent.py")

    # Test message
    test_message = "Hello from command line test"

    # Run the test agent with the --agent and --message flags
    result = subprocess.run(
        [
            "uv",
            "run",
            test_agent_path,
            "--agent",
            "test",
            "--message",
            test_message,
            "--quiet",  # Suppress progress display, etc. for cleaner output
        ],
        capture_output=True,
        text=True,
        cwd=test_dir,  # Run in the test directory to use its config
    )

    # Check that the command succeeded
    assert result.returncode == 0, f"Command failed with output: {result.stderr}"

    command_output = result.stdout
    # With the passthrough model, the output should contain the input message
    assert test_message in command_output, "Test message not found in agent response"
    # this is from show_user_output
    assert "[USER]" not in command_output, "show chat messages included in output"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_server_option_stdio(fast_agent):
    """Test that FastAgent supports --server flag with STDIO transport."""

    @fast_agent.agent(name="client", servers=["std_io"])
    async def agent_function():
        async with fast_agent.run() as agent:
            assert "connected" == await agent.send("connected")
            result = await agent.send('***CALL_TOOL test_send {"message": "stdio server test"}')
            assert "stdio server test" == result

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_server_option_stdio_and_prompt_history(fast_agent):
    """Test that FastAgent supports --server flag with STDIO transport."""

    @fast_agent.agent(name="client", servers=["std_io"])
    async def agent_function():
        async with fast_agent.run() as agent:
            assert "connected" == await agent.send("connected")
            result = await agent.send('***CALL_TOOL test_send {"message": "message one"}')
            assert "message one" == result
            result = await agent.send('***CALL_TOOL test_send {"message": "message two"}')
            assert "message two" == result

            history: GetPromptResult = await agent.get_prompt("test_history", server_name="std_io")
            assert len(history.messages) == 4
            assert "message one" == get_text(history.messages[1].content)

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_server_option_sse(fast_agent):
    """Test that FastAgent supports --server flag with SSE transport."""

    # Start the SSE server in a subprocess
    import asyncio
    import os
    import subprocess

    # Get the path to the test agent
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_agent_path = os.path.join(test_dir, "integration_agent.py")

    # Port must match what's in the fastagent.config.yaml
    port = 8723

    # Start the server process
    server_proc = subprocess.Popen(
        [
            "uv",
            "run",
            test_agent_path,
            "--server",
            "--transport",
            "sse",
            "--port",
            str(port),
            "--quiet",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=test_dir,
    )

    try:
        # Give the server a moment to start
        await asyncio.sleep(2)

        # Now connect to it via the configured MCP server
        @fast_agent.agent(name="client", servers=["sse"])
        async def agent_function():
            async with fast_agent.run() as agent:
                # Try connecting and sending a message
                assert "connected" == await agent.send("connected")
                result = await agent.send('***CALL_TOOL test_send {"message": "sse server test"}')
                assert "sse server test" == result

        await agent_function()

    finally:
        # Terminate the server process
        if server_proc.poll() is None:  # If still running
            server_proc.terminate()
            try:
                server_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                server_proc.kill()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_server_option_http(fast_agent):
    """Test that FastAgent supports --server flag with HTTP transport."""

    # Start the SSE server in a subprocess
    import asyncio
    import os
    import subprocess

    # Get the path to the test agent
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_agent_path = os.path.join(test_dir, "integration_agent.py")

    # Port must match what's in the fastagent.config.yaml
    port = 8724

    # Start the server process
    server_proc = subprocess.Popen(
        [
            "uv",
            "run",
            test_agent_path,
            "--server",
            "--transport",
            "http",
            "--port",
            str(port),
            "--quiet",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=test_dir,
    )

    try:
        # Give the server a moment to start
        await asyncio.sleep(2)

        # Now connect to it via the configured MCP server
        @fast_agent.agent(name="client", servers=["http"])
        async def agent_function():
            async with fast_agent.run() as agent:
                # Try connecting and sending a message
                assert "connected" == await agent.send("connected")
                result = await agent.send('***CALL_TOOL test_send {"message": "http server test"}')
                assert "http server test" == result

        await agent_function()

    finally:
        # Terminate the server process
        if server_proc.poll() is None:  # If still running
            server_proc.terminate()
            try:
                server_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                server_proc.kill()
