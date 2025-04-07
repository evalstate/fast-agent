import os
import subprocess

import pytest


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
            "python",
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

    # With the passthrough model, the output should contain the input message
    assert test_message in result.stdout, "Test message not found in agent response"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_server_option_stdio(fast_agent):
    """Test that FastAgent supports --server flag with STDIO transport."""

    @fast_agent.agent(name="client", servers=["std_io"])
    async def agent_function():
        async with fast_agent.run() as agent:
            assert "connected" == await agent.send("connected")
            result = await agent.send('***CALL_TOOL test.send {"message": "stdio server test"}')
            assert "stdio server test" == result

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_server_option_sse(fast_agent):
    """Test that FastAgent supports --server flag with SSE transport."""

    # Start the SSE server in a subprocess
    import asyncio
    import os
    import subprocess
    import time

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
        await asyncio.sleep(3)

        # Now connect to it via the configured MCP server
        @fast_agent.agent(name="client", servers=["sse"])
        async def agent_function():
            async with fast_agent.run() as agent:
                # Try connecting and sending a message
                assert "connected" == await agent.send("connected")
                result = await agent.send('***CALL_TOOL test.send {"message": "sse server test"}')
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
