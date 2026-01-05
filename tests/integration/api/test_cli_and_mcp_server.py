import asyncio
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import httpx
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
    """Test STDIO transport works end-to-end."""

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
    """Test STDIO transport preserves prompt history."""

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
async def test_agent_transport_option_sse(fast_agent, mcp_test_ports, wait_for_port):
    """Test that FastAgent enables server mode when --transport is provided (SSE)."""

    # Start the SSE server in a subprocess
    import os
    import subprocess

    # Get the path to the test agent
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_agent_path = os.path.join(test_dir, "integration_agent.py")

    # Port must match what's in the fastagent.config.yaml
    port = mcp_test_ports["sse"]

    # Start the server process
    server_proc = subprocess.Popen(
        [
            "uv",
            "run",
            test_agent_path,
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
        await wait_for_port("127.0.0.1", port, process=server_proc)

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
async def test_serve_request_scope_disables_session_header(mcp_test_ports, wait_for_port):
    """Request-scoped instances should not advertise an MCP session id."""

    import os
    import subprocess

    test_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(test_dir, "fastagent.config.yaml")

    port = mcp_test_ports["request_http"]

    server_proc = subprocess.Popen(
        [
            "uv",
            "run",
            "-m",
            "fast_agent.cli",
            "serve",
            "--config-path",
            config_path,
            "--transport",
            "http",
            "--port",
            str(port),
            "--instance-scope",
            "request",
            #            "--quiet",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=test_dir,
    )

    try:
        await wait_for_port("127.0.0.1", port, process=server_proc, timeout=10.0)

        async with httpx.AsyncClient(timeout=5.0) as client:
            init_payload = {
                "jsonrpc": "2.0",
                "id": "1",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-suite", "version": "0.0.0"},
                },
            }
            async with client.stream(
                "POST",
                f"http://127.0.0.1:{port}/mcp",
                headers={
                    "content-type": "application/json",
                    "accept": "application/json, text/event-stream",
                },
                json=init_payload,
            ) as response:
                assert response.status_code == 200
                assert "mcp-session-id" not in response.headers
    finally:
        if server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                server_proc.kill()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_server_option_http(fast_agent, mcp_test_ports, wait_for_port):
    """Test that FastAgent still accepts the legacy --server flag with HTTP transport."""

    # Start the SSE server in a subprocess
    import os
    import subprocess

    # Get the path to the test agent
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_agent_path = os.path.join(test_dir, "integration_agent.py")

    # Port must match what's in the fastagent.config.yaml
    port = mcp_test_ports["http"]

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
        await wait_for_port("127.0.0.1", port, process=server_proc)

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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_server_option_http_with_watch(mcp_test_ports, wait_for_port, tmp_path):
    """Server mode should start cleanly with --watch enabled."""

    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    card_path = agents_dir / "watcher.md"
    card_path.write_text(
        "---\n"
        "type: agent\n"
        "name: watcher\n"
        "---\n"
        "Echo test.\n",
        encoding="utf-8",
    )

    port = mcp_test_ports["http"]

    server_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fast_agent.cli",
            "serve",
            "--config-path",
            str(config_path),
            "--transport",
            "http",
            "--port",
            str(port),
            "--model",
            "passthrough",
            "--name",
            "fast-agent-watch-test",
            "--card",
            str(agents_dir),
            "--watch",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path,
    )

    try:
        await wait_for_port("127.0.0.1", port, process=server_proc)
        card_path.write_text(
            "---\n"
            "type: agent\n"
            "name: watcher\n"
            "---\n"
            "Echo test updated.\n",
            encoding="utf-8",
        )
        await asyncio.sleep(0.25)
        assert server_proc.poll() is None
    finally:
        if server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                server_proc.kill()
