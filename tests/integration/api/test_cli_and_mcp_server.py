import asyncio
import os
import subprocess
import sys

import httpx
import mcp_types as types
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from fast_agent.mcp.helpers.content_helpers import get_text


@pytest.mark.integration
@pytest.mark.parametrize("agent_args", [("--agent", "test"), ()], ids=["named", "default"])
@pytest.mark.parametrize(
    ("input_args", "expected"),
    [
        (("--message", "Hello from command line test"), "Hello from command line test"),
        (("--prompt-file", "prompt.txt"), "this is from the prompt file"),
    ],
    ids=["message", "prompt-file"],
)
def test_agent_message_cli(
    agent_args: tuple[str, ...], input_args: tuple[str, ...], expected: str
) -> None:
    """CLI arguments reach the selected passthrough agent in a real child process."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        [sys.executable, "integration_agent.py", *agent_args, *input_args],
        capture_output=True,
        check=False,
        text=True,
        cwd=test_dir,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    assert "▎▶ test" in result.stdout


@pytest.mark.integration
def test_agent_message_cli_quiet_flag() -> None:
    test_dir = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        [sys.executable, "integration_agent.py", "--message", "quiet response", "--quiet"],
        capture_output=True,
        check=False,
        text=True,
        cwd=test_dir,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "quiet response" in result.stdout
    assert "▎▶ test" not in result.stdout


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_server_stdio_roundtrips_and_lists_prompts(fast_agent):
    """Round-trip multiple messages and list prompts through a real stdio server."""

    @fast_agent.agent(name="client", servers=["std_io"])
    async def agent_function():
        async with fast_agent.run() as agent:
            assert "connected" == await agent.send("connected")
            result = await agent.send('***CALL_TOOL test {"message": "message one"}')
            assert "message one" == result
            result = await agent.send('***CALL_TOOL test {"message": "message two"}')
            assert "message two" == result

            prompts = await agent.list_prompts("std_io")
            assert prompts == {"std_io": []}

    await agent_function()


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

        async with streamable_http_client(f"http://127.0.0.1:{port}/mcp") as (
            read_stream,
            write_stream,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                init_result = await session.initialize()
                assert init_result.capabilities.prompts is not None
                prompt_result = await session.list_prompts()
                assert prompt_result.prompts == []
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
    """Test that --transport starts FastAgent in HTTP server mode."""

    # Start the HTTP server in a subprocess
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
                result = await agent.send('***CALL_TOOL test {"message": "http server test"}')
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
async def test_agent_server_option_http_rejects_watch(tmp_path):
    """Managed MCP serve rejects --watch because the tool surface is fixed at startup."""

    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    card_path = agents_dir / "watcher.md"
    card_path.write_text(
        "---\ntype: agent\nname: watcher\n---\nEcho test.\n",
        encoding="utf-8",
    )

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
            "0",
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

    stdout, stderr = server_proc.communicate(timeout=5)

    assert server_proc.returncode != 0
    assert "is not supported for MCP serving" in stderr
    assert stdout == ""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_server_emits_mcp_progress_notifications(
    fast_agent, mcp_test_ports, wait_for_port
):
    """Test that MCP progress notifications are emitted during tool execution."""

    import os
    import subprocess

    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_agent_path = os.path.join(test_dir, "integration_agent.py")

    port = mcp_test_ports["http"]

    server_proc = subprocess.Popen(
        [
            "uv",
            "run",
            test_agent_path,
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

        progress_events: list[tuple[float, float | None, str | None]] = []

        async def on_progress(progress: float, total: float | None, message: str | None) -> None:
            progress_events.append((progress, total, message))

        async with streamable_http_client(f"http://127.0.0.1:{port}/mcp") as (
            read_stream,
            write_stream,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                params = types.CallToolRequestParams(
                    name="test", arguments={"message": "progress check"}
                )
                request = types.CallToolRequest(method="tools/call", params=params)
                result = await session.send_request(
                    request,
                    types.CallToolResult,
                    progress_callback=on_progress,
                )

                assert result.content
                assert "progress check" in (get_text(result.content[0]) or "")

        for _ in range(20):
            if progress_events:
                break
            await asyncio.sleep(0.1)

        assert progress_events
        assert any(message and "step" in message for _, _, message in progress_events), (
            f"Unexpected progress messages: {progress_events}"
        )
    finally:
        if server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                server_proc.kill()
