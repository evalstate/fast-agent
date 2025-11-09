from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from acp import InitializeRequest, NewSessionRequest, PromptRequest
from acp.helpers import text_block
from acp.schema import ClientCapabilities, Implementation, StopReason
from acp.stdio import spawn_agent_process

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

CONFIG_PATH = TEST_DIR / "fastagent.config.yaml"
END_TURN: StopReason = "end_turn"
FAST_AGENT_CMD = (
    sys.executable,
    "-m",
    "fast_agent.cli",
    "serve",
    "--config-path",
    str(CONFIG_PATH),
    "--transport",
    "acp",
    "--model",
    "passthrough",
    "--name",
    "fast-agent-acp-test",
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_initialize_and_prompt_roundtrip() -> None:
    """Ensure the ACP transport initializes, creates a session, and echoes prompts."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-client", version="0.0.1"),
        )
        init_response = await connection.initialize(init_request)

        assert init_response.protocolVersion == 1
        assert init_response.agentCapabilities is not None
        assert init_response.agentInfo.name == "fast-agent-acp-test"
        # AgentCapabilities schema changed upstream; ensure we advertised prompt support.
        prompt_caps = getattr(init_response.agentCapabilities, "prompts", None) or getattr(
            init_response.agentCapabilities, "promptCapabilities", None
        )
        assert prompt_caps is not None

        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        prompt_text = "echo from ACP integration test"
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block(prompt_text)])
        )
        assert prompt_response.stopReason == END_TURN

        await _wait_for_notifications(client)
        last_update = client.notifications[-1]
        assert last_update.sessionId == session_id
        assert last_update.update.sessionUpdate == "agent_message_chunk"
        # Passthrough model mirrors user input, so the agent content should match the prompt.
        assert getattr(last_update.update.content, "text", None) == prompt_text


async def _wait_for_notifications(client: TestClient, timeout: float = 2.0) -> None:
    """Wait for the ACP client to receive at least one sessionUpdate."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if client.notifications:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("Expected streamed session updates")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_overlapping_prompts_are_refused() -> None:
    """
    Test that overlapping prompt requests for the same session are refused.

    Per ACP protocol, only one prompt can be active per session at a time.
    If a second prompt arrives while one is in progress, it should be immediately
    refused with stopReason="refusal".
    """
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-client", version="0.0.1"),
        )
        init_response = await connection.initialize(init_request)
        assert init_response.protocolVersion == 1

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Send two prompts truly concurrently (no sleep between them)
        # This ensures they both arrive before either completes
        prompt1_task = asyncio.create_task(
            connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block("first prompt")])
            )
        )

        # Send immediately without waiting - ensures actual overlap
        prompt2_task = asyncio.create_task(
            connection.prompt(
                PromptRequest(sessionId=session_id, prompt=[text_block("overlapping prompt")])
            )
        )

        # Wait for both to complete
        prompt1_response, prompt2_response = await asyncio.gather(prompt1_task, prompt2_task)

        # One should succeed, one should be refused
        # (We don't know which one arrives first due to async scheduling)
        responses = [prompt1_response.stopReason, prompt2_response.stopReason]
        assert "end_turn" in responses, "One prompt should succeed"
        assert "refusal" in responses, "One prompt should be refused"

        # After both complete, a new prompt should succeed
        prompt3_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_block("third prompt")])
        )
        assert prompt3_response.stopReason == END_TURN
