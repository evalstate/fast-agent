"""
Integration tests for ACP content block handling.

Tests the full flow of content blocks from ACP client -> server -> PromptMessageExtended.
"""

from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path

import pytest
from acp import InitializeRequest, NewSessionRequest, PromptRequest
from acp.helpers import text_block
from acp.schema import (
    BlobResourceContents,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    Implementation,
    StopReason,
    TextResourceContents,
)
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
    "fast-agent-acp-content-test",
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_image_content_processing() -> None:
    """Test that image content blocks are properly processed."""
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

        # Check that image is advertised as supported
        assert init_response.agentCapabilities is not None
        # Handle both "prompts" and "promptCapabilities" field names
        prompt_caps = getattr(
            init_response.agentCapabilities, "prompts", None
        ) or getattr(init_response.agentCapabilities, "promptCapabilities", None)
        assert prompt_caps is not None
        # Check if image capability is enabled
        assert getattr(prompt_caps, "image", False) is True

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Create a fake image (base64 encoded)
        fake_image_data = base64.b64encode(b"fake-image-data").decode("utf-8")

        # Send prompt with text and image
        prompt_blocks = [
            text_block("Analyze this image:"),
            ImageContentBlock(
                type="image",
                data=fake_image_data,
                mimeType="image/png",
            ),
        ]

        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=prompt_blocks)
        )

        # Should complete successfully
        assert prompt_response.stopReason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client)

        # Verify we got a response (passthrough model will echo something back)
        assert len(client.notifications) > 0
        last_update = client.notifications[-1]
        assert last_update.sessionId == session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_embedded_text_resource_processing() -> None:
    """Test that embedded text resource content blocks are properly processed."""
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

        # Check that resource is advertised as supported
        assert init_response.agentCapabilities is not None
        # Handle both "prompts" and "promptCapabilities" field names
        prompt_caps = getattr(
            init_response.agentCapabilities, "prompts", None
        ) or getattr(init_response.agentCapabilities, "promptCapabilities", None)
        assert prompt_caps is not None
        # Check if embeddedContext capability is enabled
        assert getattr(prompt_caps, "embeddedContext", False) is True

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Send prompt with text resource
        prompt_blocks = [
            text_block("Review this code:"),
            EmbeddedResourceContentBlock(
                type="resource",
                resource=TextResourceContents(
                    uri="file:///example.py",
                    mimeType="text/x-python",
                    text="def hello():\n    return 'Hello, world!'",
                ),
            ),
        ]

        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=prompt_blocks)
        )

        # Should complete successfully
        assert prompt_response.stopReason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client)

        # Verify we got a response
        assert len(client.notifications) > 0
        last_update = client.notifications[-1]
        assert last_update.sessionId == session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_embedded_blob_resource_processing() -> None:
    """Test that embedded blob resource content blocks are properly processed."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize and create session
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId
        assert session_id

        # Create fake binary data
        fake_blob_data = base64.b64encode(b"fake-binary-document-data").decode("utf-8")

        # Send prompt with blob resource
        prompt_blocks = [
            text_block("Summarize this document:"),
            EmbeddedResourceContentBlock(
                type="resource",
                resource=BlobResourceContents(
                    uri="file:///document.pdf",
                    mimeType="application/pdf",
                    blob=fake_blob_data,
                ),
            ),
        ]

        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=prompt_blocks)
        )

        # Should complete successfully
        assert prompt_response.stopReason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client)

        # Verify we got a response
        assert len(client.notifications) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_mixed_content_blocks() -> None:
    """Test that mixed content blocks (text, image, resource) work together."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize and create session
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Create mixed content
        image_data = base64.b64encode(b"fake-screenshot").decode("utf-8")

        prompt_blocks = [
            text_block("I need help with this code:"),
            EmbeddedResourceContentBlock(
                type="resource",
                resource=TextResourceContents(
                    uri="file:///app.py",
                    mimeType="text/x-python",
                    text="import sys\nprint(sys.version)",
                ),
            ),
            text_block("And here's a screenshot of the error:"),
            ImageContentBlock(
                type="image",
                data=image_data,
                mimeType="image/png",
            ),
            text_block("What's wrong?"),
        ]

        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=prompt_blocks)
        )

        # Should complete successfully
        assert prompt_response.stopReason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client)

        # Verify we got a response
        assert len(client.notifications) > 0
        last_update = client.notifications[-1]
        assert last_update.sessionId == session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_resource_only_prompt() -> None:
    """Test that a prompt with only a resource (no text blocks) works correctly."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize and create session
        init_request = InitializeRequest(
            protocolVersion=1,
            clientCapabilities=ClientCapabilities(
                fs={"readTextFile": True, "writeTextFile": True},
                terminal=False,
            ),
            clientInfo=Implementation(name="pytest-client", version="0.0.1"),
        )
        await connection.initialize(init_request)

        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Send prompt with only a resource (no text blocks)
        prompt_blocks = [
            EmbeddedResourceContentBlock(
                type="resource",
                resource=TextResourceContents(
                    uri="file:///C:/Users/shaun/AppData/Roaming/Zed/settings.json",
                    text="hello, world!",
                ),
            ),
        ]

        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=prompt_blocks)
        )

        # Should complete successfully
        assert prompt_response.stopReason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client)

        # Verify we got a response
        assert len(client.notifications) > 0
        last_update = client.notifications[-1]
        assert last_update.sessionId == session_id


async def _wait_for_notifications(client: TestClient, timeout: float = 2.0) -> None:
    """Wait for the ACP client to receive at least one sessionUpdate."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if client.notifications:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("Expected streamed session updates")
