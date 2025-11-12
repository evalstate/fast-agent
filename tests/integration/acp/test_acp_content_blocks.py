"""
Integration tests for ACP content block processing.

Tests that ImageContent, EmbeddedContent, and other content types are properly
converted from ACP format to MCP format and passed to the agent.
"""

from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path

import pytest
from acp import InitializeRequest, NewSessionRequest, PromptRequest
from acp.schema import (
    AudioContentBlock,
    BlobResourceContents,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    Implementation,
    StopReason,
    TextContentBlock,
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
async def test_acp_text_content_block() -> None:
    """Test that text content blocks are properly converted and processed."""
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

        # Send text content
        text_content = TextContentBlock(type="text", text="Hello from ACP")
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=[text_content])
        )

        assert prompt_response.stopReason == END_TURN

        await _wait_for_notifications(client)
        last_update = client.notifications[-1]
        assert last_update.sessionId == session_id
        # Passthrough model echoes the input
        assert getattr(last_update.update.content, "text", None) == "Hello from ACP"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_image_content_block() -> None:
    """Test that image content blocks are properly converted and processed."""
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

        # Verify image capability is advertised
        prompt_caps = getattr(init_response.agentCapabilities, "prompts", None)
        assert prompt_caps is not None
        assert "image" in prompt_caps.supportedTypes

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Create a simple base64-encoded image
        image_data = base64.b64encode(b"fake image data").decode("utf-8")

        # Send multipart prompt with text + image
        prompt_content = [
            TextContentBlock(type="text", text="Describe this image"),
            ImageContentBlock(type="image", data=image_data, mimeType="image/png"),
        ]
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=prompt_content)
        )

        assert prompt_response.stopReason == END_TURN
        await _wait_for_notifications(client)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_embedded_resource_content_block() -> None:
    """Test that embedded resource content blocks are properly converted."""
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

        # Verify embeddedContext capability is advertised
        prompt_caps = getattr(init_response.agentCapabilities, "prompts", None)
        assert prompt_caps is not None
        assert "embeddedContext" in prompt_caps.supportedTypes

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Create an embedded text resource
        text_resource = TextResourceContents(
            uri="file:///test/context.txt",
            mimeType="text/plain",
            text="This is important context from a file",
        )

        prompt_content = [
            TextContentBlock(type="text", text="Analyze this context"),
            EmbeddedResourceContentBlock(type="resource", resource=text_resource),
        ]

        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=prompt_content)
        )

        assert prompt_response.stopReason == END_TURN
        await _wait_for_notifications(client)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_audio_content_block() -> None:
    """Test that audio content blocks are properly converted and processed."""
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

        # Verify audio capability is advertised
        prompt_caps = getattr(init_response.agentCapabilities, "prompts", None)
        assert prompt_caps is not None
        assert "audio" in prompt_caps.supportedTypes

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Create a simple base64-encoded audio
        audio_data = base64.b64encode(b"fake audio data").decode("utf-8")

        # Send multipart prompt with text + audio
        prompt_content = [
            TextContentBlock(type="text", text="Transcribe this audio"),
            AudioContentBlock(type="audio", data=audio_data, mimeType="audio/wav"),
        ]
        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=prompt_content)
        )

        assert prompt_response.stopReason == END_TURN
        await _wait_for_notifications(client)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_multimodal_prompt() -> None:
    """Test complex multimodal prompt with multiple content types."""
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
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Create complex multimodal prompt
        image_data = base64.b64encode(b"image bytes").decode("utf-8")
        audio_data = base64.b64encode(b"audio bytes").decode("utf-8")
        blob_data = base64.b64encode(b"binary data").decode("utf-8")

        text_resource = TextResourceContents(
            uri="file:///context.txt",
            mimeType="text/plain",
            text="Context information",
        )
        blob_resource = BlobResourceContents(
            uri="file:///data.bin",
            mimeType="application/octet-stream",
            blob=blob_data,
        )

        prompt_content = [
            TextContentBlock(type="text", text="Analyze all this data:"),
            ImageContentBlock(type="image", data=image_data, mimeType="image/png"),
            AudioContentBlock(type="audio", data=audio_data, mimeType="audio/wav"),
            EmbeddedResourceContentBlock(type="resource", resource=text_resource),
            EmbeddedResourceContentBlock(type="resource", resource=blob_resource),
            TextContentBlock(type="text", text="What are your conclusions?"),
        ]

        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=prompt_content)
        )

        assert prompt_response.stopReason == END_TURN
        await _wait_for_notifications(client)

        # Verify we got a response
        assert len(client.notifications) > 0
        last_update = client.notifications[-1]
        assert last_update.sessionId == session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_content_with_annotations() -> None:
    """Test that content block annotations are preserved through conversion."""
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
        await connection.initialize(init_request)

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Create content with annotations
        annotations = {"priority": "high", "source": "user", "metadata": {"key": "value"}}

        prompt_content = [
            TextContentBlock(
                type="text",
                text="Important message",
                annotations=annotations,
            ),
        ]

        prompt_response = await connection.prompt(
            PromptRequest(sessionId=session_id, prompt=prompt_content)
        )

        assert prompt_response.stopReason == END_TURN
        await _wait_for_notifications(client)
