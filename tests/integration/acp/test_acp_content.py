"""
Integration and unit tests for ACP content conversion.

Tests both the content converter functions and end-to-end conversion through the server.
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
    AudioContentBlock,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    Implementation,
    ResourceContentBlock,
    StopReason,
    TextContentBlock,
)
from acp.schema import (
    BlobResourceContents as ACPBlobResourceContents,
)
from acp.schema import (
    TextResourceContents as ACPTextResourceContents,
)
from acp.stdio import spawn_agent_process
from mcp.types import (
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)

from fast_agent.acp.content_converter import (
    convert_acp_content_list_to_mcp,
    convert_acp_content_to_mcp,
)
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

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


# ============================================================================
# Unit-style tests for content conversion functions
# ============================================================================


@pytest.mark.integration
class TestACPContentConversion:
    """Test the ACP to MCP content conversion functions."""

    def test_text_content_conversion(self):
        """Test converting TextContentBlock to TextContent."""
        acp_text = TextContentBlock(type="text", text="Hello, world!")
        mcp_content = convert_acp_content_to_mcp(acp_text)

        assert isinstance(mcp_content, TextContent)
        assert mcp_content.type == "text"
        assert mcp_content.text == "Hello, world!"

    def test_image_content_conversion(self):
        """Test converting ImageContentBlock to ImageContent."""
        image_data = base64.b64encode(b"fake_image").decode("ascii")
        acp_image = ImageContentBlock(
            type="image",
            data=image_data,
            mimeType="image/png",
        )
        mcp_content = convert_acp_content_to_mcp(acp_image)

        assert isinstance(mcp_content, ImageContent)
        assert mcp_content.type == "image"
        assert mcp_content.data == image_data
        assert mcp_content.mimeType == "image/png"

    def test_embedded_text_resource_conversion(self):
        """Test converting embedded text resource."""
        acp_resource = EmbeddedResourceContentBlock(
            type="resource",
            resource=ACPTextResourceContents(
                uri="file:///test.txt",
                text="File contents",
                mimeType="text/plain",
            ),
        )
        mcp_content = convert_acp_content_to_mcp(acp_resource)

        assert isinstance(mcp_content, EmbeddedResource)
        assert isinstance(mcp_content.resource, TextResourceContents)
        assert mcp_content.resource.text == "File contents"

    def test_embedded_blob_resource_conversion(self):
        """Test converting embedded blob resource."""
        blob_data = base64.b64encode(b"binary").decode("ascii")
        acp_resource = EmbeddedResourceContentBlock(
            type="resource",
            resource=ACPBlobResourceContents(
                uri="file:///test.bin",
                blob=blob_data,
                mimeType="application/octet-stream",
            ),
        )
        mcp_content = convert_acp_content_to_mcp(acp_resource)

        assert isinstance(mcp_content, EmbeddedResource)
        assert isinstance(mcp_content.resource, BlobResourceContents)
        assert mcp_content.resource.blob == blob_data

    def test_audio_content_conversion(self):
        """Test converting AudioContentBlock to EmbeddedResource."""
        audio_data = base64.b64encode(b"audio").decode("ascii")
        acp_audio = AudioContentBlock(
            type="audio",
            data=audio_data,
            mimeType="audio/mpeg",
        )
        mcp_content = convert_acp_content_to_mcp(acp_audio)

        assert isinstance(mcp_content, EmbeddedResource)
        assert isinstance(mcp_content.resource, BlobResourceContents)
        assert mcp_content.resource.blob == audio_data

    def test_resource_link_conversion(self):
        """Test converting ResourceContentBlock to TextContent."""
        acp_resource = ResourceContentBlock(
            type="resource_link",
            uri="https://example.com/file",
            name="Example",
            description="Test resource",
            mimeType="text/plain",
            size=100,
        )
        mcp_content = convert_acp_content_to_mcp(acp_resource)

        assert isinstance(mcp_content, TextContent)
        assert "Example" in mcp_content.text
        assert "https://example.com/file" in mcp_content.text

    def test_mixed_content_list_conversion(self):
        """Test converting a list with multiple content types."""
        acp_content_list = [
            TextContentBlock(type="text", text="Hello"),
            ImageContentBlock(
                type="image",
                data=base64.b64encode(b"img").decode("ascii"),
                mimeType="image/png",
            ),
        ]
        mcp_content_list = convert_acp_content_list_to_mcp(acp_content_list)

        assert len(mcp_content_list) == 2
        assert isinstance(mcp_content_list[0], TextContent)
        assert isinstance(mcp_content_list[1], ImageContent)

    def test_prompt_message_integration(self):
        """Test creating PromptMessageExtended from converted content."""
        acp_content_list = [
            TextContentBlock(type="text", text="First"),
            ImageContentBlock(
                type="image",
                data=base64.b64encode(b"img").decode("ascii"),
                mimeType="image/png",
            ),
            TextContentBlock(type="text", text="Second"),
        ]
        mcp_content_list = convert_acp_content_list_to_mcp(acp_content_list)
        prompt_message = PromptMessageExtended(role="user", content=mcp_content_list)

        assert prompt_message.role == "user"
        assert len(prompt_message.content) == 3
        all_text = prompt_message.all_text()
        assert "First" in all_text
        assert "Second" in all_text


# ============================================================================
# End-to-end integration tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_image_content_conversion() -> None:
    """Test that ImageContent from ACP is properly converted."""
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
        assert init_response.agentCapabilities is not None

        # Verify we advertise image support
        prompt_caps = getattr(init_response.agentCapabilities, "prompts", None)
        assert prompt_caps is not None
        supported_types = getattr(prompt_caps, "supportedTypes", [])
        assert "image" in supported_types

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Create a fake image
        fake_image_data = base64.b64encode(b"fake_png_data").decode("ascii")

        # Send prompt with image content
        prompt_response = await connection.prompt(
            PromptRequest(
                sessionId=session_id,
                prompt=[
                    text_block("Describe this image:"),
                    ImageContentBlock(
                        type="image",
                        data=fake_image_data,
                        mimeType="image/png",
                    ),
                ],
            )
        )
        assert prompt_response.stopReason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client)
        assert len(client.notifications) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_embedded_resource_conversion() -> None:
    """Test that EmbeddedResource from ACP is properly converted."""
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

        # Verify we advertise resource support
        prompt_caps = getattr(init_response.agentCapabilities, "prompts", None)
        assert prompt_caps is not None
        supported_types = getattr(prompt_caps, "supportedTypes", [])
        assert "resource" in supported_types

        # Create session
        session_response = await connection.newSession(
            NewSessionRequest(mcpServers=[], cwd=str(TEST_DIR))
        )
        session_id = session_response.sessionId

        # Send prompt with embedded resource
        prompt_response = await connection.prompt(
            PromptRequest(
                sessionId=session_id,
                prompt=[
                    text_block("Review this code:"),
                    EmbeddedResourceContentBlock(
                        type="resource",
                        resource=ACPTextResourceContents(
                            uri="file:///test.py",
                            text="def hello():\n    return 'world'",
                            mimeType="text/x-python",
                        ),
                    ),
                ],
            )
        )
        assert prompt_response.stopReason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client)
        assert len(client.notifications) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_multipart_content_conversion() -> None:
    """Test that mixed content types are properly converted."""
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

        # Create fake image
        fake_image_data = base64.b64encode(b"screenshot_data").decode("ascii")

        # Send prompt with multiple content types
        prompt_response = await connection.prompt(
            PromptRequest(
                sessionId=session_id,
                prompt=[
                    text_block("Analyze this screenshot and code:"),
                    ImageContentBlock(
                        type="image",
                        data=fake_image_data,
                        mimeType="image/png",
                    ),
                    EmbeddedResourceContentBlock(
                        type="resource",
                        resource=ACPTextResourceContents(
                            uri="file:///app.py",
                            text="print('hello')",
                            mimeType="text/x-python",
                        ),
                    ),
                    text_block("What do you observe?"),
                ],
            )
        )
        assert prompt_response.stopReason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client)
        assert len(client.notifications) > 0


async def _wait_for_notifications(client: TestClient, timeout: float = 2.0) -> None:
    """Wait for the ACP client to receive at least one sessionUpdate."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if client.notifications:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("Expected streamed session updates")
