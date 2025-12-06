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
from acp.helpers import text_block
from acp.schema import (
    BlobResourceContents,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    FileSystemCapability,
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
        init_response = await connection.initialize(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=True, write_text_file=True),
                terminal=False,
            ),
            client_info=Implementation(name="pytest-client", version="0.0.1"),
        )

        # Check that image is advertised as supported
        agent_caps = getattr(init_response, "agent_capabilities", None) or getattr(
            init_response, "agentCapabilities", None
        )
        assert agent_caps is not None
        # Handle both "prompts" and "promptCapabilities" field names
        prompt_caps = getattr(
            agent_caps, "prompts", None
        ) or getattr(agent_caps, "promptCapabilities", None)
        assert prompt_caps is not None
        # Check if image capability is enabled
        assert getattr(prompt_caps, "image", False) is True

        # Create session
        session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
        session_id = getattr(session_response, "session_id", None) or getattr(
            session_response, "sessionId", None
        )
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

        prompt_response = await connection.prompt(session_id=session_id, prompt=prompt_blocks)

        # Should complete successfully
        stop_reason = getattr(prompt_response, "stop_reason", None) or getattr(
            prompt_response, "stopReason", None
        )
        assert stop_reason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client)

        # Verify we got a response (passthrough model will echo something back)
        assert len(client.notifications) > 0
        last_update = client.notifications[-1]
        assert last_update["session_id"] == session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_embedded_text_resource_processing() -> None:
    """Test that embedded text resource content blocks are properly processed."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize
        init_response = await connection.initialize(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=True, write_text_file=True),
                terminal=False,
            ),
            client_info=Implementation(name="pytest-client", version="0.0.1"),
        )

        # Check that resource is advertised as supported
        agent_caps = getattr(init_response, "agent_capabilities", None) or getattr(
            init_response, "agentCapabilities", None
        )
        assert agent_caps is not None
        # Handle both "prompts" and "promptCapabilities" field names
        prompt_caps = getattr(
            agent_caps, "prompts", None
        ) or getattr(agent_caps, "promptCapabilities", None)
        assert prompt_caps is not None
        # Check if embeddedContext capability is enabled
        assert getattr(prompt_caps, "embeddedContext", False) is True

        # Create session
        session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
        session_id = getattr(session_response, "session_id", None) or getattr(
            session_response, "sessionId", None
        )
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
            session_id=session_id,
            prompt=prompt_blocks,
        )

        # Should complete successfully
        stop_reason = getattr(prompt_response, "stop_reason", None) or getattr(
            prompt_response, "stopReason", None
        )
        assert stop_reason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client)

        # Verify we got a response
        assert len(client.notifications) > 0
        last_update = client.notifications[-1]
        assert last_update["session_id"] == session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_embedded_blob_resource_processing() -> None:
    """Test that embedded blob resource content blocks are properly processed."""
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize and create session
        await connection.initialize(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=True, write_text_file=True),
                terminal=False,
            ),
            client_info=Implementation(name="pytest-client", version="0.0.1"),
        )

        session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
        session_id = getattr(session_response, "session_id", None) or getattr(
            session_response, "sessionId", None
        )
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

        prompt_response = await connection.prompt(session_id=session_id, prompt=prompt_blocks)

        # Should complete successfully
        stop_reason = getattr(prompt_response, "stop_reason", None) or getattr(
            prompt_response, "stopReason", None
        )
        assert stop_reason == END_TURN

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
        await connection.initialize(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=True, write_text_file=True),
                terminal=False,
            ),
            client_info=Implementation(name="pytest-client", version="0.0.1"),
        )

        session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
        session_id = getattr(session_response, "session_id", None) or getattr(
            session_response, "sessionId", None
        )

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

        prompt_response = await connection.prompt(session_id=session_id, prompt=prompt_blocks)

        # Should complete successfully
        stop_reason = getattr(prompt_response, "stop_reason", None) or getattr(
            prompt_response, "stopReason", None
        )
        assert stop_reason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client)

        # Verify we got a response
        assert len(client.notifications) > 0
        last_update = client.notifications[-1]
        assert last_update["session_id"] == session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_acp_resource_only_prompt_not_slash_command() -> None:
    """
    Test that resource-only prompts with text starting with "/" are not treated as slash commands.

    This verifies the fix for the issue where resource content (like file contents) that
    happens to start with "/" was incorrectly being detected as a slash command.
    """
    client = TestClient()

    async with spawn_agent_process(lambda _: client, *FAST_AGENT_CMD) as (connection, _process):
        # Initialize and create session
        await connection.initialize(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=True, write_text_file=True),
                terminal=False,
            ),
            client_info=Implementation(name="pytest-client", version="0.0.1"),
        )

        session_response = await connection.new_session(mcp_servers=[], cwd=str(TEST_DIR))
        session_id = getattr(session_response, "session_id", None) or getattr(
            session_response, "sessionId", None
        )
        assert session_id

        # Send a resource-only prompt with text starting with "/"
        # This should NOT be treated as a slash command
        prompt_blocks = [
            EmbeddedResourceContentBlock(
                type="resource",
                resource=TextResourceContents(
                    uri="file:///C:/Users/shaun/AppData/Roaming/Zed/settings.json",
                    mimeType="application/json",
                    text="//hello, world!",
                ),
            ),
        ]

        prompt_response = await connection.prompt(session_id=session_id, prompt=prompt_blocks)

        # Should complete successfully with END_TURN, not be treated as an unknown slash command
        stop_reason = getattr(prompt_response, "stop_reason", None) or getattr(
            prompt_response, "stopReason", None
        )
        assert stop_reason == END_TURN

        # Wait for notifications
        await _wait_for_notifications(client)

        # Verify we got a response from the agent (passthrough model)
        # If it was incorrectly treated as a slash command, we'd get "Unknown command" response
        assert len(client.notifications) > 0
        last_update = client.notifications[-1]
        assert last_update["session_id"] == session_id

        # The response should contain the echoed resource text, not an error about unknown command
        # (passthrough model echoes the input)
        response_text = str(last_update["update"])
        assert "Unknown command" not in response_text


async def _wait_for_notifications(client: TestClient, timeout: float = 2.0) -> None:
    """Wait for the ACP client to receive at least one sessionUpdate."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if client.notifications:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("Expected streamed session updates")
