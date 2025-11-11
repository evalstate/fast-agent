"""Tests for ACP slash commands functionality."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from acp import InitializeRequest, NewSessionRequest, PromptRequest
from acp.helpers import text_block
from acp.schema import ClientCapabilities, Implementation, StopReason

from fast_agent.acp.slash_commands import SlashCommandHandler

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

CONFIG_PATH = TEST_DIR / "fastagent.config.yaml"
END_TURN: StopReason = "end_turn"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_parsing() -> None:
    """Test that slash commands are correctly parsed."""
    # Create a mock instance for testing
    from unittest.mock import Mock

    from fast_agent.core.fastagent import AgentInstance

    mock_instance = Mock(spec=AgentInstance)
    mock_instance.agents = {}

    handler = SlashCommandHandler("test-session", mock_instance, "test-agent")

    # Test valid slash command
    assert handler.is_slash_command("/status")
    assert handler.is_slash_command("/status arg1 arg2")
    assert handler.is_slash_command("  /status  ")

    # Test non-slash command
    assert not handler.is_slash_command("status")
    assert not handler.is_slash_command("just a regular prompt")

    # Test parsing
    cmd, args = handler.parse_command("/status")
    assert cmd == "status"
    assert args == ""

    cmd, args = handler.parse_command("/status arg1 arg2")
    assert cmd == "status"
    assert args == "arg1 arg2"

    cmd, args = handler.parse_command("  /status  ")
    assert cmd == "status"
    assert args == ""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_available_commands() -> None:
    """Test that available commands are returned correctly."""
    from unittest.mock import Mock

    from fast_agent.core.fastagent import AgentInstance

    mock_instance = Mock(spec=AgentInstance)
    mock_instance.agents = {}

    handler = SlashCommandHandler("test-session", mock_instance, "test-agent")

    # Get available commands
    commands = handler.get_available_commands()

    # Should have at least the status command
    assert len(commands) >= 1
    assert any(cmd["name"] == "status" for cmd in commands)

    # Check status command structure
    status_cmd = next(cmd for cmd in commands if cmd["name"] == "status")
    assert "description" in status_cmd
    assert status_cmd["description"]  # Should have a non-empty description


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_unknown_command() -> None:
    """Test that unknown commands are handled gracefully."""
    from unittest.mock import Mock

    from fast_agent.core.fastagent import AgentInstance

    mock_instance = Mock(spec=AgentInstance)
    mock_instance.agents = {}

    handler = SlashCommandHandler("test-session", mock_instance, "test-agent")

    # Execute unknown command
    response = await handler.execute_command("unknown_cmd", "")

    # Should get an error message
    assert "Unknown command" in response or "not yet implemented" in response.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_status() -> None:
    """Test the /status command execution."""
    from unittest.mock import Mock

    from fast_agent.core.fastagent import AgentInstance

    # Create a more complete mock with a mock agent
    mock_agent = Mock()
    mock_agent.message_history = []
    mock_agent._llm = None

    mock_instance = Mock(spec=AgentInstance)
    mock_instance.agents = {"test-agent": mock_agent}

    handler = SlashCommandHandler("test-session", mock_instance, "test-agent")

    # Execute status command
    response = await handler.execute_command("status", "")

    # Should contain expected sections
    assert "Fast-Agent Status" in response or "fast-agent" in response.lower()
    assert "Version" in response or "version" in response.lower()
    assert "Model" in response or "model" in response.lower()
    # Context stats should be present even if values are minimal
    assert "Turns" in response or "turns" in response.lower()
