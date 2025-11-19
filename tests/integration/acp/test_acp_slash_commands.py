"""Tests for ACP slash commands functionality."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import pytest
from mcp.types import TextContent

from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

if TYPE_CHECKING:
    from acp.schema import StopReason

    from fast_agent.core.fastagent import AgentInstance

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))


CONFIG_PATH = TEST_DIR / "fastagent.config.yaml"
END_TURN: StopReason = "end_turn"


@dataclass
class StubAgent:
    message_history: List[Any] = field(default_factory=list)
    _llm: Any = None
    cleared: bool = False
    popped: bool = False

    def clear(self) -> None:
        self.cleared = True
        self.message_history.clear()

    def pop_last_message(self):
        self.popped = True
        if not self.message_history:
            return None
        return self.message_history.pop()


@dataclass
class StubAgentInstance:
    agents: Dict[str, Any] = field(default_factory=dict)


def _handler(
    instance: StubAgentInstance,
    agent_name: str = "test-agent",
    **kwargs,
) -> SlashCommandHandler:
    return SlashCommandHandler(
        "test-session",
        cast("AgentInstance", instance),
        agent_name,
        **kwargs,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_parsing() -> None:
    """Test that slash commands are correctly parsed."""
    handler = _handler(StubAgentInstance())

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
    handler = _handler(StubAgentInstance())

    # Get available commands
    commands = handler.get_available_commands()

    # Should include primary commands
    command_names = {cmd.name for cmd in commands}
    assert "status" in command_names
    assert "save" in command_names
    assert "clear" in command_names

    # Check status command structure
    status_cmd = next(cmd for cmd in commands if cmd.name == "status")
    assert status_cmd.description  # Should have a non-empty description


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_unknown_command() -> None:
    """Test that unknown commands are handled gracefully."""
    handler = _handler(StubAgentInstance())

    # Execute unknown command
    response = await handler.execute_command("unknown_cmd", "")

    # Should get an error message
    assert "Unknown command" in response or "not yet implemented" in response.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_status() -> None:
    """Test the /status command execution."""
    stub_agent = StubAgent(message_history=[], _llm=None)
    instance = StubAgentInstance(agents={"test-agent": stub_agent})

    handler = _handler(instance)

    # Execute status command
    response = await handler.execute_command("status", "")

    # Should contain expected sections
    assert "Fast-Agent Status" in response or "fast-agent" in response.lower()
    assert "Version" in response or "version" in response.lower()
    assert "Model" in response or "model" in response.lower()
    # Context stats should be present even if values are minimal
    assert "Turns" in response or "turns" in response.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_status_reports_error_channel_entries() -> None:
    """Test that /status surfaces error channel diagnostics when available."""
    error_text = "Removed unsupported vision tool result before sending to model"
    mock_message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="response")],
        channels={FAST_AGENT_ERROR_CHANNEL: [TextContent(type="text", text=error_text)]},
    )

    stub_agent = StubAgent(message_history=[mock_message], _llm=None)
    instance = StubAgentInstance(agents={"test-agent": stub_agent})

    handler = _handler(instance)

    response = await handler.execute_command("status", "")

    assert FAST_AGENT_ERROR_CHANNEL in response
    assert error_text in response


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_status_system() -> None:
    """Test the /status system command to show system prompt."""

    @dataclass
    class AgentWithInstruction(StubAgent):
        name: str = "test-agent"
        instruction: str = "You are a helpful assistant that provides excellent support."

    stub_agent = AgentWithInstruction(message_history=[], _llm=None)
    instance = StubAgentInstance(agents={"test-agent": stub_agent})

    handler = _handler(instance)

    # Execute status system command
    response = await handler.execute_command("status", "system")

    # Should contain system prompt heading
    assert "system prompt" in response.lower()
    # Should contain the agent name
    assert "test-agent" in response.lower()
    # Should contain the instruction/system prompt
    assert stub_agent.instruction in response


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_status_system_without_instruction() -> None:
    """Test /status system when agent has no instruction attribute."""
    stub_agent = StubAgent(message_history=[], _llm=None)
    instance = StubAgentInstance(agents={"test-agent": stub_agent})

    handler = _handler(instance)

    # Execute status system command
    response = await handler.execute_command("status", "system")

    # Should contain system prompt heading
    assert "system prompt" in response.lower()
    # Should indicate no system prompt is available
    assert "no system prompt" in response.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_save_conversation() -> None:
    """Test that /save saves history and reports the filename."""

    class RecordingHistoryExporter:
        def __init__(self, default_name: str = "24_01_01_12_00-conversation.json") -> None:
            self.default_name = default_name
            self.calls: List[tuple[Any, Optional[str]]] = []

        async def save(self, agent, filename: Optional[str] = None) -> str:
            self.calls.append((agent, filename))
            return filename or self.default_name

    stub_agent = StubAgent(message_history=[])
    instance = StubAgentInstance(agents={"test-agent": stub_agent})
    exporter = RecordingHistoryExporter()

    handler = _handler(instance, history_exporter=exporter)

    response = await handler.execute_command("save", "")

    assert "save conversation" in response.lower()
    assert "24_01_01_12_00-conversation.json" in response
    assert exporter.calls == [(stub_agent, None)]

    response_with_filename = await handler.execute_command("save", "custom.md")
    assert "custom.md" in response_with_filename
    assert exporter.calls[-1] == (stub_agent, "custom.md")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_save_without_agent() -> None:
    """Test /save error handling when the agent is missing."""
    handler = _handler(StubAgentInstance(), agent_name="missing-agent")

    response = await handler.execute_command("save", "")

    assert "save conversation" in response.lower()
    assert "Unable to locate agent" in response


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_clear_history() -> None:
    """Test clearing the entire history."""
    messages = [
        PromptMessageExtended(role="user", content=[TextContent(type="text", text="hi")]),
        PromptMessageExtended(role="assistant", content=[TextContent(type="text", text="hello")]),
    ]
    stub_agent = StubAgent(message_history=messages.copy())
    instance = StubAgentInstance(agents={"test-agent": stub_agent})

    handler = _handler(instance)

    response = await handler.execute_command("clear", "")

    assert stub_agent.cleared is True
    assert stub_agent.message_history == []
    assert "clear conversation" in response.lower()
    assert "history cleared" in response.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_clear_last_entry() -> None:
    """Test clearing only the last message."""
    messages = [
        PromptMessageExtended(role="user", content=[TextContent(type="text", text="hi")]),
        PromptMessageExtended(role="assistant", content=[TextContent(type="text", text="hello")]),
    ]
    stub_agent = StubAgent(message_history=messages.copy())
    instance = StubAgentInstance(agents={"test-agent": stub_agent})

    handler = _handler(instance)

    response = await handler.execute_command("clear", "last")

    assert stub_agent.popped is True
    assert len(stub_agent.message_history) == 1
    assert stub_agent.message_history[0].role == "user"
    assert "clear last" in response.lower()
    assert "removed last" in response.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_clear_last_when_empty() -> None:
    """Test /clear last when no messages exist."""
    stub_agent = StubAgent(message_history=[])
    instance = StubAgentInstance(agents={"test-agent": stub_agent})
    handler = _handler(instance)

    response = await handler.execute_command("clear", "last")

    assert "clear last" in response.lower()
    assert "no messages" in response.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_not_detected_for_comments() -> None:
    """Test that text starting with "//" (like comments) is detected as a slash command."""
    handler = _handler(StubAgentInstance())

    # Double slash (comment-style) should still be detected as starting with "/"
    assert handler.is_slash_command("//hello, world!")
    assert handler.is_slash_command("// This is a comment")

    # However, the integration test test_acp_resource_only_prompt_not_slash_command
    # verifies that resource content with "//" is NOT treated as a slash command
    # because the slash command check only applies to pure text content, not resources
