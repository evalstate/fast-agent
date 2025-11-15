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
class StubAgentConfig:
    """Stub agent configuration for testing."""

    name: str
    instruction: str = "This is a test agent instruction"
    agent_type: str = "LLM"
    default: bool = False
    servers: List[str] = field(default_factory=list)


@dataclass
class StubAgent:
    message_history: List[Any] = field(default_factory=list)
    _llm: Any = None
    _config: Optional[StubAgentConfig] = None
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_status_shows_current_mode() -> None:
    """Test that /status shows current mode information."""
    # Create multiple agents to test mode display
    agent1_config = StubAgentConfig(
        name="code_expert",
        instruction="Expert at writing code",
        default=True,
    )
    agent2_config = StubAgentConfig(
        name="general_assistant",
        instruction="General purpose assistant",
    )

    agent1 = StubAgent(message_history=[], _config=agent1_config)
    agent2 = StubAgent(message_history=[], _config=agent2_config)

    instance = StubAgentInstance(agents={
        "code_expert": agent1,
        "general_assistant": agent2,
    })

    handler = _handler(instance, agent_name="code_expert", current_agent_name="code_expert")

    # Execute status command
    response = await handler.execute_command("status", "")

    # Should contain mode information
    assert "Session Modes" in response or "mode" in response.lower()
    assert "Current Mode" in response or "current" in response.lower()
    assert "code_expert" in response
    assert "Available Modes" in response or "available" in response.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_status_lists_all_modes() -> None:
    """Test that /status lists all available modes when multiple agents exist."""
    # Create multiple agents
    agent1_config = StubAgentConfig(
        name="code_expert",
        instruction="Expert at writing code and debugging",
    )
    agent2_config = StubAgentConfig(
        name="general_assistant",
        instruction="General purpose assistant for various tasks",
    )
    agent3_config = StubAgentConfig(
        name="data_analyst",
        instruction="Specialized in data analysis and visualization",
    )

    agent1 = StubAgent(message_history=[], _config=agent1_config)
    agent2 = StubAgent(message_history=[], _config=agent2_config)
    agent3 = StubAgent(message_history=[], _config=agent3_config)

    instance = StubAgentInstance(agents={
        "code_expert": agent1,
        "general_assistant": agent2,
        "data_analyst": agent3,
    })

    handler = _handler(instance, agent_name="code_expert", current_agent_name="general_assistant")

    # Execute status command
    response = await handler.execute_command("status", "")

    # Should list all three agents
    assert "code_expert" in response
    assert "general_assistant" in response
    assert "data_analyst" in response

    # Should mark current agent
    assert "general_assistant" in response
    assert "(active)" in response.lower() or "active" in response.lower()

    # Should show count of available modes
    assert "3" in response  # 3 available modes


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_status_agent_specific() -> None:
    """Test that /status <agent> shows detailed info for a specific agent."""
    agent_config = StubAgentConfig(
        name="code_expert",
        instruction="Expert at writing code and debugging.\nProvides detailed technical assistance.",
        agent_type="LLM",
        default=True,
        servers=["filesystem", "git"],
    )

    agent = StubAgent(message_history=[], _config=agent_config)

    instance = StubAgentInstance(agents={"code_expert": agent})

    handler = _handler(instance, agent_name="code_expert", current_agent_name="code_expert")

    # Execute status command with agent argument
    response = await handler.execute_command("status", "code_expert")

    # Should contain agent-specific information
    assert "code_expert" in response
    assert "Code Expert" in response  # Formatted name
    assert "Configuration" in response
    assert "LLM" in response  # Agent type
    assert "Expert at writing code" in response  # Instruction
    assert "Default Agent" in response  # Is default
    assert "Currently Active" in response  # Is current
    assert "MCP Servers" in response
    assert "filesystem" in response
    assert "git" in response


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_status_invalid_agent() -> None:
    """Test that /status <invalid_agent> handles invalid agent names gracefully."""
    agent = StubAgent(message_history=[])
    instance = StubAgentInstance(agents={"code_expert": agent})

    handler = _handler(instance, agent_name="code_expert")

    # Execute status command with invalid agent name
    response = await handler.execute_command("status", "nonexistent_agent")

    # Should show error message
    assert "not found" in response.lower()
    assert "nonexistent_agent" in response
    assert "Available agents" in response or "available" in response.lower()
    assert "code_expert" in response  # Should list available agents


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_status_agent_not_current() -> None:
    """Test that /status <agent> correctly shows when agent is not currently active."""
    agent1_config = StubAgentConfig(name="code_expert", instruction="Code expert")
    agent2_config = StubAgentConfig(name="general_assistant", instruction="General assistant")

    agent1 = StubAgent(message_history=[], _config=agent1_config)
    agent2 = StubAgent(message_history=[], _config=agent2_config)

    instance = StubAgentInstance(agents={
        "code_expert": agent1,
        "general_assistant": agent2,
    })

    # Current agent is code_expert, but we're checking general_assistant
    handler = _handler(instance, agent_name="code_expert", current_agent_name="code_expert")

    # Execute status command for non-active agent
    response = await handler.execute_command("status", "general_assistant")

    # Should show agent info
    assert "general_assistant" in response
    assert "General Assistant" in response

    # Should NOT show "Currently Active"
    assert "Currently Active" not in response


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_status_current_agent_name_updates() -> None:
    """Test that status reflects the current_agent_name when it changes."""
    agent1 = StubAgent(message_history=[], _config=StubAgentConfig(name="agent1"))
    agent2 = StubAgent(message_history=[], _config=StubAgentConfig(name="agent2"))

    instance = StubAgentInstance(agents={"agent1": agent1, "agent2": agent2})

    # Start with agent1 as current
    handler = _handler(instance, agent_name="agent1", current_agent_name="agent1")

    response1 = await handler.execute_command("status", "")
    assert "agent1" in response1
    # Check that agent1 is marked as active in the mode list
    assert "(active)" in response1.lower() or "current" in response1.lower()

    # Simulate mode change by updating current_agent_name
    handler.current_agent_name = "agent2"

    response2 = await handler.execute_command("status", "")
    # Now agent2 should be shown as current
    assert "agent2" in response2
    # The active marker should now be associated with agent2 context
    assert "agent2" in response2
