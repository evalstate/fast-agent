"""Tests for ACP slash commands functionality."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from mcp.types import TextContent

from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.agents.agent_types import AgentType
from fast_agent.config import get_settings, update_global_settings
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.session import display_session_name, get_session_manager, reset_session_manager
from fast_agent.session import session_manager as session_manager_module

if TYPE_CHECKING:
    from acp.schema import StopReason

    from fast_agent.core.fastagent import AgentInstance
    from fast_agent.interfaces import AgentProtocol
else:
    class AgentProtocol:  # pragma: no cover
        pass


TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))


CONFIG_PATH = TEST_DIR / "fastagent.config.yaml"
END_TURN: StopReason = "end_turn"


@dataclass
class StubAgent:
    message_history: list[Any] = field(default_factory=list)
    llm: Any = None
    cleared: bool = False
    popped: bool = False
    agent_type: AgentType = AgentType.BASIC
    name: str = "test-agent"

    def clear(self, clear_prompts: bool = False) -> None:
        self.cleared = True
        self.message_history.clear()

    def pop_last_message(self):
        self.popped = True
        if not self.message_history:
            return None
        return self.message_history.pop()


@dataclass
class StubAgentInstance:
    agents: dict[str, Any] = field(default_factory=dict)


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
    assert "clear" in command_names
    assert "history" in command_names
    assert "session" in command_names

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
    stub_agent = StubAgent(message_history=[], llm=None)
    instance = StubAgentInstance(agents={"test-agent": stub_agent})

    handler = _handler(instance)

    # Execute status command
    response = await handler.execute_command("status", "")

    # Should contain expected sections
    assert "fast-agent Status" in response or "fast-agent" in response.lower()
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

    stub_agent = StubAgent(message_history=[mock_message], llm=None)
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

    stub_agent = AgentWithInstruction(message_history=[], llm=None)
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
async def test_slash_command_session_resume_switches_current_mode(tmp_path: Path) -> None:
    """Test /session resume switches current mode when a single agent has history."""
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    session_manager_module._session_manager = None
    try:
        manager = get_session_manager()
        session = manager.create_session()

        history_message = PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="resume me")],
        )
        alpha_agent = StubAgent(name="alpha", message_history=[history_message])
        await session.save_history(cast("AgentProtocol", alpha_agent))

        beta_agent = StubAgent(name="beta")
        instance = StubAgentInstance(agents={"alpha": alpha_agent, "beta": beta_agent})
        switched: list[str] = []

        def _set_current_mode(agent_name: str) -> None:
            switched.append(agent_name)

        handler = _handler(
            instance,
            agent_name="beta",
            set_current_mode_callback=_set_current_mode,
        )
        response = await handler.execute_command(
            "session",
            f"resume {session.info.name}",
        )

        assert "Switched to agent: alpha" in response
        assert handler.current_agent_name == "alpha"
        assert switched == ["alpha"]
    finally:
        session_manager_module._session_manager = None
        os.chdir(original_cwd)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_status_system_prefers_session_instruction() -> None:
    """Test /status system prefers session-resolved instructions when available."""

    @dataclass
    class AgentWithInstruction(StubAgent):
        name: str = "test-agent"
        instruction: str = "Template instruction with {{env}}."

    stub_agent = AgentWithInstruction(message_history=[], llm=None)
    instance = StubAgentInstance(agents={"test-agent": stub_agent})

    resolved_instruction = "Resolved instruction with env."
    handler = _handler(instance, session_instructions={"test-agent": resolved_instruction})

    response = await handler.execute_command("status", "system")

    assert resolved_instruction in response
    assert stub_agent.instruction not in response


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_status_system_without_instruction() -> None:
    """Test /status system when agent has no instruction attribute."""
    stub_agent = StubAgent(message_history=[], llm=None)
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
async def test_slash_command_history_save_conversation() -> None:
    """Test that /history save saves history and reports the filename."""

    class RecordingHistoryExporter:
        def __init__(self, default_name: str = "24_01_01_12_00-conversation.json") -> None:
            self.default_name = default_name
            self.calls: list[tuple[Any, str | None]] = []

        async def save(self, agent, filename: str | None = None) -> str:
            self.calls.append((agent, filename))
            return filename or self.default_name

    stub_agent = StubAgent(message_history=[])
    instance = StubAgentInstance(agents={"test-agent": stub_agent})
    exporter = RecordingHistoryExporter()

    handler = _handler(instance, history_exporter=exporter)

    response = await handler.execute_command("history", "save")

    assert "save conversation" in response.lower()
    assert "History saved to" in response
    assert "24_01_01_12_00-conversation.json" in response
    assert exporter.calls == [(stub_agent, None)]

    response_with_filename = await handler.execute_command("history", "save custom.md")
    assert "History saved to" in response_with_filename
    assert "custom.md" in response_with_filename
    assert exporter.calls[-1] == (stub_agent, "custom.md")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_history_save_without_agent() -> None:
    """Test /history save error handling when the agent is missing."""
    handler = _handler(StubAgentInstance(), agent_name="missing-agent")

    response = await handler.execute_command("history", "save")

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
async def test_slash_command_history_load() -> None:
    """Test that /history load loads history from a file."""
    stub_agent = StubAgent(message_history=[])
    instance = StubAgentInstance(agents={"test-agent": stub_agent})

    handler = _handler(instance)

    # Create a temporary history file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        history_data = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]},
            ]
        }
        json.dump(history_data, f)
        temp_path = f.name

    try:
        response = await handler.execute_command("history", f"load {temp_path}")

        assert "load conversation" in response.lower()
        assert "loaded 2 messages" in response.lower()
        assert temp_path in response
        assert len(stub_agent.message_history) == 2
        assert stub_agent.cleared is True  # History should be cleared before loading
    finally:
        Path(temp_path).unlink()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_history_load_without_filename() -> None:
    """Test /history load error handling when no filename is provided."""
    stub_agent = StubAgent(message_history=[])
    instance = StubAgentInstance(agents={"test-agent": stub_agent})

    handler = _handler(instance)

    response = await handler.execute_command("history", "load")

    assert "load conversation" in response.lower()
    assert "filename required" in response.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_history_load_file_not_found() -> None:
    """Test /history load error handling when file does not exist."""
    stub_agent = StubAgent(message_history=[])
    instance = StubAgentInstance(agents={"test-agent": stub_agent})

    handler = _handler(instance)

    response = await handler.execute_command("history", "load nonexistent_file.json")

    assert "load conversation" in response.lower()
    assert "file not found" in response.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_history_load_without_agent() -> None:
    """Test /history load error handling when the agent is missing."""
    handler = _handler(StubAgentInstance(), agent_name="missing-agent")

    response = await handler.execute_command("history", "load somefile.json")

    assert "load conversation" in response.lower()
    assert "Unable to locate agent" in response


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_history_available_in_commands() -> None:
    """Test that /history is in the available commands list."""
    handler = _handler(StubAgentInstance())

    commands = handler.get_available_commands()
    command_names = {cmd.name for cmd in commands}

    assert "history" in command_names

    history_cmd = next(cmd for cmd in commands if cmd.name == "history")
    assert history_cmd.description
    assert history_cmd.input is not None  # Should have input hint


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_session_list_no_sessions(tmp_path, monkeypatch) -> None:
    """Test /session list output when no sessions exist."""
    monkeypatch.chdir(tmp_path)

    import fast_agent.session.session_manager as session_module

    monkeypatch.setattr(session_module, "_session_manager", None)

    handler = _handler(StubAgentInstance())
    response = await handler.execute_command("session", "list")

    assert "no sessions" in response.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_session_pin_sets_metadata(tmp_path: Path) -> None:
    """Test /session pin marks the session as pinned."""
    old_settings = get_settings()
    env_dir = tmp_path / "env"
    override = old_settings.model_copy(update={"environment_dir": str(env_dir)})
    update_global_settings(override)
    reset_session_manager()

    try:
        manager = get_session_manager()
        session = manager.create_session()
        label = display_session_name(session.info.name)

        handler = _handler(
            StubAgentInstance(agents={"test-agent": StubAgent(message_history=[])}),
        )
        response = await handler.execute_command(
            "session",
            f"pin on {session.info.name}",
        )

        assert "Pinned session" in response
        assert label in response

        metadata = json.loads(
            (manager.base_dir / session.info.name / "session.json").read_text()
        )
        assert metadata["metadata"].get("pinned") is True

        list_response = await handler.execute_command("session", "list")
        assert "pin" in list_response.lower()
        assert label in list_response
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_card_loads_and_attaches() -> None:
    """Test /card loads AgentCards and attaches tools when requested."""
    stub_agent = StubAgent(message_history=[])
    instance = StubAgentInstance(agents={"test-agent": stub_agent})

    async def _card_loader(filename: str, parent_agent: str | None = None):
        new_instance = StubAgentInstance(
            agents={"test-agent": stub_agent, "alpha": StubAgent(name="alpha")}
        )
        return new_instance, ["alpha"], ["alpha"]

    handler = _handler(instance, card_loader=_card_loader)
    response = await handler.execute_command("card", "card.yml --tool")

    assert "Loaded AgentCard(s): alpha" in response
    assert "Attached agent tool(s): alpha" in response


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_agent_attach_and_detach() -> None:
    """Test /agent --tool attach and remove flows."""
    stub_agent = StubAgent(message_history=[])
    alpha_agent = StubAgent(message_history=[], name="alpha")
    instance = StubAgentInstance(agents={"test-agent": stub_agent, "alpha": alpha_agent})

    async def _attach_agent(parent_agent: str, child_agents: list[str]):
        return instance, child_agents

    async def _detach_agent(parent_agent: str, child_agents: list[str]):
        return instance, child_agents

    handler = _handler(
        instance,
        attach_agent_callback=_attach_agent,
        detach_agent_callback=_detach_agent,
    )

    response = await handler.execute_command("agent", "alpha --tool")
    assert "Attached agent tool(s): alpha" in response

    response = await handler.execute_command("agent", "alpha --tool remove")
    assert "Detached agent tool(s): alpha" in response


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_agent_dump() -> None:
    """Test /agent --dump returns agent card output."""
    stub_agent = StubAgent(message_history=[])
    instance = StubAgentInstance(agents={"test-agent": stub_agent})

    async def _dump_agent(agent_name: str) -> str:
        return "agent-card: test-agent"

    handler = _handler(instance, dump_agent_callback=_dump_agent)
    response = await handler.execute_command("agent", "--dump")

    assert "agent-card: test-agent" in response


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slash_command_reload_agent_cards() -> None:
    """Test /reload reports changes and no-changes states."""
    stub_agent = StubAgent(message_history=[])
    instance = StubAgentInstance(agents={"test-agent": stub_agent})

    async def _reload_changed() -> bool:
        return True

    handler = _handler(instance, reload_callback=_reload_changed)
    response = await handler.execute_command("reload", "")
    assert "AgentCards reloaded" in response

    async def _reload_no_change() -> bool:
        return False

    handler = _handler(instance, reload_callback=_reload_no_change)
    response = await handler.execute_command("reload", "")
    assert "No AgentCard changes detected" in response
