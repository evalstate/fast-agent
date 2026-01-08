"""Tests for HashAgentCommand (# command) functionality."""

import pytest
from prompt_toolkit.document import Document

from fast_agent.ui.command_payloads import HashAgentCommand
from fast_agent.ui.enhanced_prompt import AgentCompleter, parse_special_input


class TestParseHashAgentCommand:
    """Tests for parsing #agent message syntax."""

    def test_parse_hash_agent_with_message(self):
        """Test parsing #agent_name message returns HashAgentCommand."""
        result = parse_special_input("#myagent hello world")
        assert isinstance(result, HashAgentCommand)
        assert result.agent_name == "myagent"
        assert result.message == "hello world"
        assert result.kind == "hash_agent"

    def test_parse_hash_agent_without_message(self):
        """Test parsing #agent_name without message returns empty message."""
        result = parse_special_input("#myagent")
        assert isinstance(result, HashAgentCommand)
        assert result.agent_name == "myagent"
        assert result.message == ""

    def test_parse_hash_agent_preserves_message_spaces(self):
        """Test that spaces in the message are preserved."""
        result = parse_special_input("#agent this is a   long   message")
        assert isinstance(result, HashAgentCommand)
        assert result.message == "this is a   long   message"

    def test_parse_hash_agent_strips_agent_name(self):
        """Test that agent name is stripped of whitespace."""
        result = parse_special_input("#  agent_name  message")
        assert isinstance(result, HashAgentCommand)
        assert result.agent_name == "agent_name"
        assert result.message == "message"

    def test_parse_hash_only_returns_plain_text(self):
        """Test that # alone returns original text."""
        result = parse_special_input("#")
        # Just "#" returns original text since there's no agent name
        assert result == "#"

    def test_parse_hash_agent_multiline_message(self):
        """Test parsing with newlines in message."""
        result = parse_special_input("#agent line1\nline2")
        assert isinstance(result, HashAgentCommand)
        assert result.agent_name == "agent"
        assert "line1\nline2" in result.message


class TestHashAgentCompleter:
    """Tests for # command completion."""

    def test_hash_completer_shows_agents(self):
        """Test that # prefix shows agent completions."""
        completer = AgentCompleter(agents=["test_agent", "other_agent"])

        doc = Document("#te", cursor_position=3)
        completions = list(completer.get_completions(doc, None))
        texts = [c.text for c in completions]

        assert len(completions) == 1
        # Should include space after agent name for message input
        assert "test_agent " in texts

    def test_hash_completer_shows_all_matching_agents(self):
        """Test that all matching agents are shown."""
        completer = AgentCompleter(agents=["agent_one", "agent_two", "other"])

        doc = Document("#agent", cursor_position=6)
        completions = list(completer.get_completions(doc, None))
        texts = [c.text for c in completions]

        assert len(completions) == 2
        assert "agent_one " in texts
        assert "agent_two " in texts

    def test_hash_completer_case_insensitive(self):
        """Test that completion is case insensitive."""
        completer = AgentCompleter(agents=["TestAgent"])

        doc = Document("#test", cursor_position=5)
        completions = list(completer.get_completions(doc, None))
        texts = [c.text for c in completions]

        assert "TestAgent " in texts

    def test_hash_completer_stops_after_space(self):
        """Test that completion stops after agent name and space."""
        completer = AgentCompleter(agents=["test_agent"])

        doc = Document("#test_agent ", cursor_position=12)
        completions = list(completer.get_completions(doc, None))

        # Should return nothing - user is now typing the message
        assert len(completions) == 0

    def test_hash_completer_metadata_shows_type(self):
        """Test that completion metadata shows agent type."""
        from fast_agent.agents.agent_types import AgentType

        completer = AgentCompleter(
            agents=["basic_agent", "orchestrator"],
            agent_types={
                "basic_agent": AgentType.BASIC,
                "orchestrator": AgentType.ORCHESTRATOR,
            },
        )

        doc = Document("#basic", cursor_position=6)
        completions = list(completer.get_completions(doc, None))

        assert len(completions) == 1
        # Metadata should include "# " prefix to distinguish from @ completions
        meta = str(completions[0].display_meta) if completions[0].display_meta else ""
        assert "# " in meta

    def test_hash_completer_no_agents(self):
        """Test completion with no agents."""
        completer = AgentCompleter(agents=[])

        doc = Document("#te", cursor_position=3)
        completions = list(completer.get_completions(doc, None))

        assert len(completions) == 0

    def test_hash_completer_empty_prefix(self):
        """Test completion with just # shows all agents."""
        completer = AgentCompleter(agents=["agent1", "agent2"])

        doc = Document("#", cursor_position=1)
        completions = list(completer.get_completions(doc, None))
        texts = [c.text for c in completions]

        assert len(completions) == 2
        assert "agent1 " in texts
        assert "agent2 " in texts


class TestHashAgentCommandPayload:
    """Tests for HashAgentCommand dataclass."""

    def test_hash_agent_command_immutable(self):
        """Test that HashAgentCommand is immutable (frozen)."""
        cmd = HashAgentCommand(agent_name="test", message="hello")

        with pytest.raises(AttributeError):
            cmd.agent_name = "other"  # type: ignore[misc]

    def test_hash_agent_command_kind(self):
        """Test that kind is always 'hash_agent'."""
        cmd = HashAgentCommand(agent_name="test", message="hello")
        assert cmd.kind == "hash_agent"

    def test_hash_agent_command_equality(self):
        """Test equality comparison."""
        cmd1 = HashAgentCommand(agent_name="test", message="hello")
        cmd2 = HashAgentCommand(agent_name="test", message="hello")
        cmd3 = HashAgentCommand(agent_name="test", message="world")

        assert cmd1 == cmd2
        assert cmd1 != cmd3
