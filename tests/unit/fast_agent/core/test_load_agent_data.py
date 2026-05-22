"""Tests for the in-memory agent definition loader.

Covers build_loaded_card_from_dict (single-card construction) and the
synthetic memory:// path marker the FastAgent layer uses to distinguish
file-based from in-memory cards.

FastAgent.load_agents_from_dicts has its own integration test in jarvis
because exercising it standalone requires a fully initialised FastAgent
which depends on a runnable config file. Here we test the public
loader primitives.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fast_agent.core.agent_card_loader import (
    build_loaded_card_from_dict,
    is_memory_card_path,
)
from fast_agent.core.exceptions import AgentConfigError


class TestBuildLoadedCardFromDict:
    """build_loaded_card_from_dict produces a LoadedAgentCard tagged
    with a synthetic memory:// path, matching the validation behaviour
    of file-based loads."""

    def test_minimal_dict_builds_card(self) -> None:
        card = build_loaded_card_from_dict(
            {"name": "Helper", "instruction": "You help."}
        )
        assert card.name == "Helper"
        # Python's Path("memory://X") collapses // to a single /; the
        # important invariant is the "memory:" prefix (checked via
        # the marker function below).
        assert is_memory_card_path(card.path)
        assert card.agent_data["config"].instruction.strip() == "You help."

    def test_defaults_type_to_agent(self) -> None:
        """File-based cards default to type=agent when omitted. The
        in-memory loader must match so callers do not need to specify
        type just because they skipped writing a file.

        Note: card type "agent" maps to AgentType.BASIC internally
        (see CARD_TYPE_TO_AGENT_TYPE), so agent_data["type"] reads
        as "basic" — that's the AgentType enum value, not the card
        type label."""
        card = build_loaded_card_from_dict(
            {"name": "X", "instruction": "x"}
        )
        assert card.agent_data["type"] == "basic"

    def test_explicit_type_respected(self) -> None:
        card = build_loaded_card_from_dict(
            {
                "name": "Router",
                "instruction": "route",
                "type": "router",
                "agents": ["A"],
            }
        )
        # router type stored on agent_data
        assert card.agent_data["type"] == "router"

    def test_name_override(self) -> None:
        """An explicit `name` arg overrides the dict's name field —
        useful when the caller's key (e.g. DB primary key) is the
        canonical source of truth."""
        card = build_loaded_card_from_dict(
            {"name": "FromDict", "instruction": "x"}, name="FromArg"
        )
        assert card.name == "FromArg"
        assert is_memory_card_path(card.path)

    def test_missing_name_raises(self) -> None:
        with pytest.raises(AgentConfigError, match="Agent name is required"):
            build_loaded_card_from_dict({"instruction": "x"})

    def test_empty_name_raises(self) -> None:
        with pytest.raises(AgentConfigError, match="Agent name is required"):
            build_loaded_card_from_dict({"name": "   ", "instruction": "x"})

    def test_invalid_field_raises(self) -> None:
        """Same validation as file-based cards — unknown fields are
        rejected to catch typos early."""
        with pytest.raises(AgentConfigError):
            build_loaded_card_from_dict(
                {"name": "X", "instruction": "x", "noSuchField": 1}
            )

    def test_servers_skills_tools_passed_through(self) -> None:
        card = build_loaded_card_from_dict(
            {
                "name": "Full",
                "instruction": "do",
                "servers": ["s1", "s2"],
                "tools": {"s1": ["t1"]},
                "skills": [".fast-agent/skills/research"],
                "model": "anthropic.claude-sonnet",
                "use_history": False,
            }
        )
        cfg = card.agent_data["config"]
        assert list(cfg.servers) == ["s1", "s2"]
        assert dict(cfg.tools) == {"s1": ["t1"]}
        assert cfg.skills == [".fast-agent/skills/research"]
        assert cfg.model == "anthropic.claude-sonnet"
        assert cfg.use_history is False


class TestIsMemoryCardPath:
    """The marker function callers use to tell file paths from
    in-memory tags inside _agent_card_sources."""

    def test_memory_url_str(self) -> None:
        assert is_memory_card_path("memory://Foo")

    def test_memory_url_path(self) -> None:
        assert is_memory_card_path(Path("memory://Foo"))

    def test_filesystem_path(self) -> None:
        assert not is_memory_card_path("/tmp/agent_cards/Foo.md")

    def test_relative_filesystem_path(self) -> None:
        assert not is_memory_card_path(Path(".fast-agent/agent_cards/Foo.md"))

    def test_empty_string(self) -> None:
        assert not is_memory_card_path("")
