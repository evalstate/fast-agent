"""Tests for `<source>` provenance and `/skills enable|disable` toggles."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.commands.handlers.skills import handle_skills_command
from fast_agent.context import Context
from fast_agent.skills.registry import SkillManifest, format_skills_for_prompt

if TYPE_CHECKING:
    from pathlib import Path


def _outcome_text(outcome) -> str:
    parts: list[str] = []
    for msg in outcome.messages:
        text = msg.text
        parts.append(text.plain if hasattr(text, "plain") else str(text))
    return "\n".join(parts)


def _fs_manifest(tmp_path: Path, name: str) -> SkillManifest:
    skill_dir = tmp_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    md = skill_dir / "SKILL.md"
    md.write_text(
        f"---\nname: {name}\ndescription: d\n---\nbody\n", encoding="utf-8"
    )
    return SkillManifest(name=name, description="d", body="body", path=md)


def _mcp_manifest(name: str, server: str = "github") -> SkillManifest:
    return SkillManifest(
        name=name,
        description=f"The {name} skill",
        body="",
        path=None,
        uri=f"skill://{name}/SKILL.md",
        server_name=server,
    )


# --- provenance rendering ------------------------------------------------


def test_format_emits_filesystem_source(tmp_path: Path) -> None:
    m = _fs_manifest(tmp_path, "alpha")
    out = format_skills_for_prompt([m])
    assert "<source>filesystem:" in out
    # Source line includes the skill's directory.
    assert str(tmp_path / "alpha") in out


def test_format_emits_mcp_server_source() -> None:
    m = _mcp_manifest("git-workflow", server="github")
    out = format_skills_for_prompt([m])
    assert "<source>mcp-server: github</source>" in out


def test_format_filters_disabled_skills(tmp_path: Path) -> None:
    fs = _fs_manifest(tmp_path, "alpha")
    mcp = _mcp_manifest("beta", server="github")
    out = format_skills_for_prompt(
        [fs, mcp], disabled_skill_names={"beta"}
    )
    # Only alpha makes it into the rendering.
    assert "<name>alpha</name>" in out
    assert "<name>beta</name>" not in out


def test_format_empty_when_all_disabled(tmp_path: Path) -> None:
    """If every manifest is disabled, the rendering must not leave an
    empty <available_skills> block in the prompt — that's confusing for
    the model. Return empty string instead, same as the no-manifests
    case."""
    fs = _fs_manifest(tmp_path, "alpha")
    out = format_skills_for_prompt([fs], disabled_skill_names={"alpha"})
    assert out == ""


def test_format_case_insensitive_disable(tmp_path: Path) -> None:
    fs = _fs_manifest(tmp_path, "alpha")
    out = format_skills_for_prompt([fs], disabled_skill_names={"ALPHA"})
    assert "<name>alpha</name>" not in out


# --- agent toggle methods ------------------------------------------------


@pytest.mark.asyncio
async def test_disable_then_enable_round_trip(tmp_path: Path) -> None:
    """disable() filters the SkillReader; enable() restores it. The model
    can read the skill before/after but not while disabled."""
    fs = _fs_manifest(tmp_path, "alpha")
    config = AgentConfig(
        name="test", instruction="x", servers=[], skills=tmp_path
    )
    config.skill_manifests = [fs]
    agent = McpAgent(config=config, context=Context())

    # Pre-disable: reader admits the path.
    result = await agent._skill_reader.execute({"path": str(fs.path)})
    assert not result.isError

    assert agent.disable_skill("alpha") is True
    # While disabled the reader rejects the path.
    result = await agent._skill_reader.execute({"path": str(fs.path)})
    assert result.isError
    assert "not within an allowed skill directory" in result.content[0].text

    assert agent.enable_skill("alpha") is True
    result = await agent._skill_reader.execute({"path": str(fs.path)})
    assert not result.isError


def test_disable_unknown_returns_false(tmp_path: Path) -> None:
    """Trying to disable a skill the agent doesn't have must not silently
    add the name to the set — otherwise an enable would do nothing and
    the user has no way to tell the typo apart from the no-op."""
    fs = _fs_manifest(tmp_path, "alpha")
    config = AgentConfig(
        name="test", instruction="x", servers=[], skills=tmp_path
    )
    config.skill_manifests = [fs]
    agent = McpAgent(config=config, context=Context())

    assert agent.disable_skill("nonexistent") is False
    assert agent.disabled_skill_names == set()


def test_disable_idempotent(tmp_path: Path) -> None:
    fs = _fs_manifest(tmp_path, "alpha")
    config = AgentConfig(
        name="test", instruction="x", servers=[], skills=tmp_path
    )
    config.skill_manifests = [fs]
    agent = McpAgent(config=config, context=Context())

    assert agent.disable_skill("alpha") is True
    # Second call: no state change, returns False.
    assert agent.disable_skill("alpha") is False


# --- slash command wiring ------------------------------------------------


def _ctx_with_agent(agent_obj):
    provider = MagicMock()
    provider._agent = MagicMock(return_value=agent_obj)
    ctx = SimpleNamespace(
        agent_provider=provider,
        current_agent_name="default",
        io=MagicMock(),
        settings=None,
    )
    ctx.resolve_settings = MagicMock(return_value=MagicMock())
    return ctx


@pytest.mark.asyncio
async def test_disable_command_invokes_agent() -> None:
    agent = SimpleNamespace(disable_skill=MagicMock(return_value=True))
    ctx = _ctx_with_agent(agent)

    outcome = await handle_skills_command(
        ctx, agent_name="default", action="disable", argument="alpha"
    )
    text = _outcome_text(outcome)
    assert "Disabled skill: alpha" in text
    agent.disable_skill.assert_called_once_with("alpha")


@pytest.mark.asyncio
async def test_disable_command_no_argument() -> None:
    agent = SimpleNamespace(disable_skill=MagicMock(return_value=True))
    ctx = _ctx_with_agent(agent)
    outcome = await handle_skills_command(
        ctx, agent_name="default", action="disable", argument=None
    )
    text = _outcome_text(outcome)
    assert "Usage:" in text
    agent.disable_skill.assert_not_called()


@pytest.mark.asyncio
async def test_disable_command_when_skill_unknown() -> None:
    agent = SimpleNamespace(disable_skill=MagicMock(return_value=False))
    ctx = _ctx_with_agent(agent)
    outcome = await handle_skills_command(
        ctx, agent_name="default", action="disable", argument="nope"
    )
    text = _outcome_text(outcome)
    assert "not active" in text or "already disabled" in text


@pytest.mark.asyncio
async def test_enable_command_invokes_agent() -> None:
    agent = SimpleNamespace(enable_skill=MagicMock(return_value=True))
    ctx = _ctx_with_agent(agent)
    outcome = await handle_skills_command(
        ctx, agent_name="default", action="enable", argument="alpha"
    )
    text = _outcome_text(outcome)
    assert "Enabled skill: alpha" in text
    agent.enable_skill.assert_called_once_with("alpha")
