"""Tests for `/skills preview <name>` — pre-load inspection surface.

SEP-2640 §Security Implications: hosts SHOULD let users inspect a
skill's content before it is loaded into model context. The model
decides autonomously when to call `read_skill`, so this command is the
only pre-load surface available to users.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from mcp.types import CallToolResult, TextContent

from fast_agent.commands.handlers.skills import handle_skills_command
from fast_agent.skills.registry import SkillManifest

if TYPE_CHECKING:
    from pathlib import Path


def _outcome_text(outcome) -> str:
    parts: list[str] = []
    for msg in outcome.messages:
        text = msg.text
        parts.append(text.plain if hasattr(text, "plain") else str(text))
    return "\n".join(parts)


def _ctx(agent_obj):
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


def _make_agent(manifests: list[SkillManifest], read_result: CallToolResult):
    """Build a stand-in agent with a SkillReader that returns `read_result`."""
    reader = MagicMock()

    async def execute(args):
        return read_result

    reader.execute = execute
    return SimpleNamespace(_skill_manifests=manifests, _skill_reader=reader)


@pytest.mark.asyncio
async def test_preview_filesystem_skill_renders_body(tmp_path: Path) -> None:
    md = tmp_path / "alpha" / "SKILL.md"
    md.parent.mkdir(parents=True)
    md.write_text("---\nname: alpha\ndescription: x\n---\n# alpha body\n", encoding="utf-8")
    manifest = SkillManifest(name="alpha", description="x", body="b", path=md)
    read_result = CallToolResult(
        content=[TextContent(type="text", text="# alpha body")],
        isError=False,
    )
    agent = _make_agent([manifest], read_result)

    outcome = await handle_skills_command(
        _ctx(agent), agent_name="default", action="preview", argument="alpha"
    )
    text = _outcome_text(outcome)
    assert "Skill preview: alpha" in text
    assert "filesystem:" in text
    assert "# alpha body" in text


@pytest.mark.asyncio
async def test_preview_mcp_skill_uses_uri_and_shows_server() -> None:
    manifest = SkillManifest(
        name="pull-requests",
        description="PR review",
        body="",
        path=None,
        uri="skill://pull-requests/SKILL.md",
        server_name="github",
    )
    read_result = CallToolResult(
        content=[TextContent(type="text", text="# pull-requests body")],
        isError=False,
    )
    agent = _make_agent([manifest], read_result)

    outcome = await handle_skills_command(
        _ctx(agent), agent_name="default", action="preview", argument="pull-requests"
    )
    text = _outcome_text(outcome)
    assert "Skill preview: pull-requests" in text
    assert "mcp-server: github" in text
    assert "skill://pull-requests/SKILL.md" in text
    assert "# pull-requests body" in text


@pytest.mark.asyncio
async def test_preview_case_insensitive_name_match() -> None:
    manifest = SkillManifest(
        name="Refunds",  # capitalized in frontmatter
        description="x",
        body="",
        path=None,
        uri="skill://refunds/SKILL.md",
        server_name="srv",
    )
    read_result = CallToolResult(
        content=[TextContent(type="text", text="# refunds body")],
        isError=False,
    )
    agent = _make_agent([manifest], read_result)

    outcome = await handle_skills_command(
        _ctx(agent), agent_name="default", action="preview", argument="refunds"
    )
    text = _outcome_text(outcome)
    assert "Skill preview: Refunds" in text


@pytest.mark.asyncio
async def test_preview_unknown_skill() -> None:
    agent = _make_agent([], CallToolResult(content=[], isError=False))

    outcome = await handle_skills_command(
        _ctx(agent), agent_name="default", action="preview", argument="ghost"
    )
    text = _outcome_text(outcome)
    assert "No skill named 'ghost'" in text


@pytest.mark.asyncio
async def test_preview_no_argument() -> None:
    agent = _make_agent([], CallToolResult(content=[], isError=False))

    outcome = await handle_skills_command(
        _ctx(agent), agent_name="default", action="preview", argument=None
    )
    text = _outcome_text(outcome)
    assert "Usage: /skills preview" in text


@pytest.mark.asyncio
async def test_preview_propagates_read_error(tmp_path: Path) -> None:
    """If the reader rejects (e.g. server returned not-found, or the skill
    is disabled), the user sees the failure rather than a silent empty
    preview."""
    md = tmp_path / "alpha" / "SKILL.md"
    md.parent.mkdir(parents=True)
    md.write_text("---\nname: alpha\ndescription: x\n---\nbody\n", encoding="utf-8")
    manifest = SkillManifest(name="alpha", description="x", body="b", path=md)
    read_result = CallToolResult(
        content=[TextContent(type="text", text="Access denied: nope.")],
        isError=True,
    )
    agent = _make_agent([manifest], read_result)

    outcome = await handle_skills_command(
        _ctx(agent), agent_name="default", action="preview", argument="alpha"
    )
    text = _outcome_text(outcome)
    assert "Preview failed:" in text
    assert "Access denied" in text


@pytest.mark.asyncio
async def test_preview_aliased_actions() -> None:
    """`/skills inspect` and `/skills show` route to the same handler so
    users discover the surface under several names."""
    manifest = SkillManifest(
        name="alpha",
        description="x",
        body="",
        path=None,
        uri="skill://alpha/SKILL.md",
        server_name="srv",
    )
    read_result = CallToolResult(
        content=[TextContent(type="text", text="# alpha")],
        isError=False,
    )
    for action in ("preview", "inspect", "show"):
        agent = _make_agent([manifest], read_result)
        outcome = await handle_skills_command(
            _ctx(agent), agent_name="default", action=action, argument="alpha"
        )
        text = _outcome_text(outcome)
        assert "Skill preview: alpha" in text, f"action {action} should preview"
