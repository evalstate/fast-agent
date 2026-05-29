"""Tests for `/skills templates` and `/skills resolve` slash command handlers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest

from fast_agent.commands.handlers.skills import handle_skills_command
from fast_agent.mcp.mcp_skills_loader import SkillTemplateEntry
from fast_agent.skills.registry import SkillManifest


def _outcome_text(outcome) -> str:
    """Flatten messages into a single string for assertion convenience."""
    parts: list[str] = []
    for msg in outcome.messages:
        text = msg.text
        if hasattr(text, "plain"):
            parts.append(text.plain)
        else:
            parts.append(str(text))
    return "\n".join(parts)


class _FakeIO:
    """Minimal CommandIO stand-in: scripted prompt_selection / prompt_text."""

    def __init__(
        self,
        *,
        selections: Sequence[str | None] = (),
        texts: Sequence[str | None] = (),
    ) -> None:
        self._selections = list(selections)
        self._texts = list(texts)
        self.emitted: list[Any] = []

    async def emit(self, message) -> None:
        self.emitted.append(message)

    async def prompt_text(self, prompt, *, default=None, allow_empty=True):
        return self._texts.pop(0) if self._texts else None

    async def prompt_selection(self, prompt, *, options, allow_cancel=False, default=None):
        return self._selections.pop(0) if self._selections else None

    async def prompt_argument(self, *args, **kwargs):
        return None

    async def prompt_model_selection(self, *args, **kwargs):
        return None

    # The protocol has more methods, but the handlers under test only
    # exercise emit / prompt_selection / prompt_text.
    def __getattr__(self, name):
        # Anything else returns a no-op AsyncMock to keep duck typing happy.
        return AsyncMock()


def _ctx_with_agent(agent_obj, *, io: _FakeIO | None = None):
    """Build a CommandContext with an agent_provider exposing `agent_obj`."""
    provider = MagicMock()
    provider._agent = MagicMock(return_value=agent_obj)
    ctx = SimpleNamespace(
        agent_provider=provider,
        current_agent_name="default",
        io=io or _FakeIO(),
        settings=None,
    )
    ctx.resolve_settings = MagicMock(return_value=MagicMock())  # not used by template handlers
    return ctx


@pytest.mark.asyncio
async def test_templates_action_lists_discovered_entries() -> None:
    template = SkillTemplateEntry(
        server_name="github",
        url_template="skill://docs/{product}/SKILL.md",
        description="Per-product documentation skill",
    )
    agent = SimpleNamespace(
        skill_template_entries=[template],
        complete_skill_template_argument=AsyncMock(return_value=[]),
        register_resolved_skill_template=AsyncMock(),
    )
    ctx = _ctx_with_agent(agent)

    outcome = await handle_skills_command(
        ctx, agent_name="default", action="templates", argument=None
    )
    text = _outcome_text(outcome)
    assert "Skill templates" in text
    assert "skill://docs/{product}/SKILL.md" in text
    assert "github" in text  # provenance line
    assert "product" in text  # variable listing


@pytest.mark.asyncio
async def test_templates_action_with_no_templates() -> None:
    agent = SimpleNamespace(skill_template_entries=[])
    ctx = _ctx_with_agent(agent)

    outcome = await handle_skills_command(
        ctx, agent_name="default", action="templates", argument=None
    )
    text = _outcome_text(outcome)
    assert "No skill templates" in text


@pytest.mark.asyncio
async def test_resolve_action_walks_completion_and_registers() -> None:
    template = SkillTemplateEntry(
        server_name="github",
        url_template="skill://docs/{product}/SKILL.md",
        description="Per-product documentation skill",
    )
    resolved_manifest = SkillManifest(
        name="anvil",
        description="Anvil docs",
        body="body",
        path=None,
        uri="skill://docs/anvil/SKILL.md",
        server_name="github",
    )
    agent = SimpleNamespace(
        skill_template_entries=[template],
        complete_skill_template_argument=AsyncMock(return_value=["anvil", "hammer", "saw"]),
        register_resolved_skill_template=AsyncMock(return_value=resolved_manifest),
    )
    ctx = _ctx_with_agent(agent, io=_FakeIO(selections=["anvil"]))

    outcome = await handle_skills_command(
        ctx, agent_name="default", action="resolve", argument="1"
    )
    text = _outcome_text(outcome)
    assert "Registered skill: anvil" in text
    # complete_skill_template_argument should have been called for `product`.
    agent.complete_skill_template_argument.assert_awaited_once()
    args, kwargs = agent.complete_skill_template_argument.call_args
    # Could be called positionally or by keyword depending on style above.
    assert kwargs.get("argument_name", args[1] if len(args) > 1 else None) == "product"
    agent.register_resolved_skill_template.assert_awaited_once_with(
        template, {"product": "anvil"}
    )


@pytest.mark.asyncio
async def test_resolve_action_with_var_overrides_skips_completion() -> None:
    """`var=value` overrides bypass completion — needed for scripted runs
    and for variables whose space the server doesn't enumerate."""
    template = SkillTemplateEntry(
        server_name="github",
        url_template="skill://docs/{product}/SKILL.md",
        description="x",
    )
    resolved_manifest = SkillManifest(
        name="hammer",
        description="x",
        body="b",
        path=None,
        uri="skill://docs/hammer/SKILL.md",
        server_name="github",
    )
    completion_mock = AsyncMock(return_value=[])
    agent = SimpleNamespace(
        skill_template_entries=[template],
        complete_skill_template_argument=completion_mock,
        register_resolved_skill_template=AsyncMock(return_value=resolved_manifest),
    )
    ctx = _ctx_with_agent(agent, io=_FakeIO())

    outcome = await handle_skills_command(
        ctx, agent_name="default", action="resolve", argument="1 product=hammer"
    )
    text = _outcome_text(outcome)
    assert "Registered skill: hammer" in text
    completion_mock.assert_not_called()  # override skipped the completion roundtrip
    agent.register_resolved_skill_template.assert_awaited_once_with(
        template, {"product": "hammer"}
    )


@pytest.mark.asyncio
async def test_resolve_action_invalid_index() -> None:
    agent = SimpleNamespace(
        skill_template_entries=[
            SkillTemplateEntry(
                server_name="x",
                url_template="skill://{a}/SKILL.md",
                description="x",
            )
        ],
    )
    ctx = _ctx_with_agent(agent, io=_FakeIO())

    outcome = await handle_skills_command(
        ctx, agent_name="default", action="resolve", argument="99"
    )
    text = _outcome_text(outcome)
    assert "No template at index 99" in text


@pytest.mark.asyncio
async def test_resolve_action_no_templates() -> None:
    agent = SimpleNamespace(skill_template_entries=[])
    ctx = _ctx_with_agent(agent, io=_FakeIO())

    outcome = await handle_skills_command(
        ctx, agent_name="default", action="resolve", argument="1"
    )
    text = _outcome_text(outcome)
    assert "No skill templates available" in text


@pytest.mark.asyncio
async def test_resolve_action_user_cancels_selection() -> None:
    template = SkillTemplateEntry(
        server_name="github",
        url_template="skill://docs/{product}/SKILL.md",
        description="x",
    )
    agent = SimpleNamespace(
        skill_template_entries=[template],
        complete_skill_template_argument=AsyncMock(return_value=["anvil"]),
        register_resolved_skill_template=AsyncMock(),
    )
    ctx = _ctx_with_agent(agent, io=_FakeIO(selections=[None]))

    outcome = await handle_skills_command(
        ctx, agent_name="default", action="resolve", argument="1"
    )
    text = _outcome_text(outcome)
    assert "Resolution cancelled" in text
    agent.register_resolved_skill_template.assert_not_called()
