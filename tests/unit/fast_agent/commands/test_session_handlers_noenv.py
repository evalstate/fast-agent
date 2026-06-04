from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from mcp.types import TextContent

from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers import sessions as session_handlers
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.session import ResumeSessionAgentsResult


class _StubIO:
    def __init__(self) -> None:
        self.history_overviews: list[tuple[str, list[PromptMessageExtended], object | None]] = []

    async def emit(self, message):
        return None

    async def prompt_text(self, prompt: str, *, default=None, allow_empty=True):
        return default

    async def prompt_selection(
        self, prompt: str, *, options, allow_cancel=False, default=None
    ):
        return default

    async def prompt_model_selection(
        self,
        *,
        initial_provider=None,
        default_model=None,
    ):
        del initial_provider, default_model
        return None

    async def prompt_argument(self, arg_name: str, *, description=None, required=True):
        return None

    async def display_history_turn(self, agent_name: str, turn, *, turn_index=None, total_turns=None):
        return None

    async def display_history_overview(self, agent_name: str, history, usage=None):
        self.history_overviews.append((agent_name, list(history), usage))
        return None

    async def display_usage_report(self, agents):
        return None

    async def display_system_prompt(self, agent_name: str, system_prompt: str, *, server_count=0):
        return None


class _StubAgentProvider:
    def _agent(self, name: str):  # noqa: ARG002
        return object()

    def resolve_target_agent_name(self, agent_name: str | None = None):
        return agent_name or "agent"

    def visible_agent_names(self, *, force_include: str | None = None):
        del force_include
        return ["agent"]

    def registered_agent_names(self):
        return ["agent"]

    def registered_agents(self):
        return {"agent": object()}

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None):  # noqa: ARG002
        return {}


class _Agent:
    def __init__(self, name: str, *, history: list[PromptMessageExtended] | None = None) -> None:
        self.name = name
        self.message_history = list(history or [])
        self.usage_accumulator = None
        self.llm = None
        self.config = SimpleNamespace(model="passthrough")


class _ResumeAgentProvider:
    def __init__(self, agents: dict[str, _Agent]) -> None:
        self._agents = agents

    def _agent(self, name: str):
        return self._agents[name]

    def resolve_target_agent_name(self, agent_name: str | None = None):
        return agent_name or "alpha"

    def visible_agent_names(self, *, force_include: str | None = None):
        del force_include
        return list(self._agents)

    def registered_agent_names(self):
        return list(self._agents)

    def registered_agents(self):
        return dict(self._agents)

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None):
        del namespace, agent_name
        return {}


def _build_noenv_context() -> CommandContext:
    return CommandContext(
        agent_provider=_StubAgentProvider(),
        current_agent_name="agent",
        io=_StubIO(),
        noenv=True,
    )


def _build_context(*, session_cwd: Path | None = None) -> CommandContext:
    return CommandContext(
        agent_provider=_StubAgentProvider(),
        current_agent_name="agent",
        io=_StubIO(),
        session_cwd=session_cwd,
    )


def _assistant_message(text: str) -> PromptMessageExtended:
    return PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text=text)],
    )


@pytest.mark.asyncio
async def test_noenv_list_sessions_returns_disabled_message() -> None:
    outcome = await session_handlers.handle_list_sessions(
        _build_noenv_context(),
        show_help=True,
    )

    assert outcome.messages
    assert str(outcome.messages[0].text) == session_handlers.NOENV_SESSION_MESSAGE
    assert outcome.messages[0].channel == "warning"


@pytest.mark.asyncio
async def test_noenv_resume_session_returns_disabled_message() -> None:
    outcome = await session_handlers.handle_resume_session(
        _build_noenv_context(),
        agent_name="agent",
        session_id="latest",
    )

    assert outcome.messages
    assert str(outcome.messages[0].text) == session_handlers.NOENV_SESSION_MESSAGE
    assert outcome.messages[0].channel == "warning"


def test_strip_wrapping_quotes_removes_matching_outer_quotes() -> None:
    assert session_handlers._strip_wrapping_quotes('"quoted title"') == "quoted title"
    assert session_handlers._strip_wrapping_quotes("'quoted title'") == "quoted title"


def test_strip_wrapping_quotes_preserves_unmatched_quotes() -> None:
    assert session_handlers._strip_wrapping_quotes('"quoted title') == '"quoted title'
    assert session_handlers._strip_wrapping_quotes("plain title") == "plain title"


@pytest.mark.asyncio
async def test_create_session_uses_context_session_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager_calls: list[Path | None] = []

    class _Manager:
        def create_session(self, name: str | None = None):
            del name
            return SimpleNamespace(info=SimpleNamespace(metadata={}, name="s-1"))

    def fake_get_session_manager(
        *,
        cwd: Path | None = None,
        environment_override=None,
        respect_env_override: bool = True,
    ):
        del environment_override, respect_env_override
        manager_calls.append(cwd)
        return _Manager()

    monkeypatch.setattr("fast_agent.session.get_session_manager", fake_get_session_manager)

    outcome = await session_handlers.handle_create_session(
        _build_context(session_cwd=workspace.resolve()),
        session_name="Title",
    )

    assert outcome.messages
    assert manager_calls == [workspace.resolve()]


@pytest.mark.asyncio
async def test_resume_session_switches_to_hydrated_active_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alpha = _Agent("alpha", history=[_assistant_message("alpha preview")])
    beta = _Agent("beta", history=[_assistant_message("beta preview")])
    io = _StubIO()
    ctx = CommandContext(
        agent_provider=_ResumeAgentProvider({"alpha": alpha, "beta": beta}),
        current_agent_name="alpha",
        io=io,
    )
    session = SimpleNamespace(info=SimpleNamespace(name="s-1", metadata={}))
    async def _resume_session_agents_async(*args, **kwargs):
        del args, kwargs
        return ResumeSessionAgentsResult(
            session=cast("Any", session),
            loaded={"alpha": Path("history_alpha.json")},
            missing_agents=[],
            active_agent="beta",
        )

    manager = SimpleNamespace(resume_session_agents_async=_resume_session_agents_async)

    monkeypatch.setattr("fast_agent.session.get_session_manager", lambda **kwargs: manager)

    outcome = await session_handlers.handle_resume_session(
        ctx,
        agent_name="alpha",
        session_id="latest",
    )

    assert outcome.switch_agent == "beta"
    assert any("Switched to agent: beta" in str(message.text) for message in outcome.messages)
    assert io.history_overviews
    assert io.history_overviews[0][0] == "beta"
    assert any("beta preview" in str(message.text) for message in outcome.messages)
