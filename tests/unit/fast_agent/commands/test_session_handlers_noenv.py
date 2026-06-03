from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from mcp.types import TextContent

from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers import sessions as session_handlers
from fast_agent.commands.results import CommandOutcome
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.session import ResumeSessionAgentsResult, Session

if TYPE_CHECKING:
    from fast_agent.session.session_manager import SessionManager


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


class _PinSessionManager:
    def __init__(
        self,
        *,
        current_session: object | None = None,
        listed_names: list[str] | None = None,
        sessions: dict[str, object] | None = None,
        resolved_names: dict[str, str | None] | None = None,
    ) -> None:
        self.current_session = current_session
        self._listed_names = list(listed_names or [])
        self._sessions = dict(sessions or {})
        self._resolved_names = dict(resolved_names or {})

    def resolve_session_name(self, name: str | None) -> str | None:
        if name is None:
            return None
        return self._resolved_names.get(name, name)

    def get_session(self, name: str) -> object | None:
        return self._sessions.get(name)

    def list_sessions(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=name) for name in self._listed_names]


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
    assert session_handlers._strip_wrapping_quotes('"  quoted title  "') == "quoted title"
    assert session_handlers._strip_wrapping_quotes('"   "') is None


def test_strip_wrapping_quotes_preserves_unmatched_quotes() -> None:
    assert session_handlers._strip_wrapping_quotes('"quoted title') == '"quoted title'
    assert session_handlers._strip_wrapping_quotes("plain title") == "plain title"


def test_resolve_pin_state_toggles_current_state() -> None:
    assert session_handlers._resolve_pin_state(None, current=False).desired is True
    assert session_handlers._resolve_pin_state("  ", current=False).desired is True
    assert session_handlers._resolve_pin_state("toggle", current=True).desired is False
    assert session_handlers._resolve_pin_state(" TOGGLE ", current=True).desired is False


@pytest.mark.parametrize("value", ["on", "true", "yes", "enable", "enabled"])
def test_resolve_pin_state_accepts_on_aliases(value: str) -> None:
    assert session_handlers._resolve_pin_state(value, current=False).desired is True


@pytest.mark.parametrize("value", ["off", "false", "no", "disable", "disabled"])
def test_resolve_pin_state_accepts_off_aliases(value: str) -> None:
    assert session_handlers._resolve_pin_state(value, current=True).desired is False


@pytest.mark.parametrize("value", ["1", "0"])
def test_resolve_pin_state_rejects_numeric_aliases(value: str) -> None:
    result = session_handlers._resolve_pin_state(value, current=False)

    assert result.desired is None
    assert result.error == "Usage: /session pin [on|off|id|number]"


def test_resolve_pin_state_reports_invalid_value() -> None:
    result = session_handlers._resolve_pin_state("maybe", current=False)

    assert result.desired is None
    assert result.error == "Usage: /session pin [on|off|id|number]"


def test_session_for_pin_prefers_explicit_target() -> None:
    session = SimpleNamespace(info=SimpleNamespace(name="target", metadata={}))
    manager = _PinSessionManager(
        current_session=object(),
        sessions={"resolved-target": session},
        resolved_names={"target": "resolved-target"},
    )
    outcome = CommandOutcome()

    resolved = session_handlers._session_for_pin(
        cast("SessionManager", manager),
        outcome,
        target="target",
    )

    assert resolved is cast("Session", session)
    assert outcome.messages == []


def test_session_for_pin_trims_explicit_target() -> None:
    session = SimpleNamespace(info=SimpleNamespace(name="target", metadata={}))
    manager = _PinSessionManager(
        current_session=object(),
        sessions={"resolved-target": session},
        resolved_names={"target": "resolved-target"},
    )
    outcome = CommandOutcome()

    resolved = session_handlers._session_for_pin(
        cast("SessionManager", manager),
        outcome,
        target=" target ",
    )

    assert resolved is cast("Session", session)
    assert outcome.messages == []


def test_session_for_pin_falls_back_to_first_listed_session() -> None:
    session = SimpleNamespace(info=SimpleNamespace(name="first", metadata={}))
    manager = _PinSessionManager(
        listed_names=["first"],
        sessions={"first": session},
    )
    outcome = CommandOutcome()

    resolved = session_handlers._session_for_pin(
        cast("SessionManager", manager),
        outcome,
        target=None,
    )

    assert resolved is cast("Session", session)
    assert outcome.messages == []


def test_session_for_pin_reports_missing_session() -> None:
    manager = _PinSessionManager()
    outcome = CommandOutcome()

    resolved = session_handlers._session_for_pin(
        cast("SessionManager", manager),
        outcome,
        target=None,
    )

    assert resolved is None
    assert [str(message.text) for message in outcome.messages] == [
        "No session available to pin."
    ]


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
async def test_create_session_can_replace_explicit_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None, dict[str, str] | None]] = []

    class _Manager:
        def delete_session(self, name: str) -> bool:
            calls.append(("delete", name, None))
            return True

        def create_session_with_id(self, session_id: str, metadata: dict | None = None):
            calls.append(("create_with_id", session_id, metadata))
            return SimpleNamespace(
                info=SimpleNamespace(metadata=metadata or {}, name=session_id)
            )

    monkeypatch.setattr("fast_agent.session.get_session_manager", lambda **kwargs: _Manager())

    outcome = await session_handlers.handle_create_session(
        _build_context(),
        session_name="Fresh start",
        session_id="acp-session-1",
        replace_existing=True,
    )

    assert calls == [
        ("delete", "acp-session-1", None),
        ("create_with_id", "acp-session-1", {"title": "Fresh start"}),
    ]
    assert outcome.messages
    assert "Created session: Fresh start" in str(outcome.messages[0].text)


@pytest.mark.asyncio
async def test_clear_sessions_all_pluralizes_deleted_count(monkeypatch: pytest.MonkeyPatch) -> None:
    deleted: list[str] = []

    class _Manager:
        def list_sessions(self):
            return [
                SimpleNamespace(name="session-1"),
                SimpleNamespace(name="session-2"),
            ]

        def delete_session(self, name: str) -> bool:
            deleted.append(name)
            return True

    monkeypatch.setattr("fast_agent.session.get_session_manager", lambda **kwargs: _Manager())

    outcome = await session_handlers.handle_clear_sessions(
        _build_context(),
        target="all",
    )

    assert deleted == ["session-1", "session-2"]
    assert [str(message.text) for message in outcome.messages] == ["Deleted 2 sessions."]


@pytest.mark.asyncio
async def test_clear_sessions_trims_all_target(monkeypatch: pytest.MonkeyPatch) -> None:
    deleted: list[str] = []

    class _Manager:
        def list_sessions(self):
            return [
                SimpleNamespace(name="session-1"),
                SimpleNamespace(name="session-2"),
            ]

        def delete_session(self, name: str) -> bool:
            deleted.append(name)
            return True

    monkeypatch.setattr("fast_agent.session.get_session_manager", lambda **kwargs: _Manager())

    outcome = await session_handlers.handle_clear_sessions(
        _build_context(),
        target=" all ",
    )

    assert deleted == ["session-1", "session-2"]
    assert [str(message.text) for message in outcome.messages] == ["Deleted 2 sessions."]


@pytest.mark.asyncio
async def test_clear_sessions_normalizes_all_target_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted: list[str] = []

    class _Manager:
        def list_sessions(self):
            return [
                SimpleNamespace(name="session-1"),
                SimpleNamespace(name="session-2"),
            ]

        def delete_session(self, name: str) -> bool:
            deleted.append(name)
            return True

    monkeypatch.setattr("fast_agent.session.get_session_manager", lambda **kwargs: _Manager())

    outcome = await session_handlers.handle_clear_sessions(
        _build_context(),
        target="ALL",
    )

    assert deleted == ["session-1", "session-2"]
    assert [str(message.text) for message in outcome.messages] == ["Deleted 2 sessions."]


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
