from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fast_agent.commands.context import CommandContext, StaticAgentProvider
from fast_agent.ui.interactive.command_context import build_command_context

if TYPE_CHECKING:
    from fast_agent.session.session_manager import SessionManager


class _LegacyProvider(StaticAgentProvider):
    def __init__(self) -> None:
        super().__init__({"main": object()})
        self._no_home_mode = True


class _Provider(StaticAgentProvider):
    no_home_mode = False


def test_build_command_context_reads_legacy_no_home_storage() -> None:
    context = build_command_context(cast("Any", _LegacyProvider()), "main")

    assert context.no_home is True
    assert context.current_agent_name == "main"
    assert context.sessions_enabled is False


def test_build_command_context_does_not_resolve_global_session_manager(
    monkeypatch: Any,
) -> None:
    def fail_get_session_manager(**kwargs: object) -> object:
        del kwargs
        raise AssertionError("global session manager should not be resolved")

    monkeypatch.setattr("fast_agent.session.get_session_manager", fail_get_session_manager)
    context = build_command_context(cast("Any", _Provider({"main": object()})), "main")

    assert context.sessions_enabled is False
    assert context.session_runtime is None


def test_build_command_context_uses_supplied_session_manager() -> None:
    manager = cast("SessionManager", object())
    context = build_command_context(
        cast("Any", _Provider({"main": object()})),
        "main",
        session_manager=manager,
    )

    assert context.resolve_session_manager() is manager


def test_no_home_context_rejects_session_capability() -> None:
    manager = cast("SessionManager", object())

    try:
        CommandContext(
            agent_provider=StaticAgentProvider({"main": object()}),
            current_agent_name="main",
            io=cast("Any", object()),
            no_home=True,
            session_manager=manager,
        )
    except ValueError as exc:
        assert str(exc) == "no_home command contexts cannot enable sessions."
    else:
        raise AssertionError("expected ValueError")
