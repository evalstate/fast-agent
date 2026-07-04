from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent.commands.handlers.sessions import NOENV_SESSION_MESSAGE
from fast_agent.ui.command_payloads import ExportSessionCommand
from fast_agent.ui.interactive import command_dispatch

if TYPE_CHECKING:
    from fast_agent.commands.results import CommandOutcome
    from fast_agent.core.agent_app import AgentApp


class _NoHomePromptProvider:
    no_home_mode = True

    def _agent(self, name: str) -> object:
        del name
        return object()

    def registered_agents(self) -> dict[str, object]:
        return {"agent": object()}


class _PromptProvider(_NoHomePromptProvider):
    no_home_mode = False


class _SessionManager:
    current_session = None


@pytest.mark.asyncio
async def test_no_home_session_export_dispatch_does_not_resolve_session_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[CommandOutcome] = []

    async def collect_outcome(_context: object, outcome: CommandOutcome) -> None:
        emitted.append(outcome)

    def fail_get_session_manager(**_kwargs: Any) -> object:
        raise AssertionError("session manager should not be resolved in --no-home mode")

    monkeypatch.setattr(command_dispatch, "emit_command_outcome", collect_outcome)
    monkeypatch.setattr("fast_agent.session.get_session_manager", fail_get_session_manager)

    result = await command_dispatch._dispatch_session_payload(
        ExportSessionCommand(
            target="latest",
            agent_name=None,
            output_path=None,
            hf_url=None,
            hf_dataset="evalstate/test-traces",
            hf_dataset_path=None,
            privacy_filter=False,
            privacy_filter_path=None,
            download_privacy_filter=False,
            privacy_filter_device=None,
            privacy_filter_variant=None,
            show_redactions=False,
            show_help=False,
            error=None,
        ),
        prompt_provider=cast("AgentApp", _NoHomePromptProvider()),
        agent="agent",
        session_manager=None,
    )

    assert result is not None
    assert result.handled is True
    assert emitted
    assert str(emitted[0].messages[0].text) == NOENV_SESSION_MESSAGE


@pytest.mark.asyncio
async def test_session_export_dispatch_uses_supplied_session_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[CommandOutcome] = []

    async def collect_outcome(_context: object, outcome: CommandOutcome) -> None:
        emitted.append(outcome)

    def fail_get_session_manager(**_kwargs: Any) -> object:
        raise AssertionError("global session manager should not be resolved")

    monkeypatch.setattr(command_dispatch, "emit_command_outcome", collect_outcome)
    monkeypatch.setattr("fast_agent.session.get_session_manager", fail_get_session_manager)

    result = await command_dispatch._dispatch_session_payload(
        ExportSessionCommand(
            target=None,
            agent_name=None,
            output_path=None,
            hf_url=None,
            hf_dataset="evalstate/test-traces",
            hf_dataset_path=None,
            privacy_filter=False,
            privacy_filter_path=None,
            download_privacy_filter=False,
            privacy_filter_device=None,
            privacy_filter_variant=None,
            show_redactions=False,
            show_help=False,
            error=None,
        ),
        prompt_provider=cast("AgentApp", _PromptProvider()),
        agent="agent",
        session_manager=cast("Any", _SessionManager()),
    )

    assert result is not None
    assert result.handled is True
    assert emitted
    assert str(emitted[0].messages[0].text) == "No active session to export."
