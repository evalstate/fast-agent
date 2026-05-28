from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent.commands.handlers.sessions import NOENV_SESSION_MESSAGE
from fast_agent.ui.command_payloads import ExportSessionCommand
from fast_agent.ui.interactive import command_dispatch

if TYPE_CHECKING:
    from fast_agent.commands.results import CommandOutcome
    from fast_agent.core.agent_app import AgentApp


class _NoenvPromptProvider:
    noenv_mode = True

    def _agent(self, name: str) -> object:
        del name
        return object()

    def registered_agents(self) -> dict[str, object]:
        return {"agent": object()}


@pytest.mark.asyncio
async def test_noenv_session_export_dispatch_does_not_resolve_session_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[CommandOutcome] = []

    async def collect_outcome(_context: object, outcome: CommandOutcome) -> None:
        emitted.append(outcome)

    def fail_get_session_manager(**_kwargs: Any) -> object:
        raise AssertionError("session manager should not be resolved in --noenv mode")

    monkeypatch.setattr(command_dispatch, "emit_command_outcome", collect_outcome)
    monkeypatch.setattr("fast_agent.session.get_session_manager", fail_get_session_manager)

    result = await command_dispatch._dispatch_session_payload(
        ExportSessionCommand(
            target="latest",
            agent_name=None,
            output_path=None,
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
        prompt_provider=cast("AgentApp", _NoenvPromptProvider()),
        agent="agent",
    )

    assert result is not None
    assert result.handled is True
    assert emitted
    assert str(emitted[0].messages[0].text) == NOENV_SESSION_MESSAGE
