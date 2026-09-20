import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from fast_agent.cli.runtime import copilot_activation as activation
from fast_agent.commands.context import AgentProvider
from fast_agent.llm.provider_types import Provider
from fast_agent.ui import model_picker
from fast_agent.ui.adapters.tui_io import TuiCommandIO
from fast_agent.ui.model_picker import ModelPickerResult
from fast_agent.ui.model_picker_common import ProviderActivation


def activation_selection(provider: Provider, spec: str) -> ModelPickerResult:
    return ModelPickerResult(
        provider=provider.config_name,
        provider_available=False,
        selected_model=spec,
        resolved_model=spec,
        source="curated",
        refer_to_docs=False,
        activation_action=ProviderActivation(provider),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_succeeds", [False, True])
async def test_failed_copilot_activation_reopens_picker_before_selecting(
    monkeypatch: pytest.MonkeyPatch, retry_succeeds: bool
) -> None:
    io = TuiCommandIO(Mock(spec=AgentProvider), "alpha", config_payload={})
    activate = AsyncMock(side_effect=[False, True])
    monkeypatch.setattr(activation, "activate_copilot", activate)
    original_spec = "generic.llama3.2"
    failed_spec = "copilot.gpt-4.1"
    successful_spec = "copilot.gpt-5-mini"
    picker_calls = 0

    async def pick(**kwargs: object) -> ModelPickerResult | None:
        nonlocal picker_calls
        picker_calls += 1
        # Reopening must not promote the failed selection to the active/default spec.
        assert kwargs["initial_model_spec"] == original_spec
        if picker_calls == 1:
            return activation_selection(Provider.COPILOT, failed_spec)
        assert picker_calls == 2
        assert kwargs["initial_provider"] == Provider.COPILOT.config_name
        activate.assert_awaited_once()
        if retry_succeeds:
            return activation_selection(Provider.COPILOT, successful_spec)
        return None

    monkeypatch.setattr(model_picker, "run_model_picker_async", pick)

    selected = await io.prompt_model_selection(default_model=original_spec)

    assert picker_calls == 2
    assert selected == (successful_spec if retry_succeeds else None)
    assert activate.await_count == (2 if retry_succeeds else 1)


@pytest.mark.asyncio
async def test_other_provider_activation_failure_still_exits_picker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    io = TuiCommandIO(Mock(spec=AgentProvider), "alpha", config_payload={})
    picker = AsyncMock(
        return_value=activation_selection(Provider.CODEX_RESPONSES, "codexresponses.gpt-5")
    )
    monkeypatch.setattr(model_picker, "run_model_picker_async", picker)
    monkeypatch.setattr(TuiCommandIO, "_handle_model_activation", AsyncMock(return_value=False))

    assert await io.prompt_model_selection() is None
    picker.assert_awaited_once()


@pytest.mark.asyncio
async def test_tui_task_cancellation_stops_directly_awaited_copilot_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    io = TuiCommandIO(Mock(spec=AgentProvider), "alpha", config_payload={})
    picker = AsyncMock(return_value=activation_selection(Provider.COPILOT, "copilot.gpt-4.1"))
    monkeypatch.setattr(model_picker, "run_model_picker_async", picker)
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    broker = Mock(has_credentials=AsyncMock(return_value=False))
    monkeypatch.setattr(activation, "CopilotBroker", Mock(return_value=broker))
    monkeypatch.setattr(activation.typer, "confirm", Mock(return_value=True))
    monkeypatch.setattr("fast_agent.ui.console.ensure_blocking_console", Mock())
    started = asyncio.Event()
    stopped = asyncio.Event()
    saved = Mock()
    polling_tasks: list[asyncio.Task | None] = []

    async def poll() -> None:
        polling_tasks.append(asyncio.current_task())
        started.set()
        try:
            await asyncio.Event().wait()
            saved()
        finally:
            stopped.set()

    monkeypatch.setattr(activation, "login_copilot_oauth_async", poll)
    selection = asyncio.create_task(io.prompt_model_selection())
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        assert polling_tasks == [selection]
    finally:
        selection.cancel()
        with pytest.raises(asyncio.CancelledError):
            await selection

    assert stopped.is_set()
    saved.assert_not_called()
    picker.assert_awaited_once()
    broker.has_credentials.assert_awaited_once()
