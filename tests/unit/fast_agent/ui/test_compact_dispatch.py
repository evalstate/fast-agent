from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import pytest

from fast_agent.commands.results import CommandOutcome
from fast_agent.ui.command_payloads import CompactCommand
from fast_agent.ui.interactive import command_dispatch

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp


class _Provider:
    def _agent(self, name: str) -> object:
        del name
        return object()


class _FakeProgressDisplay:
    def __init__(self) -> None:
        self.events: list[str] = []

    def resume(self, *, debounce_seconds: float = 0.0) -> None:
        del debounce_seconds
        self.events.append("resume")

    def pause(self, *, cancel_deferred_on_noop: bool = False) -> None:
        del cancel_deferred_on_noop
        self.events.append("pause")


@pytest.fixture
def patched_dispatch(monkeypatch: pytest.MonkeyPatch) -> _FakeProgressDisplay:
    progress = _FakeProgressDisplay()
    monkeypatch.setattr(command_dispatch, "progress_display", progress)
    monkeypatch.setattr(
        command_dispatch,
        "build_command_context",
        lambda provider, agent, **_kwargs: object(),
    )

    async def _noop_emit(_context: object, _outcome: CommandOutcome) -> None:
        return None

    monkeypatch.setattr(command_dispatch, "emit_command_outcome", _noop_emit)
    return progress


@pytest.mark.asyncio
async def test_compact_run_wraps_model_call_in_progress_display(
    monkeypatch: pytest.MonkeyPatch,
    patched_dispatch: _FakeProgressDisplay,
) -> None:
    progress = patched_dispatch
    captured: dict[str, Any] = {}

    async def fake_run(context: object, *, agent_name: str, instructions: str | None):
        # The display must already be resumed when the model-calling handler runs.
        captured["events_at_handler"] = list(progress.events)
        captured["instructions"] = instructions
        return CommandOutcome()

    monkeypatch.setattr(command_dispatch.compact_handlers, "handle_compact", fake_run)

    result = await command_dispatch._dispatch_compact_payload(
        CompactCommand(action="run", instructions="focus on X"),
        prompt_provider=cast("AgentApp", _Provider()),
        agent="default",
        session_manager=None,
    )

    assert result is not None and result.handled is True
    assert captured["instructions"] == "focus on X"
    # Resumed before the handler, paused after.
    assert captured["events_at_handler"] == ["resume"]
    assert progress.events == ["resume", "pause"]


@pytest.mark.asyncio
async def test_compact_run_pauses_display_even_on_error(
    monkeypatch: pytest.MonkeyPatch,
    patched_dispatch: _FakeProgressDisplay,
) -> None:
    progress = patched_dispatch

    async def boom(context: object, *, agent_name: str, instructions: str | None):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(command_dispatch.compact_handlers, "handle_compact", boom)

    with pytest.raises(RuntimeError, match="model exploded"):
        await command_dispatch._dispatch_compact_payload(
            CompactCommand(action="run"),
            prompt_provider=cast("AgentApp", _Provider()),
            agent="default",
            session_manager=None,
        )

    # The display is restored to paused even when the handler raises.
    assert progress.events == ["resume", "pause"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "handler_name"),
    [("preview", "handle_compact_preview"), ("prompt", "handle_compact_prompt")],
)
async def test_compact_preview_and_prompt_do_not_touch_progress_display(
    monkeypatch: pytest.MonkeyPatch,
    patched_dispatch: _FakeProgressDisplay,
    action: Literal["preview", "prompt"],
    handler_name: str,
) -> None:
    progress = patched_dispatch
    called = False

    async def fake_handler(context: object, *, agent_name: str):
        nonlocal called
        called = True
        return CommandOutcome()

    monkeypatch.setattr(command_dispatch.compact_handlers, handler_name, fake_handler)

    result = await command_dispatch._dispatch_compact_payload(
        CompactCommand(action=action),
        prompt_provider=cast("AgentApp", _Provider()),
        agent="default",
        session_manager=None,
    )

    assert result is not None and result.handled is True
    assert called is True
    # No model call -> no streaming display management.
    assert progress.events == []


@pytest.mark.asyncio
async def test_compact_dispatch_ignores_non_compact_payload(
    patched_dispatch: _FakeProgressDisplay,
) -> None:
    from fast_agent.ui.command_payloads import ShowUsageCommand

    result = await command_dispatch._dispatch_compact_payload(
        ShowUsageCommand(),
        prompt_provider=cast("AgentApp", _Provider()),
        agent="default",
        session_manager=None,
    )
    assert result is None
