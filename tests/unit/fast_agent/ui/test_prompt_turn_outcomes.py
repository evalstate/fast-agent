from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui.interactive_prompt import InteractivePrompt, PromptLoopRuntimeState

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stop,response,expected",
    [
        (LlmStopReason.END_TURN, "done", "D;0"),
        (LlmStopReason.ERROR, "provider error", "D;1"),
        (LlmStopReason.TIMEOUT, "timed out", "D;1"),
        (LlmStopReason.CANCELLED, "cancelled", "D;130"),
        (LlmStopReason.MAX_ITERATIONS, "budget reached", "D;0"),
        (None, "▲ **System Error:** retries exhausted", "D;1"),
        (LlmStopReason.CANCELLED, "▲ **System Error:** retries exhausted", "D;1"),
    ],
)
async def test_regular_turn_outcome_marks(monkeypatch, stop, response, expected) -> None:
    marks = []
    monkeypatch.setattr("fast_agent.ui.interactive_prompt.emit_prompt_mark", marks.append)
    prompt = InteractivePrompt()
    monkeypatch.setattr(prompt, "_last_assistant_stop_reason", lambda **kwargs: stop)

    async def send(*args):
        return response

    await prompt._send_regular_message(
        send_func=send,
        prompt_payload="test",
        agent_name="agent",
        prompt_provider=cast("AgentApp", object()),
        runtime_state=PromptLoopRuntimeState(),
    )
    assert marks == ["C", expected]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,expected",
    [
        (RuntimeError("failed"), "D;1"),
        (KeyboardInterrupt(), "D;130"),
        (asyncio.CancelledError(), "D;130"),
    ],
)
async def test_regular_turn_exceptions_close_with_outcome(monkeypatch, error, expected) -> None:
    marks = []
    monkeypatch.setattr("fast_agent.ui.interactive_prompt.emit_prompt_mark", marks.append)

    async def send(*args):
        raise error

    async def run():
        return await InteractivePrompt()._send_regular_message(
            send_func=send,
            prompt_payload="test",
            agent_name="agent",
            prompt_provider=cast("AgentApp", object()),
            runtime_state=PromptLoopRuntimeState(),
        )

    if isinstance(error, RuntimeError):
        with pytest.raises(RuntimeError):
            await run()
    else:
        await run()
    assert marks == ["C", expected]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cancelled,failed,response,expected",
    [
        (False, False, "done", "D;0"),
        (True, False, "cancelled", "D;130"),
        (False, True, "failed", "D;1"),
        (False, False, "▲ **System Error:** retries exhausted", "D;1"),
        (True, False, "▲ **System Error:** retries exhausted", "D;1"),
    ],
)
async def test_hash_turn_outcomes(monkeypatch, cancelled, failed, response, expected) -> None:
    marks = []
    monkeypatch.setattr("fast_agent.ui.interactive_prompt.emit_prompt_mark", marks.append)

    async def send(*args):
        return response

    await InteractivePrompt()._execute_hash_send(
        send_func=send,
        target_agent="helper",
        message="test",
        quiet=True,
        clear_progress_for_agent=lambda agent: None,
        clear_ctrl_c_interrupt=lambda: None,
        handle_inflight_cancel=lambda: None,
        last_assistant_message_cancelled=lambda agent: cancelled,
        last_assistant_message_failed=lambda agent: failed,
    )
    assert marks == ["C", expected]


@pytest.mark.asyncio
async def test_hash_exception_reports_error(monkeypatch) -> None:
    marks = []
    monkeypatch.setattr("fast_agent.ui.interactive_prompt.emit_prompt_mark", marks.append)

    async def send(*args):
        raise RuntimeError("provider failed")

    await InteractivePrompt()._execute_hash_send(
        send_func=send,
        target_agent="helper",
        message="test",
        quiet=True,
        clear_progress_for_agent=lambda agent: None,
        clear_ctrl_c_interrupt=lambda: None,
        handle_inflight_cancel=lambda: None,
        last_assistant_message_cancelled=lambda agent: False,
    )
    assert marks == ["C", "D;1"]


@pytest.mark.asyncio
async def test_task_cancellation_is_not_reported_as_error(monkeypatch) -> None:
    marks = []
    monkeypatch.setattr("fast_agent.ui.interactive_prompt.emit_prompt_mark", marks.append)
    started = asyncio.Event()

    async def send(*args):
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    task = asyncio.create_task(
        InteractivePrompt()._send_regular_message(
            send_func=send,
            prompt_payload="test",
            agent_name="agent",
            prompt_provider=cast("AgentApp", object()),
            runtime_state=PromptLoopRuntimeState(),
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert marks == ["C", "D;130"]
