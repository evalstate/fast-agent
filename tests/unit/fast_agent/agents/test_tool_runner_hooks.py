import asyncio

import pytest
from mcp import CallToolRequest
from mcp.types import CallToolRequestParams, ContentBlock, ImageContent, Tool
from rich.text import Text

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.constants import FAST_AGENT_PENDING_MEDIA_ATTACHMENTS
from fast_agent.core.prompt import Prompt
from fast_agent.hooks import show_hook_message
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.model_info import ModelInfo
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.tool_execution_handler import NoOpToolExecutionHandler
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui.console_display import ConsoleDisplay


def tool_one() -> int:
    return 1


def tool_two() -> int:
    return 2


def stage_media() -> str:
    return "staged"


class TwoStepToolUseLlm(PassthroughLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls: list[list[str]] = []
        self._turn = 0

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        self._turn += 1
        self.calls.append(
            [
                get_text(block) or ""
                for msg in multipart_messages
                for block in (msg.content or [])
                if get_text(block)
            ]
        )

        if self._turn == 1:
            tool_calls = {
                "id_one": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="tool_one", arguments={}),
                ),
                "id_two": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="tool_two", arguments={}),
                ),
            }
            return Prompt.assistant(
                "use tools",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls=tool_calls,
            )

        return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


class HookedToolAgent(ToolAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events: list[str] = []
        self._injected = False

    def _tool_runner_hooks(self) -> ToolRunnerHooks | None:
        async def before_llm_call(runner, messages):
            self.events.append(f"before_llm_call:{runner.iteration}")
            if not self._injected:
                runner.append_messages("extra from hook")
                self._injected = True

        async def after_llm_call(runner, message):
            self.events.append(f"after_llm_call:{message.stop_reason}")

        async def before_tool_call(runner, message):
            self.events.append(f"before_tool_call:{len(message.tool_calls or {})}")

        async def after_tool_call(runner, message):
            self.events.append(f"after_tool_call:{len(message.tool_results or {})}")

        return ToolRunnerHooks(
            before_llm_call=before_llm_call,
            after_llm_call=after_llm_call,
            before_tool_call=before_tool_call,
            after_tool_call=after_tool_call,
        )


class MediaStagingLlm(PassthroughLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls: list[list[tuple[str, list[str], list[str]]]] = []
        self._turn = 0

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name="media-test",
            provider=Provider.GENERIC,
            context_window=None,
            max_output_tokens=None,
            tokenizes=["image/png"],
            json_mode=None,
            reasoning=None,
        )

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        self._turn += 1
        self.calls.append(
            [
                (
                    msg.role,
                    [block.type for block in (msg.content or [])],
                    sorted((msg.channels or {}).keys()),
                )
                for msg in multipart_messages
            ]
        )
        if self._turn == 1:
            return Prompt.assistant(
                "use tool",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "id_stage_media": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(name="stage_media", arguments={}),
                    )
                },
            )
        return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


class MediaStagingToolAgent(ToolAgent):
    def _consume_pending_media_attachments(self) -> list[ContentBlock]:
        return [ImageContent(type="image", data="abcd", mimeType="image/png")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_runner_hooks_fire_and_can_inject_messages():
    llm = TwoStepToolUseLlm()
    agent = HookedToolAgent(AgentConfig("hooked"), [tool_one, tool_two])
    agent._llm = llm

    result = await agent.generate("hi")
    assert result.last_text() == "done"

    assert any("extra from hook" in entry for entry in llm.calls[0])

    assert agent.events == [
        "before_llm_call:0",
        f"after_llm_call:{LlmStopReason.TOOL_USE}",
        "before_tool_call:2",
        "after_tool_call:2",
        "before_llm_call:1",
        f"after_llm_call:{LlmStopReason.END_TURN}",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_runner_stages_pending_media_as_followup_user_message():
    llm = MediaStagingLlm()
    agent = MediaStagingToolAgent(AgentConfig("media"), [stage_media])
    agent._llm = llm

    result = await agent.generate("hi")

    assert result.last_text() == "done"
    assert len(llm.calls) == 2
    second_call = llm.calls[1]
    assert all(FAST_AGENT_PENDING_MEDIA_ATTACHMENTS not in channels for _, _, channels in second_call)
    assert second_call[-1][0] == "user"
    assert second_call[-1][1] == ["image"]


# Track tool invocations globally for the regression test
_tool_invocations: list[str] = []


def tracked_tool_a() -> str:
    _tool_invocations.append("tool_a")
    return "result_a"


def tracked_tool_b() -> str:
    _tool_invocations.append("tool_b")
    return "result_b"


class TwoRoundToolUseLlm(PassthroughLLM):
    """LLM that returns tool_use twice before completing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._turn = 0

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        self._turn += 1

        if self._turn == 1:
            # First round: call tool_a
            return Prompt.assistant(
                "calling tool_a",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "call_1": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(name="tracked_tool_a", arguments={}),
                    ),
                },
            )

        if self._turn == 2:
            # Second round: call tool_b
            return Prompt.assistant(
                "calling tool_b",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "call_2": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(name="tracked_tool_b", arguments={}),
                    ),
                },
            )

        return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_two_tool_use_rounds_both_execute():
    """Regression test: ensure second tool-use round executes new tools, not cached response."""
    _tool_invocations.clear()

    llm = TwoRoundToolUseLlm()
    agent = ToolAgent(AgentConfig("test"), [tracked_tool_a, tracked_tool_b])
    agent._llm = llm

    result = await agent.generate("hi")
    assert result.last_text() == "done"

    # Both tools must have been called - if caching bug exists, only tool_a would be called
    assert _tool_invocations == ["tool_a", "tool_b"], (
        f"Expected both tools to execute, got: {_tool_invocations}"
    )


# Tests for after_turn_complete hook
_after_turn_complete_calls: list[tuple[int, str | None]] = []


class AfterTurnCompleteToolAgent(ToolAgent):
    """Agent that tracks after_turn_complete hook calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _tool_runner_hooks(self) -> ToolRunnerHooks | None:
        async def after_turn_complete(runner, message):
            _after_turn_complete_calls.append(
                (runner.iteration, message.stop_reason.value if message.stop_reason else None)
            )

        return ToolRunnerHooks(after_turn_complete=after_turn_complete)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_after_turn_complete_hook_fires():
    """Test that after_turn_complete hook is called once after tool loop completes."""
    _after_turn_complete_calls.clear()

    llm = TwoRoundToolUseLlm()
    agent = AfterTurnCompleteToolAgent(AgentConfig("test"), [tracked_tool_a, tracked_tool_b])
    agent._llm = llm

    result = await agent.generate("hi")
    assert result.last_text() == "done"

    # Hook should be called exactly once, after all iterations complete
    assert len(_after_turn_complete_calls) == 1, (
        f"Expected 1 after_turn_complete call, got {len(_after_turn_complete_calls)}"
    )

    # Should be called with final iteration count and END_TURN stop reason
    iteration, stop_reason = _after_turn_complete_calls[0]
    assert iteration == 2, f"Expected iteration 2, got {iteration}"
    assert stop_reason == LlmStopReason.END_TURN.value


@pytest.mark.unit
@pytest.mark.asyncio
async def test_after_turn_complete_receives_final_message():
    """Test that after_turn_complete hook receives the final response message."""
    captured_messages: list[PromptMessageExtended] = []

    class CaptureAgent(ToolAgent):
        def _tool_runner_hooks(self) -> ToolRunnerHooks | None:
            async def after_turn_complete(runner, message):
                captured_messages.append(message)

            return ToolRunnerHooks(after_turn_complete=after_turn_complete)

    llm = TwoRoundToolUseLlm()
    agent = CaptureAgent(AgentConfig("test"), [tracked_tool_a, tracked_tool_b])
    agent._llm = llm

    await agent.generate("hi")

    assert len(captured_messages) == 1
    msg = captured_messages[0]
    assert msg.role == "assistant"
    assert msg.stop_reason == LlmStopReason.END_TURN
    # Verify it's the final "done" message, not an intermediate tool call
    from fast_agent.mcp.helpers.content_helpers import get_text

    text = get_text(msg.content[0]) if msg.content else ""
    assert text == "done"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_after_turn_complete_with_loop_progress_hooks():
    """Ensure after_turn_complete survives merge with loop progress hooks."""
    captured: list[tuple[int, LlmStopReason | None]] = []

    class ProgressHookAgent(ToolAgent):
        def _tool_runner_hooks(self) -> ToolRunnerHooks | None:
            async def after_turn_complete(runner, message):
                captured.append((runner.iteration, message.stop_reason))

            return ToolRunnerHooks(after_turn_complete=after_turn_complete)

    llm = TwoRoundToolUseLlm()
    agent = ProgressHookAgent(AgentConfig("progress-test"), [tracked_tool_a, tracked_tool_b])
    agent._llm = llm

    request_params = RequestParams(
        emit_loop_progress=True,
        tool_execution_handler=NoOpToolExecutionHandler(),
    )
    await agent.generate("hi", request_params=request_params)

    assert captured == [(2, LlmStopReason.END_TURN)]


class FailingBeforeToolHookAgent(ToolAgent):
    def _tool_runner_hooks(self) -> ToolRunnerHooks | None:
        async def before_tool_call(runner, message):
            raise RuntimeError("hook boom")

        return ToolRunnerHooks(before_tool_call=before_tool_call)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_hook_error_returns_tool_result():
    llm = TwoStepToolUseLlm()
    agent = FailingBeforeToolHookAgent(AgentConfig("hook-error"), [tool_one, tool_two])
    agent._llm = llm

    result = await agent.generate("hi")
    assert result.stop_reason == LlmStopReason.ERROR


# ─── on_pause_cancel hook — instant LLM pause via task cancel ──────────


class CancelOnceLlm(PassthroughLLM):
    """LLM that raises CancelledError on the first call, then succeeds.

    Simulates PauseController calling ``task.cancel()`` while an LLM
    stream is in flight. The on_pause_cancel hook should intercept,
    say "retry", and the second call returns a normal END_TURN.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calls = 0

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages,
        request_params=None,
        tools=None,
        is_template=False,
    ):
        self._calls += 1
        if self._calls == 1:
            import asyncio as _asyncio
            raise _asyncio.CancelledError()
        return Prompt.assistant("done after retry", stop_reason=LlmStopReason.END_TURN)


class CancelRetryHookAgent(ToolAgent):
    """Agent that retries the LLM call once via on_pause_cancel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retry_count = 0

    def _tool_runner_hooks(self):
        async def on_pause_cancel(runner):
            # First call: simulate "we were paused", request retry.
            # Second call (if any): "genuine cancel", propagate.
            self.retry_count += 1
            return self.retry_count == 1

        return ToolRunnerHooks(on_pause_cancel=on_pause_cancel)


class OneTurnLlm(PassthroughLLM):
    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        del multipart_messages, request_params, tools, is_template
        return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


class CapturingConsoleDisplay(ConsoleDisplay):
    def __init__(self) -> None:
        super().__init__()
        self.status_messages: list[str] = []

    def show_status_message(self, content: Text) -> None:
        self.status_messages.append(content.plain)


class HookMessageDeferringAgent(ToolAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.display = CapturingConsoleDisplay()
        self.status_counts_during_hook: list[int] = []

    def _should_stream(self) -> bool:
        return False

    def _display_user_messages(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
    ) -> None:
        del messages, request_params

    async def show_assistant_message(
        self,
        message: PromptMessageExtended,
        bottom_items=None,
        highlight_items=None,
        max_item_length=None,
        name=None,
        model=None,
        additional_message=None,
        render_markdown: bool | None = None,
        show_hook_indicator: bool | None = None,
        render_message: bool = True,
        show_reprint_banner: bool = False,
    ) -> None:
        del (
            message,
            bottom_items,
            highlight_items,
            max_item_length,
            name,
            model,
            additional_message,
            render_markdown,
            show_hook_indicator,
            render_message,
            show_reprint_banner,
        )

    def _tool_runner_hooks(self) -> ToolRunnerHooks | None:
        async def after_llm_call(runner, message):
            del runner, message
            show_hook_message(self, "42ms", hook_name="llm_time")
            display = self.display
            assert isinstance(display, CapturingConsoleDisplay)
            self.status_counts_during_hook.append(len(display.status_messages))

        return ToolRunnerHooks(after_llm_call=after_llm_call)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_on_pause_cancel_retries_llm_call():
    """The tool_runner must catch CancelledError from the LLM call and
    re-issue it when on_pause_cancel returns True. End result: the
    chat request completes normally despite the mid-call cancellation,
    matching PauseController's pause+resume semantics.
    """
    llm = CancelOnceLlm()
    agent = CancelRetryHookAgent(AgentConfig("cancel-retry"))
    agent._llm = llm

    result = await agent.generate("hi")

    assert result.last_text() == "done after retry"
    assert agent.retry_count == 1
    assert llm._calls == 2  # first cancelled, second succeeded


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hook_messages_flush_after_hook_boundary() -> None:
    llm = OneTurnLlm()
    agent = HookMessageDeferringAgent(AgentConfig("hook-message"), [tool_one])
    agent._llm = llm

    result = await agent.generate("hi")

    display = agent.display
    assert isinstance(display, CapturingConsoleDisplay)
    assert result.last_text() == "done"
    assert agent.status_counts_during_hook == [0]
    assert len(display.status_messages) == 1
    assert "llm_time" in display.status_messages[0]
    assert display.status_messages[0].endswith("42ms")


class GenuineCancelHookAgent(ToolAgent):
    """Agent whose on_pause_cancel always says "genuine cancel, propagate"."""

    def _tool_runner_hooks(self):
        async def on_pause_cancel(runner):
            return False

        return ToolRunnerHooks(on_pause_cancel=on_pause_cancel)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_on_pause_cancel_returning_false_propagates():
    """When on_pause_cancel returns False (= not paused, real cancel),
    the CancelledError must propagate up so the chat request actually
    fails — otherwise legitimate client-disconnect cancels would be
    silently swallowed.
    """
    import asyncio
    llm = CancelOnceLlm()
    agent = GenuineCancelHookAgent(AgentConfig("genuine-cancel"))
    agent._llm = llm

    with pytest.raises(asyncio.CancelledError):
        await agent.generate("hi")

    assert llm._calls == 1  # no retry happened


# ─── Strategy B: subprocess pause_before_tool must NOT cancel ──────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pause_before_tool_does_not_register_cancel_target():
    """Subprocess regression for fast-agent#3 F1.

    When ``pause_before_tool`` fires (i.e. the subprocess agent is
    about to execute a tool), it must NOT register the current task
    as ``_current_llm_task``. Otherwise a SIGUSR1 arriving mid-tool
    would cancel the chat-request task and tear down the whole turn
    (no on_pause_cancel retry contract covers the tool phase).

    The split (pause_before_llm registers, pause_before_tool doesn't)
    is the load-bearing strategy-B invariant.
    """
    from fast_agent.spawn import pause_signal_handler as psh

    # Reset module-level state to known baseline.
    psh._current_llm_task = None
    psh._pause_event = None  # let _ensure_event() rebuild a fresh one

    # Pre-condition.
    assert psh._current_llm_task is None

    # Tool checkpoint fires. Must not touch _current_llm_task.
    await psh.pause_before_tool(runner=None, request=None)
    assert psh._current_llm_task is None, \
        "pause_before_tool must not register the cancel target (strategy B)"

    # LLM checkpoint fires. Must register.
    await psh.pause_before_llm(runner=None, messages=None)
    assert psh._current_llm_task is not None, \
        "pause_before_llm must register the cancel target"

    # After LLM finishes, ref is cleared so a SIGUSR1 during the
    # following tool phase has no task to cancel.
    await psh.pause_after_llm(runner=None, message=None)
    assert psh._current_llm_task is None, \
        "pause_after_llm must clear the cancel target before tool phase"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pause_block_helper_is_no_op_when_not_paused():
    """``_block_if_paused`` must return immediately if the event is set
    (= not paused). Both ``pause_before_llm`` and ``pause_before_tool``
    delegate to it, so this pins the cheap-path invariant.
    """
    from fast_agent.spawn import pause_signal_handler as psh

    psh._pause_event = None  # reset
    event = psh._ensure_event()
    assert event.is_set(), "default state must be not-paused"

    # Should return immediately, not block.
    await asyncio.wait_for(psh._block_if_paused(), timeout=0.1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pause_block_helper_blocks_then_releases_on_resume():
    """When the event is cleared (= pause requested), ``_block_if_paused``
    awaits ``event.wait()`` and returns once the event is set again.
    This is the core resume contract the SIGUSR2 handler depends on.
    """
    from fast_agent.spawn import pause_signal_handler as psh

    psh._pause_event = None
    event = psh._ensure_event()
    event.clear()  # paused

    task = asyncio.create_task(psh._block_if_paused())
    await asyncio.sleep(0.05)
    assert not task.done(), "must block while event is cleared"

    event.set()  # resume
    await asyncio.wait_for(task, timeout=0.5)


# ─── Path B: LLM returns stop_reason=CANCELLED (no exception) ──────────


class CancelStopReasonLlm(PassthroughLLM):
    """Simulates the openai/anthropic graceful-cancel path: the provider
    catches its own CancelledError and returns ``stop_reason=CANCELLED``
    instead of propagating. tool_runner must STILL dispatch to
    ``on_pause_cancel`` so the pause flow doesn't get stuck.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calls = 0

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages,
        request_params=None,
        tools=None,
        is_template=False,
    ):
        self._calls += 1
        if self._calls == 1:
            return Prompt.assistant("", stop_reason=LlmStopReason.CANCELLED)
        return Prompt.assistant("retry succeeded", stop_reason=LlmStopReason.END_TURN)


class CancelRetryStopReasonAgent(ToolAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retry_count = 0

    def _tool_runner_hooks(self):
        async def on_pause_cancel(runner):
            self.retry_count += 1
            return self.retry_count == 1

        return ToolRunnerHooks(on_pause_cancel=on_pause_cancel)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_on_pause_cancel_fires_on_stop_reason_cancelled():
    """Regression for 2026-05-24 stuck-Pausing bug.

    openai + anthropic providers catch CancelledError internally and
    return a response with ``stop_reason=CANCELLED`` rather than
    propagating the exception. Pre-fix, tool_runner only dispatched to
    ``on_pause_cancel`` on raised CancelledError → graceful-cancel
    path bypassed the hook → controller never emitted ``agent_paused``
    → UI stuck on "Pausing…" forever. Path B in __anext__ now
    inspects ``stop_reason`` after the call returns and dispatches
    the same hook, so both provider styles behave identically.
    """
    llm = CancelStopReasonLlm()
    agent = CancelRetryStopReasonAgent(AgentConfig("cancel-stop-reason"))
    agent._llm = llm

    result = await agent.generate("hi")

    assert result.last_text() == "retry succeeded"
    assert agent.retry_count == 1
    assert llm._calls == 2


class GenuineCancelStopReasonAgent(ToolAgent):
    """Hook always says "no retry, propagate" — verifies the
    CANCELLED message is preserved when the hook declines.
    """

    def _tool_runner_hooks(self):
        async def on_pause_cancel(runner):
            return False

        return ToolRunnerHooks(on_pause_cancel=on_pause_cancel)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_reason_cancelled_falls_through_when_hook_declines():
    """When ``on_pause_cancel`` returns False on the CANCELLED branch,
    the CANCELLED response must pass through unchanged so
    ``until_done`` runs its rollback path. Pins the "don't swallow
    genuine cancels" invariant for Path B.
    """
    llm = CancelStopReasonLlm()
    agent = GenuineCancelStopReasonAgent(AgentConfig("genuine-cancel-stop"))
    agent._llm = llm

    result = await agent.generate("hi")

    assert result.stop_reason == LlmStopReason.CANCELLED
    assert llm._calls == 1  # no retry


# ─── E2E: real cancellation flow through provider-like LLM ──────────


class CancellableSlowLlm(PassthroughLLM):
    """Mimics openai/anthropic provider faithfully:

    - First call: ``await asyncio.sleep`` to simulate streaming.
      When the surrounding task is cancelled, the sleep raises
      CancelledError → the provider's ``except CancelledError`` catches
      and returns ``stop_reason=CANCELLED`` (mirroring
      ``llm_openai.py:931-938`` and ``llm_anthropic.py:1656``).
    - Subsequent calls: returns the real END_TURN response (the retry).

    Closer to reality than a mock that just returns CANCELLED directly,
    because:
      1. The CancelledError is actually raised by asyncio (not faked).
      2. The provider's except-handler is exercised (replicates
         producer code path inside this test class).
      3. tool_runner.__anext__'s Path B is reached via the same code
         path it would in production.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calls = 0
        self.cancel_event: asyncio.Event | None = None

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages,
        request_params=None,
        tools=None,
        is_template=False,
    ):
        self._calls += 1
        if self._calls == 1:
            # Block in a way that's actually cancellable. Real provider
            # blocks on ``await client.chat.completions.create(...)``;
            # we substitute a sleep that the test cancels externally.
            try:
                if self.cancel_event is not None:
                    self.cancel_event.set()  # signal "I'm now in the LLM call"
                await asyncio.sleep(30)  # would be the streaming await
            except asyncio.CancelledError:
                # Mirror llm_openai.py:931-938 — graceful cancel return.
                return Prompt.assistant("", stop_reason=LlmStopReason.CANCELLED)
            return Prompt.assistant("never reached", stop_reason=LlmStopReason.END_TURN)
        return Prompt.assistant("resumed work", stop_reason=LlmStopReason.END_TURN)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_e2e_provider_like_cancel_with_inline_pause_hook():
    """End-to-end happy path closest to production:

    - Inline pause hook implementing the same contract as the parent
      repo's ``PauseController`` (the controller itself lives in
      ``services/`` outside this submodule, so we reproduce its
      behavior here rather than reaching across).
    - Real ``tool_runner`` retry loop.
    - LLM simulator that handles CancelledError exactly like
      ``llm_openai.py`` does (returns ``stop_reason=CANCELLED``).
    - External pause: ``pause_controller.pause(name)`` cancels the
      in-flight task while it's awaiting the LLM "stream".
    - External resume: ``pause_controller.resume(name)`` unblocks the
      hook → tool_runner reissues the LLM call → returns final result.

    Pre-fix (no Path B in __anext__): the CANCELLED response would
    bubble up unhandled, on_pause_cancel never fired, agent.generate()
    would return a CANCELLED response immediately instead of awaiting
    resume. This test would fail with stop_reason=CANCELLED instead
    of END_TURN.
    """
    # ── Wire real PauseController hooks onto the agent ──
    # We can't import the parent-repo pause_controller from here (it's
    # in services/, not on path). Re-implement the same hook contract
    # inline so the test stays in submodule scope but exercises the
    # exact tool_runner integration.
    pause_event = asyncio.Event()
    pause_event.set()  # start unpaused
    captured_states = []

    async def on_before_llm_call(runner, messages):
        if not pause_event.is_set():
            await pause_event.wait()

    async def on_pause_cancel(runner):
        captured_states.append("cancel_detected")
        if not pause_event.is_set():
            captured_states.append("awaiting_resume")
            await pause_event.wait()
            captured_states.append("resumed")
            return True  # retry
        return False  # genuine cancel

    class _E2EAgent(ToolAgent):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)

        def _tool_runner_hooks(self):
            return ToolRunnerHooks(
                before_llm_call=on_before_llm_call,
                on_pause_cancel=on_pause_cancel,
            )

    llm = CancellableSlowLlm()
    llm.cancel_event = asyncio.Event()  # signal when LLM is "streaming"
    agent = _E2EAgent(AgentConfig("e2e-pause-resume"))
    agent._llm = llm

    # ── Run agent.generate in a task; wait until LLM call is mid-stream ──
    gen_task = asyncio.create_task(agent.generate("do work"))
    await asyncio.wait_for(llm.cancel_event.wait(), timeout=2.0)

    # ── External pause: clear event + cancel the task ──
    pause_event.clear()
    gen_task.cancel()
    # Yield so cancel propagates through provider → returns CANCELLED →
    # tool_runner Path B → on_pause_cancel awaits.
    for _ in range(10):
        await asyncio.sleep(0)
        if "awaiting_resume" in captured_states:
            break
    assert "cancel_detected" in captured_states, \
        "Path B in tool_runner.__anext__ failed to dispatch on_pause_cancel"
    assert "awaiting_resume" in captured_states, \
        "on_pause_cancel didn't reach the await event.wait() step"
    assert not gen_task.done(), \
        "agent.generate() must remain hanging while hook awaits resume"

    # ── External resume: set event → hook returns True → retry LLM ──
    pause_event.set()
    result = await asyncio.wait_for(gen_task, timeout=2.0)

    assert result.last_text() == "resumed work", \
        "after resume, tool_runner must reissue the LLM call and return its result"
    assert llm._calls == 2, "exactly one retry should happen"
    assert "resumed" in captured_states


class OneToolUseThenDoneLlm(PassthroughLLM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._turn = 0

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        del multipart_messages, request_params, tools, is_template
        self._turn += 1
        if self._turn == 1:
            return Prompt.assistant(
                "calling tool_one",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "call_1": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(name="tool_one", arguments={}),
                    ),
                },
            )
        return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


class ToolEventCapturingDisplay(CapturingConsoleDisplay):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def show_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, object] | None,
        bottom_items: list[str] | None = None,
        highlight_index: int | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        metadata: dict[str, object] | None = None,
        tool_call_id: str | None = None,
        type_label: str | None = None,
        show_hook_indicator: bool = False,
    ) -> None:
        del (
            tool_args,
            bottom_items,
            highlight_index,
            max_item_length,
            name,
            metadata,
            tool_call_id,
            type_label,
            show_hook_indicator,
        )
        self.events.append(f"tool_call:{tool_name}")

    def show_tool_result(
        self,
        result,
        name: str | None = None,
        tool_name: str | None = None,
        skybridge_config=None,
        timing_ms: float | None = None,
        tool_call_id: str | None = None,
        type_label: str | None = None,
        truncate_content: bool = True,
        show_hook_indicator: bool = False,
    ) -> None:
        del (
            result,
            name,
            skybridge_config,
            timing_ms,
            tool_call_id,
            type_label,
            truncate_content,
            show_hook_indicator,
        )
        self.events.append(f"tool_result:{tool_name}")

    def show_status_message(self, content: Text) -> None:
        super().show_status_message(content)
        self.events.append(f"status:{content.plain}")


class ToolUseHookMessageDeferringAgent(ToolAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.display = ToolEventCapturingDisplay()
        self.event_counts_during_after_llm: list[int] = []

    def _should_stream(self) -> bool:
        return False

    def _display_user_messages(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
    ) -> None:
        del messages, request_params

    async def show_assistant_message(
        self,
        message: PromptMessageExtended,
        bottom_items=None,
        highlight_items=None,
        max_item_length=None,
        name=None,
        model=None,
        additional_message=None,
        render_markdown: bool | None = None,
        show_hook_indicator: bool | None = None,
        render_message: bool = True,
        show_reprint_banner: bool = False,
    ) -> None:
        del (
            message,
            bottom_items,
            highlight_items,
            max_item_length,
            name,
            model,
            additional_message,
            render_markdown,
            show_hook_indicator,
            render_message,
            show_reprint_banner,
        )

    def _tool_runner_hooks(self) -> ToolRunnerHooks | None:
        async def after_llm_call(runner, message):
            del runner
            if message.stop_reason == LlmStopReason.TOOL_USE:
                show_hook_message(self, "42ms", hook_name="llm_time")
                display = self.display
                assert isinstance(display, ToolEventCapturingDisplay)
                self.event_counts_during_after_llm.append(len(display.events))

        return ToolRunnerHooks(after_llm_call=after_llm_call)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_use_hook_messages_flush_after_tool_results() -> None:
    llm = OneToolUseThenDoneLlm()
    agent = ToolUseHookMessageDeferringAgent(AgentConfig("hook-message-tool-use"), [tool_one])
    agent._llm = llm

    result = await agent.generate("hi")

    display = agent.display
    assert isinstance(display, ToolEventCapturingDisplay)
    assert result.last_text() == "done"
    assert agent.event_counts_during_after_llm == [0]
    assert display.events == [
        "tool_call:tool_one",
        "tool_result:tool_one",
        "status:▎ extension llm_time — 42ms",
    ]
