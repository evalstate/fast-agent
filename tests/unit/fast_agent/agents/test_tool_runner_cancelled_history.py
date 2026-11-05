import asyncio
import os
from unittest.mock import AsyncMock

import pytest
from mcp import CallToolRequest
from mcp_types import CallToolRequestParams, CallToolResult, TextContent, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.agents.tool_runner import ToolRunner, ToolRunnerHooks
from fast_agent.config import get_settings, update_global_settings
from fast_agent.constants import FAST_AGENT_RUNTIME_HOME
from fast_agent.core.queued_input import queued_input_source
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.helpers.content_helpers import get_text, text_content
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompts.prompt_load import load_prompt
from fast_agent.session import SessionManager, reset_session_manager, set_session_manager
from fast_agent.types.llm_stop_reason import LlmStopReason


async def cancel_tool() -> None:
    raise asyncio.CancelledError("cancelled by test")


async def ok_tool() -> str:
    return "ok"


def _register_session_manager(home) -> SessionManager:
    os.environ["FAST_AGENT_HOME"] = str(home)
    os.environ[FAST_AGENT_RUNTIME_HOME] = str(home)
    manager = SessionManager(home_override=home)
    set_session_manager(manager)
    return manager


class ExternalCancelledToolUseLlm(PassthroughLLM):
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
            tool_calls = {
                "slow_call": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="slow_tool", arguments={}),
                )
            }
            return Prompt.assistant(
                "use tool",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls=tool_calls,
            )

        return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


class CancelledToolUseLlm(PassthroughLLM):
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
            tool_calls = {
                "cancel_call": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="cancel_tool", arguments={}),
                )
            }
            return Prompt.assistant(
                "use tool",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls=tool_calls,
            )

        return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


class CancelledStopReasonLlm(PassthroughLLM):
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
            tool_calls = {
                "cancel_call": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="ok_tool", arguments={}),
                )
            }
            return Prompt.assistant(
                "use tool",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls=tool_calls,
            )

        return Prompt.assistant("", stop_reason=LlmStopReason.CANCELLED)


class ExplodingSecondTurnLlm(PassthroughLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._turn = 0
        self.retry_count = 2
        self.retry_backoff_seconds = 0.0

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        self._turn += 1
        if self._turn == 1:
            tool_calls = {
                "explode_call": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="ok_tool", arguments={}),
                )
            }
            return Prompt.assistant(
                "use tool",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls=tool_calls,
            )

        raise RuntimeError("llm boom")


class BlockingSecondTurnLlm(PassthroughLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._turn = 0
        self.followup_started = asyncio.Event()

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        self._turn += 1
        if self._turn == 1:
            return Prompt.assistant(
                "use tool",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "ok_call": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(name="ok_tool", arguments={}),
                    )
                },
            )

        self.followup_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class ContinuedToolResultLlm(PassthroughLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen_last_message: PromptMessageExtended | None = None

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        self.seen_last_message = multipart_messages[-1].model_copy(deep=True)
        tool_result_text = " ".join(
            (get_text(tool_result.content[0]) or "")
            for tool_result in (self.seen_last_message.tool_results or {}).values()
            if tool_result.content
        )
        combined_text = "\n".join(
            text for text in [tool_result_text, self.seen_last_message.all_text()] if text
        )
        return Prompt.assistant(
            combined_text,
            stop_reason=LlmStopReason.END_TURN,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_tool_loop_resets_history() -> None:
    llm = CancelledToolUseLlm()
    agent = ToolAgent(AgentConfig("cancelled"), [cancel_tool])
    agent._llm = llm

    agent.load_message_history(
        [
            Prompt.user("previous"),
            Prompt.assistant("ok", stop_reason=LlmStopReason.END_TURN),
        ]
    )

    with pytest.raises(asyncio.CancelledError):
        await agent.generate("trigger")

    history = agent.message_history
    assert len(history) == 5
    assert history[-2].role == "assistant"
    assert history[-2].stop_reason == LlmStopReason.TOOL_USE
    assert history[-1].role == "user"
    tool_results = history[-1].tool_results
    assert tool_results is not None
    assert "cancel_call" in tool_results
    last_text = history[-1].last_text()
    assert last_text is not None
    assert "The user interrupted this tool call" in last_text

    rollback_state = getattr(agent, "_last_turn_history_state", None)
    assert rollback_state is not None
    assert getattr(rollback_state, "status", None) == "appended_interrupted_tool_result"
    assert getattr(rollback_state, "removed_messages", None) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_stop_reason_preserves_history_and_skips_after_turn_hooks() -> None:
    llm = CancelledStopReasonLlm()
    agent = ToolAgent(AgentConfig("cancelled-stop"), [ok_tool])
    agent._llm = llm

    hook_called = False

    async def after_turn_complete(_runner, _message) -> None:
        nonlocal hook_called
        hook_called = True

    agent.tool_runner_hooks = ToolRunnerHooks(after_turn_complete=after_turn_complete)

    agent.load_message_history(
        [
            Prompt.user("previous"),
            Prompt.assistant("ok", stop_reason=LlmStopReason.END_TURN),
        ]
    )

    result = await agent.generate("trigger")

    assert result.stop_reason == LlmStopReason.CANCELLED
    assert hook_called is False

    history = agent.message_history
    assert len(history) == 6
    assert history[-1].role == "assistant"
    assert history[-1].stop_reason == LlmStopReason.CANCELLED

    rollback_state = getattr(agent, "_last_turn_history_state", None)
    assert rollback_state is not None
    assert getattr(rollback_state, "status", None) == "history_unchanged"
    assert getattr(rollback_state, "removed_messages", None) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_external_cancel_after_completed_tool_preserves_real_result() -> None:
    llm = BlockingSecondTurnLlm()
    agent = ToolAgent(AgentConfig("cancelled-followup"), [ok_tool])
    agent._llm = llm

    task = asyncio.create_task(agent.generate("trigger"))
    await asyncio.wait_for(llm.followup_started.wait(), timeout=1.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    history = agent.message_history
    assert history[-1].role == "user"
    assert history[-1].tool_results is not None
    result = history[-1].tool_results["ok_call"]
    [content] = result.content
    assert isinstance(content, TextContent)
    assert content.text == "ok"
    assert result.is_error in (False, None)

    rollback_state = agent.last_turn_history_state
    assert rollback_state is not None
    assert rollback_state.status == "appended_completed_tool_result"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_tool_loop_with_use_history_false_keeps_history_unchanged() -> None:
    llm = CancelledStopReasonLlm()
    agent = ToolAgent(AgentConfig("cancelled-no-history", use_history=False), [ok_tool])
    agent._llm = llm

    baseline_history = [
        Prompt.user("seed"),
        Prompt.assistant("seed-response", stop_reason=LlmStopReason.END_TURN),
    ]
    agent.load_message_history(baseline_history)

    result = await agent.generate("trigger", RequestParams(use_history=False))

    assert result.stop_reason == LlmStopReason.CANCELLED
    assert len(agent.message_history) == len(baseline_history)
    for index, message in enumerate(agent.message_history):
        assert message.role == baseline_history[index].role
        assert message.last_text() == baseline_history[index].last_text()

    rollback_state = getattr(agent, "_last_turn_history_state", None)
    assert rollback_state is not None
    assert getattr(rollback_state, "status", None) == "history_disabled"
    assert getattr(rollback_state, "removed_messages", None) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_interrupted_tool_result_append_is_idempotent() -> None:
    agent = ToolAgent(AgentConfig("idempotent"), [ok_tool])
    pending_tool_call = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name="ok_tool", arguments={}),
    )
    agent.load_message_history(
        [
            Prompt.user("hello"),
            Prompt.assistant(
                "pending",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={"call-1": pending_tool_call},
            ),
        ]
    )

    runner = ToolRunner(agent=agent, messages=[])

    first = runner._reset_history_after_cancelled_turn()
    second = runner._reset_history_after_cancelled_turn()

    assert first.status == "appended_interrupted_tool_result"
    assert second.status == "history_unchanged"

    history = agent.message_history
    assert len(history) == 3
    assert history[-1].role == "user"
    assert history[-1].tool_results is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_auto_heals_stale_pending_tool_call_with_interrupted_marker() -> None:
    llm = PassthroughLLM()
    agent = ToolAgent(AgentConfig("stale-history"), [ok_tool])
    agent._llm = llm

    pending_tool_call = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name="ok_tool", arguments={}),
    )
    agent.load_message_history(
        [
            Prompt.user("hello"),
            Prompt.assistant(
                "pending",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={"call-1": pending_tool_call},
            ),
        ]
    )

    result = await agent.generate("new turn")

    assert result.stop_reason == LlmStopReason.END_TURN
    history = agent.message_history
    assert len(history) == 5
    assert history[1].role == "assistant"
    assert history[1].stop_reason == LlmStopReason.TOOL_USE
    assert history[2].role == "user"
    assert history[2].tool_results is not None
    assert "call-1" in history[2].tool_results
    interrupted_text = history[2].last_text()
    assert interrupted_text is not None
    assert "The user interrupted this tool call" in interrupted_text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_continues_from_staged_tool_result_history() -> None:
    llm = ContinuedToolResultLlm()
    agent = ToolAgent(AgentConfig("staged-tool-result"), [ok_tool])
    agent._llm = llm

    pending_tool_call = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name="ok_tool", arguments={}),
    )
    agent.load_message_history(
        [
            Prompt.user("hello"),
            Prompt.assistant(
                "pending",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={"call-1": pending_tool_call},
            ),
            PromptMessageExtended(
                role="user",
                content=[text_content("ok")],
                tool_results={
                    "call-1": CallToolResult(content=[text_content("ok")]),
                },
            ),
        ]
    )

    result = await agent.generate("new turn")

    assert result.stop_reason == LlmStopReason.END_TURN
    assert result.last_text() == "ok\nnew turn"
    assert llm.seen_last_message is not None
    assert llm.seen_last_message.tool_results is not None
    assert "call-1" in llm.seen_last_message.tool_results
    assert llm.seen_last_message.all_text() == "new turn"

    history = agent.message_history
    assert len(history) == 5
    assert history[2].role == "user"
    resumed_results = history[2].tool_results
    assert resumed_results is not None
    assert "call-1" in resumed_results
    assert resumed_results["call-1"].is_error in (False, None)
    assert history[2].last_text() == "ok"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_turn_is_persisted_to_session_history(tmp_path) -> None:
    old_settings = get_settings()
    override = old_settings.model_copy(update={"home": str(tmp_path / "env")})
    update_global_settings(override)
    reset_session_manager()

    try:
        manager = _register_session_manager(tmp_path / "env")
        llm = CancelledToolUseLlm()
        agent = ToolAgent(AgentConfig("cancelled-persist"), [cancel_tool])
        agent._llm = llm

        agent.load_message_history(
            [
                Prompt.user("previous"),
                Prompt.assistant("ok", stop_reason=LlmStopReason.END_TURN),
            ]
        )

        with pytest.raises(asyncio.CancelledError):
            await agent.generate("trigger")

        session = manager.current_session
        assert session is not None

        history_path = session.latest_history_path(agent.name)
        assert history_path is not None
        assert history_path.exists()

        saved_messages = load_prompt(history_path)
        assert saved_messages
        assert saved_messages[-1].role == "user"
        saved_text = saved_messages[-1].last_text()
        assert saved_text is not None
        assert "The user interrupted this tool call" in saved_text
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_externally_cancelled_turn_is_persisted_to_session_history(tmp_path) -> None:
    old_settings = get_settings()
    override = old_settings.model_copy(update={"home": str(tmp_path / "env")})
    update_global_settings(override)
    reset_session_manager()

    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    async def slow_tool() -> str:
        tool_started.set()
        await release_tool.wait()
        return "ok"

    try:
        manager = _register_session_manager(tmp_path / "env")
        llm = ExternalCancelledToolUseLlm()
        agent = ToolAgent(AgentConfig("cancelled-persist-external"), [slow_tool])
        agent._llm = llm

        agent.load_message_history(
            [
                Prompt.user("previous"),
                Prompt.assistant("ok", stop_reason=LlmStopReason.END_TURN),
            ]
        )

        task = asyncio.create_task(agent.generate("trigger"))
        await asyncio.wait_for(tool_started.wait(), timeout=1.0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        session = manager.current_session
        assert session is not None

        history_path = session.latest_history_path(agent.name)
        assert history_path is not None
        assert history_path.exists()

        saved_messages = load_prompt(history_path)
        assert saved_messages
        assert saved_messages[-1].role == "user"
        saved_text = saved_messages[-1].last_text()
        assert saved_text is not None
        assert "The user interrupted this tool call" in saved_text
    finally:
        release_tool.set()
        update_global_settings(old_settings)
        reset_session_manager()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("failing_round", [1, 2])
async def test_unhandled_after_llm_hook_error_persists_tool_loop_history(
    tmp_path, monkeypatch, failing_round: int
) -> None:
    old_settings = get_settings()
    override = old_settings.model_copy(update={"home": str(tmp_path / "env")})
    update_global_settings(override)
    reset_session_manager()

    try:
        manager = _register_session_manager(tmp_path / "env")
        manager.create_session(metadata={"agent_name": "after-llm-hook-persist"})
        llm = CancelledStopReasonLlm()
        responses = [
            Prompt.assistant(
                "use tool",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    f"call{round_number}": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(name="ok_tool", arguments={}),
                    )
                },
            )
            for round_number in range(1, failing_round + 1)
        ]
        monkeypatch.setattr(
            llm, "_apply_prompt_provider_specific", AsyncMock(side_effect=responses)
        )
        agent = ToolAgent(AgentConfig("after-llm-hook-persist"), [ok_tool])
        agent._llm = llm

        async def after_llm_call(runner, _message) -> None:
            if runner.iteration == failing_round - 1:
                raise RuntimeError("after llm boom")

        agent.tool_runner_hooks = ToolRunnerHooks(after_llm_call=after_llm_call)

        with pytest.raises(RuntimeError, match="after llm boom"):
            await agent.generate("trigger")

        session = manager.current_session
        assert session is not None

        history_path = session.latest_history_path(agent.name)
        assert history_path is not None
        assert history_path.exists()

        saved_messages = load_prompt(history_path)
        for history in (saved_messages, agent.message_history):
            assert history[-1].role == "assistant"
            assert history[-1].stop_reason == LlmStopReason.TOOL_USE
            assert set(history[-1].tool_calls or {}) == {f"call{failing_round}"}
            assert sum(bool(message.tool_results) for message in history) == failing_round - 1

        rollback = ToolRunner.reconcile_interrupted_history(agent, use_history=True)
        assert rollback.status == "appended_interrupted_tool_result"
        results = agent.message_history[-1].tool_results
        assert results is not None
        assert set(results) == {f"call{failing_round}"}
        assert all(result.is_error for result in results.values())
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_second_llm_error_after_tool_use_persists_resumable_checkpoint(tmp_path) -> None:
    old_settings = get_settings()
    override = old_settings.model_copy(update={"home": str(tmp_path / "env")})
    update_global_settings(override)
    reset_session_manager()

    try:
        manager = _register_session_manager(tmp_path / "env")
        manager.create_session(metadata={"agent_name": "checkpointed-mid-loop"})
        llm = ExplodingSecondTurnLlm()
        agent = ToolAgent(AgentConfig("checkpointed-mid-loop"), [ok_tool])
        agent._llm = llm

        with pytest.raises(RuntimeError, match="llm boom"):
            await agent.generate("trigger")

        session = manager.load_latest_session(require_content=True)
        assert session is not None

        history_path = session.latest_history_path(agent.name)
        assert history_path is not None
        assert history_path.exists()

        saved_messages = load_prompt(history_path)
        assert saved_messages
        assert saved_messages[-1].role == "user"
        assert saved_messages[-1].tool_results is not None
        assert "explode_call" in saved_messages[-1].tool_results
        saved_result = saved_messages[-1].tool_results["explode_call"]
        assert len(saved_result.content) == 1
        saved_content = saved_result.content[0]
        assert isinstance(saved_content, TextContent)
        assert saved_content.text == "ok"
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resumed_tool_result_history_does_not_rerun_completed_tool(tmp_path) -> None:
    old_settings = get_settings()
    override = old_settings.model_copy(update={"home": str(tmp_path / "env")})
    update_global_settings(override)
    reset_session_manager()

    tool_runs = 0

    async def ok_tool() -> str:
        nonlocal tool_runs
        tool_runs += 1
        return f"ok {tool_runs}"

    try:
        manager = _register_session_manager(tmp_path / "env")
        manager.create_session(metadata={"agent_name": "checkpointed-side-effect"})
        exploding_llm = ExplodingSecondTurnLlm()
        agent = ToolAgent(AgentConfig("checkpointed-side-effect"), [ok_tool])
        agent._llm = exploding_llm

        with pytest.raises(RuntimeError, match="llm boom"):
            await agent.generate("trigger")

        assert tool_runs == 1

        resumed_llm = ContinuedToolResultLlm()
        resumed_agent = ToolAgent(AgentConfig("checkpointed-side-effect"), [ok_tool])
        resumed_agent._llm = resumed_llm

        resumed = await manager.resume_session_agents_async(
            {resumed_agent.name: resumed_agent},
            fallback_agent_name=resumed_agent.name,
        )
        assert resumed is not None

        result = await resumed_agent.generate("after resume")

        assert result.stop_reason == LlmStopReason.END_TURN
        assert result.last_text() == "ok 1\nafter resume"
        assert tool_runs == 1
        assert resumed_llm.seen_last_message is not None
        assert resumed_llm.seen_last_message.tool_results is not None
        assert "explode_call" in resumed_llm.seen_last_message.tool_results
        assert resumed_llm.seen_last_message.all_text() == "after resume"
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["before_llm_hook", "llm"])
async def test_failed_continuation_preserves_consumed_input_in_live_and_saved_history(
    tmp_path, failure_phase
) -> None:
    old_settings = get_settings()
    update_global_settings(old_settings.model_copy(update={"home": str(tmp_path / "env")}))
    reset_session_manager()

    try:
        manager = _register_session_manager(tmp_path / "env")
        manager.create_session(metadata={"agent_name": "queued-failure"})
        agent = ToolAgent(AgentConfig("queued-failure"), [ok_tool])
        agent._llm = ExplodingSecondTurnLlm()

        async def before_llm(runner, messages):
            if runner.iteration > 0:
                raise RuntimeError("hook boom")

        if failure_phase == "before_llm_hook":
            agent.tool_runner_hooks = ToolRunnerHooks(before_llm_call=before_llm)

        pending = ["important report", "important report"]

        def take_messages():
            result = list(pending)
            pending.clear()
            return result

        with queued_input_source(agent, take_messages):
            with pytest.raises(RuntimeError, match="boom"):
                await agent.generate("trigger")
        assert not pending

        session = manager.current_session
        assert session is not None
        path = session.latest_history_path(agent.name)
        assert path is not None
        saved = load_prompt(path)
        for history in (saved, agent.message_history):
            assert history[-3].tool_results
            assert [message.first_text() for message in history[-2:]] == [
                "important report",
                "important report",
            ]

        # Continuing in this same agent must retain the input, without replaying tools
        # or overwriting the checkpoint with the pre-failure history.
        agent.tool_runner_hooks = None
        agent._llm = PassthroughLLM()
        await agent.generate("continue")
        await manager.save_current_session(agent)
        path = session.latest_history_path(agent.name)
        assert path is not None
        for history in (agent.message_history, load_prompt(path)):
            assert sum(message.first_text() == "important report" for message in history) == 2
            assert sum(bool(message.tool_results) for message in history) == 1
    finally:
        update_global_settings(old_settings)
        reset_session_manager()
