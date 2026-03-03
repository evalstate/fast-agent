import asyncio

import pytest
from mcp import CallToolRequest
from mcp.types import CallToolRequestParams, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.agents.tool_runner import ToolRunner, ToolRunnerHooks
from fast_agent.config import get_settings, update_global_settings
from fast_agent.core.prompt import Prompt
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompts.prompt_load import load_prompt
from fast_agent.session import get_session_manager, reset_session_manager
from fast_agent.types.llm_stop_reason import LlmStopReason


async def cancel_tool() -> None:
    raise asyncio.CancelledError("cancelled by test")


async def ok_tool() -> str:
    return "ok"


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
async def test_cancelled_turn_is_persisted_to_session_history(tmp_path) -> None:
    old_settings = get_settings()
    override = old_settings.model_copy(update={"environment_dir": str(tmp_path / "env")})
    update_global_settings(override)
    reset_session_manager()

    try:
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

        manager = get_session_manager()
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
    override = old_settings.model_copy(update={"environment_dir": str(tmp_path / "env")})
    update_global_settings(override)
    reset_session_manager()

    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    async def slow_tool() -> str:
        tool_started.set()
        await release_tool.wait()
        return "ok"

    try:
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

        manager = get_session_manager()
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
