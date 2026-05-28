from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent.acp.server.prompt_flow import ACPPromptFlow, PromptFlowHost
from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.core.prompt import Prompt
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from fast_agent.types import PromptMessageExtended


class _FakeHost:
    def __init__(self) -> None:
        self._connection = object()
        self.status_updates: list[tuple[str, Any, int | None]] = []

    @staticmethod
    def _merge_tool_runner_hooks(
        base: ToolRunnerHooks | None, extra: ToolRunnerHooks | None
    ) -> ToolRunnerHooks | None:
        if base is None:
            return extra
        if extra is None:
            return base

        def merge(one: Any, two: Any) -> Any:
            if one is None:
                return two
            if two is None:
                return one

            async def merged(*args: Any, **kwargs: Any) -> None:
                await one(*args, **kwargs)
                await two(*args, **kwargs)

            return merged

        return ToolRunnerHooks(
            before_llm_call=merge(base.before_llm_call, extra.before_llm_call),
            after_llm_call=merge(base.after_llm_call, extra.after_llm_call),
            before_tool_call=merge(base.before_tool_call, extra.before_tool_call),
            after_tool_call=merge(base.after_tool_call, extra.after_tool_call),
            after_turn_complete=merge(base.after_turn_complete, extra.after_turn_complete),
        )

    async def _send_status_line_update(
        self, session_id: str, agent: Any, turn_start_index: int | None
    ) -> None:
        self.status_updates.append((session_id, agent, turn_start_index))


class _FakeHookCapableAgent:
    def __init__(self) -> None:
        self._tool_runner_hooks: ToolRunnerHooks | None = None
        self.hook_stop_reasons: list[LlmStopReason | str | None] = []

    @property
    def tool_runner_hooks(self) -> ToolRunnerHooks | None:
        return self._tool_runner_hooks

    @tool_runner_hooks.setter
    def tool_runner_hooks(self, value: ToolRunnerHooks | None) -> None:
        self._tool_runner_hooks = value

    async def generate(
        self, prompt_message: Any, request_params: Any = None
    ) -> PromptMessageExtended:
        del prompt_message, request_params
        message = Prompt.assistant("using tools", stop_reason=LlmStopReason.TOOL_USE)
        assert self._tool_runner_hooks is not None
        after_llm_call = self._tool_runner_hooks.after_llm_call
        assert after_llm_call is not None
        await after_llm_call(cast("Any", None), message)
        self.hook_stop_reasons.append(message.stop_reason)
        return message


@pytest.mark.asyncio
async def test_run_with_status_hooks_sends_interim_update_for_tool_use() -> None:
    host = _FakeHost()
    flow = ACPPromptFlow(cast("PromptFlowHost", host))
    agent = _FakeHookCapableAgent()

    result = await flow._run_with_status_hooks(
        agent=agent,
        session_id="session-1",
        turn_start_index=3,
        prompt_message=Prompt.user("hi"),
        session_request_params=None,
    )

    message = cast("PromptMessageExtended", result["result"])
    assert message.stop_reason == LlmStopReason.TOOL_USE
    assert agent.hook_stop_reasons == [LlmStopReason.TOOL_USE]
    assert host.status_updates == [("session-1", agent, 3)]
    assert agent.tool_runner_hooks is None
