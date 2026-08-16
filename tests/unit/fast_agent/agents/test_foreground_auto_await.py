from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from mcp import CallToolRequest
from mcp_types import CallToolRequestParams, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import Settings, ShellSettings
from fast_agent.context import Context
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.tools.execution_environment import (
    ShellExecution,
    ShellExecutionCallbacks,
    ShellExecutionOptions,
    ShellExecutionRequest,
    ShellExecutionResult,
    ShellRuntimeInfo,
)
from fast_agent.tools.shell_process import process_result_metadata
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from mcp_types import Tool

    from fast_agent.llm.request_params import RequestParams


class _CompletesAfterInitialYieldEnvironment:
    def __init__(self) -> None:
        self.requests: list[ShellExecutionRequest] = []

    async def open(self) -> None:
        return None

    @property
    def cwd(self) -> str:
        return "/workspace"

    def runtime_info(self) -> ShellRuntimeInfo:
        return ShellRuntimeInfo(name="bash", kind="remote", provider="auto-await-test")

    async def execute(
        self,
        request: ShellExecutionRequest,
        *,
        callbacks: ShellExecutionCallbacks | None = None,
    ) -> ShellExecution:
        self.requests.append(request)
        if callbacks is not None:
            await callbacks.on_started(4321)
        await asyncio.sleep(0.04)
        if callbacks is not None:
            await callbacks.on_stdout("build complete\n")
        return ShellExecution(
            result=ShellExecutionResult(stdout="", stderr="", exit_code=0),
            options=ShellExecutionOptions(),
        )

    async def close(self) -> None:
        return None


class _WaitOnlyWhenShellResultIsRunningLlm(PassthroughLLM):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0
        self.process_wait_calls = 0

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        del request_params, tools, is_template
        self.call_count += 1
        if self.call_count == 1:
            return PromptMessageExtended(
                role="assistant",
                content=[text_content("run build")],
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "call-shell": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(
                            name="bash",
                            arguments={"command": "slow-build"},
                        ),
                    )
                },
            )

        latest = multipart_messages[-1]
        assert latest.tool_results is not None
        latest_result = next(iter(latest.tool_results.values()))
        metadata = process_result_metadata(latest_result)
        assert metadata is not None
        if metadata["process_status"] == "running":
            self.process_wait_calls += 1
            return PromptMessageExtended(
                role="assistant",
                content=[text_content("wait for build")],
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "call-process": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(
                            name="process",
                            arguments={
                                "process_id": metadata["process_id"],
                                "action": "wait",
                            },
                        ),
                    )
                },
            )
        return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


async def _run_agent(
    *, auto_await_max_seconds: int
) -> tuple[McpAgent, _WaitOnlyWhenShellResultIsRunningLlm]:
    environment = _CompletesAfterInitialYieldEnvironment()
    agent = McpAgent(
        config=AgentConfig(name="auto-await", shell=True),
        context=Context(
            config=Settings(
                shell_execution=ShellSettings(
                    tool_profile="minimal_process",
                    show_bash=False,
                    foreground_auto_await_max_seconds=auto_await_max_seconds,
                )
            )
        ),
        shell_environment=environment,
    )
    runtime = agent.shell_runtime
    assert runtime is not None
    runtime._idle_yield_seconds = 0.01
    runtime._foreground_yield_seconds = 1

    llm = _WaitOnlyWhenShellResultIsRunningLlm()
    agent._llm = llm
    await agent.generate("build it")
    return agent, llm


@pytest.mark.asyncio
async def test_auto_await_avoids_process_scheduling_inference_and_synthetic_history() -> None:
    agent, llm = await _run_agent(auto_await_max_seconds=1)

    assert llm.call_count == 2
    assert llm.process_wait_calls == 0
    assert [message.role for message in agent.message_history] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    tool_names = [
        call.params.name
        for message in agent.message_history
        for call in (message.tool_calls or {}).values()
    ]
    assert tool_names == ["bash"]
    shell_result_message = agent.message_history[2]
    assert shell_result_message.tool_results is not None
    shell_result = shell_result_message.tool_results["call-shell"]
    metadata = process_result_metadata(shell_result)
    assert metadata is not None
    assert metadata["process_status"] == "completed"
    assert metadata["foreground_auto_await"]["initial_yield_reason"] == "idle"
    assert isinstance(shell_result.content[0], TextContent)
    assert "build complete" in shell_result.content[0].text

    assert agent.shell_runtime is not None
    await agent.shell_runtime.close()
    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_zero_auto_await_preserves_real_model_managed_process_wait() -> None:
    agent, llm = await _run_agent(auto_await_max_seconds=0)

    assert llm.call_count == 3
    assert llm.process_wait_calls == 1
    tool_names = [
        call.params.name
        for message in agent.message_history
        for call in (message.tool_calls or {}).values()
    ]
    assert tool_names == ["bash", "process"]
    initial_result_message = agent.message_history[2]
    assert initial_result_message.tool_results is not None
    initial_result = initial_result_message.tool_results["call-shell"]
    initial_metadata = process_result_metadata(initial_result)
    assert initial_metadata is not None
    assert initial_metadata["process_yield_reason"] == "idle"
    assert initial_metadata["foreground_auto_await"]["outcome"] == "disabled"

    assert agent.shell_runtime is not None
    await agent.shell_runtime.close()
    await agent._aggregator.close()
