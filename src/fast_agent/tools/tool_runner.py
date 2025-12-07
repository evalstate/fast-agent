"""Async tool runner abstraction with parallel execution and hook points.

This provides a small orchestrator that repeatedly asks an LLM for a response,
executes any requested tool calls (optionally in parallel), and feeds the tool
results back into the conversation until the model stops requesting tools.

It is intentionally lightweight: callers supply the generation function, a tool
executor, and optional hooks for UI/telemetry. This keeps the runner reusable
outside the core Agent classes while allowing Agents to plug in custom routing
logic (e.g., MCP servers, shell runtime, filesystem runtime).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

from mcp.server.fastmcp.tools.base import Tool as FastMCPTool
from mcp.types import CallToolResult, Tool

from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL, FAST_AGENT_TOOL_TIMING
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui.console import console

ToolExecutor = Callable[[str, dict[str, Any] | None, str | None], Awaitable[CallToolResult]]
ToolValidator = Callable[[str], bool]
ToolResultFactory = Callable[
    [dict[str, CallToolResult], dict[str, dict[str, float | str | None]] | None, str | None],
    PromptMessageExtended,
]


@dataclass
class ToolRunnerHooks:
    """Optional callbacks for UI/telemetry integration."""

    on_tool_call: Callable[[str, dict[str, Any] | None, str | None], Awaitable[None] | None] | None = None
    on_tool_result: Callable[
        [str, CallToolResult, float, str | None], Awaitable[None] | None
    ] | None = None


class ToolRunner:
    """Async-only tool runner with parallel tool execution."""

    def __init__(
        self,
        *,
        generate_fn: Callable[[list[PromptMessageExtended]], Awaitable[PromptMessageExtended]],
        tools: Iterable[Tool],
        messages: list[PromptMessageExtended],
        max_iterations: int | None = None,
        use_history: bool = True,
        parallel_tool_calls: bool = True,
        concurrency_limit: int | None = None,
        tool_executor: ToolExecutor,
        tool_validator: ToolValidator | None = None,
        tool_result_factory: ToolResultFactory | None = None,
        hooks: ToolRunnerHooks | None = None,
    ) -> None:
        self._generate_fn = generate_fn
        self._messages = list(messages)
        self._max_iterations = max_iterations
        self._use_history = use_history
        self._parallel_tool_calls = parallel_tool_calls
        self._semaphore = asyncio.Semaphore(concurrency_limit) if concurrency_limit else None
        self._tool_executor = tool_executor
        self._tool_validator = tool_validator
        self._tool_result_factory = tool_result_factory or self._default_tool_result_factory
        self._hooks = hooks or ToolRunnerHooks()

        self._last_assistant: PromptMessageExtended | None = None
        self._iterator = self._run()

    def __aiter__(self):
        return self

    async def __anext__(self) -> PromptMessageExtended:
        return await self._iterator.__anext__()

    async def run_until_done(self) -> PromptMessageExtended:
        """Exhaust the iterator and return the last assistant message."""

        async for _ in self:
            pass
        assert self._last_assistant is not None
        return self._last_assistant

    async def _maybe_call(self, cb: Callable, *args: Any, **kwargs: Any) -> None:
        """Call a hook that may be sync or async."""

        if cb is None:
            return
        result = cb(*args, **kwargs)
        if asyncio.iscoroutine(result):
            await result

    async def _execute_tool_call(
        self, *, tool_use_id: str, tool_name: str, tool_args: dict[str, Any] | None
    ) -> tuple[str, CallToolResult, float]:
        """Execute a single tool call and return timing."""

        await self._maybe_call(self._hooks.on_tool_call, tool_name, tool_args, tool_use_id)

        start_time = asyncio.get_event_loop().time()
        result = await self._tool_executor(tool_name, tool_args, tool_use_id)
        duration_ms = round((asyncio.get_event_loop().time() - start_time) * 1000, 2)

        await self._maybe_call(self._hooks.on_tool_result, tool_name, result, duration_ms, tool_use_id)
        return tool_use_id, result, duration_ms

    async def _run(self):
        iterations = 0

        while True:
            console.print("[dim]ToolRunner: requesting assistant response[/dim]")
            assistant = await self._generate_fn(self._messages)
            self._last_assistant = assistant

            console.print(
                f"[dim]ToolRunner: got stop_reason={assistant.stop_reason} "
                f"tool_calls={list((assistant.tool_calls or {}).keys()) if assistant.tool_calls else []}[/dim]"
            )

            # Stop if no tool use was requested
            if assistant.stop_reason != LlmStopReason.TOOL_USE or not assistant.tool_calls:
                console.print("[dim]ToolRunner: no tool calls, stopping[/dim]")
                yield assistant
                return

            tool_calls = assistant.tool_calls
            tool_results: dict[str, CallToolResult] = {}
            tool_timings: dict[str, dict[str, float | str | None]] = {}
            tool_loop_error: str | None = None

            # Optional validation before dispatch
            if self._tool_validator:
                for tool_request in tool_calls.values():
                    if not self._tool_validator(tool_request.params.name):
                        tool_loop_error = f"Tool '{tool_request.params.name}' is not available"
                        break

            if tool_loop_error:
                error_result = CallToolResult(content=[text_content(tool_loop_error)], isError=True)
                # Attach the error to each tool call for clarity
                for correlation_id in tool_calls.keys():
                    tool_results[correlation_id] = error_result
                assistant.stop_reason = LlmStopReason.ERROR
                if assistant.content is None:
                    assistant.content = []
                assistant.content.append(text_content(tool_loop_error))
                yield assistant
                return

            async def _call_one(correlation_id: str, tool_name: str, tool_args: dict[str, Any] | None):
                if self._semaphore:
                    async with self._semaphore:
                        return await self._execute_tool_call(
                            tool_use_id=correlation_id, tool_name=tool_name, tool_args=tool_args
                        )
                return await self._execute_tool_call(
                    tool_use_id=correlation_id, tool_name=tool_name, tool_args=tool_args
                )

            tasks = [
                _call_one(correlation_id, call.params.name, call.params.arguments or {})
                for correlation_id, call in tool_calls.items()
            ]

            if self._parallel_tool_calls:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                results = []
                for task in tasks:
                    results.append(await task)

            for item, request in zip(results, tool_calls.items()):
                correlation_id = request[0]
                if isinstance(item, Exception):
                    error_text = f"Error executing tool '{tool_calls[correlation_id].params.name}': {item}"
                    tool_results[correlation_id] = CallToolResult(
                        content=[text_content(error_text)], isError=True
                    )
                    tool_timings[correlation_id] = {"timing_ms": None, "transport_channel": None}
                    continue

                tool_use_id, result, duration_ms = item
                tool_results[tool_use_id] = result
                tool_timings[tool_use_id] = {"timing_ms": duration_ms, "transport_channel": getattr(result, "transport_channel", None)}

            tool_message = self._tool_result_factory(tool_results, tool_timings, None)

            if self._use_history:
                self._messages = [tool_message]
            else:
                self._messages.extend([assistant, tool_message])

            iterations += 1
            if self._max_iterations is not None and iterations > self._max_iterations:
                assistant.stop_reason = LlmStopReason.MAX_ITERATIONS
                yield assistant
                return

            yield assistant

    @staticmethod
    def _default_tool_result_factory(
        tool_results: dict[str, CallToolResult],
        tool_timings: dict[str, dict[str, float | str | None]] | None,
        tool_loop_error: str | None,
    ) -> PromptMessageExtended:
        channels = None
        content = []

        if tool_loop_error:
            content.append(text_content(tool_loop_error))
            channels = {FAST_AGENT_ERROR_CHANNEL: [text_content(tool_loop_error)]}

        if tool_timings:
            if channels is None:
                channels = {}
            channels[FAST_AGENT_TOOL_TIMING] = [text_content(json.dumps(tool_timings))]

        return PromptMessageExtended(role="user", content=content, tool_results=tool_results, channels=channels)


# Developer-facing helpers -------------------------------------------------

def tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    **kwargs: Any,
):
    """Decorator to create a FastMCP tool from a function with optional naming.

    Mirrors the ergonomics of Anthropic's `@tool`/`@async_tool` helpers while
    emitting a FastMCP tool object that can be registered with agents or
    passed into the ToolRunner directly.
    """

    def _wrap(fn: Callable) -> FastMCPTool:
        return FastMCPTool.from_function(fn, name=name, description=description, **kwargs)

    if func is None:
        return _wrap
    return _wrap(func)


def async_tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    **kwargs: Any,
):
    """Alias for `tool` to emphasize async usage; FastMCP handles async funcs."""

    return tool(func, name=name, description=description, **kwargs)
