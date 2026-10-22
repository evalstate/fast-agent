"""Programmatic tool calling runtime backed by a pydantic-monty sandbox.

The model sees one tool, ``execute_python``. Its code runs in a persistent Monty REPL where
each agent tool is an async function. Nested calls are dispatched back through the agent's
normal tool path, so MCP routing, permissions and local runtimes still apply; only printed
output and the final expression value are returned to the model.
"""

from __future__ import annotations

import asyncio
import fnmatch
import importlib.util
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mcp_types import CallToolResult, TextContent, Tool

from fast_agent.core.exceptions import AgentConfigError
from fast_agent.ptc.stubs import PtcFunction, build_functions, render_stubs
from fast_agent.ptc.values import render_value, tool_result_to_value

if TYPE_CHECKING:
    from pydantic_monty import AsyncMonty, AsyncMontySession

    from fast_agent.config import ProgrammaticToolCallingSettings

PTC_TOOL_NAME = "execute_python"
PTC_TOOL_META = {"variant": "code", "code_arg": "code", "language": "python"}
_MAX_SUSPENSIONS = 100_000

ToolDispatcher = Callable[[str, dict[str, Any], str], Awaitable[CallToolResult]]
"""Dispatch a nested call: (tool name, arguments, nested call id) -> result."""

_DESCRIPTION = """\
Run Python code to orchestrate and compose tool calls (programmatic tool calling).

- Code runs in Monty, a sandboxed subset of Python 3: no filesystem, network, environment \
or third-party packages. Importable modules: asyncio, json, math, re, datetime, time, random, \
collections, itertools, functools, dataclasses, copy, base64, typing.
- Every tool listed below is a pre-defined async function. Call it with keyword arguments \
and `await`, e.g. `result = await some_tool(arg="value")`. Do not import or redefine them.
- Run independent calls concurrently with `import asyncio` then \
`a, b = await asyncio.gather(tool_a(x=1), tool_b(y=2))` (positional awaitables only).
- Tool results are Python values: structured content as a dict, JSON text parsed into \
dict/list, other text as str. A failing tool raises RuntimeError, which you can catch.
- Only `print()` output and the value of the final expression are returned to you. Filter, \
aggregate and summarise large tool results in code rather than returning them whole.
- Variables and functions persist between calls, so later code can reuse earlier results.
{type_check_note}\
- Top-level `await` is supported. Not supported: class inheritance, generators/`yield`, \
`del`, `async with`/`async for`, and `import *`.

Available functions:
```python
{stubs}```
"""
_TYPE_CHECK_NOTE = (
    "- Code is type-checked against these signatures first; on type errors nothing runs.\n"
)


@dataclass(slots=True)
class _ExecutionStats:
    completed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    started: int = 0


class ProgrammaticToolRuntime:
    """Owns the Monty pool and REPL session used to run model-written tool orchestration."""

    def __init__(self, settings: ProgrammaticToolCallingSettings) -> None:
        if importlib.util.find_spec("pydantic_monty") is None:
            raise AgentConfigError(
                "Programmatic tool calling requires pydantic-monty",
                "Install it with: uv pip install 'fast-agent-mcp[ptc]'",
            )
        self._settings = settings
        self._functions: dict[str, PtcFunction] = {}
        self._stubs = render_stubs({})
        self._pool: AsyncMonty | None = None
        self._pool_stack = AsyncExitStack()
        self._session: AsyncMontySession | None = None
        self._session_stack = AsyncExitStack()
        self._session_stubs: str | None = None
        self._lock = asyncio.Lock()
        self._executions = 0

    @property
    def functions(self) -> Mapping[str, PtcFunction]:
        return self._functions

    def model_tools(self, tools: Sequence[Tool]) -> list[Tool]:
        """Replace the agent's tools with the code tool, keeping configured direct tools."""
        direct = [tool for tool in tools if self._is_direct(tool.name)]
        nested = [tool for tool in tools if not self._is_direct(tool.name)]
        self._functions = build_functions(nested)
        self._stubs = render_stubs(self._functions)
        return [self.tool(), *direct]

    def tool(self) -> Tool:
        note = _TYPE_CHECK_NOTE if self._settings.type_check else ""
        stubs = "\n\n".join(function.stub() for function in self._functions.values())
        return Tool(
            name=PTC_TOOL_NAME,
            description=_DESCRIPTION.format(type_check_note=note, stubs=f"{stubs}\n"),
            input_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source to run. Not JSON, not markdown fenced.",
                    }
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            meta=dict(PTC_TOOL_META),
        )

    async def execute(
        self, arguments: Mapping[str, Any], dispatch: ToolDispatcher
    ) -> CallToolResult:
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return _error_result("`code` must be a non-empty string of Python source.")

        from pydantic_monty import (
            MontyCrashedError,
            MontyError,
            MontyRuntimeError,
            MontySyntaxError,
            MontyTypingError,
        )

        async with self._lock:
            self._executions += 1
            stats = _ExecutionStats()
            output: list[str] = []
            lookup = {
                name: self._host_function(function, dispatch, stats, execution=self._executions)
                for name, function in self._functions.items()
            }
            started = time.perf_counter()
            session = await self._ensure_session()
            try:
                value = await session.feed_run(
                    code,
                    external_lookup=lookup,
                    print_callback=lambda _stream, text: output.append(text),
                )
            except MontyTypingError as exc:
                return _error_result(
                    "Type check failed; the code was NOT executed. Fix it and retry.\n"
                    + exc.display()
                )
            except MontySyntaxError as exc:
                return _error_result(f"Syntax error; the code was NOT executed.\n{exc.display()}")
            except MontyRuntimeError as exc:
                detail = exc.display()
                if "time limit exceeded" in detail:
                    # Monty gives no heap guarantees after a timeout: start a fresh session.
                    await self._reset_session()
                    detail += "\nThe sandbox session was reset; earlier variables are gone."
                return self._failure(detail, output, stats, started)
            except MontyCrashedError as exc:
                await self._reset_session()
                detail = f"{exc}\nThe sandbox crashed and was reset; earlier variables are gone."
                return self._failure(detail, output, stats, started)
            except MontyError as exc:
                return self._failure(str(exc), output, stats, started)

            return self._success(value, output, stats, started)

    async def close(self) -> None:
        await self._session_stack.aclose()
        await self._pool_stack.aclose()
        self._session = None
        self._pool = None

    def _is_direct(self, tool_name: str) -> bool:
        return tool_name != PTC_TOOL_NAME and any(
            fnmatch.fnmatchcase(tool_name, pattern) for pattern in self._settings.direct_tools
        )

    def _host_function(
        self,
        function: PtcFunction,
        dispatch: ToolDispatcher,
        stats: _ExecutionStats,
        *,
        execution: int,
    ) -> Callable[..., Awaitable[Any]]:
        async def call(*args: Any, **kwargs: Any) -> Any:
            if args:
                raise TypeError(f"{function.python_name}() accepts keyword arguments only")
            stats.started += 1
            call_id = f"ptc-{execution}-{stats.started}"
            result = await dispatch(function.tool_name, function.bind_arguments(kwargs), call_id)
            (stats.failed if result.is_error else stats.completed).append(function.tool_name)
            return tool_result_to_value(result)

        return call

    async def _ensure_session(self) -> AsyncMontySession:
        if self._session is not None and self._session_stubs == self._stubs:
            return self._session
        if self._session is not None:
            # The tool surface changed; stubs are fixed per checkout, so start a new session.
            await self._reset_session()

        if self._pool is None:
            from pydantic_monty import AsyncMonty

            self._pool = await self._pool_stack.enter_async_context(
                AsyncMonty(min_processes=1, max_processes=2)
            )
        session = self._pool.checkout(
            script_name="ptc.py",
            type_check=self._settings.type_check,
            type_check_stubs=self._stubs if self._settings.type_check else None,
            type_check_format="concise",
            limits={
                "max_feed_duration_secs": self._settings.max_duration_seconds,
                "max_memory": self._settings.max_memory_mb * 1024 * 1024,
                "max_suspensions": _MAX_SUSPENSIONS,
            },
        )
        self._session = await self._session_stack.enter_async_context(session)
        self._session_stubs = self._stubs
        return self._session

    async def _reset_session(self) -> None:
        await self._session_stack.aclose()
        self._session_stack = AsyncExitStack()
        self._session = None
        self._session_stubs = None

    def _success(
        self, value: Any, output: list[str], stats: _ExecutionStats, started: float
    ) -> CallToolResult:
        sections = [_header("completed", stats, started)]
        printed = "".join(output).rstrip()
        if printed:
            sections.append(f"Output:\n{printed}")
        if value is not None:
            sections.append(f"Result:\n{render_value(value)}")
        if not printed and value is None:
            sections.append("(no output; print() values or end with an expression)")
        return CallToolResult(content=[TextContent(type="text", text="\n".join(sections))])

    def _failure(
        self, detail: str, output: list[str], stats: _ExecutionStats, started: float
    ) -> CallToolResult:
        sections = [_header("failed", stats, started)]
        printed = "".join(output).rstrip()
        if printed:
            sections.append(f"Output before the error:\n{printed}")
        if stats.completed:
            sections.append(
                "Tool calls that completed before the error (avoid repeating side effects): "
                + ", ".join(stats.completed)
            )
        sections.append(f"Error:\n{detail}")
        return _error_result("\n".join(sections))


def _header(status: str, stats: _ExecutionStats, started: float) -> str:
    elapsed = time.perf_counter() - started
    calls = len(stats.completed) + len(stats.failed)
    failed = f", {len(stats.failed)} failed" if stats.failed else ""
    return f"Script {status} in {elapsed:.2f}s ({calls} tool calls{failed})"


def _error_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)
