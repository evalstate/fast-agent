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
from io import StringIO
from typing import TYPE_CHECKING, Any

from mcp_types import CallToolResult, ContentBlock, TextContent, Tool

from fast_agent.constants import FAST_AGENT_PTC_CALLS, FAST_AGENT_SUBAGENT_RESULT_METADATA
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.mcp.tool_result_metadata import (
    add_url_elicitation_required_payloads,
    url_elicitation_required_payloads,
)
from fast_agent.ptc.stubs import PtcFunction, build_functions, render_stubs, tool_ptc_options
from fast_agent.ptc.values import MediaStore, render_value, tool_result_to_value
from fast_agent.tools.invocation_context import NESTED_PTC_CALL_SEPARATOR, programmatic_tool_call
from fast_agent.tools.tool_surface import (
    ProgrammaticSurface,
    SurfaceTool,
    ToolExposure,
    ToolSurface,
    estimate_tokens,
    tool_definition_tokens,
)

if TYPE_CHECKING:
    from pydantic_monty import AsyncMonty, AsyncMontySession

    from fast_agent.config import ProgrammaticToolCallingSettings
    from fast_agent.mcp.url_elicitation_required import URLElicitationRequiredDisplayPayload

PTC_TOOL_NAME = "execute_python"
PTC_TOOL_META = {"variant": "code", "code_arg": "code", "language": "python"}
_MAX_SUSPENSIONS = 100_000
# Bound host-side print capture independently of Monty's sandbox memory limit. This protects
# host memory, not model context: bound_tool_result_for_llm still caps what the model sees, so
# do not tune this value for context size.
_MAX_PRINT_CHARS = 1_000_000

# Nested result metadata copied onto the execute_python result, which is what history keeps.
_CARRIED_RESULT_META = (FAST_AGENT_SUBAGENT_RESULT_METADATA,)

ToolDispatcher = Callable[[str, dict[str, Any], str], Awaitable[CallToolResult]]
"""Dispatch a nested call: (tool name, arguments, nested call id) -> result."""

_INSTRUCTIONS = """\
Run Python code to orchestrate and compose tool calls (programmatic tool calling).

- Code runs in Monty, a sandboxed subset of Python 3: no filesystem, network, environment \
or third-party packages. Importable modules: asyncio, json, math, re, datetime, time, random, \
collections, itertools, functools, dataclasses, copy, base64, typing.
- Every tool listed below is a pre-defined async function. Call it with keyword arguments \
and `await`, e.g. `result = await some_tool(arg="value")`. Do not import or redefine them.
- Parameters shown with `= ...` are optional: omit them to use the tool's default. Pass `None` \
only where the annotation allows it.
- Run independent calls concurrently with `import asyncio` then \
`a, b = await asyncio.gather(tool_a(x=1), tool_b(y=2))` (positional awaitables only).
- Tool results are Python values: structured content as a dict, JSON text parsed into \
dict/list, other text as str. A failing tool raises RuntimeError, which you can catch.
- Images, audio and binary resources in tool results arrive as handle dicts with a `media_id` \
key. To see media yourself, include its handles in the final expression, e.g. `shot` or \
`[a, b]`; printed handles are not attached.
- Only `print()` output and the value of the final expression are returned to you. Filter, \
aggregate and summarise large tool results in code rather than returning them whole.
- Variables and functions persist between calls, so later code can reuse earlier results.
{type_check_note}\
- Top-level `await` is supported. Not supported: class inheritance, generators/`yield`, \
`del`, `async with`/`async for`, and `import *`.
"""
_FUNCTIONS_BLOCK = """
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
    calls: list[dict[str, Any]] = field(default_factory=list)
    session_restarted: bool = False
    # Nested results are consumed in the sandbox, so URL elicitations must be carried to the
    # execute_python result for the user to see them.
    url_elicitations: list[URLElicitationRequiredDisplayPayload] = field(default_factory=list)


@dataclass(slots=True)
class _PrintedOutput:
    buffer: StringIO = field(default_factory=StringIO)
    size: int = 0
    truncated: bool = False

    def append(self, _stream: str, text: str) -> None:
        remaining = _MAX_PRINT_CHARS - self.size
        if len(text) > remaining:
            self.truncated = True
            text = text[:remaining]
        if text:
            self.buffer.write(text)
            self.size += len(text)

    def render(self) -> str:
        printed = self.buffer.getvalue().rstrip()
        if self.truncated:
            printed += "\n[Printed output truncated]"
        return printed


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
        self._nested_tools: dict[str, Tool] = {}
        self._stubs = render_stubs({})
        self._pool: AsyncMonty | None = None
        self._pool_stack = AsyncExitStack()
        self._session: AsyncMontySession | None = None
        self._session_stack = AsyncExitStack()
        self._session_stubs: str | None = None
        self._lock = asyncio.Lock()
        self._executions = 0
        self._media = MediaStore()

    @property
    def functions(self) -> Mapping[str, PtcFunction]:
        return self._functions

    def model_tools(self, tools: Sequence[Tool]) -> list[Tool]:
        """Replace the agent's tools with the code tool, keeping configured direct tools."""
        direct = [tool for tool in tools if self._direct_reason(tool) is not None]
        nested = [tool for tool in tools if self._direct_reason(tool) is None]
        self._functions = build_functions(nested)
        self._nested_tools = {tool.name: tool for tool in nested}
        self._stubs = render_stubs(self._functions)
        return [self.tool(), *direct]

    def surface(self, model_tools: Sequence[Tool]) -> ToolSurface:
        """Describe ``model_tools`` (as returned by ``model_tools()``) and the sandbox functions."""
        exposed = [
            SurfaceTool(
                tool=tool,
                exposure=ToolExposure.MODEL,
                context_tokens=tool_definition_tokens(tool),
                direct_reason=self._direct_reason(tool),
            )
            for tool in model_tools
        ]
        nested = [
            SurfaceTool(
                tool=self._nested_tools[function.tool_name],
                exposure=ToolExposure.PROGRAMMATIC,
                context_tokens=estimate_tokens(function.stub()),
                python_name=function.python_name,
                signature=function.signature(),
                stub=function.stub(),
            )
            for function in self._functions.values()
        ]
        return ToolSurface(
            tools=(*exposed, *nested),
            programmatic=ProgrammaticSurface(
                tool_name=PTC_TOOL_NAME,
                instructions=self._instructions(),
                type_check=self._settings.type_check,
                max_duration_seconds=self._settings.max_duration_seconds,
                max_memory_mb=self._settings.max_memory_mb,
            ),
        )

    def _instructions(self) -> str:
        note = _TYPE_CHECK_NOTE if self._settings.type_check else ""
        return _INSTRUCTIONS.format(type_check_note=note)

    def tool(self) -> Tool:
        stubs = "\n\n".join(function.stub() for function in self._functions.values())
        return Tool(
            name=PTC_TOOL_NAME,
            description=self._instructions() + _FUNCTIONS_BLOCK.format(stubs=f"{stubs}\n"),
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
        self,
        arguments: Mapping[str, Any],
        dispatch: ToolDispatcher,
        *,
        parent_call_id: str | None = None,
    ) -> CallToolResult:
        """Run model code; nested call ids derive from ``parent_call_id`` when it is known."""
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
            output = _PrintedOutput()
            dispatches: list[asyncio.Task[Any]] = []
            call_prefix = (
                f"{parent_call_id}{NESTED_PTC_CALL_SEPARATOR}"
                if parent_call_id
                else f"ptc-{self._executions}-"
            )
            lookup = {
                name: self._host_function(
                    function, dispatch, stats, dispatches, call_prefix=call_prefix
                )
                for name, function in self._functions.items()
            }
            started = time.perf_counter()
            try:
                session, stats.session_restarted = await self._ensure_session()
                try:
                    value = await session.feed_run(
                        code,
                        external_lookup=lookup,
                        print_callback=output.append,
                    )
                except Exception:
                    # A failed gather can leave sibling tools running. Settle them before
                    # reporting completed side effects and copying nested result metadata.
                    await asyncio.gather(*dispatches, return_exceptions=True)
                    raise
                await asyncio.gather(*dispatches, return_exceptions=True)
            except asyncio.CancelledError:
                for task in dispatches:
                    task.cancel()
                await asyncio.gather(*dispatches, return_exceptions=True)
                # Monty finishes a cancelled checkout; it cannot be reused on the next turn.
                await self._reset_session()
                raise
            except MontyTypingError as exc:
                return _error_result(
                    "Type check failed; the code was NOT executed. Fix it and retry.\n"
                    + _restart_note(stats)
                    + exc.display()
                )
            except MontySyntaxError as exc:
                return _error_result(
                    "Syntax error; the code was NOT executed.\n"
                    + _restart_note(stats)
                    + exc.display()
                )
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

    def invalidate_session(self) -> None:
        """Discard conversation state on the next execution, which can close it asynchronously."""
        self._session_stubs = None

    async def close(self) -> None:
        await self._session_stack.aclose()
        await self._pool_stack.aclose()
        self._session = None
        self._pool = None

    def _direct_reason(self, tool: Tool) -> str | None:
        """Return why ``tool`` stays directly callable, or None when it runs in the sandbox."""
        if tool.name == PTC_TOOL_NAME:
            return None
        if tool_ptc_options(tool).get("expose") == "direct":
            return "declared by tool"
        pattern = next(
            (
                pattern
                for pattern in self._settings.direct_tools
                if fnmatch.fnmatchcase(tool.name, pattern)
            ),
            None,
        )
        return None if pattern is None else f'ptc.direct_tools "{pattern}"'

    def _host_function(
        self,
        function: PtcFunction,
        dispatch: ToolDispatcher,
        stats: _ExecutionStats,
        dispatches: list[asyncio.Task[Any]],
        *,
        call_prefix: str,
    ) -> Callable[..., Awaitable[Any]]:
        async def run(*args: Any, **kwargs: Any) -> Any:
            if args:
                raise TypeError(f"{function.python_name}() accepts keyword arguments only")
            stats.started += 1
            call_id = f"{call_prefix}{stats.started}"
            with programmatic_tool_call(call_id):
                result = await dispatch(
                    function.tool_name, function.bind_arguments(kwargs), call_id
                )
            (stats.failed if result.is_error else stats.completed).append(function.tool_name)
            stats.url_elicitations.extend(url_elicitation_required_payloads(result))
            stats.calls.append(_call_record(call_id, function.tool_name, result))
            return tool_result_to_value(result, self._media)

        async def call(*args: Any, **kwargs: Any) -> Any:
            task = asyncio.create_task(run(*args, **kwargs))
            dispatches.append(task)
            # Own dispatch lifetime independently of Monty's callback tasks. Script errors
            # drain these tasks; turn cancellation explicitly cancels and settles them.
            return await asyncio.shield(task)

        return call

    async def _ensure_session(self) -> tuple[AsyncMontySession, bool]:
        """Return the REPL session and whether a tool-surface change just restarted it."""
        if self._session is not None and self._session_stubs == self._stubs:
            return self._session, False
        # A cleared _session_stubs means a conversation reset, which needs no notice.
        restarted = self._session is not None and self._session_stubs is not None
        if self._session is not None:
            # Conversation resets and tool-surface changes both require a fresh checkout.
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
        return self._session, restarted

    async def _reset_session(self) -> None:
        await self._session_stack.aclose()
        self._session_stack = AsyncExitStack()
        self._session = None
        self._session_stubs = None
        self._media.clear()

    def _success(
        self, value: Any, output: _PrintedOutput, stats: _ExecutionStats, started: float
    ) -> CallToolResult:
        sections = [_header("completed", stats, started)]
        printed = output.render()
        if printed:
            sections.append(f"Output:\n{printed}")
        if value is not None:
            sections.append(f"Result:\n{render_value(value)}")
        if not printed and value is None:
            sections.append("(no output; print() values or end with an expression)")
        media = self._media.referenced(value)
        if media:
            sections.append(f"Attached {len(media)} media item(s) referenced by the result.")
        content: list[ContentBlock] = [TextContent(type="text", text="\n".join(sections))]
        result = CallToolResult(content=[*content, *media])
        _add_nested_metadata(result, stats)
        return result

    def _failure(
        self, detail: str, output: _PrintedOutput, stats: _ExecutionStats, started: float
    ) -> CallToolResult:
        sections = [_header("failed", stats, started)]
        printed = output.render()
        if printed:
            sections.append(f"Output before the error:\n{printed}")
        if stats.completed:
            sections.append(
                "Tool calls that completed (avoid repeating side effects): "
                + ", ".join(stats.completed)
            )
        sections.append(f"Error:\n{detail}")
        result = _error_result("\n".join(sections))
        _add_nested_metadata(result, stats)
        return result


def _call_record(call_id: str, tool_name: str, result: CallToolResult) -> dict[str, Any]:
    record: dict[str, Any] = {"id": call_id, "tool": tool_name, "is_error": result.is_error}
    meta = result.meta or {}
    record.update({key: meta[key] for key in _CARRIED_RESULT_META if key in meta})
    return record


def _add_nested_metadata(result: CallToolResult, stats: _ExecutionStats) -> None:
    add_url_elicitation_required_payloads(result, stats.url_elicitations)
    if stats.calls:
        result.meta = {**(result.meta or {}), FAST_AGENT_PTC_CALLS: stats.calls}


def _header(status: str, stats: _ExecutionStats, started: float) -> str:
    elapsed = time.perf_counter() - started
    calls = len(stats.completed) + len(stats.failed)
    failed = f", {len(stats.failed)} failed" if stats.failed else ""
    header = f"Script {status} in {elapsed:.2f}s ({calls} tool calls{failed})"
    note = _restart_note(stats).rstrip()
    return f"{header}\n{note}" if note else header


def _restart_note(stats: _ExecutionStats) -> str:
    if not stats.session_restarted:
        return ""
    return (
        "Note: the available tools changed, so the sandbox session was restarted before this "
        "run; variables from earlier runs are gone.\n"
    )


def _error_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)
