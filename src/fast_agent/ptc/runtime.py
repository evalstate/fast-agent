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
from typing import TYPE_CHECKING, Any, Literal

from mcp_types import CallToolResult, ContentBlock, TextContent, Tool

from fast_agent.constants import FAST_AGENT_PTC_CALLS, FAST_AGENT_SUBAGENT_RESULT_METADATA
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.mcp.tool_result_metadata import (
    add_url_elicitation_required_payloads,
    url_elicitation_required_payloads,
)
from fast_agent.ptc.stubs import PtcFunction, build_functions, render_stubs, tool_ptc_options
from fast_agent.ptc.syntax import describe_syntax_error
from fast_agent.ptc.values import MediaStore, render_value, tool_result_to_value
from fast_agent.tools.apply_patch_tool import (
    APPLY_PATCH_INPUT_FIELD as FREEFORM_TOOL_INPUT_FIELD,
)
from fast_agent.tools.apply_patch_tool import OPENAI_RESPONSES_CUSTOM_TOOL_META_KEY
from fast_agent.tools.invocation_context import NESTED_PTC_CALL_SEPARATOR, programmatic_tool_call
from fast_agent.tools.tool_surface import (
    ProgrammaticSurface,
    SurfaceTool,
    ToolExposure,
    ToolSurface,
    estimate_tokens,
    tool_definition_tokens,
)
from fast_agent.utils.tool_names import is_shell_command_tool_name

if TYPE_CHECKING:
    from pydantic_monty import AsyncMonty, AsyncMontySession

    from fast_agent.config import ProgrammaticToolCallingSettings
    from fast_agent.mcp.url_elicitation_required import URLElicitationRequiredDisplayPayload

PTC_TOOL_NAME = "execute_python"
PTC_TOOL_META = {"variant": "code", "code_arg": "code", "language": "python"}
PtcToolFormat = Literal["json", "freeform"]
# Responses custom tool calls carry their raw input under this argument name.
_FREEFORM_CODE_ARG = FREEFORM_TOOL_INPUT_FIELD
_FREEFORM_TOOL_META = {
    **PTC_TOOL_META,
    "code_arg": _FREEFORM_CODE_ARG,
    "code_format": "raw",
}
_FREEFORM_GRAMMAR = r"""
start: SOURCE
SOURCE: /[\s\S]+/
"""
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
Run Python that calls tools as functions. Only `print()` output and the final expression \
value are returned to you: filter and aggregate large results in code.
{input_note}\
- Sandbox: a subset of Python 3.14 with no file access, network, environment or packages. \
Importable modules: asyncio, base64, binascii, bisect, collections, copy, csv, dataclasses, \
datetime, functools, hashlib, itertools, json, math, os.path, random, re, sys, time, tomllib, \
typing, unicodedata. Some are partial (functools has only reduce and partial). Not available: \
enum, contextlib, io, string, struct, operator, heapq, decimal, subprocess.
{shell_python_note}\
- Each function below is a pre-defined async tool: `r = await tool(arg="value")`, keyword \
arguments only, no import needed. `= ...` marks an optional parameter; pass `None` only \
where the type allows it.
{shared_note}\
- Run independent calls together (asyncio is pre-imported): \
`a, b = await asyncio.gather(f(x=1), g(y=2))`; keep dependent steps sequential. `settle(aw)` returns an exception instead of raising, so \
`await asyncio.gather(*[settle(f(x=x)) for x in xs])` returns every result \
(`return_exceptions=` is unsupported).
{shell_batch_note}\
- Results are Python values: structured content as a dict, JSON text parsed, other text as \
str. A failing tool raises RuntimeError.
- Images, audio and binary resources arrive as `{{"media_id": ...}}` handles. To see them, \
return the handles in the final expression; printed handles are not attached. A structured \
result with such blocks is `{{"structured_content": ..., "content": [...]}}`.
- Variables persist between calls.{type_check_note}
- Top-level `await` works. Unsupported: class inheritance, decorators, `yield`, `del`, \
`async with`/`async for`, `except*`, `import *`.
"""
# Helpers defined in every sandbox session before model code runs. asyncio is pre-imported
# because models routinely call asyncio.gather without importing it.
_PRELUDE = """\
import asyncio


async def settle(awaitable):
    try:
        return await awaitable
    except Exception as exc:
        return exc
"""
_FUNCTIONS_BLOCK = """
Available functions:
```python
{stubs}```
"""
_FREEFORM_INPUT_NOTE = (
    "This is a FREEFORM tool: send the Python source itself, not JSON, quotes or markdown fences.\n"
)
_SHELL_BATCH_NOTE = (
    "- Run independent shell commands as separate gathered `shell` calls, not chained with "
    "separators; each result keeps its own stdout, stderr and exit_code.\n"
)
_SHELL_PYTHON_NOTE = (
    "- For full CPython or packages, run a script with the shell tool (e.g. `python3 x.py`), "
    "outside this sandbox.\n"
)
_SHARED_NOTE = (
    "- {functions} can also be called directly as tools: directly for a one-off step, "
    "especially with quotes, heredocs or source code; from code for several steps or to "
    "process results.\n"
)
_TYPE_CHECK_NOTE = " Code is type-checked against these signatures first; on errors nothing runs."


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
        # Functions whose tools the model can also call directly.
        self._shared_functions: tuple[str, ...] = ()
        self._has_shell_tools = False
        self._has_nested_shell_tools = False
        self._tool_format: PtcToolFormat = "json"
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

    def resolve_tool_format(self, model_default: PtcToolFormat | None) -> PtcToolFormat:
        """Apply ``ptc.tool_format``, using the model's default when it is ``auto``."""
        if self._settings.tool_format != "auto":
            return self._settings.tool_format
        return model_default or "json"

    def model_tools(
        self, tools: Sequence[Tool], *, tool_format: PtcToolFormat = "json"
    ) -> list[Tool]:
        """Replace the agent's tools with the code tool, keeping configured direct tools."""
        self._tool_format = tool_format
        direct = [tool for tool in tools if self._direct_reason(tool) is not None]
        nested = [tool for tool in tools if self._is_nested(tool)]
        self._has_shell_tools = any(
            is_shell_command_tool_name(tool.name) for tool in (*direct, *nested)
        )
        self._has_nested_shell_tools = any(is_shell_command_tool_name(tool.name) for tool in nested)
        self._functions = build_functions(nested)
        self._nested_tools = {tool.name: tool for tool in nested}
        direct_names = {tool.name for tool in direct}
        self._shared_functions = tuple(
            function.python_name
            for function in self._functions.values()
            if function.tool_name in direct_names
        )
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
                tool_format=self._tool_format,
                instructions=self._instructions(),
                type_check=self._settings.type_check,
                max_duration_seconds=self._settings.max_duration_seconds,
                max_memory_mb=self._settings.max_memory_mb,
            ),
        )

    def _instructions(self) -> str:
        shared = ", ".join(f"`{name}`" for name in self._shared_functions)
        return _INSTRUCTIONS.format(
            shell_python_note=_SHELL_PYTHON_NOTE if self._has_shell_tools else "",
            shell_batch_note=_SHELL_BATCH_NOTE if self._has_nested_shell_tools else "",
            shared_note=_SHARED_NOTE.format(functions=shared) if shared else "",
            input_note=_FREEFORM_INPUT_NOTE if self._tool_format == "freeform" else "",
            type_check_note=_TYPE_CHECK_NOTE if self._settings.type_check else "",
        )

    def tool_meta(self) -> dict[str, Any]:
        """Display metadata for the code tool in its current input format."""
        return dict(_FREEFORM_TOOL_META if self._tool_format == "freeform" else PTC_TOOL_META)

    def tool(self) -> Tool:
        stubs = "\n\n".join(function.stub() for function in self._functions.values())
        freeform = self._tool_format == "freeform"
        code_arg = _FREEFORM_CODE_ARG if freeform else "code"
        meta = self.tool_meta()
        if freeform:
            # Providers without custom tools ignore this and send the JSON schema below.
            meta[OPENAI_RESPONSES_CUSTOM_TOOL_META_KEY] = {
                "type": "custom",
                "format": {"type": "grammar", "syntax": "lark", "definition": _FREEFORM_GRAMMAR},
            }
        return Tool(
            name=PTC_TOOL_NAME,
            description=self._instructions() + _FUNCTIONS_BLOCK.format(stubs=f"{stubs}\n"),
            input_schema={
                "type": "object",
                "properties": {
                    code_arg: {
                        "type": "string",
                        "description": "Python source to run. Not JSON, not markdown fenced.",
                    }
                },
                "required": [code_arg],
                "additionalProperties": False,
            },
            meta=meta,
        )

    async def execute(
        self,
        arguments: Mapping[str, Any],
        dispatch: ToolDispatcher,
        *,
        parent_call_id: str | None = None,
    ) -> CallToolResult:
        """Run model code; nested call ids derive from ``parent_call_id`` when it is known."""
        # Accept both input formats: history can contain calls made before a model switch.
        code = arguments.get("code", arguments.get(_FREEFORM_CODE_ARG))
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
                # The type checker also reports syntax errors, mixed with errors they cause.
                detail = exc.display()
                report = self._describe_syntax_error(code) if "invalid-syntax" in detail else None
                if report is not None:
                    return _syntax_error_result(report, stats)
                return _error_result(
                    "Type check failed; the code was NOT executed. Fix it and retry.\n"
                    + _restart_note(stats)
                    + detail
                )
            except MontySyntaxError as exc:
                return _syntax_error_result(
                    self._describe_syntax_error(code) or exc.display(), stats
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

    def _describe_syntax_error(self, code: str) -> str | None:
        return describe_syntax_error(code, direct_functions=self._shared_functions)

    def invalidate_session(self) -> None:
        """Discard conversation state on the next execution, which can close it asynchronously."""
        self._session_stubs = None

    async def close(self) -> None:
        await self._session_stack.aclose()
        await self._pool_stack.aclose()
        self._session = None
        self._pool = None

    def _direct_reason(self, tool: Tool) -> str | None:
        """Return why ``tool`` stays directly callable, or None when only code can call it."""
        if tool.name == PTC_TOOL_NAME:
            return None
        pattern = _matching_pattern(tool, self._settings.direct_tools)
        if pattern is not None:
            return f'ptc.direct_tools "{pattern}"'
        if _matching_pattern(tool, self._settings.nested_tools) is not None:
            return None
        expose = tool_ptc_options(tool).get("expose")
        if expose == "direct":
            return "declared by tool"
        if expose == "both":
            return "declared by tool; also callable from code"
        return None

    def _is_nested(self, tool: Tool) -> bool:
        """Return whether sandbox code can call ``tool``; ``ptc.direct_tools`` excludes it."""
        if tool.name == PTC_TOOL_NAME:
            return False
        if _matching_pattern(tool, self._settings.direct_tools) is not None:
            return False
        if _matching_pattern(tool, self._settings.nested_tools) is not None:
            return True
        return tool_ptc_options(tool).get("expose", "nested") in {"nested", "both"}

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
            return tool_result_to_value(result, self._media, name=function.python_name)

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
        # Defined in the sandbox (not a host function) so it can await sandbox coroutines.
        await self._session.feed_run(_PRELUDE)
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


def _matching_pattern(tool: Tool, patterns: Sequence[str]) -> str | None:
    return next((pattern for pattern in patterns if fnmatch.fnmatchcase(tool.name, pattern)), None)


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


def _syntax_error_result(report: str, stats: _ExecutionStats) -> CallToolResult:
    return _error_result(
        "Syntax error; the code was NOT executed.\n" + _restart_note(stats) + report
    )


def _error_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)
