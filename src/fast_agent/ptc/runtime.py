"""Programmatic tool calling runtime backed by a pydantic-monty sandbox.

The model sees one tool, ``execute_python``. Each run's code executes in a fresh Monty sandbox
where each agent tool is an async function; values a run keeps with ``store()`` are what later
runs can ``load()``. Nested calls are dispatched back through the agent's
normal tool path, so MCP routing, permissions and local runtimes still apply; only printed
output and the final expression value are returned to the model.
"""

from __future__ import annotations

import asyncio
import fnmatch
import importlib.util
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from io import StringIO
from typing import TYPE_CHECKING, Any, Literal

from mcp_types import CallToolResult, ContentBlock, TextContent, Tool

from fast_agent.constants import (
    FAST_AGENT_PTC_CALLS,
    FAST_AGENT_PTC_VALUE,
    FAST_AGENT_SUBAGENT_RESULT_METADATA,
)
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.mcp.tool_result_metadata import (
    add_url_elicitation_required_payloads,
    url_elicitation_required_payloads,
)
from fast_agent.ptc.store import STORE_STUBS, SandboxStore
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
from fast_agent.utils.text import summarize_command
from fast_agent.utils.tool_names import (
    POLL_PROCESS_TOOL_NAME,
    PROCESS_TOOL_NAME,
    is_shell_command_tool_name,
)

if TYPE_CHECKING:
    from pydantic_monty import AsyncMonty, AsyncMontySession

    from fast_agent.config import ProgrammaticToolCallingSettings
    from fast_agent.mcp.url_elicitation_required import URLElicitationRequiredDisplayPayload
    from fast_agent.ptc.files import SandboxFilesystem

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

PTC_BATCHING_INSTRUCTION = (
    "Batch independent searches, file reads, and other read-only calls in a single "
    "`execute_python` invocation using `asyncio.gather`. Prefer this over separate invocations "
    "or sequential awaits when no result depends on another. Inspect every result; use "
    "`settle()` when each result must be collected despite individual failures. Keep dependent "
    "steps, approvals, and potentially conflicting mutations sequential."
)

_INSTRUCTIONS = """\
Run Python code. Only `print()` output and the final expression value are returned to you: \
filter and aggregate large results in code.
{input_note}\
- {batching_instruction}
- asyncio is pre-imported: `a, b = await asyncio.gather(f(x=1), g(y=2))`. \
`settle(aw)` returns an exception instead of raising, so \
`await asyncio.gather(*[settle(f(x=x)) for x in xs])` returns every result \
(`return_exceptions=` is unsupported).
{shell_batch_note}\
{functions_note}\
{io_note}\
- Sandbox: a subset of Python 3.14 with no environment or packages. \
Importable modules: asyncio, base64, binascii, bisect, collections, copy, csv, dataclasses, \
datetime, decimal, functools, hashlib, itertools, json, math, os.path, random, re, sys, time, \
tomllib, typing, unicodedata. Some are partial (functools has only reduce and partial). Not \
available: enum, contextlib, io, string, struct, operator, heapq, subprocess.
{shell_python_note}\
{shared_note}\
- Results are Python values: structured content as a dict, JSON text parsed, other text as \
str. A failing call raises RuntimeError.
- Images, audio and binary resources arrive as `{{"media_id": ...}}` handles. To see them, \
return the handles in the final expression; printed handles are not attached. A structured \
result with such blocks is `{{"structured_content": ..., "content": [...]}}`.
- {store_note}{type_check_note}
- Top-level `await` works. Unsupported: class inheritance, method decorators (`@staticmethod`, \
`@classmethod`, `@property`), `yield`, `del`, \
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
# The sandbox callables are always "functions"; "tool" means only something the model can call
# directly (execute_python, or a function also exposed as a direct tool). Models act on "tool"
# by emitting a direct tool call, so the wording must not blur the two.
_FUNCTIONS_NOTE = (
    "- The functions listed below are already defined in this Python{only_code}. Where other "
    'instructions say to use a tool of the same name (e.g. "use the `{tool}` tool"), call '
    "`{call}` here.\n"
    '- Call them as `r = await name(arg="value")`: async, keyword arguments only, no import '
    "needed. `= ...` marks an optional parameter; pass `None` only where the type allows it.\n"
)
_SHELL_BATCH_NOTE = (
    "- Run independent shell commands as separate gathered `{function}` calls, not chained with "
    "separators; each result keeps its own stdout, stderr and exit_code.\n"
)
_SHELL_PYTHON_NOTE = (
    "- For full CPython or packages, write a script and run it with `python3` via `{call}`, "
    "outside this sandbox.\n"
)
_DIRECT_SHELL_PYTHON_NOTE = (
    "- For full CPython or packages, run a script (e.g. `python3 x.py`) with the `{tool}` tool, "
    "outside this sandbox.\n"
)
_SHARED_NOTE = (
    "- {functions} can also be called directly as tools: directly for a one-off step, "
    "especially with quotes, heredocs or source code; from code for several steps or to "
    "process results.\n"
)
_IO_NOTE = (
    "- The functions are your only access to files, commands and the network, e.g. "
    "`r = {example}`. Python's own `open()`, `os`, `pathlib` and `subprocess` cannot reach "
    "files or processes here.\n"
)
_FILES_NOTE = (
    "- `open()`, `pathlib.Path` and the `os` file functions (`os.listdir`, `os.walk`, "
    "`os.path.exists`, `os.makedirs`, ...) work on the real files; relative paths start in the "
    "working directory. File access is experimental local Python pass-through, not a "
    "filesystem sandbox; it does not invoke tools or tool permission prompts.{commands}\n"
)
_READ_LIMITS_NOTE = (
    "- File reads load the whole file, even for `read(n)` or `seek()`: at most {binary} binary "
    "or {text} text. `struct` and `int.from_bytes` are unavailable.{advice}\n"
)
_READ_LIMITS_ADVICE = (
    " For large or binary files, run commands such as `head -c`, `xxd` or `python3` with "
    "`{shell}` instead."
)
_COMMANDS_NOTE = (
    " Commands and the network need the functions, e.g. `r = {example}`; "
    "`subprocess` is unavailable."
)
# Sandbox errors from trying to do I/O directly instead of through tool functions.
_DIRECT_IO_ERROR = re.compile(
    r"PermissionError|`open` used when not defined|Module `os` has no member"
    r"|Cannot resolve imported module `(?:subprocess|shutil|glob|io|socket|urllib|http|requests)"
)
# The same, when file operations are served by tools: only processes and other I/O are missing.
_UNSUPPORTED_IO_ERROR = re.compile(
    r"Module `os` has no member"
    r"|Cannot resolve imported module `(?:subprocess|shutil|glob|io|socket|urllib|http|requests)"
)
# pydantic-monty (1.3, 1.4) runs os.walk, os.scandir and map, but its type-checker stubs omit them.
_STUB_GAP = re.compile(
    r"Module `os` has no member `(?:walk|scandir)`|Name `map` used when not defined"
)
# How Monty renders a coroutine from a function call that was never awaited.
_UNAWAITED_CALL = re.compile(r"<coroutine external_future\(\d+\)>")
_MAX_HINT_EXAMPLES = 4
_TYPE_CHECK_NOTE = " Code is type-checked against these signatures first; on errors nothing runs."
# Each run gets a fresh sandbox; the store is the only state that carries between runs.
_STORE_NOTE = (
    "Each run starts in a fresh sandbox: variables, functions and imports do not carry over. "
    'Keep values later runs need with `await store("rows", rows)` and get them back with '
    '`rows = await load("rows")`; `await stored()` lists what is kept.'
)


@dataclass(slots=True)
class _ExecutionStats:
    completed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    started: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)
    # Nested results are consumed in the sandbox, so URL elicitations must be carried to the
    # execute_python result for the user to see them.
    url_elicitations: list[URLElicitationRequiredDisplayPayload] = field(default_factory=list)
    # Commands the shell returned while still running, by process id, with their command
    # text; a later wait or stop that sees them end removes them.
    running: dict[str, str] = field(default_factory=dict)

    def track_process(self, arguments: Mapping[str, Any], result: CallToolResult) -> None:
        value = (result.meta or {}).get(FAST_AGENT_PTC_VALUE)
        if not isinstance(value, Mapping) or not isinstance(value.get("process_id"), str):
            return
        process_id = value["process_id"]
        if value.get("status") != "running":
            self.running.pop(process_id, None)
        elif process_id not in self.running:
            self.running[process_id] = str(arguments.get("command") or "")


@dataclass(slots=True)
class _NestedCalls:
    """Runs one execution's nested tool calls, for sandbox functions and file operations."""

    dispatch: ToolDispatcher
    stats: _ExecutionStats
    call_prefix: str
    # Outstanding dispatches only: a finished task holds its full result, which sandbox code may
    # long since have discarded, so keeping them all would grow host memory with fetched data.
    tasks: set[asyncio.Task[CallToolResult]] = field(default_factory=set)

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> CallToolResult:
        task = asyncio.create_task(self._run(tool_name, arguments))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        # Own dispatch lifetime independently of Monty's callback tasks. Script errors
        # drain these tasks; turn cancellation explicitly cancels and settles them.
        return await asyncio.shield(task)

    async def _run(self, tool_name: str, arguments: dict[str, Any]) -> CallToolResult:
        stats = self.stats
        stats.started += 1
        call_id = f"{self.call_prefix}{stats.started}"
        # Snapshot before dispatch: tools may mutate arguments, and parallel calls may
        # finish out of order. This ledger is metadata, never model-visible output.
        record = _started_call_record(call_id, tool_name, arguments)
        stats.calls.append(record)
        try:
            with programmatic_tool_call(call_id):
                result = await self.dispatch(tool_name, arguments, call_id)
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            raise
        except Exception:
            record.update(status="raised", is_error=True)
            stats.failed.append(tool_name)
            raise
        record.update(_call_record(call_id, tool_name, result), status="completed")
        (stats.failed if result.is_error else stats.completed).append(tool_name)
        stats.track_process(arguments, result)
        stats.url_elicitations.extend(url_elicitation_required_payloads(result))
        return result


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
    """Owns the Monty pool, a fresh sandbox per run, and the store that runs share."""

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
        self._shell_function: PtcFunction | None = None
        self._direct_shell_tool: str | None = None
        self._tool_format: PtcToolFormat = "json"
        self._stubs = render_stubs({})
        self._pool: AsyncMonty | None = None
        self._pool_stack = AsyncExitStack()
        self._store = SandboxStore()
        self._file_read = False
        self._file_write = False
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
        self,
        tools: Sequence[Tool],
        *,
        tool_format: PtcToolFormat = "json",
        file_read: bool = False,
        file_write: bool = False,
    ) -> list[Tool]:
        """Replace the agent's tools with the code tool, keeping configured direct tools."""
        self._tool_format = tool_format
        direct = [tool for tool in tools if self._direct_reason(tool) is not None]
        nested = [tool for tool in tools if self._is_nested(tool)]
        self._functions = build_functions(nested)
        self._shell_function = next(
            (
                function
                for function in self._functions.values()
                if is_shell_command_tool_name(function.tool_name)
            ),
            None,
        )
        self._direct_shell_tool = next(
            (tool.name for tool in direct if is_shell_command_tool_name(tool.name)), None
        )
        self._file_read = file_read
        self._file_write = file_write
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
            batching_instruction=PTC_BATCHING_INSTRUCTION,
            functions_note=self._functions_note(),
            io_note=self._io_note(),
            shell_python_note=self._shell_python_note(),
            shell_batch_note=_SHELL_BATCH_NOTE.format(function=self._shell_function.python_name)
            if self._shell_function
            else "",
            shared_note=_SHARED_NOTE.format(functions=shared) if shared else "",
            input_note=_FREEFORM_INPUT_NOTE if self._tool_format == "freeform" else "",
            store_note=_STORE_NOTE,
            type_check_note=_TYPE_CHECK_NOTE if self._settings.type_check else "",
        )

    def _functions_note(self) -> str:
        if not self._functions:
            return ""
        example = self._example_function()
        return _FUNCTIONS_NOTE.format(
            only_code="" if self._shared_functions else ", and code is the only way to call them",
            tool=example.tool_name,
            call=f"{example.example_call()}",
        )

    def _shell_python_note(self) -> str:
        if self._shell_function is not None:
            return _SHELL_PYTHON_NOTE.format(call=self._shell_function.example_call())
        if self._direct_shell_tool is not None:
            return _DIRECT_SHELL_PYTHON_NOTE.format(tool=self._direct_shell_tool)
        return ""

    def _io_note(self) -> str:
        example = self._example_function().example_call() if self._functions else None
        if self._file_read or self._file_write:
            from fast_agent.ptc.files import MAX_BINARY_READ_BYTES, MAX_TEXT_READ_BYTES, format_size

            files_note = _FILES_NOTE.format(
                commands=_COMMANDS_NOTE.format(example=example) if example else "",
            )
            return files_note + _READ_LIMITS_NOTE.format(
                binary=format_size(MAX_BINARY_READ_BYTES),
                text=format_size(MAX_TEXT_READ_BYTES),
                advice=_READ_LIMITS_ADVICE.format(shell=self._shell_function.python_name)
                if self._shell_function
                else "",
            )
        return _IO_NOTE.format(example=example) if example else ""

    def _io_guidance(self) -> str:
        """Tell sandbox code that cannot do some I/O itself which functions can."""
        if not self._functions:
            return "No functions are available to do it."
        examples = ", ".join(
            f"`{function.example_call()}`"
            for function in list(self._functions.values())[:_MAX_HINT_EXAMPLES]
        )
        return f"Call the functions from this code instead, e.g. {examples}."

    def _example_function(self) -> PtcFunction:
        functions = list(self._functions.values())
        shell = (
            function for function in functions if is_shell_command_tool_name(function.tool_name)
        )
        return next(shell, functions[0])

    def _direct_io_hint(self, detail: str) -> str:
        """Point code that tried to do I/O itself at the tool functions that can."""
        if not self._functions:
            return ""
        if self._file_read or self._file_write:
            unsupported = (
                line
                for line in detail.splitlines()
                if _UNSUPPORTED_IO_ERROR.search(line) and not _STUB_GAP.search(line)
            )
            if next(unsupported, None) is None:
                return ""
            return (
                "\nHint: that is not available in the sandbox. Use `open()`, `pathlib.Path` "
                "(including `Path.glob`) or `os` for files. " + self._io_guidance()
            )
        if _DIRECT_IO_ERROR.search(detail) is None:
            return ""
        return "\nHint: sandbox code cannot touch files or processes itself. " + self._io_guidance()

    def unavailable_tool_hint(self, name: str) -> str | None:
        """Explain how to reach a tool the model called directly that only code can call."""
        function = next(
            (
                function
                for function in self._functions.values()
                if name in (function.tool_name, function.python_name)
                and function.python_name not in self._shared_functions
            ),
            None,
        )
        if function is None:
            return None
        return (
            f"`{name}` is not a direct tool; it is a function inside `{PTC_TOOL_NAME}`. "
            f"Call `{PTC_TOOL_NAME}` with code such as `r = {function.example_call()}` "
            "then `print(r)`."
        )

    def tool_meta(self) -> dict[str, Any]:
        """Display metadata for the code tool in its current input format."""
        return dict(_FREEFORM_TOOL_META if self._tool_format == "freeform" else PTC_TOOL_META)

    def tool(self) -> Tool:
        stubs = "\n\n".join(
            [*(function.stub() for function in self._functions.values()), STORE_STUBS.rstrip()]
        )
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
        cwd: str | None = None,
    ) -> CallToolResult:
        """Run model code; nested call ids derive from ``parent_call_id`` when it is known.

        ``cwd`` is the tools' working directory, against which sandbox code resolves relative
        file paths; it must be an absolute POSIX path.
        """
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
            calls = _NestedCalls(
                dispatch,
                stats,
                call_prefix=(
                    f"{parent_call_id}{NESTED_PTC_CALL_SEPARATOR}"
                    if parent_call_id
                    else f"ptc-{self._executions}-"
                ),
            )
            from fast_agent.ptc.files import SandboxFilesystem

            files = SandboxFilesystem(read=self._file_read, write=self._file_write)
            dispatches = calls.tasks
            lookup: dict[str, Callable[..., Awaitable[Any]]] = {
                name: self._host_function(function, calls)
                for name, function in self._functions.items()
            }
            lookup.update(self._store.functions())
            started = time.perf_counter()
            try:
                async with await self._checkout() as session:
                    await session.feed_run(_PRELUDE)
                    try:
                        value = await self._feed(session, code, lookup, output, files, cwd)
                    except Exception:
                        # A failed gather can leave sibling tools running. Settle them before
                        # reporting completed side effects and copying nested result metadata.
                        await asyncio.gather(*dispatches, return_exceptions=True)
                        raise
                    await asyncio.gather(*dispatches, return_exceptions=True)
            except asyncio.CancelledError:
                for task in list(dispatches):
                    task.cancel()
                await asyncio.gather(*dispatches, return_exceptions=True)
                raise
            except MontyTypingError as exc:
                # The type checker also reports syntax errors, mixed with errors they cause.
                detail = exc.display()
                report = self._describe_syntax_error(code) if "invalid-syntax" in detail else None
                if report is not None:
                    return _syntax_error_result(report)
                return _error_result(
                    "Type check failed; the code was NOT executed. Fix it and retry.\n"
                    + detail
                    + self._direct_io_hint(detail)
                )
            except MontySyntaxError as exc:
                return _syntax_error_result(self._describe_syntax_error(code) or exc.display())
            except MontyRuntimeError as exc:
                detail = exc.display()
                return self._failure(detail + self._direct_io_hint(detail), output, stats, started)
            except MontyCrashedError as exc:
                return self._failure(f"{exc}\nThe sandbox crashed.", output, stats, started)
            except MontyError as exc:
                return self._failure(str(exc), output, stats, started)
            finally:
                await files.close()

            return self._success(value, output, stats, started)

    def _describe_syntax_error(self, code: str) -> str | None:
        return describe_syntax_error(code, direct_functions=self._shared_functions)

    @property
    def store(self) -> SandboxStore:
        """Values sandbox code keeps between runs; sessions persist and restore it."""
        return self._store

    def reset_conversation(self) -> None:
        """Forget what earlier runs of this conversation kept: stored values and media."""
        self._store.clear()
        self._media.clear()

    async def close(self) -> None:
        await self._pool_stack.aclose()
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
        self, function: PtcFunction, calls: _NestedCalls
    ) -> Callable[..., Awaitable[Any]]:
        async def call(*args: Any, **kwargs: Any) -> Any:
            if args:
                raise TypeError(f"{function.python_name}() accepts keyword arguments only")
            result = await calls.call(function.tool_name, function.bind_arguments(kwargs))
            return tool_result_to_value(result, self._media, name=function.python_name)

        return call

    async def _checkout(self) -> AsyncMontySession:
        """Check out a fresh sandbox for one run; nothing but the store carries between runs."""
        if self._pool is None:
            from pydantic_monty import AsyncMonty

            self._pool = await self._pool_stack.enter_async_context(
                AsyncMonty(min_processes=1, max_processes=2)
            )
        return self._pool.checkout(
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

    async def _feed(
        self,
        session: AsyncMontySession,
        code: str,
        lookup: dict[str, Callable[..., Awaitable[Any]]],
        output: _PrintedOutput,
        files: SandboxFilesystem,
        cwd: str | None,
    ) -> Any:
        from pydantic_monty import MontyTypingError

        try:
            return await session.feed_run(
                code,
                external_lookup=lookup,
                print_callback=output.append,
                os=files,
                cwd=cwd,
            )
        except MontyTypingError as exc:
            if not _only_stub_gaps(exc.display()):
                raise
        return await session.feed_run(
            code,
            external_lookup=lookup,
            print_callback=output.append,
            os=files,
            cwd=cwd,
            skip_type_check=True,
        )

    def _success(
        self, value: Any, output: _PrintedOutput, stats: _ExecutionStats, started: float
    ) -> CallToolResult:
        sections = [_header("completed", stats, started)]
        printed = output.render()
        if printed:
            sections.append(f"Output:\n{printed}")
        if value is not None:
            sections.append(f"Result:\n{render_value(value)}")
        if not printed and value is None and not stats.running:
            sections.append("(no output; print() values or end with an expression)")
        if _UNAWAITED_CALL.search("\n".join(sections)):
            sections.append(
                "Note: a function call was not awaited, so its result was lost (the call still "
                "ran). Use `r = await name(...)`."
            )
        if stats.running:
            sections.append(self._running_note(stats))
        media = self._media.referenced(value)
        if media:
            sections.append(f"Attached {len(media)} media item(s) referenced by the result.")
        content: list[ContentBlock] = [TextContent(type="text", text="\n".join(sections))]
        result = CallToolResult(content=[*content, *media])
        _add_nested_metadata(result, stats)
        return result

    def _running_note(self, stats: _ExecutionStats) -> str:
        """Name the commands the shell returned before they finished: their output is partial."""
        count = len(stats.running)
        noun, pronoun = ("command", "it") if count == 1 else ("commands", "them")
        header = (
            f"Note: {count} {noun} returned while still running, so the output this script "
            f"saw from {pronoun} is incomplete:"
        )
        lines = [header]
        lines.extend(
            f"- {process_id}: `{summarize_command(command)}`" if command else f"- {process_id}"
            for process_id, command in stats.running.items()
        )
        process_id = next(iter(stats.running))
        wait = self._process_wait_call(process_id)
        how = f"Wait for {pronoun} with `{wait}`" if wait else f"Wait for {pronoun}"
        lines.append(f"{how}, or pass `check=True` to make a command wait until it finishes.")
        return "\n".join(lines)

    def _process_wait_call(self, process_id: str) -> str | None:
        for function in self._functions.values():
            name = function.tool_name.casefold()
            if name == PROCESS_TOOL_NAME:
                return f'await {function.python_name}(action="wait", process_id="{process_id}")'
            if name == POLL_PROCESS_TOOL_NAME:
                return f'await {function.python_name}(process_id="{process_id}")'
        return None

    def _failure(
        self, detail: str, output: _PrintedOutput, stats: _ExecutionStats, started: float
    ) -> CallToolResult:
        sections = [_header("failed", stats, started)]
        printed = output.render()
        if printed:
            sections.append(f"Output before the error:\n{printed}")
        if stats.completed:
            sections.append(
                "Calls that completed (avoid repeating side effects): " + ", ".join(stats.completed)
            )
        sections.append(f"Error:\n{detail}")
        if stats.running:
            sections.append(self._running_note(stats))
        result = _error_result("\n".join(sections))
        _add_nested_metadata(result, stats)
        return result


def _only_stub_gaps(diagnostics: str) -> bool:
    """Whether type checking failed only on functions Monty runs but its stubs omit."""
    errors = [line for line in diagnostics.splitlines() if "error[" in line]
    return bool(errors) and all(_STUB_GAP.search(line) for line in errors)


def _matching_pattern(tool: Tool, patterns: Sequence[str]) -> str | None:
    return next((pattern for pattern in patterns if fnmatch.fnmatchcase(tool.name, pattern)), None)


def _started_call_record(call_id: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    # Tool arguments cross an untyped boundary. JSON round-tripping both detaches nested
    # mutable values and ensures the audit metadata can be persisted/exported.
    record: dict[str, Any] = {
        "id": call_id,
        "tool": tool_name,
        "arguments": None,
        "is_error": None,
        "status": "in_flight",
    }
    try:
        record["arguments"] = json.loads(json.dumps(arguments, allow_nan=False))
    except (TypeError, ValueError, RecursionError):
        # Do not change dispatch behavior or stringify arbitrary objects into the trace.
        record["arguments_error"] = "not_json_serializable"
    return record


def _call_record(call_id: str, tool_name: str, result: CallToolResult) -> dict[str, Any]:
    record: dict[str, Any] = {"id": call_id, "tool": tool_name, "is_error": result.is_error}
    meta = result.meta or {}
    record.update({key: meta[key] for key in _CARRIED_RESULT_META if key in meta})
    return record


def _add_nested_metadata(result: CallToolResult, stats: _ExecutionStats) -> None:
    add_url_elicitation_required_payloads(result, stats.url_elicitations)
    result.meta = {**(result.meta or {}), FAST_AGENT_PTC_CALLS: stats.calls}


def _header(status: str, stats: _ExecutionStats, started: float) -> str:
    elapsed = time.perf_counter() - started
    calls = len(stats.completed) + len(stats.failed)
    failed = f", {len(stats.failed)} failed" if stats.failed else ""
    return f"Script {status} in {elapsed:.2f}s ({calls} call{'' if calls == 1 else 's'}{failed})"


def _syntax_error_result(report: str) -> CallToolResult:
    return _error_result("Syntax error; the code was NOT executed.\n" + report)


def _error_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)
