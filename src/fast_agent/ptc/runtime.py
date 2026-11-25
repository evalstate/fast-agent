"""Programmatic tool calling runtime backed by a pydantic-monty sandbox.

The model sees one tool, ``exec_sandbox``. Each run's code executes in a fresh Monty sandbox
where each agent tool is an async function; values a run keeps with ``store()`` are what later
runs can ``load()``. Nested calls are dispatched back through the agent's
normal tool path, so MCP routing, permissions and local runtimes still apply; only printed
output and the final expression value are returned to the model.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field, replace
from io import StringIO
from typing import TYPE_CHECKING, Any, Literal

from mcp_types import CallToolResult, ContentBlock, TextContent, Tool

from fast_agent.constants import (
    FAST_AGENT_PTC_CALLS,
    FAST_AGENT_PTC_VALUE,
    FAST_AGENT_SUBAGENT_RESULT_METADATA,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.tool_result_metadata import (
    add_url_elicitation_required_payloads,
    url_elicitation_required_payloads,
)
from fast_agent.ptc.catalog import (
    PTC_SEARCH_EXTRA_BYTE_THRESHOLD,
    PtcCatalog,
    PtcNameLedger,
    SourceRoster,
    ToolRoute,
    build_roster,
    function_source,
    unavailable_tool_message,
)
from fast_agent.ptc.output import (
    OUTPUT_HELPER_NAMES,
    OUTPUT_PRELUDE,
    OUTPUT_STUBS,
    PtcOutputGuidance,
    shell_output_note,
)
from fast_agent.ptc.search import (
    SEARCH_FUNCTION_NAME,
    SEARCH_STUBS,
    CellSearchState,
    search_tools,
)
from fast_agent.ptc.store import STORE_STUBS, SandboxStore
from fast_agent.ptc.stubs import PtcFunction, build_functions, render_stubs, tool_ptc_options
from fast_agent.ptc.syntax import describe_syntax_error
from fast_agent.ptc.values import MediaStore, render_value, tool_result_to_value
from fast_agent.ptc.vocabulary import (
    LEGACY_FUNCTIONS,
    PTC_VOCABULARY_META,
    PTC_VOCABULARY_VERSION,
    is_builtin_tool,
    legacy_name_hint,
    undefined_names,
)
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
)

if TYPE_CHECKING:
    from pydantic_monty import AsyncMonty, AsyncMontySession

    from fast_agent.config import ProgrammaticToolCallingSettings
    from fast_agent.mcp.url_elicitation_required import URLElicitationRequiredDisplayPayload
    from fast_agent.ptc.files import SandboxFilesystem

PTC_TOOL_NAME = "exec_sandbox"
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

# Nested result metadata copied onto the exec_sandbox result, which is what history keeps.
_CARRIED_RESULT_META = (FAST_AGENT_SUBAGENT_RESULT_METADATA,)

ToolDispatcher = Callable[[str, dict[str, Any], str], Awaitable[CallToolResult]]
"""Dispatch a nested call: (tool name, arguments, nested call id) -> result."""
SearchObserver = Callable[[str, dict[str, Any], str], None]
"""Observe a ``search_tools`` call: (call id, arguments, result text)."""

PTC_BATCHING_INSTRUCTION_TEMPLATE = (
    "Batch independent searches, file reads, and other read-only calls in a single "
    "`{tool}` invocation using `asyncio.gather`. Prefer this over separate invocations "
    "or sequential awaits when no result depends on another. Inspect every result; use "
    "`settle()` when each result must be collected despite individual failures. Keep dependent "
    "steps, approvals, and potentially conflicting mutations sequential."
)
PTC_BATCHING_INSTRUCTION = PTC_BATCHING_INSTRUCTION_TEMPLATE.format(tool=PTC_TOOL_NAME)

# The system prompt carries the batching guidance (PTC_BATCHING_INSTRUCTION); the description
# says what the sandbox is, how to call functions and what comes back. Keep every claim here
# checked against Monty: models act on it literally.
_INSTRUCTIONS = """\
Run a Python script in a sandbox: a restricted Python 3.14 interpreter (Monty), separate from \
the machine's Python. Only `print()` output and the final expression value come back to you \
(capped at about {output_limit}; longer output loses its middle): compute counts, sorts, \
filters, and compact per-item summaries in code rather than printing raw dumps.
{input_note}\
- Sandbox Python: the only importable modules are annotationlib, asyncio, base64, binascii, bisect, \
collections, colorsys, copy, csv, dataclasses, datetime, decimal, fnmatch, functools, hashlib, \
itertools, json, keyword, math, os, os.path, pathlib, random, re, stat, statistics, sys, time, \
tomllib, typing and unicodedata. No other standard library (difflib, heapq, io, string, struct, \
subprocess, ...), no packages, no environment variables. Unsupported syntax: `super()`, \
subclassing built-ins (except `object`), `@staticmethod`/`@classmethod`/`@property`, `yield`, \
`del`, `async with`/`async for`, `except*`.
- Python string syntax only: use `'...'`, `"..."`, or `'''...'''`/`\"\"\"...\"\"\"`, never \
backticks (`` `...` ``); avoid nesting the same triple-quote delimiter inside a string \
(write multi-line scripts to a file instead of embedding heredocs).
{shell_python_note}\
{functions_note}\
{search_note}\
- Independent calls can run concurrently: `a, b = await asyncio.gather(f(x=1), g(y=2))` \
(asyncio, collections, datetime, itertools, json, math, os, pathlib and re are pre-imported). `settle(aw)` returns an exception instead of raising, so \
`await asyncio.gather(*[settle(f(x=x)) for x in xs])` collects every result \
(`return_exceptions=` is unsupported).
{shell_batch_note}\
{shell_output_note}\
{io_note}\
{shared_note}\
- Function results: structured content as a dict, JSON text parsed, other text as a str. A \
failing call raises RuntimeError{shell_exit_note}.
- Images, audio and binary resources arrive as `{{"media_id": ...}}` handles: return them in \
the final expression to see them; printed handles are not attached. A structured result with \
such blocks is `{{"structured_content": ..., "content": [...]}}`.
- {store_note}{type_check_note}
"""
# Helpers defined in every sandbox session before model code runs. Common data/async
# modules are pre-imported because models routinely use them without importing in fresh runs.
_PRELUDE = """\
import asyncio
import collections
import datetime
import itertools
import json
import math
import os
import pathlib
import re


_builtin_sorted = sorted


def sorted(iterable, /, *, key=None, reverse=False):
    items = list(iterable)
    if key is None and items and all(isinstance(item, pathlib.Path) for item in items):
        return _builtin_sorted(items, key=lambda p: p.parts, reverse=reverse)
    return _builtin_sorted(items, key=key, reverse=reverse)


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
_SEARCH_NOTE = (
    "- Additional functions from connected tool sources are searchable on demand: run "
    '`print(await search_tools(query="operation object"))` (or scope with `server="..."`) to '
    "get full Python signatures, then call them in the next step or if you already know their "
    "signature.\n"
)
_SEARCH_EXAMPLE_CALL = 'await search_tools(query="...")'
_FREEFORM_INPUT_NOTE = (
    "This is a FREEFORM tool: send the Python source itself, not JSON, quotes or markdown fences.\n"
)
# The sandbox callables are always "functions"; "tool" means only something the model can call
# directly (the code tool, or a function also exposed as a direct tool). Models act on "tool"
# by emitting a direct tool call, so the wording must not blur the two.
_FUNCTIONS_NOTE = (
    "- The functions below are defined in the sandbox{only_code}. Call them as "
    '`r = await name(arg="value")` (async, keyword arguments only, no import). `= ...` marks an '
    "optional parameter; pass `None` only where the type allows it. Where other instructions "
    "{mapping}\n"
)
_TOOL_MAPPING = (
    'say to use a tool of the same name (e.g. "use the `{tool}` tool"), call `{call}` here.'
)
# With only search_tools upfront, every tool other instructions name is a deferred function.
_SEARCH_MAPPING = "say to use a tool, call the function of the same name here."
_SEARCH_IO = "find them with `search_tools`"
_SHELL_BATCH_NOTE = (
    "- Run independent shell commands as separate gathered `{function}` calls rather than "
    "chained with `;` or `&&`, so each keeps its own stdout, stderr and exit_code: "
    "`cmds = [...]` then `rs = await asyncio.gather(*[{function}(command=c) for c in cmds])`.\n"
)
_SHELL_EXIT_NOTE = (
    "; a shell command that exits non-zero does not raise: check `r['exit_code']`, or pass "
    "`check=True`"
)
# Running the machine's Python from a file avoids a heredoc inside a string inside a shell call.
_SHELL_PYTHON_NOTE = (
    "- For other modules, packages or heavy computation, use the machine's Python: write the "
    "script to a file, then run it with `{call_python}` (install `python3` if it is missing).\n"
)
_DIRECT_SHELL_PYTHON_NOTE = (
    "- For other modules, packages or heavy computation, run a script with the machine's "
    "`python3` using the `{tool}` tool (install `python3` if it is missing).\n"
)
_SHARED_NOTE = (
    "- {functions} can also be called directly as tools: directly for a one-off step, "
    "especially with quotes, heredocs or source code; from code for several steps or to "
    "process results.\n"
)
_IO_NOTE = (
    "- The functions are your only access to files, commands and the network, {how}; "
    "`open()`, `os`, `pathlib` and `subprocess` cannot reach them here.\n"
)
_FILES_NOTE = (
    "- Files are the machine's real files: `open()`, `pathlib.Path` and the `os` file functions "
    "(`os.listdir`, `os.walk`, `os.path.exists`, `os.makedirs`, ...) read and write them, "
    "relative to the working directory.{commands}\n"
)
_READ_LIMITS_NOTE = (
    "- A read loads the whole file (at most {binary} binary or {text} text), even for `read(n)` "
    "or `seek()`.{advice}\n"
)
_READ_LIMITS_ADVICE = (
    " For larger or binary files, use commands such as `head -c` or `xxd` via `{shell}`."
)
_COMMANDS_NOTE = " Commands and the network go through the functions, {how}."
# Sandbox errors from trying to do I/O directly instead of through tool functions.
_DIRECT_IO_ERROR = re.compile(
    r"PermissionError|Module `os` has no member"
    r"|Cannot resolve imported module `(?:subprocess|shutil|glob|io|socket|urllib|http|requests)"
)
# The same, when file operations are served by tools: only processes and other I/O are missing.
_UNSUPPORTED_IO_ERROR = re.compile(
    r"Module `os` has no member"
    r"|Cannot resolve imported module `(?:subprocess|shutil|glob|io|socket|urllib|http|requests)"
)
# Built-in methods and standard-library modules the sandbox lacks. Models otherwise work around
# them in sandbox code (e.g. a hand-rolled str.maketrans) instead of using the system Python.
_BUILTIN_TYPES = (
    r"(?:str|bytes|bytearray|int|float|complex|list|dict|set|frozenset|tuple|bool|range)"
)
_MISSING_BUILTIN = re.compile(
    rf"AttributeError: (?:type object '{_BUILTIN_TYPES}'|'{_BUILTIN_TYPES}' object) "
    r"has no attribute"
)
_MISSING_MODULE = re.compile(r"(?:Cannot resolve imported module `|No module named ')([^'`\n]+)")
# A literal list inside `gather(*[... for x in [...]])` whose comprehension `]` was dropped.
_GATHER_BRACKETS = re.compile(r"closing parenthesis '\)' does not match opening parenthesis '\['")
_GATHER_HINT = (
    "\nHint: in `asyncio.gather(*[f(x) for x in [...]])` the comprehension needs its own `]` "
    "before `)`. Binding the list first (`items = [...]`) avoids this."
)
# pydantic-monty (through 1.7) runs os.walk, os.scandir and map, but its type-checker stubs omit them.
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
    # exec_sandbox result for the user to see them.
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
    execution_started: float = field(default_factory=time.perf_counter)
    # Outstanding dispatches only: a finished task holds its full result, which sandbox code may
    # long since have discarded, so keeping them all would grow host memory with fetched data.
    tasks: set[asyncio.Task[CallToolResult]] = field(default_factory=set)

    async def call(
        self, tool_name: str, arguments: dict[str, Any], *, function_name: str | None = None
    ) -> CallToolResult:
        task = asyncio.create_task(self._run(tool_name, arguments, function_name or tool_name))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        # Own dispatch lifetime independently of Monty's callback tasks. Script errors
        # drain these tasks; turn cancellation explicitly cancels and settles them.
        return await asyncio.shield(task)

    async def _run(
        self, tool_name: str, arguments: dict[str, Any], function_name: str
    ) -> CallToolResult:
        stats = self.stats
        stats.started += 1
        call_id = f"{self.call_prefix}{stats.started}"
        # Snapshot before dispatch: tools may mutate arguments, and parallel calls may
        # finish out of order. This ledger is metadata, never model-visible output.
        call_started = time.perf_counter()
        record = _started_call_record(call_id, tool_name, arguments)
        record["function"] = function_name
        record["start_offset_ms"] = round(
            max(0.0, (call_started - self.execution_started) * 1000), 2
        )
        stats.calls.append(record)
        try:
            with programmatic_tool_call(call_id):
                result = await self.dispatch(tool_name, arguments, call_id)
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            raise
        except Exception:
            record.update(status="raised", is_error=True)
            stats.failed.append(function_name)
            raise
        finally:
            record["timing_ms"] = round(max(0.0, (time.perf_counter() - call_started) * 1000), 2)
        record.update(_call_record(call_id, tool_name, result), status="completed")
        (stats.failed if result.is_error else stats.completed).append(function_name)
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

    def __init__(
        self, settings: ProgrammaticToolCallingSettings, *, ledger: PtcNameLedger | None = None
    ) -> None:
        self._settings = settings
        # ptc-vocab-v0: route override can change without resetting the sandbox store.
        self.function_names = settings.function_names
        # The name avoids "python": models treat a Python-named tool as the machine's Python.
        self.tool_name = PTC_TOOL_NAME
        # Defer extra function definitions behind search_tools; the agent applies CLI overrides.
        self.search_enabled = settings.search
        # When True (e.g. --ptc-search), defer extra functions even below the stub byte budget.
        self.search_forced = False
        self._direct_patterns = list(settings.direct_tools)
        self._warned_patterns: set[str] = set()
        self._ledger = ledger if ledger is not None else PtcNameLedger()
        self._catalog = PtcCatalog(type_check_stubs=render_stubs({}))
        self._direct_shell_tool: str | None = None
        self._tool_format: PtcToolFormat = "json"
        # Shell output guidance in effect: "off" unless sandbox code can run local shell commands.
        self._output_guidance: PtcOutputGuidance = "off"
        self._result_byte_limit = settings.output_byte_limit
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
        """Every function sandbox code can call, including deferred ones."""
        return self._catalog.functions

    @property
    def catalog(self) -> PtcCatalog:
        """The catalogue built by the latest ``model_tools()`` call."""
        return self._catalog

    @property
    def batching_instruction(self) -> str:
        """System-prompt guidance on batching calls in the code tool."""
        return PTC_BATCHING_INSTRUCTION_TEMPLATE.format(tool=self.tool_name)

    def resolve_tool_format(self, model_default: PtcToolFormat | None) -> PtcToolFormat:
        """Apply ``ptc.tool_format``, using the model's default when it is ``auto``."""
        if self._settings.tool_format != "auto":
            return self._settings.tool_format
        return model_default or "json"

    def resolve_output_guidance(self, model_default: PtcOutputGuidance | None) -> PtcOutputGuidance:
        """Apply ``ptc.output_guidance``, using the model's default when it is ``auto``."""
        if self._settings.output_guidance != "auto":
            return self._settings.output_guidance
        return model_default or "helpers"

    def model_tools(
        self,
        tools: Sequence[Tool],
        *,
        tool_format: PtcToolFormat = "json",
        output_guidance: PtcOutputGuidance = "off",
        result_byte_limit: int | None = None,
        file_read: bool = False,
        file_write: bool = False,
        sources: Mapping[str, str] | None = None,
        unavailable_sources: Mapping[str, str] | None = None,
        routes: Mapping[str, ToolRoute] | None = None,
    ) -> list[Tool]:
        """Replace the agent's tools with the code tool, keeping configured direct tools.

        ``result_byte_limit`` is how much of a code tool result the model sees, for the
        shell output guidance and search results. ``sources`` overrides the source group of
        tools by listed name; ``unavailable_sources`` notes configured sources without
        loaded functions; ``routes`` (the tools' origins by listed name) is kept with the
        catalogue for dispatching its runs' calls.
        """
        self._tool_format = tool_format
        names = {tool.name for tool in tools}
        builtin_names = {tool.name for tool in tools if is_builtin_tool(tool)}
        self._direct_patterns = []
        # ptc-vocab-v0: remove direct-selector migration after matched rollout.
        for pattern in self._settings.direct_tools:
            replacement = LEGACY_FUNCTIONS.get(pattern)
            if self.vocabulary == 1 and pattern not in names and replacement in builtin_names:
                self._direct_patterns.append(replacement)
                if pattern not in self._warned_patterns:
                    get_logger(__name__).warning(
                        f"ptc.direct_tools: '{pattern}' is now '{replacement}'; update your config."
                    )
                    self._warned_patterns.add(pattern)
            else:
                self._direct_patterns.append(pattern)
        direct = [tool for tool in tools if self._direct_reason(tool) is not None]
        nested = [tool for tool in tools if self._is_nested(tool)]
        local_shell = any(_returns_shell_output(tool) for tool in nested)
        self._output_guidance = output_guidance if local_shell else "off"
        self._result_byte_limit = (
            self._settings.output_byte_limit if result_byte_limit is None else result_byte_limit
        )
        self._direct_shell_tool = next(
            (tool.name for tool in direct if _returns_shell_output(tool)), None
        )
        self._file_read = file_read
        self._file_write = file_write
        self._catalog = self._build_catalog(
            nested,
            direct_names={tool.name for tool in direct},
            sources=sources or {},
            unavailable_sources=unavailable_sources or {},
            routes=routes or {},
        )
        return [self.tool(), *direct]

    def _build_catalog(
        self,
        nested: Sequence[Tool],
        *,
        direct_names: set[str],
        sources: Mapping[str, str],
        unavailable_sources: Mapping[str, str],
        routes: Mapping[str, ToolRoute],
    ) -> PtcCatalog:
        canonical = self.vocabulary == 1
        reserved = OUTPUT_HELPER_NAMES if self._output_guidance != "off" else frozenset()
        functions = build_functions(
            nested,
            profile_names=not canonical,
            reserved_names=(reserved | {SEARCH_FUNCTION_NAME}) if canonical else reserved,
            ledger=self._ledger if canonical else None,
            routes=routes,
        )
        tools = {tool.name: tool for tool in nested}
        function_sources = {
            name: function_source(
                tools[function.tool_name], sources, routes.get(function.tool_name)
            )
            for name, function in functions.items()
        }
        # Built-ins and functions the model can also call directly always stay upfront.
        extras = [
            name
            for name, function in functions.items()
            if not is_builtin_tool(tools[function.tool_name])
            and function.tool_name not in direct_names
        ]
        extra_stub_bytes = sum(len(functions[name].stub().encode("utf-8")) for name in extras)
        search_active = (
            self.search_enabled
            and canonical
            and bool(extras)
            and (self.search_forced or extra_stub_bytes > PTC_SEARCH_EXTRA_BYTE_THRESHOLD)
        )
        deferred = frozenset(extras) if search_active else frozenset()
        stubs = render_stubs(functions)
        if self._output_guidance != "off":
            stubs += OUTPUT_STUBS
        if search_active:
            stubs += SEARCH_STUBS
        return PtcCatalog(
            functions=functions,
            tools=tools,
            sources=function_sources,
            deferred=deferred,
            search_active=search_active,
            roster=(
                build_roster(function_sources, deferred, unavailable_sources)
                if search_active
                else SourceRoster()
            ),
            shared_functions=tuple(
                function.python_name
                for function in functions.values()
                if function.tool_name in direct_names
            ),
            shell_function=next(
                (
                    function
                    for name, function in functions.items()
                    if name not in deferred and _returns_shell_output(tools[function.tool_name])
                ),
                None,
            ),
            type_check_stubs=stubs,
            retired=self._ledger.retired(functions) if canonical else frozenset(),
            routes=routes,
        )

    def _prelude(self, catalog: PtcCatalog) -> str:
        # Old calls stay catchable after a refresh, without advertising missing tools.
        helpers = self._output_guidance != "off"
        retired = catalog.retired - OUTPUT_HELPER_NAMES if helpers else catalog.retired
        unavailable = "".join(
            f"\nasync def {name}(*args, **kwargs):\n"
            f"    raise RuntimeError({unavailable_tool_message(name)!r})\n"
            for name in sorted(retired)
        )
        return _PRELUDE + unavailable + (OUTPUT_PRELUDE if helpers else "")

    def surface(
        self, model_tools: Sequence[Tool], *, seen_signatures: frozenset[str] = frozenset()
    ) -> ToolSurface:
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
        catalog = self._catalog
        nested: list[SurfaceTool] = []
        for name, function in catalog.functions.items():
            stub = function.stub()
            tokens = estimate_tokens(stub)
            deferred = name in catalog.deferred
            signature = function.signature()
            nested.append(
                SurfaceTool(
                    tool=catalog.tools[function.tool_name],
                    exposure=ToolExposure.PROGRAMMATIC,
                    context_tokens=0 if deferred else tokens,
                    python_name=function.python_name,
                    signature=signature,
                    stub=stub,
                    deferred=deferred,
                    stub_tokens=tokens,
                    source=catalog.sources[name],
                    seen=deferred and signature in seen_signatures,
                )
            )
        return ToolSurface(
            tools=(*exposed, *nested),
            programmatic=ProgrammaticSurface(
                tool_name=self.tool_name,
                vocabulary=self.vocabulary,
                tool_format=self._tool_format,
                instructions=self._instructions(),
                type_check=self._settings.type_check,
                max_duration_seconds=self._settings.max_duration_seconds,
                max_memory_mb=self._settings.max_memory_mb,
                search_active=catalog.search_active,
                retired=catalog.retired,
            ),
        )

    @property
    def vocabulary(self) -> int:
        """Zero identifies the temporary per-model profile arm, one the canonical surface."""
        return PTC_VOCABULARY_VERSION if self.function_names == "canonical" else 0

    def _instructions(self) -> str:
        catalog = self._catalog
        shared = ", ".join(f"`{name}`" for name in catalog.shared_functions)
        shell = catalog.shell_function
        return _INSTRUCTIONS.format(
            output_limit=f"{max(1, round(self._result_byte_limit / 1000))} KB",
            shell_exit_note=_SHELL_EXIT_NOTE if shell else "",
            functions_note=self._functions_note(catalog),
            search_note=_SEARCH_NOTE if catalog.search_active else "",
            io_note=self._io_note(catalog),
            shell_python_note=self._shell_python_note(catalog),
            shell_output_note=self._shell_output_note(catalog),
            shell_batch_note=_SHELL_BATCH_NOTE.format(function=shell.python_name) if shell else "",
            shared_note=_SHARED_NOTE.format(functions=shared) if shared else "",
            input_note=_FREEFORM_INPUT_NOTE if self._tool_format == "freeform" else "",
            store_note=_STORE_NOTE,
            type_check_note=_TYPE_CHECK_NOTE if self._settings.type_check else "",
        )

    def _functions_note(self, catalog: PtcCatalog) -> str:
        example = _example_function(catalog)
        if example is not None:
            mapping = _TOOL_MAPPING.format(tool=example.tool_name, call=example.example_call())
        elif catalog.search_active:
            mapping = _SEARCH_MAPPING
        else:
            return ""
        return _FUNCTIONS_NOTE.format(
            only_code="" if catalog.shared_functions else ", and code is the only way to call them",
            mapping=mapping,
        )

    def _shell_output_note(self, catalog: PtcCatalog) -> str:
        shell = catalog.shell_function
        if self._output_guidance == "off" or shell is None:
            return ""
        wait, read = None, None
        for function in catalog.upfront:
            name = function.tool_name.casefold()
            if name == PROCESS_TOOL_NAME:
                wait = f"await {function.python_name}(action='wait', process_id=r['process_id'])"
                read = (
                    f"await {function.python_name}(action='read_output', process_id=..., "
                    "query='error')"
                )
                break
            if name == POLL_PROCESS_TOOL_NAME:
                wait = f"await {function.python_name}(process_id=r['process_id'])"
                break
        return shell_output_note(
            self._output_guidance,
            call=shell.example_call(),
            result_byte_limit=self._result_byte_limit,
            wait=wait,
            read=read,
        )

    def _shell_python_note(self, catalog: PtcCatalog) -> str:
        if catalog.shell_function is not None:
            return _SHELL_PYTHON_NOTE.format(
                call_python=f'await {catalog.shell_function.python_name}(command="python3 script.py")'
            )
        if self._direct_shell_tool is not None:
            return _DIRECT_SHELL_PYTHON_NOTE.format(tool=self._direct_shell_tool)
        return ""

    def _io_note(self, catalog: PtcCatalog) -> str:
        example = _example_function(catalog)
        how = f"e.g. `r = {example.example_call()}`" if example else None
        if how is None and catalog.search_active:
            how = _SEARCH_IO
        if self._file_read or self._file_write:
            from fast_agent.ptc.files import MAX_BINARY_READ_BYTES, MAX_TEXT_READ_BYTES, format_size

            files_note = _FILES_NOTE.format(
                commands=_COMMANDS_NOTE.format(how=how) if how else "",
            )
            return files_note + _READ_LIMITS_NOTE.format(
                binary=format_size(MAX_BINARY_READ_BYTES),
                text=format_size(MAX_TEXT_READ_BYTES),
                advice=_READ_LIMITS_ADVICE.format(shell=catalog.shell_function.python_name)
                if catalog.shell_function
                else "",
            )
        return _IO_NOTE.format(how=how) if how else ""

    def _io_guidance(self, catalog: PtcCatalog) -> str:
        """Tell sandbox code that cannot do some I/O itself which functions can."""
        calls = [function.example_call() for function in catalog.upfront[:_MAX_HINT_EXAMPLES]]
        if catalog.search_active:
            calls.append(_SEARCH_EXAMPLE_CALL)
        if not calls:
            return "No functions are available to do it."
        examples = ", ".join(f"`{call}`" for call in calls)
        return f"Call the functions from this code instead, e.g. {examples}."

    def _direct_io_hint(self, detail: str, catalog: PtcCatalog) -> str:
        """Point code that tried to do I/O itself at the tool functions that can."""
        if not catalog.functions:
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
                "(including `Path.glob`) or `os` for files. " + self._io_guidance(catalog)
            )
        if _DIRECT_IO_ERROR.search(detail) is None:
            return ""
        return "\nHint: sandbox code cannot touch files or processes itself. " + self._io_guidance(
            catalog
        )

    def unavailable_tool_hint(self, name: str) -> str | None:
        """Explain how to reach a tool the model called directly that only code can call."""
        # ptc-vocab-v0: remove legacy direct-call lookup after matched rollout.
        catalog = self._catalog
        lookup_name = name
        if self.vocabulary == 1 and name not in catalog.functions:
            lookup_name = LEGACY_FUNCTIONS.get(name, name)
        function = next(
            (
                function
                for function in catalog.functions.values()
                if lookup_name in (function.tool_name, function.python_name)
                and function.python_name not in catalog.shared_functions
            ),
            None,
        )
        if function is None:
            return None
        return (
            f"`{name}` is not a direct tool; it is a function inside `{self.tool_name}`. "
            f"Call `{self.tool_name}` with code such as `r = {function.example_call()}` "
            "then `print(r)`."
        )

    def tool_meta(self) -> dict[str, Any]:
        """Display metadata for the code tool in its current input format."""
        return {
            **(_FREEFORM_TOOL_META if self._tool_format == "freeform" else PTC_TOOL_META),
            PTC_VOCABULARY_META: self.vocabulary,
        }

    def tool(self) -> Tool:
        catalog = self._catalog
        stubs = "\n\n".join(
            [
                *(function.stub() for function in catalog.upfront),
                *([SEARCH_STUBS.rstrip()] if catalog.search_active else []),
                STORE_STUBS.rstrip(),
            ]
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
            name=self.tool_name,
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
        catalog: PtcCatalog | None = None,
        parent_call_id: str | None = None,
        cwd: str | None = None,
        on_search: SearchObserver | None = None,
    ) -> CallToolResult:
        """Run model code; nested call ids derive from ``parent_call_id`` when it is known.

        ``catalog`` is the catalogue the run uses, by default the latest; a caller that routes
        nested calls by a catalogue's ``routes`` passes the one it captured. ``cwd`` is the
        tools' working directory, against which sandbox code resolves relative file paths; it
        must be an absolute POSIX path. ``on_search`` observes ``search_tools`` calls, which
        the runtime answers itself rather than dispatching.
        """
        result = await self._execute(
            arguments,
            dispatch,
            catalog=catalog,
            parent_call_id=parent_call_id,
            cwd=cwd,
            on_search=on_search,
        )
        result.meta = {**(result.meta or {}), PTC_VOCABULARY_META: self.vocabulary}
        return result

    async def _execute(
        self,
        arguments: Mapping[str, Any],
        dispatch: ToolDispatcher,
        *,
        catalog: PtcCatalog | None,
        parent_call_id: str | None,
        cwd: str | None,
        on_search: SearchObserver | None,
    ) -> CallToolResult:
        """Execute validated code and collect nested calls."""
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
            # One catalogue for the whole run: a tool list refresh applies from the next run.
            catalog = self._catalog if catalog is None else catalog
            prelude = self._prelude(catalog)
            self._executions += 1
            stats = _ExecutionStats()
            output = _PrintedOutput()
            started = time.perf_counter()
            calls = _NestedCalls(
                dispatch,
                stats,
                call_prefix=(
                    f"{parent_call_id}{NESTED_PTC_CALL_SEPARATOR}"
                    if parent_call_id
                    else f"ptc-{self._executions}-"
                ),
                execution_started=started,
            )
            from fast_agent.ptc.files import SandboxFilesystem

            files = SandboxFilesystem(read=self._file_read, write=self._file_write)
            dispatches = calls.tasks
            lookup: dict[str, Callable[..., Awaitable[Any]]] = {
                name: self._host_function(function, calls)
                for name, function in catalog.functions.items()
            }
            if catalog.search_active:
                lookup[SEARCH_FUNCTION_NAME] = _search_function(
                    catalog, calls, on_search, byte_limit=self._result_byte_limit
                )
            lookup.update(self._store.functions())
            try:
                async with await self._checkout(catalog) as session:
                    await session.feed_run(prelude)
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
                report = (
                    self._describe_syntax_error(code, catalog)
                    if "invalid-syntax" in detail
                    else None
                )
                if report is not None:
                    return _syntax_error_result(report)
                return _error_result(
                    "Type check failed; the code was NOT executed. Fix it and retry.\n"
                    + detail
                    + self._error_hints(detail, catalog)
                    + _argument_type_hints(detail, catalog)
                    + self._name_hints(undefined_names(detail), catalog)
                )
            except MontySyntaxError as exc:
                return _syntax_error_result(
                    self._describe_syntax_error(code, catalog) or exc.display()
                )
            except MontyRuntimeError as exc:
                detail = exc.display()
                return self._failure(
                    detail
                    + self._error_hints(detail, catalog)
                    + self._name_hints(undefined_names(detail, ran=True), catalog),
                    output,
                    stats,
                    started,
                    catalog,
                )
            except MontyCrashedError as exc:
                return self._failure(
                    f"{exc}\nThe sandbox crashed.", output, stats, started, catalog
                )
            except MontyError as exc:
                return self._failure(str(exc), output, stats, started, catalog)
            finally:
                await files.close()

            return self._success(value, output, stats, started, catalog)

    def _describe_syntax_error(self, code: str, catalog: PtcCatalog) -> str | None:
        report = describe_syntax_error(code, direct_functions=catalog.shared_functions)
        if report is not None and _GATHER_BRACKETS.search(report) and "gather(*[" in code:
            report += _GATHER_HINT
        return report

    def _error_hints(self, detail: str, catalog: PtcCatalog) -> str:
        return self._direct_io_hint(detail, catalog) or _sandbox_gap_hint(detail, catalog)

    def _name_hints(self, undefined: set[str], catalog: PtcCatalog) -> str:
        """Explain renamed functions and inactive search."""
        # ptc-vocab-v0: remove undefined-name hints after matched rollout.
        legacy = legacy_name_hint(undefined, catalog.functions) if self.vocabulary == 1 else ""
        if SEARCH_FUNCTION_NAME in undefined and not catalog.search_active:
            return legacy + (
                f"\nHint: `{SEARCH_FUNCTION_NAME}` is not active because all available functions "
                f"are already listed in `{PTC_TOOL_NAME}`."
            )
        return legacy

    @property
    def store(self) -> SandboxStore:
        """Values sandbox code keeps between runs; sessions persist and restore it."""
        return self._store

    def reset_conversation(self) -> None:
        """Forget what earlier runs of this conversation kept: stored values, media and names."""
        self._store.clear()
        self._media.clear()
        self._ledger.clear()
        # The tools are unchanged; only names from earlier in the conversation are forgotten.
        self._catalog = replace(self._catalog, retired=frozenset())

    async def close(self) -> None:
        await self._pool_stack.aclose()
        self._pool = None

    def _direct_reason(self, tool: Tool) -> str | None:
        """Return why ``tool`` stays directly callable, or None when only code can call it."""
        if tool.name == self.tool_name:
            return None
        pattern = _matching_pattern(tool, self._direct_patterns)
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
        if tool.name == self.tool_name:
            return False
        if _matching_pattern(tool, self._direct_patterns) is not None:
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
            result = await calls.call(
                function.tool_name,
                function.bind_arguments(kwargs),
                function_name=function.python_name,
            )
            return tool_result_to_value(
                result,
                self._media,
                name=function.python_name,
                unwrap_result=function.unwrap_result,
            )

        return call

    async def _checkout(self, catalog: PtcCatalog) -> AsyncMontySession:
        """Check out a fresh sandbox for one run; nothing but the store carries between runs."""
        if self._pool is None:
            from pydantic_monty import AsyncMonty

            self._pool = await self._pool_stack.enter_async_context(
                AsyncMonty(min_processes=1, max_processes=2)
            )
        return self._pool.checkout(
            script_name="ptc.py",
            type_check=self._settings.type_check,
            type_check_stubs=catalog.type_check_stubs if self._settings.type_check else None,
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
            if not _only_false_positives(exc.display(), code):
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
        self,
        value: Any,
        output: _PrintedOutput,
        stats: _ExecutionStats,
        started: float,
        catalog: PtcCatalog,
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
            sections.append(_running_note(stats, catalog))
        media = self._media.referenced(value)
        if media:
            sections.append(f"Attached {len(media)} media item(s) referenced by the result.")
        content: list[ContentBlock] = [TextContent(type="text", text="\n".join(sections))]
        result = CallToolResult(content=[*content, *media])
        _add_nested_metadata(result, stats)
        return result

    def _failure(
        self,
        detail: str,
        output: _PrintedOutput,
        stats: _ExecutionStats,
        started: float,
        catalog: PtcCatalog,
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
            sections.append(_running_note(stats, catalog))
        result = _error_result("\n".join(sections))
        _add_nested_metadata(result, stats)
        return result


def _running_note(stats: _ExecutionStats, catalog: PtcCatalog) -> str:
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
    wait = _process_wait_call(process_id, catalog)
    how = f"Wait for {pronoun} with `{wait}`" if wait else f"Wait for {pronoun}"
    lines.append(f"{how}, or pass `check=True` to make a command wait until it finishes.")
    return "\n".join(lines)


def _process_wait_call(process_id: str, catalog: PtcCatalog) -> str | None:
    for function in catalog.functions.values():
        name = function.tool_name.casefold()
        if name == PROCESS_TOOL_NAME:
            return f'await {function.python_name}(action="wait", process_id="{process_id}")'
        if name == POLL_PROCESS_TOOL_NAME:
            return f'await {function.python_name}(process_id="{process_id}")'
    return None


_EXPECTED_FOUND = re.compile(r"Expected `(.+?)`, found `(.+?)`$", re.MULTILINE)
_TYPED_DICT_KEY_ERROR = re.compile(r"\[(?:invalid-key|missing-typed-dict-key)\]|on TypedDict `")
_INLINE_ARGUMENTS_HINT = (
    "\nHint: write Literal and dict arguments inline in the call; variables and `**kwargs` dicts "
    "infer plain `str`, `list` or `dict` types. To build one separately, annotate it `Any` "
    "(`from typing import Any`)."
)


def _argument_type_hints(detail: str, catalog: PtcCatalog) -> str:
    """Name the keys of rejected dict arguments; point non-literal arguments at inline values.

    The checker names a dict parameter's private TypedDict but not its keys, and rejects a
    variable holding a valid value because its inferred type is wider than a Literal/TypedDict.
    """
    mismatches = _EXPECTED_FOUND.findall(detail)
    rejected = [
        (function, typed)
        for function in catalog.functions.values()
        for typed in function.typed_dicts
        if f"TypedDict `{typed.name}`" in detail
        or any(
            re.search(rf"\b{re.escape(typed.name)}\b", expected) and "dict[" in found
            for expected, found in mismatches
        )
    ]
    shapes: dict[tuple[str, str], list[str]] = {}
    for function, typed in rejected:
        shapes.setdefault((function.python_name, typed.path), []).append(typed.shape)
    hints = "".join(
        f"\nHint: `{path}` of `{name}` takes {' | '.join(options)}."
        for (name, path), options in shapes.items()
    )
    names = [
        typed.name for function in catalog.functions.values() for typed in function.typed_dicts
    ]
    # A literal written inline infers Literal or TypedDict members; a variable infers neither.
    widened = not _TYPED_DICT_KEY_ERROR.search(detail) and any(
        any(marker in expected and marker not in found for marker in ("Literal[", *names))
        for expected, found in mismatches
    )
    return hints + (_INLINE_ARGUMENTS_HINT if widened else "")


def _sandbox_gap_hint(detail: str, catalog: PtcCatalog) -> str:
    """Point code that hit a missing built-in method or module at the system Python."""
    if "gather() does not yet support keyword arguments" in detail:
        return (
            "\nHint: `asyncio.gather()` does not accept `return_exceptions=`; wrap each "
            "awaitable with `settle(...)` instead: "
            "`await asyncio.gather(*[settle(f(...)) for ...])`."
        )
    if (
        "async with" in detail
        or "has no member `Semaphore`" in detail
        or "no attribute 'Semaphore'" in detail
    ):
        return (
            "\nHint: `async with` and `asyncio.Semaphore` are not supported in the sandbox; "
            "throttle concurrency by batching calls into chunks with `asyncio.gather(...)`."
        )
    mod_match = _MISSING_MODULE.search(detail)
    if mod_match is not None:
        mod_name = mod_match.group(1).split(".")[0]
        sources = {s.casefold(): s for s in catalog.sources.values()}
        if mod_name.casefold() in sources:
            src = sources[mod_name.casefold()]
            if catalog.search_active:
                return (
                    f"\nHint: `{mod_name}` is a tool source, not an importable module. "
                    f'Discover its functions with `await search_tools(server="{src}")` '
                    "and call them directly with `await`."
                )
            return (
                f"\nHint: `{mod_name}` is a tool source, not an importable module. "
                "Call its pre-bound functions directly with `await`."
            )
    if not (_MISSING_BUILTIN.search(detail) or mod_match is not None):
        return ""
    hint = (
        "\nHint: the sandbox's Python lacks this; it has a reduced standard library and "
        "built-in methods, and no packages."
    )
    if catalog.shell_function is not None:
        hint += (
            " Run code that needs it with the system Python: write it to a file and run "
            f'`await {catalog.shell_function.python_name}(command="python3 script.py")`.'
        )
    return hint


def _search_function(
    catalog: PtcCatalog, calls: _NestedCalls, on_search: SearchObserver | None, *, byte_limit: int
) -> Callable[..., Awaitable[str]]:
    """The host side of ``search_tools``: answered from ``catalog``, recorded like a call.

    ``byte_limit`` is how much of the run's result the model sees. Sibling calls within one
    ``exec_sandbox`` execution share per-cell deduplication and byte-budget state.
    """
    cell_state = CellSearchState()

    async def call(*args: Any, **kwargs: Any) -> str:
        if args:
            raise TypeError(f"{SEARCH_FUNCTION_NAME}() accepts keyword arguments only")
        stats = calls.stats
        stats.started += 1
        call_id = f"{calls.call_prefix}{stats.started}"
        record = _started_call_record(call_id, SEARCH_FUNCTION_NAME, kwargs)
        record["function"] = SEARCH_FUNCTION_NAME
        stats.calls.append(record)
        try:
            text = search_tools(catalog, **kwargs, byte_limit=byte_limit, cell_state=cell_state)
        except Exception:
            record.update(status="raised", is_error=True)
            stats.failed.append(SEARCH_FUNCTION_NAME)
            raise
        record.update(status="completed", is_error=False)
        stats.completed.append(SEARCH_FUNCTION_NAME)
        if on_search is not None:
            # Displaying the search must never fail the sandbox code that made it.
            with suppress(Exception):
                on_search(call_id, dict(kwargs), text)
        return text

    return call


def _example_function(catalog: PtcCatalog) -> PtcFunction | None:
    """The function the code tool's description uses in examples: never a deferred one."""
    return catalog.shell_function or next(iter(catalog.upfront), None)


def _returns_shell_output(tool: Tool) -> bool:
    """Whether sandbox code gets a shell command's stdout and stderr from ``tool``."""
    returns = tool_ptc_options(tool).get("returns")
    properties = returns.get("properties") if isinstance(returns, Mapping) else None
    return isinstance(properties, Mapping) and {"stdout", "stderr"} <= properties.keys()


_ARGUMENT_TYPE_ERROR = re.compile(
    r"^ptc\.py:(\d+):(\d+): error\[invalid-argument-type\].*Expected `(.+?)`, found `(.+?)`$"
)
_INFERRED_SHAPE_ERROR = re.compile(
    r"error\[(?:unresolved-attribute|not-subscriptable|not-iterable|unsupported-operator)\]"
    r"|error\[invalid-argument-type\] Method `__getitem__`"
)
_STRING_LITERALS = ('Literal["', 'list[Literal["')
_WIDENED_STRINGS = frozenset({"str", "list[str]", "Unknown", "Any"})
_INFERRED_DYNAMIC_TYPE = re.compile(r"\b(?:Unknown|Any)\b| \| ")


def _only_false_positives(diagnostics: str, code: str) -> bool:
    """Whether type checking failed only on code that runs as written.

    Monty runs some functions its stubs omit. The checker also rejects argument values it
    cannot type precisely (``**dict`` unpacking, widened ``str`` vs ``Literal``, inferred
    unions or ``Unknown``/``Any`` from dynamic dict/list payloads) and dynamic dict/list
    shapes (``unresolved-attribute``, ``not-subscriptable``, ``not-iterable``,
    ``unsupported-operator``). High-precision checks (undefined names, bad imports, concrete
    wrong tool arguments, invalid await) still fail fast before execution.
    """
    lines = code.splitlines()
    errors = [line for line in diagnostics.splitlines() if "error[" in line]
    return bool(errors) and all(
        _STUB_GAP.search(error)
        or _INFERRED_SHAPE_ERROR.search(error)
        or _unknowable_argument(error, lines)
        for error in errors
    )


def _unknowable_argument(error: str, lines: list[str]) -> bool:
    match = _ARGUMENT_TYPE_ERROR.match(error)
    if match is None:
        return False
    line, column, expected, found = match.groups()
    index = int(line) - 1
    if index < len(lines) and lines[index][int(column) - 1 :].startswith("**"):
        return True
    if _INFERRED_DYNAMIC_TYPE.search(found):
        return True
    return expected.startswith(_STRING_LITERALS) and found in _WIDENED_STRINGS


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
