"""Model-facing prompt text and base Monty prelude for the ``exec_sandbox`` code tool.

Separating the tool description templates and base prelude from ``runtime.py`` keeps everything
the model sees readable in one place, while ``output.py`` holds the optional sync prelude helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.ptc.output import SLICE_FILE_NOTE, PtcOutputGuidance, shell_output_note
from fast_agent.utils.tool_names import POLL_PROCESS_TOOL_NAME, PROCESS_TOOL_NAME

if TYPE_CHECKING:
    from fast_agent.ptc.catalog import PtcCatalog
    from fast_agent.ptc.stubs import PtcFunction

PTC_TOOL_NAME = "exec_sandbox"

PTC_BATCHING_INSTRUCTION_TEMPLATE = (
    "Batch independent searches, file reads, and other read-only calls in a single "
    "`{tool}` invocation using `asyncio.gather`. Prefer this over separate invocations "
    "or sequential awaits when no result depends on another. Inspect every result; use "
    "`settle()` when each result must be collected despite individual failures. Keep dependent "
    "steps, approvals, and potentially conflicting mutations sequential."
)
PTC_BATCHING_INSTRUCTION = PTC_BATCHING_INSTRUCTION_TEMPLATE.format(tool=PTC_TOOL_NAME)

# Helpers defined in every sandbox session before model code runs. Common data/async
# modules are pre-imported because models routinely use them without importing in fresh runs.
BASE_PRELUDE = """\
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
SEARCH_EXAMPLE_CALL = 'await search_tools(query="...")'
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
    "relative to the working directory.{slice_note}{commands}\n"
)
_READ_LIMITS_NOTE = (
    "- A read loads the whole file (at most {binary} binary or {text} text), even for `read(n)` "
    "or `seek()`.{advice}\n"
)
_READ_LIMITS_ADVICE = (
    " For larger or binary files, use commands such as `head -c` or `xxd` via `{shell}`."
)
_COMMANDS_NOTE = " Commands and the network go through the functions, {how}."
_TYPE_CHECK_NOTE = " Code is type-checked against these signatures first; on errors nothing runs."
# Each run gets a fresh sandbox; the store is the only state that carries between runs.
_STORE_NOTE = (
    "Each run starts in a fresh sandbox: variables, functions and imports do not carry over. "
    'Keep values later runs need with `await store("rows", rows)` and get them back with '
    '`rows = await load("rows")`; `await stored()` lists what is kept.'
)


def _example_function(catalog: PtcCatalog) -> PtcFunction | None:
    """The function the code tool's description uses in examples: never a deferred one."""
    return catalog.shell_function or next(iter(catalog.upfront), None)


def _functions_note(catalog: PtcCatalog) -> str:
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


def _shell_output_note(
    catalog: PtcCatalog,
    *,
    output_guidance: PtcOutputGuidance,
    result_byte_limit: int,
) -> str:
    shell = catalog.shell_function
    if output_guidance == "off" or shell is None:
        return ""
    wait, read = None, None
    for function in catalog.upfront:
        name = function.tool_name.casefold()
        if name == PROCESS_TOOL_NAME:
            wait = f"await {function.python_name}(action='wait', process_id=r['process_id'])"
            read = (
                f"await {function.python_name}(action='read_output', process_id=..., query='error')"
            )
            break
        if name == POLL_PROCESS_TOOL_NAME:
            wait = f"await {function.python_name}(process_id=r['process_id'])"
            break
    return shell_output_note(
        output_guidance,
        call=shell.example_call(),
        result_byte_limit=result_byte_limit,
        wait=wait,
        read=read,
    )


def _shell_python_note(catalog: PtcCatalog, *, direct_shell_tool: str | None) -> str:
    if catalog.shell_function is not None:
        return _SHELL_PYTHON_NOTE.format(
            call_python=f'await {catalog.shell_function.python_name}(command="python3 script.py")'
        )
    if direct_shell_tool is not None:
        return _DIRECT_SHELL_PYTHON_NOTE.format(tool=direct_shell_tool)
    return ""


def _io_note(catalog: PtcCatalog, *, file_read: bool, file_write: bool) -> str:
    example = _example_function(catalog)
    how = f"e.g. `r = {example.example_call()}`" if example else None
    if how is None and catalog.search_active:
        how = _SEARCH_IO
    if file_read or file_write:
        from fast_agent.ptc.files import MAX_BINARY_READ_BYTES, MAX_TEXT_READ_BYTES, format_size

        files_note = _FILES_NOTE.format(
            slice_note=SLICE_FILE_NOTE if file_read else "",
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


def render_instructions(
    catalog: PtcCatalog,
    *,
    output_guidance: PtcOutputGuidance,
    result_byte_limit: int,
    file_read: bool,
    file_write: bool,
    freeform: bool,
    direct_shell_tool: str | None,
    type_check: bool,
) -> str:
    """Render the prose instructions section of ``exec_sandbox``."""
    shared = ", ".join(f"`{name}`" for name in catalog.shared_functions)
    shell = catalog.shell_function
    return _INSTRUCTIONS.format(
        output_limit=f"{max(1, round(result_byte_limit / 1000))} KB",
        shell_exit_note=_SHELL_EXIT_NOTE if shell else "",
        functions_note=_functions_note(catalog),
        search_note=_SEARCH_NOTE if catalog.search_active else "",
        io_note=_io_note(catalog, file_read=file_read, file_write=file_write),
        shell_python_note=_shell_python_note(catalog, direct_shell_tool=direct_shell_tool),
        shell_output_note=_shell_output_note(
            catalog,
            output_guidance=output_guidance,
            result_byte_limit=result_byte_limit,
        ),
        shell_batch_note=_SHELL_BATCH_NOTE.format(function=shell.python_name) if shell else "",
        shared_note=_SHARED_NOTE.format(functions=shared) if shared else "",
        input_note=_FREEFORM_INPUT_NOTE if freeform else "",
        store_note=_STORE_NOTE,
        type_check_note=_TYPE_CHECK_NOTE if type_check else "",
    )


def render_tool_description(instructions: str, *, stubs: str) -> str:
    """Append the ``Available functions:`` stub block to ``instructions``."""
    return instructions + _FUNCTIONS_BLOCK.format(stubs=f"{stubs}\n")
