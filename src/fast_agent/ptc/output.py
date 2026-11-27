"""Sandbox prelude helpers and shell/file output guidance for the code tool.

Each :class:`SandboxHelper` bundles a sync Python helper injected into Monty before model code
runs: its reserved global ``names``, its Monty type-checker ``stubs``, and its ``prelude`` source.
Because they are plain Python in the sandbox session (not host functions), they cost no tool calls.
"""

from dataclasses import dataclass
from typing import Literal

PtcOutputGuidance = Literal["off", "print", "helpers"]
"""``print`` explains what to print, with ``show(r)`` as a shortcut; ``helpers`` leads with
``show(r)``; ``off`` adds neither the guidance nor ``show``."""


@dataclass(frozen=True, slots=True)
class SandboxHelper:
    """A sync Python helper bundle injected into the Monty prelude."""

    names: frozenset[str]
    stubs: str
    prelude: str


# Keep plain Python: Monty does not support decorators, classes, generators, open(..., errors=...),
# or Path.read_text(errors=...). Fall back to binary decode on ValueError (Monty raises
# UnicodeDecodeError, a ValueError).
SLICE_FILE_HELPER = SandboxHelper(
    names=frozenset({"slice_file"}),
    stubs="""\
import pathlib
def slice_file(path: str | pathlib.Path, start: int = 1, end: int | None = None) -> str: ...
""",
    prelude="""\
def slice_file(path: str | pathlib.Path, start: int = 1, end: int | None = None) -> str:
    path_str = str(path)
    try:
        with open(path_str) as f:
            text = f.read()
    except ValueError:
        with open(path_str, "rb") as f:
            text = f.read().decode("utf-8", "replace")
    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        return "=== " + path_str + " (empty) ==="
    lo = max(1, start)
    hi = total if end is None else min(total, end)
    if lo > total or hi < lo:
        return "=== " + path_str + ":" + str(lo) + "-" + str(hi) + " (of " + str(total) + " lines) ==="
    width = max(4, len(str(hi)))
    header = "=== " + path_str + ":" + str(lo) + "-" + str(hi) + " (of " + str(total) + ") ==="
    body = [str(i).rjust(width) + ": " + lines[i - 1] for i in range(lo, hi + 1)]
    return "\\n".join([header] + body)
""",
)

SLICE_FILE_NOTE = (
    " Use `print(slice_file(path, start=1, end=None))` (pre-defined, no `await`) to inspect "
    "1-based inclusive numbered line ranges with a header instead of writing your own reader."
)

SHOW_HELPER = SandboxHelper(
    names=frozenset({"show"}),
    stubs="""\
def show(r: dict[str, Any], n: int = 40, pattern: str | None = None) -> str: ...
""",
    prelude="""\
import re
from typing import Any


def _grep(text: str, pattern: str, context: int = 0, limit: int = 50) -> str:
    lines = text.splitlines()
    matches = [i for i, line in enumerate(lines) if re.search(pattern, line, re.IGNORECASE)]
    selected = set()
    for i in matches[:max(0, limit)]:
        for j in range(max(0, i - max(0, context)), min(len(lines), i + max(0, context) + 1)):
            selected.add(j)
    result = [str(i + 1) + ": " + lines[i] for i in sorted(selected)]
    remaining = len(matches) - max(0, limit)
    if remaining > 0:
        result.append("[" + str(remaining) + " more matches]")
    return "\\n".join(result)


def _view(text: str, n: int) -> str:
    lines = text.splitlines()
    n = max(0, n)
    if len(lines) <= 2 * n:
        return "\\n".join(lines)
    k = max(1, n // 4) if n > 0 else 0
    omitted = len(lines) - k - n
    marker = (
        "[... " + str(omitted) + " of " + str(len(lines)) + " lines omitted; "
        + "print the stream or use show(r, pattern=...) to see them ...]"
    )
    return "\\n".join(lines[:k] + [marker] + lines[len(lines) - n:])


def show(r: dict[str, Any], n: int = 40, pattern: str | None = None) -> str:
    fields = []
    for key in ["exit_code", "status", "process_id"]:
        value = r.get(key)
        if value is not None:
            fields.append(key + "=" + str(value))
    if r.get("truncated"):
        fields.append("truncated=True")
    result = [" ".join(fields)]
    stderr = r.get("stderr") or ""
    if stderr:
        result.extend(["--- stderr", _grep(stderr, pattern) if pattern is not None else _view(stderr, n)])
    stdout = r.get("stdout") or ""
    result.extend(["--- stdout", _grep(stdout, pattern) if pattern is not None else _view(stdout, n)])
    return "\\n".join(result)
""",
)

_RESULT_SHAPE = (
    "returns a dict of stdout, stderr, exit_code, status ('completed', 'failed', 'running', "
    "...), process_id and truncated (True if a stream over a million characters lost its "
    "middle)"
)
_PRINT_NOTE = (
    "- Shell: `r = {call}` " + _RESULT_SHAPE + ". Print only what you need, not `r` (a dict "
    "prints as one escaped line): `r['exit_code']`, the end of build and test logs "
    "(`r['stdout'][-3000:]`), or lines you pick out. `show(r)` prints exit_code, status and both "
    "streams (long ones as their first 10 and last 40 lines), or `show(r, pattern='error|fail')` "
    "for numbered matches. You see at most about {limit} of printed output; a longer print loses "
    "its middle.{running}{stored}{retained}\n"
)
_PRINT_RUNNING = " If `status` is 'running', wait with `{wait}`, then print the end of its output."
_HELPERS_NOTE = (
    "- Shell: `r = {call}` " + _RESULT_SHAPE + ". Print `show(r)`, not `r` (a dict prints as "
    "one escaped line): it gives exit_code and status, then stderr and stdout, each in full up "
    "to 80 lines, otherwise the first 10 and last 40 lines around an omitted-lines marker; "
    "`show(r, pattern='error|fail')` gives numbered matching lines instead. You see at most about "
    "{limit} of printed output; a longer print loses its middle.{running}{stored}{retained}\n"
)
_HELPERS_RUNNING = " If `status` is 'running', `print(show({wait}))` waits for it."
_STORE_NOTE = ' To use `r` in a later run, `await store("r", r)`.'
_RETAINED_NOTE = " `{read}` searches kept output, if any."


def shell_output_note(
    guidance: PtcOutputGuidance,
    *,
    call: str,
    result_byte_limit: int,
    wait: str | None,
    read: str | None,
) -> str:
    """Render the guidance for ``call``; ``wait`` and ``read`` are process calls, if any."""
    if guidance == "off":
        return ""
    helpers = guidance == "helpers"
    running = (_HELPERS_RUNNING if helpers else _PRINT_RUNNING).format(wait=wait) if wait else ""
    retained = _RETAINED_NOTE.format(read=read) if read else ""
    return (_HELPERS_NOTE if helpers else _PRINT_NOTE).format(
        call=call,
        limit=f"{max(1, round(result_byte_limit / 1000))} KB",
        running=running,
        stored=_STORE_NOTE,
        retained=retained,
    )
