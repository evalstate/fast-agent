"""Shell output guidance for the code tool, and the sandbox helpers it describes.

Sandbox code receives a shell command's complete stdout and stderr, but the model sees only what
the code prints, and models tend to print whole results. The guidance says what to print instead;
``tail``, ``head``, ``grep`` and ``show`` are plain Python defined in each sandbox session (not
host functions), so they cost no tool calls.
"""

from typing import Literal

PtcOutputGuidance = Literal["off", "print", "helpers"]
"""``print`` explains what to print, with the helpers as shortcuts; ``helpers`` leads with
``show(r)``; ``off`` adds neither the guidance nor the helpers."""

OUTPUT_HELPER_NAMES = frozenset({"tail", "head", "grep", "show"})

OUTPUT_STUBS = """\
def tail(text: str, n: int = 40) -> str: ...
def head(text: str, n: int = 40) -> str: ...
def grep(text: str, pattern: str, context: int = 0, limit: int = 50) -> str: ...
def show(r: dict[str, Any], n: int = 40, pattern: str | None = None) -> str: ...
"""

# Keep these plain functions: Monty does not support decorators, classes or generators.
OUTPUT_PRELUDE = """\
import re
from typing import Any


def tail(text: str, n: int = 40) -> str:
    return "\\n".join(text.splitlines()[-n:]) if n > 0 else ""


def head(text: str, n: int = 40) -> str:
    return "\\n".join(text.splitlines()[:n]) if n > 0 else ""


def grep(text: str, pattern: str, context: int = 0, limit: int = 50) -> str:
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


def show(r: dict[str, Any], n: int = 40, pattern: str | None = None) -> str:
    fields = []
    for key in ["exit_code", "status", "process_id"]:
        value = r.get(key)
        if value is not None:
            fields.append(key + "=" + str(value))
    result = [" ".join(fields)]
    stderr = r.get("stderr") or ""
    if stderr:
        result.extend(["--- stderr", tail(stderr, n)])
    stdout = r.get("stdout") or ""
    result.extend(["--- stdout", grep(stdout, pattern) if pattern is not None else tail(stdout, n)])
    return "\\n".join(result)
"""

_PRINT_NOTE = (
    "- Shell results: `r = {call}` gives your code the complete stdout and stderr, but you "
    "only see what you print, cut to about {limit} (a longer print loses its middle). Don't "
    "`print(r)` or return `r`: it shows everything on one escaped line. Print `r['exit_code']` "
    "or `r['status']` and only the part you need: the end of build, install and test logs "
    "(`r['stdout'][-3000:]`), or the lines you pick out.{running}{retained} Shortcuts: "
    "`show(r)` gives exit_code, status and the stderr/stdout tails, "
    "`show(r, pattern='error|fail')` the matching lines; `tail(text, n)`, `head(text, n)` and "
    "`grep(text, pattern, context=0, limit=50)` work on any text.\n"
)
_PRINT_RUNNING = (
    " If `status` is `'running'`, wait with `{wait}`, then print the end of its output."
)
_PRINT_RETAINED = " Output you didn't print can still be retrieved later: `{read}`."

_HELPERS_NOTE = (
    "- Shell output: `r = {call}` gives your code the full stdout and stderr, but you only see "
    "what you print, cut to about {limit} (a longer print loses its middle). Never `print(r)` "
    "or return `r`: it shows everything on one escaped line. Print `show(r)` instead: "
    "exit_code, status, process_id and the last 40 lines of stderr and stdout; "
    "`show(r, pattern='error|fail|Traceback')` shows numbered matching lines instead. "
    "`tail(text, n=40)`, `head(text, n=40)` and `grep(text, pattern, context=0, limit=50)` "
    "work on any text.{running}{retained}\n"
)
_HELPERS_RUNNING = " If status is 'running', `print(show({wait}))`."
_HELPERS_RETAINED = " Output you didn't print can be retrieved later: `{read}`."


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
    retained = (_HELPERS_RETAINED if helpers else _PRINT_RETAINED).format(read=read) if read else ""
    return (_HELPERS_NOTE if helpers else _PRINT_NOTE).format(
        call=call,
        limit=f"{max(1, round(result_byte_limit / 1000))} KB",
        running=running,
        retained=retained,
    )
