"""Shell output guidance for the code tool, and the sandbox helpers it describes.

Sandbox code receives bounded stdout and stderr captured since the previous read; the model sees what
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


def _view(text: str, n: int) -> str:
    lines = text.splitlines()
    if len(lines) <= 2 * n:
        return "\\n".join(lines)
    k = max(1, n // 4)
    omitted = len(lines) - k - n
    marker = (
        "[... " + str(omitted) + " of " + str(len(lines)) + " lines omitted; "
        + "print the stream or use head()/grep() to see them ...]"
    )
    return "\\n".join(lines[:k] + [marker] + lines[-n:])


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
        result.extend(["--- stderr", grep(stderr, pattern) if pattern is not None else _view(stderr, n)])
    stdout = r.get("stdout") or ""
    result.extend(["--- stdout", grep(stdout, pattern) if pattern is not None else _view(stdout, n)])
    return "\\n".join(result)
"""

PY_HELPER_NAME = "py"
PY_STUB = """\
async def py(code: str, check: bool = False) -> dict[str, Any]: ...
"""
# Runs ``code`` with the machine's CPython through the local shell function ({shell}), so
# sandbox code never has to embed a heredoc in a shell command string.
PY_PRELUDE = """\
from typing import Any


async def py(code: str, check: bool = False) -> dict[str, Any]:
    command = (
        "FAST_AGENT_PY=$(command -v python3 || command -v python) && "
        + '"$FAST_AGENT_PY" - <<' + "'FAST_AGENT_PY_EOF'\\n"
        + code
        + "\\nFAST_AGENT_PY_EOF"
    )
    return await {shell}(command=command, check=check)
"""

_PRINT_NOTE = (
    "- Shell results: `r = {call}` gives your code bounded stdout and stderr since the previous read; you "
    "only see what you print, cut to about {limit} (a longer print loses its middle). Don't "
    "`print(r)` or return `r`: it shows everything on one escaped line. Print `r['exit_code']` "
    "or `r['status']` and only the part you need: the end of build, install and test logs "
    "(`r['stdout'][-3000:]`), or the lines you pick out. Check stderr and `r['truncated']` "
    "for capture loss.{running}{stored}{retained} Shortcuts: "
    "`show(r)` gives exit_code, status, process_id, any capture truncation and stderr/stdout "
    "(in full up to 80 lines, else the first 10 and last 40 around an omitted-lines marker), "
    "`show(r, pattern='error|fail')` matching lines from both streams; `tail(text, n)`, `head(text, n)` and "
    "`grep(text, pattern, context=0, limit=50)` work on any text.\n"
)
_PRINT_RUNNING = (
    " If `status` is `'running'`, wait with `{wait}`, then print the end of its output."
)
_STORE_NOTE = (
    ' If you may need more later, `await store("result", r)` before ending this call; '
    '`r = await load("result")` recovers that capture, not output already dropped.'
)
_RETAINED_NOTE = (
    " `{read}` searches available bounded retained output, not all unprinted output. "
    "Short output may not be retained; use store/load for it."
)

_HELPERS_NOTE = (
    "- Shell output: `r = {call}` gives your code bounded stdout and stderr since the previous read; you see "
    "what you print, cut to about {limit} (a longer print loses its middle). Never `print(r)` "
    "or return `r`: it shows everything on one escaped line. Print `show(r)` instead: "
    "exit_code, status, process_id, `truncated=True` if capture lost output, and stderr and "
    "stdout: in full up to 80 lines, else the first 10 and last 40 around an omitted-lines "
    "marker. show() is for command results: print the full text of files you need to read, "
    "not show() of a `cat`. "
    "`show(r, pattern='error|fail|Traceback')` searches both streams for numbered matching lines. "
    "`tail(text, n=40)`, `head(text, n=40)` and `grep(text, pattern, context=0, limit=50)` "
    "work on any text.{running}{stored}{retained}\n"
)
_HELPERS_RUNNING = " If status is 'running', `print(show({wait}))`."


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
