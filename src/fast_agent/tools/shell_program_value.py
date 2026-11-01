"""The value programmatic tool calling (PTC) sandbox code receives from the shell tool.

The model reads shell results as display text: interleaved output, ``[stderr]`` markers and a
process trailer, with a non-zero exit reported as an error. Sandbox code needs data instead, so a
shell call made from ``execute_python`` returns separate stdout/stderr, the exit code and the
process status, and a non-zero exit is a value rather than an exception unless ``check=True``.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict

from fast_agent.constants import FAST_AGENT_PTC_ERROR, FAST_AGENT_PTC_VALUE
from fast_agent.tools.shell_process import process_result_metadata

if TYPE_CHECKING:
    from mcp_types import CallToolResult, Tool

# Characters kept per stream for sandbox code. Sandbox code filters output before it reaches the
# model, so this bounds host memory rather than model context.
PTC_STREAM_CHAR_LIMIT = 1_000_000
PTC_CHECK_ARGUMENT = "check"

_CHECK_SCHEMA = {
    "type": "boolean",
    "description": (
        "Wait until the command finishes, then raise RuntimeError if it failed, timed out or "
        "was stopped. By default a command still running at the foreground limit is returned "
        "with status 'running', and a non-zero exit is returned in exit_code and does not raise."
    ),
}
_RETURNS_SCHEMA = {
    "type": "object",
    "properties": {
        "stdout": {"type": "string"},
        "stderr": {"type": "string"},
        "exit_code": {
            "type": ["integer", "null"],
            "description": "Null while running, or when timed out or stopped.",
        },
        "status": {
            "type": "string",
            "enum": ["completed", "failed", "running", "timed_out", "terminated", "cancelled"],
            "description": "completed means exit code 0; failed means a non-zero exit code.",
        },
        "process_id": {
            "type": ["string", "null"],
            "description": "Managed process handle, for a command that is still running.",
        },
        "truncated": {
            "type": "boolean",
            "description": "True when the middle of a very large stdout/stderr was omitted.",
        },
    },
    "required": ["stdout", "stderr", "exit_code", "status", "process_id", "truncated"],
}
_PROCESS_RETURNS_NOTE = (
    "Returns: status/wait (and polling) give the same dict as the shell function, with the "
    "stdout/stderr produced since the previous read of that process; stop gives "
    '{"process_id", "status"}; list gives a list of process dicts; read_output gives a dict '
    'whose "content" holds retained combined output.'
)
_SUCCESS_STATUSES = frozenset({"completed", "running"})
# Output kept per stream in a check=True exception message; the full streams stay available
# to code that calls without check.
_CHECK_OUTPUT_TAIL_CHARS = 1_000


class ShellProgramValue(TypedDict):
    stdout: str
    stderr: str
    exit_code: int | None
    status: str
    process_id: str | None
    truncated: bool


@dataclass(slots=True)
class _BoundedText:
    """Keep the head and tail of a stream, dropping the middle beyond ``limit`` characters."""

    limit: int
    head: list[str] = field(default_factory=list)
    head_size: int = 0
    tail: deque[str] = field(default_factory=deque)
    tail_size: int = 0
    total: int = 0

    def append(self, text: str) -> None:
        self.total += len(text)
        head_room = self.limit // 2 - self.head_size
        if head_room > 0:
            self.head.append(text[:head_room])
            self.head_size += min(len(text), head_room)
            text = text[head_room:]
        if not text:
            return
        self.tail.append(text)
        self.tail_size += len(text)
        tail_limit = self.limit - self.limit // 2
        while self.tail_size - len(self.tail[0]) >= tail_limit:
            self.tail_size -= len(self.tail.popleft())

    @property
    def truncated(self) -> bool:
        return self.total > self.limit

    def render(self) -> str:
        head = "".join(self.head)
        tail = "".join(self.tail)
        if not self.truncated:
            return head + tail
        tail = tail[-(self.limit - self.limit // 2) :]
        omitted = self.total - len(head) - len(tail)
        return f"{head}\n[... {omitted} characters omitted ...]\n{tail}"


@dataclass(frozen=True, slots=True)
class StreamText:
    """stdout/stderr text produced since the previous read."""

    stdout: str = ""
    stderr: str = ""
    truncated: bool = False


@dataclass(slots=True)
class ShellStreamCapture:
    """Separate stdout/stderr text for sandbox code, read incrementally with ``take``."""

    limit: int = PTC_STREAM_CHAR_LIMIT
    stdout: _BoundedText = field(init=False)
    stderr: _BoundedText = field(init=False)

    def __post_init__(self) -> None:
        self.stdout = _BoundedText(self.limit)
        self.stderr = _BoundedText(self.limit)

    def append(self, text: str, *, is_stderr: bool) -> None:
        (self.stderr if is_stderr else self.stdout).append(text)

    def take(self) -> StreamText:
        """Return the text captured since the previous ``take`` and start afresh."""
        text = StreamText(
            stdout=self.stdout.render(),
            stderr=self.stderr.render(),
            truncated=self.stdout.truncated or self.stderr.truncated,
        )
        self.__post_init__()
        return text


def programmatic_shell_tool(tool: Tool, *, contract: Tool | None = None) -> Tool:
    """Declare the sandbox-facing contract of a shell tool.

    ``contract`` supplies the description and keyword-argument schema for sandbox code when the
    model-facing wire format does not translate to keyword arguments, such as raw freeform
    command text.
    """
    contract = contract or tool
    schema = copy.deepcopy(contract.input_schema)
    schema.setdefault("properties", {})[PTC_CHECK_ARGUMENT] = _CHECK_SCHEMA
    # Under PTC, shell commands run from code only, so --ptc measures programmatic tool calling
    # rather than an optional code tool. Use `ptc.direct_tools` to also expose them directly.
    options: dict[str, Any] = {
        "expose": "nested",
        "input_schema": schema,
        "returns": _RETURNS_SCHEMA,
    }
    if contract is not tool and contract.description:
        options["description"] = contract.description
    meta = dict(tool.meta or {})
    meta["fast_agent"] = {**(meta.get("fast_agent") or {}), "ptc": options}
    return tool.model_copy(update={"meta": meta})


def programmatic_process_tool(tool: Tool) -> Tool:
    """Document the values a process lifecycle tool returns to sandbox code."""
    options = {
        "expose": "nested",
        "description": f"{tool.description or ''}\n\n{_PROCESS_RETURNS_NOTE}".strip(),
    }
    meta = dict(tool.meta or {})
    meta["fast_agent"] = {**(meta.get("fast_agent") or {}), "ptc": options}
    return tool.model_copy(update={"meta": meta})


def pop_check_argument(arguments: dict[str, Any] | None) -> tuple[dict[str, Any] | None, bool]:
    """Split the sandbox-only ``check`` option from the shell tool arguments."""
    if not arguments or PTC_CHECK_ARGUMENT not in arguments:
        return arguments, False
    remaining = dict(arguments)
    check = remaining.pop(PTC_CHECK_ARGUMENT)
    if type(check) is not bool:
        raise ValueError(f"Error: '{PTC_CHECK_ARGUMENT}' argument must be a boolean")
    return remaining, check


def attach_process_program_value(
    result: CallToolResult, output: StreamText, *, check: bool = False
) -> CallToolResult:
    """Give sandbox code a structured value for a command or process that ran.

    Results without process metadata (invalid arguments, unknown process, a command that could
    not start) and, with ``check``, unsuccessful commands keep no value, so sandbox code sees
    RuntimeError. A ``check`` failure carries a short message (exit code or status plus output
    tails) rather than the model-facing display text.
    """
    metadata = process_result_metadata(result)
    if metadata is None:
        return result
    status = metadata.get("process_status", "failed")
    exit_code = metadata.get("exit_code")
    if status in {"failed", "unavailable"} and exit_code is None:
        return result
    if check and status not in _SUCCESS_STATUSES:
        result.meta = {
            **(result.meta or {}),
            FAST_AGENT_PTC_ERROR: _check_failure_message(status, exit_code, output),
        }
        # A process stopped by another call reports terminated/cancelled without an error
        # flag; check must still raise, and only error results raise in sandbox code.
        result.is_error = True
        return result
    value = ShellProgramValue(
        stdout=output.stdout,
        stderr=output.stderr,
        exit_code=exit_code,
        status=status,
        process_id=metadata.get("process_id"),
        truncated=output.truncated,
    )
    return with_program_value(result, dict(value))


def _check_failure_message(status: str, exit_code: int | None, output: StreamText) -> str:
    """Describe a ``check=True`` failure the way CalledProcessError would, with output tails."""
    if exit_code is not None:
        message = f"command failed with exit code {exit_code}"
    elif status == "timed_out":
        message = "command timed out"
    else:
        message = f"command {status}"
    for label, text in (("stderr", output.stderr), ("stdout", output.stdout)):
        tail = text.strip()[-_CHECK_OUTPUT_TAIL_CHARS:]
        if tail:
            message += f"\n{label} (tail):\n{tail}"
    return message


def attach_stop_program_value(result: CallToolResult) -> CallToolResult:
    """Return ``{"process_id", "status"}`` for a stop request that reached a process."""
    metadata = process_result_metadata(result)
    if metadata is None or result.is_error:
        return result
    value = {"process_id": metadata.get("process_id"), "status": metadata.get("process_status")}
    return with_program_value(result, value)


def with_program_value(result: CallToolResult, value: object) -> CallToolResult:
    result.meta = {**(result.meta or {}), FAST_AGENT_PTC_VALUE: value}
    return result
