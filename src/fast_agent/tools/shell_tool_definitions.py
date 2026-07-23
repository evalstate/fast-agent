from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from mcp.types import Tool

from fast_agent.constants import MAX_TERMINAL_OUTPUT_BYTE_LIMIT
from fast_agent.tools.filesystem_tool_args import (
    coerce_optional_string_argument,
    coerce_positive_int_argument,
    coerce_required_string_argument,
    coerce_tool_arguments,
)
from fast_agent.tools.shell_command import classify_shell_detachment
from fast_agent.utils.tool_names import (
    BASH_TOOL_NAME,
    EXECUTE_TOOL_NAME,
    POLL_PROCESS_TOOL_NAME,
    PROCESS_TOOL_NAME,
    TERMINATE_PROCESS_TOOL_NAME,
)

MAX_IDLE_YIELD_SECONDS = 30
PROCESS_OUTPUT_DEBOUNCE_SECONDS = 2.0

_EXECUTE_ARGUMENTS = frozenset(
    {
        "command",
        "cwd",
        "background",
        "lifecycle",
        "yield_after_idle_sec",
        "output_byte_limit",
    }
)
_POLL_PROCESS_ARGUMENTS = frozenset(
    {"process_id", "wait_sec", "wake_on_output"}
)
_TERMINATE_PROCESS_ARGUMENTS = frozenset({"process_id"})
_MINIMAL_BASH_ARGUMENTS = frozenset({"command", "run_in_background"})
_MINIMAL_PROCESS_ARGUMENTS = frozenset({"process_id", "action", "wait_sec"})


@dataclass(frozen=True, slots=True)
class ShellExecuteArguments:
    command: str
    cwd: str | None
    background: bool
    lifecycle: Literal["session", "persistent"]
    yield_after_idle_sec: int | None
    output_byte_limit: int | None


@dataclass(frozen=True, slots=True)
class PollProcessArguments:
    process_id: str
    wait_sec: int
    wake_on_output: bool


@dataclass(frozen=True, slots=True)
class MinimalProcessArguments:
    process_id: str
    action: Literal["status", "wait", "stop"]
    wait_sec: int | None


def build_execute_tool(*, shell_name: str) -> Tool:
    return Tool(
        name=EXECUTE_TOOL_NAME,
        description=(
            f"Run one shell command in {shell_name}. Most commands return when they "
            "exit. If a foreground command remains active for 10 seconds without "
            "output or 30 seconds total, it keeps running and returns a process ID; "
            "use poll_process to monitor it or terminate_process to stop it. Set "
            "`background=true` for known long-running commands. Explicit background "
            "commands default to `lifecycle='persistent'` and remain running after "
            "the agent runtime exits; set `lifecycle='session'` for temporary "
            "concurrent jobs that should be terminated at shutdown. Persistent "
            "output is monitorable while the runtime is active and continues to a "
            "reported spool path after shutdown. Automatically yielded foreground "
            "commands remain session-scoped. Do not append '&'. "
            "`cwd` and `output_byte_limit` apply only to this command. Pipelines "
            "report the final command's status unless you enable `pipefail`."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Command string only - no shell executable prefix "
                        "(correct: 'pwd', incorrect: 'bash -c pwd')."
                    ),
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory for this command only.",
                },
                "background": {
                    "type": "boolean",
                    "description": (
                        "Return promptly while the command continues running as a "
                        "managed process. By default it remains running after the "
                        "agent runtime exits. Set lifecycle='session' for temporary "
                        "concurrent work that should be terminated at shutdown. Do "
                        "not append '&' to the command."
                    ),
                },
                "lifecycle": {
                    "type": "string",
                    "enum": ["session", "persistent"],
                    "default": "persistent",
                    "description": (
                        "Lifetime of a background command. Omitted lifecycle defaults "
                        "to 'persistent' when background=true. 'session' terminates "
                        "the process when the agent runtime exits. 'persistent' leaves "
                        "it running in the execution environment after the agent exits "
                        "and writes subsequent output to its reported, size-unbounded "
                        "spool path. "
                        "Automatically yielded foreground commands are always "
                        "session-scoped."
                    ),
                },
                "yield_after_idle_sec": {
                    "type": "integer",
                    "description": (
                        "Optional seconds without output before returning a live "
                        "process ID without stopping the command. Defaults to 10."
                    ),
                    "minimum": 1,
                    "maximum": MAX_IDLE_YIELD_SECONDS,
                },
                "output_byte_limit": {
                    "type": "integer",
                    "description": (
                        "Optional maximum output bytes returned to the model for this "
                        "command (clamped to "
                        f"{MAX_TERMINAL_OUTPUT_BYTE_LIMIT}). Complete output is not "
                        "retained after truncation."
                    ),
                    "minimum": 1,
                    "maximum": MAX_TERMINAL_OUTPUT_BYTE_LIMIT,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    )


def build_poll_process_tool(
    *,
    default_wait_seconds: int,
    max_wait_seconds: int,
) -> Tool:
    return Tool(
        name=POLL_PROCESS_TOOL_NAME,
        description=(
            "Wait for a managed shell process to exit or for the polling deadline. "
            "Completion always returns promptly. Routine stdout/stderr is buffered "
            "and included when the call returns, but does not end the wait by default. "
            "Omit `wait_sec` to use the model default declared in the schema, or use "
            "0 for a non-blocking status check. Set `wake_on_output=true` only when "
            "new output must affect the next action immediately; output-triggered "
            "returns are debounced until output has been quiet for "
            f"{PROCESS_OUTPUT_DEBOUNCE_SECONDS:g} seconds, while continuous output "
            "remains buffered until completion or the deadline. Repeated polls return "
            "only output not returned previously."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "process_id": {
                    "type": "string",
                    "description": "Process ID returned by execute.",
                },
                "wait_sec": {
                    "type": "integer",
                    "default": default_wait_seconds,
                    "description": (
                        f"Optional wait in seconds, from 0 through {max_wait_seconds}."
                    ),
                    "minimum": 0,
                    "maximum": max_wait_seconds,
                },
                "wake_on_output": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Return after new output has been quiet for "
                        f"{PROCESS_OUTPUT_DEBOUNCE_SECONDS:g} seconds. Defaults to "
                        "false so routine output remains buffered until the process "
                        "completes or wait_sec elapses. Continuous output does not "
                        "return early."
                    ),
                },
            },
            "required": ["process_id"],
            "additionalProperties": False,
        },
    )


def build_terminate_process_tool() -> Tool:
    return Tool(
        name=TERMINATE_PROCESS_TOOL_NAME,
        description=(
            "Terminate a managed shell process and its process group. Returns success "
            "if the process was terminated or had already exited."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "process_id": {
                    "type": "string",
                    "description": "Process ID returned by execute.",
                }
            },
            "required": ["process_id"],
            "additionalProperties": False,
        },
    )


def build_minimal_bash_tool(*, shell_name: str) -> Tool:
    return Tool(
        name=BASH_TOOL_NAME,
        description=(
            f"Run one shell command in {shell_name}. Set "
            "`run_in_background=true` for a server, service, or other "
            "long-running command; it returns a managed process ID and remains "
            "running for the verifier. Do not use shell `&`, `nohup`, or `disown` "
            "to detach services. Foreground commands that take time may yield a "
            "managed process ID; use Process to inspect, wait for, or stop it."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "run_in_background": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Use true for servers, services, and other commands that "
                        "must remain running. Do not also add shell `&`, `nohup`, "
                        "or `disown`."
                    ),
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    )


def build_minimal_process_tool(
    *,
    default_wait_seconds: int,
    max_wait_seconds: int,
) -> Tool:
    return Tool(
        name=PROCESS_TOOL_NAME,
        description=(
            "Inspect, wait for, or stop a managed process returned by Bash. "
            "`status` returns immediately. `wait` accepts an optional `wait_sec`; "
            "when omitted it uses the configured model-specific polling interval "
            "(with a nonzero fallback when the model has none). "
            f"Use {default_wait_seconds} seconds unless more frequent monitoring "
            "is needed. `stop` terminates the process group."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "process_id": {
                    "type": "string",
                    "description": "Managed process ID returned by Bash.",
                },
                "action": {
                    "type": "string",
                    "enum": ["status", "wait", "stop"],
                    "default": "status",
                },
                "wait_sec": {
                    "type": "integer",
                    "description": (
                        "Optional wait in seconds for action='wait', from 0 through "
                        f"{max_wait_seconds}. Values below 10 are clamped to 10."
                    ),
                    "minimum": 0,
                    "maximum": max_wait_seconds,
                },
            },
            "required": ["process_id"],
            "additionalProperties": False,
        },
    )


def set_poll_process_tool_default_wait_seconds(
    tool: Tool,
    *,
    default_wait_seconds: int,
) -> None:
    properties = tool.inputSchema.get("properties")
    if not isinstance(properties, dict):
        return
    wait_schema = properties.get("wait_sec")
    if isinstance(wait_schema, dict):
        wait_schema["default"] = default_wait_seconds


def _reject_unknown_arguments(
    payload: dict[str, Any],
    allowed: frozenset[str],
    *,
    tool_name: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if not unknown:
        return
    rendered = ", ".join(repr(name) for name in unknown)
    raise ValueError(f"Error: unknown {tool_name} argument(s): {rendered}")


def parse_execute_arguments(
    arguments: dict[str, Any] | None,
) -> ShellExecuteArguments:
    payload = coerce_tool_arguments(arguments)
    unknown = sorted(set(payload) - _EXECUTE_ARGUMENTS)
    if unknown:
        if unknown in (["timeout"], ["timeout_sec"]):
            raise ValueError(
                f"Error: unknown argument {unknown[0]!r}; use "
                "'yield_after_idle_sec' to return a live process ID without stopping "
                "the command"
            )
        rendered = ", ".join(repr(name) for name in unknown)
        raise ValueError(f"Error: unknown execute argument(s): {rendered}")

    yield_after_idle_sec = coerce_positive_int_argument(
        payload.get("yield_after_idle_sec"),
        "yield_after_idle_sec",
    )
    if (
        yield_after_idle_sec is not None
        and yield_after_idle_sec > MAX_IDLE_YIELD_SECONDS
    ):
        raise ValueError(
            "Error: 'yield_after_idle_sec' argument must be at most "
            f"{MAX_IDLE_YIELD_SECONDS}"
        )
    background = payload.get("background", False)
    if type(background) is not bool:
        raise ValueError("Error: 'background' argument must be a boolean")
    lifecycle = payload.get(
        "lifecycle",
        "persistent" if background else "session",
    )
    if lifecycle not in {"session", "persistent"}:
        raise ValueError(
            "Error: 'lifecycle' argument must be 'session' or 'persistent'"
        )
    if lifecycle == "persistent" and not background:
        raise ValueError("Error: lifecycle='persistent' requires background=true")

    output_byte_limit = coerce_positive_int_argument(
        payload.get("output_byte_limit"),
        "output_byte_limit",
    )
    if output_byte_limit is not None:
        output_byte_limit = min(
            output_byte_limit,
            MAX_TERMINAL_OUTPUT_BYTE_LIMIT,
        )

    return ShellExecuteArguments(
        command=coerce_required_string_argument(
            payload.get("command"),
            "command",
            strip=True,
        ),
        cwd=coerce_optional_string_argument(
            payload.get("cwd"),
            "cwd",
            empty_as_none=True,
            strip=True,
        ),
        background=background,
        lifecycle=cast("Literal['session', 'persistent']", lifecycle),
        yield_after_idle_sec=yield_after_idle_sec,
        output_byte_limit=output_byte_limit,
    )


def parse_poll_process_arguments(
    arguments: dict[str, Any] | None,
    *,
    default_wait_seconds: int,
    max_wait_seconds: int,
) -> PollProcessArguments:
    payload = coerce_tool_arguments(arguments)
    _reject_unknown_arguments(
        payload,
        _POLL_PROCESS_ARGUMENTS,
        tool_name="poll_process",
    )
    wait_sec = payload.get("wait_sec", default_wait_seconds)
    if type(wait_sec) is not int or wait_sec < 0:
        raise ValueError(
            "Error: 'wait_sec' argument must be a non-negative integer"
        )
    if wait_sec > max_wait_seconds:
        raise ValueError(
            f"Error: 'wait_sec' argument must be at most {max_wait_seconds}"
        )
    wake_on_output = payload.get("wake_on_output", False)
    if type(wake_on_output) is not bool:
        raise ValueError("Error: 'wake_on_output' argument must be a boolean")
    return PollProcessArguments(
        process_id=coerce_required_string_argument(
            payload.get("process_id"),
            "process_id",
            strip=True,
        ),
        wait_sec=wait_sec,
        wake_on_output=wake_on_output,
    )


def parse_minimal_bash_arguments(
    arguments: dict[str, Any] | None,
) -> ShellExecuteArguments:
    payload = coerce_tool_arguments(arguments)
    _reject_unknown_arguments(
        payload,
        _MINIMAL_BASH_ARGUMENTS,
        tool_name="Bash",
    )
    run_in_background = payload.get("run_in_background", False)
    if type(run_in_background) is not bool:
        raise ValueError("Error: 'run_in_background' argument must be a boolean")
    command = coerce_required_string_argument(
        payload.get("command"),
        "command",
        strip=True,
    )
    if classify_shell_detachment(
        command,
        run_in_background=run_in_background,
    ) != "none":
        raise ValueError(
            "Shell-level backgrounding was not executed.\n"
            "Submit only the long-running service command with "
            "run_in_background=true. Use Process to inspect or stop it, "
            "and run readiness checks in a separate Bash call."
        )
    return ShellExecuteArguments(
        command=command,
        cwd=None,
        background=run_in_background,
        lifecycle="persistent" if run_in_background else "session",
        yield_after_idle_sec=None,
        output_byte_limit=None,
    )


def parse_minimal_process_arguments(
    arguments: dict[str, Any] | None,
    *,
    min_wait_seconds: int,
    max_wait_seconds: int,
) -> MinimalProcessArguments:
    payload = coerce_tool_arguments(arguments)
    _reject_unknown_arguments(
        payload,
        _MINIMAL_PROCESS_ARGUMENTS,
        tool_name="Process",
    )
    action = payload.get("action", "status")
    if action not in {"status", "wait", "stop"}:
        raise ValueError("Error: 'action' must be 'status', 'wait', or 'stop'")
    wait_sec = payload.get("wait_sec")
    if action == "wait" and wait_sec is not None:
        if type(wait_sec) is not int or wait_sec < 0:
            raise ValueError(
                "Error: 'wait_sec' argument must be a non-negative integer"
            )
        if wait_sec > max_wait_seconds:
            raise ValueError(
                f"Error: 'wait_sec' argument must be at most {max_wait_seconds}"
            )
        wait_sec = min(max(wait_sec, min_wait_seconds), max_wait_seconds)
    elif action != "wait":
        wait_sec = None
    return MinimalProcessArguments(
        process_id=coerce_required_string_argument(
            payload.get("process_id"),
            "process_id",
            strip=True,
        ),
        action=cast("Literal['status', 'wait', 'stop']", action),
        wait_sec=wait_sec,
    )


def parse_terminate_process_arguments(
    arguments: dict[str, Any] | None,
) -> str:
    payload = coerce_tool_arguments(arguments)
    _reject_unknown_arguments(
        payload,
        _TERMINATE_PROCESS_ARGUMENTS,
        tool_name="terminate_process",
    )
    return coerce_required_string_argument(
        payload.get("process_id"),
        "process_id",
        strip=True,
    )
