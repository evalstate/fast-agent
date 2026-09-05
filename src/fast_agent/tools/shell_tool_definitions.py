from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

from mcp_types import Tool

from fast_agent.constants import MAX_PROCESS_OUTPUT_QUERY_CHARS, MAX_TERMINAL_OUTPUT_BYTE_LIMIT
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
    GROK_SHELL_TOOL_NAME,
    LUNA_EXEC_TOOL_NAME,
    POLL_PROCESS_TOOL_NAME,
    PROCESS_TOOL_NAME,
    TERMINATE_PROCESS_TOOL_NAME,
)

MAX_IDLE_YIELD_SECONDS = 30
MAX_ALIGNED_SHELL_TIMEOUT_SECONDS = 600
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
_POLL_PROCESS_ARGUMENTS = frozenset({"process_id", "wait_sec", "wake_on_output"})
_TERMINATE_PROCESS_ARGUMENTS = frozenset({"process_id"})
_MINIMAL_BASH_ARGUMENTS = frozenset({"command", "description", "run_in_background"})
_MINIMAL_PROCESS_ARGUMENTS = frozenset(
    {
        "process_id",
        "action",
        "wait_sec",
        "offset",
        "limit",
        "query",
    }
)
_GROK_SHELL_ARGUMENTS = frozenset({"command", "working_directory", "background", "timeout"})
_LUNA_EXEC_ARGUMENTS = _GROK_SHELL_ARGUMENTS


@dataclass(frozen=True, slots=True)
class ShellExecuteArguments:
    command: str
    cwd: str | None
    background: bool
    lifecycle: Literal["session", "persistent"]
    yield_after_idle_sec: int | None
    output_byte_limit: int | None
    hard_timeout_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class PollProcessArguments:
    process_id: str
    wait_sec: int
    wake_on_output: bool


@dataclass(frozen=True, slots=True)
class MinimalProcessListArguments:
    action: Literal["list"]


@dataclass(frozen=True, slots=True)
class MinimalProcessLifecycleArguments:
    process_id: str
    action: Literal["status", "wait", "stop"]
    wait_sec: int | None
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class MinimalProcessReadOutputArguments:
    process_id: str
    action: Literal["read_output"]
    offset: int
    limit: int | None
    query: str | None


MinimalProcessArguments: TypeAlias = (
    MinimalProcessListArguments
    | MinimalProcessLifecycleArguments
    | MinimalProcessReadOutputArguments
)


def build_execute_tool(*, shell_name: str) -> Tool:
    return Tool(
        name=EXECUTE_TOOL_NAME,
        description=(
            f"Run one shell command in {shell_name}. Most commands return when they "
            "exit. Foreground work remains in this call up to a bounded total-runtime "
            "cap, while retaining the normal 10-second no-output and 30-second total "
            "yield checks. If the command is still active at the cap, it keeps "
            "running and returns a process ID; "
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
        input_schema={
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
                        "Optional seconds without output before entering bounded "
                        "foreground auto-await. The total-runtime cap remains fixed "
                        "from process start. If the command outlives it, a live process "
                        "ID is returned without stopping. Defaults to 10."
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
        input_schema={
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
        input_schema={
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


def build_minimal_bash_tool(
    *,
    shell_name: str,
    tool_name: str = BASH_TOOL_NAME,
    require_description: bool = False,
    extended_guidance: bool = False,
) -> Tool:
    process_guidance = (
        "Foreground commands are awaited across the initial runtime yield and may "
        "return a managed process ID only if they outlive the bounded total-runtime "
        "auto-await cap. "
        "Do not assume a yielded process completed: use process with `wait` or "
        "`status` before relying on its output or ending the task, unless it is "
        "an intentionally persistent service. If output is truncated and a "
        "retained-output path is reported, inspect the relevant ranges with "
        "read_text_file or a targeted search before drawing conclusions. Before "
        "ending, run a task-relevant verification and inspect its result."
        if extended_guidance
        else (
            "Foreground commands are auto-awaited up to a bounded total-runtime cap "
            "and may then yield a managed process ID; "
            "use process to inspect, wait for, or stop it."
        )
    )
    properties: dict[str, Any] = {
        "command": {
            "type": "string",
            "description": "Shell command string only, without a shell executable prefix.",
        },
        "run_in_background": {
            "type": "boolean",
            "default": False,
            "description": (
                "Use true for servers, services, and other commands that "
                "must remain running. Do not also add shell `&`, `nohup`, "
                "or `disown`."
            ),
        },
    }
    required = ["command"]
    if require_description:
        properties["description"] = {
            "type": "string",
            "description": "Short operator-facing description of what the command does.",
        }
        required.append("description")
    return Tool(
        name=tool_name,
        description=(
            f"Run one shell command in {shell_name}. Set "
            "`run_in_background=true` for a server, service, or other "
            "long-running command; it returns a managed process ID and remains "
            "running for the verifier. Do not use shell `&`, `nohup`, or `disown` "
            f"to detach services. {process_guidance}"
        ),
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    )


def build_minimal_process_tool(
    *,
    default_wait_seconds: int,
    max_wait_seconds: int,
    shell_tool_name: str = BASH_TOOL_NAME,
    extended_guidance: bool = False,
) -> Tool:
    completion_guidance = (
        f" When {shell_tool_name} yields a foreground process ID, use `wait` or `status` until "
        "completion before relying on its result or ending the task."
        if extended_guidance
        else ""
    )
    return Tool(
        name=PROCESS_TOOL_NAME,
        description=(
            f"Manage processes returned by {shell_tool_name}. `list` needs no process "
            "ID; `status` returns immediately; `wait` defaults to "
            f"{default_wait_seconds} seconds.{completion_guidance} `stop` terminates "
            "the process group. `read_output` reads only that process's bounded "
            "retained output, not arbitrary files."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "process_id": {
                    "type": "string",
                    "description": (
                        f"Managed process ID returned by {shell_tool_name}. Required for status, "
                        "wait, stop, and read_output; omit for list."
                    ),
                },
                "action": {
                    "type": "string",
                    "enum": ["list", "status", "wait", "stop", "read_output"],
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
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Optional retained-output byte offset for action='read_output'. "
                        "Defaults to 0. Omit when using query unless the search should "
                        "start after a known byte offset."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_TERMINAL_OUTPUT_BYTE_LIMIT,
                    "description": (
                        "Optional output preview byte cap for wait/status, or read_output "
                        "byte limit. Does not limit execution or retained output. "
                        "Omit for the default preview; capped "
                        f"at {MAX_TERMINAL_OUTPUT_BYTE_LIMIT}."
                    ),
                },
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_PROCESS_OUTPUT_QUERY_CHARS,
                    "description": (
                        "Optional case-sensitive literal search for action='read_output'. "
                        "Returns matching retained-output lines within limit."
                    ),
                },
            },
            "additionalProperties": False,
        },
    )


def build_grok_shell_tool(*, shell_name: str) -> Tool:
    return Tool(
        name=GROK_SHELL_TOOL_NAME,
        description=(
            f"Run one shell command in {shell_name}. Omit `timeout` to use normal "
            "bounded foreground auto-await with a total-runtime cap; a managed process "
            "ID is returned only if the command remains active at that cap. When "
            "`timeout` is present, wait "
            "synchronously for completion and terminate the process group if the hard "
            "deadline expires. Set `background=true` only for a server, service, or "
            "other command that must remain running for later checks or the verifier. "
            "Do not use shell `&`, `nohup`, or `disown` to detach services."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "working_directory": {
                    "type": "string",
                    "description": "Optional working directory for this command only.",
                },
                "background": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Return promptly while the command continues as a managed "
                        "persistent process. Do not combine with `timeout`."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_ALIGNED_SHELL_TIMEOUT_SECONDS,
                    "description": (
                        "Optional foreground hard deadline in seconds. Suppresses "
                        "normal auto-yield and terminates the process group on expiry."
                    ),
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    )


def build_luna_exec_tool(*, shell_name: str) -> Tool:
    return Tool(
        name=LUNA_EXEC_TOOL_NAME,
        description=(
            f"Run one shell command in {shell_name}. Keep finite commands whose "
            "result or exit status matters in the foreground. Omit `timeout` for "
            "ordinary commands, including builds, tests, installs, downloads, "
            "compilation, training, and scripts, even when they may be slow; the "
            "runtime auto-awaits them up to a bounded total-runtime cap and returns a "
            "managed process ID only if they remain active at that cap. Set "
            "`background=true` only for a server or service that "
            "must remain running. Use `timeout` only when the user explicitly "
            "requests a hard deadline or when intentionally bounding disposable "
            "exploratory work whose termination is acceptable. Timeout expiry "
            "terminates the process group; inspect useful partial output or "
            "artifacts, and complete required work without blindly repeating the "
            "destructive deadline. Do not use shell `&`, `nohup`, or `disown`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command string without a shell executable prefix.",
                },
                "working_directory": {
                    "type": "string",
                    "description": "Optional working directory for this command only.",
                },
                "background": {
                    "type": "boolean",
                    "description": (
                        "Only for a server or service that must remain running. "
                        "Do not combine with timeout."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_ALIGNED_SHELL_TIMEOUT_SECONDS,
                    "description": (
                        "Optional foreground hard deadline in seconds. Suppresses "
                        "normal auto-yield and terminates the process group on expiry."
                    ),
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    )


def set_poll_process_tool_default_wait_seconds(
    tool: Tool,
    *,
    default_wait_seconds: int,
) -> None:
    properties = tool.input_schema.get("properties")
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
                f"Error: unknown argument {unknown[0]!r}; omit it to use the bounded "
                "foreground total-runtime cap, or set background=true to return a "
                "live process ID promptly"
            )
        rendered = ", ".join(repr(name) for name in unknown)
        raise ValueError(f"Error: unknown execute argument(s): {rendered}")

    yield_after_idle_sec = coerce_positive_int_argument(
        payload.get("yield_after_idle_sec"),
        "yield_after_idle_sec",
    )
    if yield_after_idle_sec is not None and yield_after_idle_sec > MAX_IDLE_YIELD_SECONDS:
        raise ValueError(
            f"Error: 'yield_after_idle_sec' argument must be at most {MAX_IDLE_YIELD_SECONDS}"
        )
    background = payload.get("background", False)
    if type(background) is not bool:
        raise ValueError("Error: 'background' argument must be a boolean")
    lifecycle = payload.get(
        "lifecycle",
        "persistent" if background else "session",
    )
    if lifecycle not in {"session", "persistent"}:
        raise ValueError("Error: 'lifecycle' argument must be 'session' or 'persistent'")
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
        raise ValueError("Error: 'wait_sec' argument must be a non-negative integer")
    if wait_sec > max_wait_seconds:
        raise ValueError(f"Error: 'wait_sec' argument must be at most {max_wait_seconds}")
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
    *,
    tool_name: str = BASH_TOOL_NAME,
    require_description: bool = False,
) -> ShellExecuteArguments:
    payload = coerce_tool_arguments(arguments)
    _reject_unknown_arguments(
        payload,
        _MINIMAL_BASH_ARGUMENTS,
        tool_name=tool_name,
    )
    if require_description:
        coerce_required_string_argument(
            payload.get("description"),
            "description",
            strip=True,
        )
    elif "description" in payload:
        coerce_required_string_argument(
            payload.get("description"),
            "description",
            strip=True,
        )
    run_in_background = payload.get("run_in_background", False)
    if type(run_in_background) is not bool:
        raise ValueError("Error: 'run_in_background' argument must be a boolean")
    command = coerce_required_string_argument(
        payload.get("command"),
        "command",
        strip=True,
    )
    if (
        classify_shell_detachment(
            command,
            run_in_background=run_in_background,
        )
        != "none"
    ):
        raise ValueError(
            "Shell-level backgrounding was not executed.\n"
            "Submit only the long-running service command with "
            "run_in_background=true. Use process to inspect or stop it, "
            f"and run readiness checks in a separate {tool_name} call."
        )
    return ShellExecuteArguments(
        command=command,
        cwd=None,
        background=run_in_background,
        lifecycle="persistent" if run_in_background else "session",
        yield_after_idle_sec=None,
        output_byte_limit=None,
    )


def parse_grok_shell_arguments(
    arguments: dict[str, Any] | None,
) -> ShellExecuteArguments:
    return _parse_aligned_shell_arguments(
        arguments,
        tool_name=GROK_SHELL_TOOL_NAME,
        allowed_arguments=_GROK_SHELL_ARGUMENTS,
    )


def parse_luna_exec_arguments(
    arguments: dict[str, Any] | None,
) -> ShellExecuteArguments:
    return _parse_aligned_shell_arguments(
        arguments,
        tool_name=LUNA_EXEC_TOOL_NAME,
        allowed_arguments=_LUNA_EXEC_ARGUMENTS,
    )


def _parse_aligned_shell_arguments(
    arguments: dict[str, Any] | None,
    *,
    tool_name: str,
    allowed_arguments: frozenset[str],
) -> ShellExecuteArguments:
    payload = coerce_tool_arguments(arguments)
    _reject_unknown_arguments(
        payload,
        allowed_arguments,
        tool_name=tool_name,
    )
    background = payload.get("background", False)
    if type(background) is not bool:
        raise ValueError("Error: 'background' argument must be a boolean")
    timeout = coerce_positive_int_argument(payload.get("timeout"), "timeout")
    if timeout is not None and timeout > MAX_ALIGNED_SHELL_TIMEOUT_SECONDS:
        raise ValueError(
            f"Error: 'timeout' argument must be at most {MAX_ALIGNED_SHELL_TIMEOUT_SECONDS}"
        )
    if background and timeout is not None:
        raise ValueError("Error: 'background=true' cannot be combined with 'timeout'")
    command = coerce_required_string_argument(
        payload.get("command"),
        "command",
        strip=True,
    )
    if classify_shell_detachment(command, run_in_background=background) != "none":
        raise ValueError(
            "Shell-level backgrounding was not executed.\n"
            "Submit only the long-running service command with background=true, "
            "then use a managed-process tool for lifecycle operations."
        )
    return ShellExecuteArguments(
        command=command,
        cwd=coerce_optional_string_argument(
            payload.get("working_directory"),
            "working_directory",
            empty_as_none=True,
            strip=True,
        ),
        background=background,
        lifecycle="persistent" if background else "session",
        yield_after_idle_sec=None,
        output_byte_limit=None,
        hard_timeout_seconds=timeout,
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
        tool_name="process",
    )
    action = payload.get("action", "status")
    valid_actions = {"list", "status", "wait", "stop", "read_output"}
    if action not in valid_actions:
        raise ValueError(
            "Error: 'action' must be 'list', 'status', 'wait', 'stop', or 'read_output'"
        )
    if action == "list":
        if "process_id" in payload:
            raise ValueError("Error: 'process_id' must be omitted for action='list'")
        if "wait_sec" in payload:
            raise ValueError("Error: 'wait_sec' must be omitted for action='list'")
        if {"offset", "limit", "query"} & payload.keys():
            raise ValueError(
                "Error: 'offset', 'limit', and 'query' must be omitted for action='list'"
            )
        return MinimalProcessListArguments(action="list")

    process_id = coerce_required_string_argument(
        payload.get("process_id"),
        "process_id",
        strip=True,
    )
    limit = payload.get("limit")
    if limit is not None:
        if type(limit) is not int or limit <= 0:
            raise ValueError("Error: 'limit' argument must be a positive integer")
        if limit > MAX_TERMINAL_OUTPUT_BYTE_LIMIT:
            raise ValueError(
                f"Error: 'limit' argument must be at most {MAX_TERMINAL_OUTPUT_BYTE_LIMIT}"
            )
    if action == "read_output":
        if "wait_sec" in payload:
            raise ValueError("Error: 'wait_sec' must be omitted for action='read_output'")
        offset = payload.get("offset", 0)
        if type(offset) is not int or offset < 0:
            raise ValueError("Error: 'offset' argument must be a non-negative integer")
        query = (
            coerce_required_string_argument(
                payload.get("query"),
                "query",
                strip=True,
            )
            if "query" in payload
            else None
        )
        if query is not None and len(query) > MAX_PROCESS_OUTPUT_QUERY_CHARS:
            raise ValueError(
                f"Error: 'query' argument must be at most {MAX_PROCESS_OUTPUT_QUERY_CHARS} "
                "characters"
            )
        return MinimalProcessReadOutputArguments(
            process_id=process_id,
            action="read_output",
            offset=offset,
            limit=limit,
            query=query,
        )

    if {"offset", "query"} & payload.keys():
        raise ValueError("Error: 'offset' and 'query' require action='read_output'")
    if action == "stop" and "limit" in payload:
        raise ValueError("Error: 'limit' must be omitted for action='stop'")
    wait_sec = payload.get("wait_sec")
    if action == "wait" and wait_sec is not None:
        if type(wait_sec) is not int or wait_sec < 0:
            raise ValueError("Error: 'wait_sec' argument must be a non-negative integer")
        if wait_sec > max_wait_seconds:
            raise ValueError(f"Error: 'wait_sec' argument must be at most {max_wait_seconds}")
        wait_sec = min(max(wait_sec, min_wait_seconds), max_wait_seconds)
    elif action != "wait":
        wait_sec = None
    return MinimalProcessLifecycleArguments(
        process_id=process_id,
        action=cast("Literal['status', 'wait', 'stop']", action),
        wait_sec=wait_sec,
        limit=limit,
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
