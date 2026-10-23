"""Contract for the value sandbox code receives from shell calls made in execute_python."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

import pytest

from fast_agent.config import Settings, ShellSettings
from fast_agent.constants import FAST_AGENT_PTC_VALUE
from fast_agent.ptc.stubs import build_functions
from fast_agent.tools.invocation_context import programmatic_tool_call
from fast_agent.tools.shell_process import process_result_metadata
from fast_agent.tools.shell_program_value import ShellStreamCapture
from fast_agent.tools.shell_runtime import ShellRuntime

if TYPE_CHECKING:
    from pathlib import Path

    from mcp_types import CallToolResult

    from fast_agent.tools.shell_profiles import ShellToolProfile


def _runtime(profile: ShellToolProfile, **kwargs: Any) -> ShellRuntime:
    return ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("shell-program-value-test"),
        model_shell_tool_name="shell",
        config=Settings(shell_execution=ShellSettings(tool_profile=profile, show_bash=False)),
        **kwargs,
    )


async def _call_from_sandbox(
    runtime: ShellRuntime, arguments: dict[str, Any], *, tool: str | None = None
) -> CallToolResult:
    with programmatic_tool_call("ptc-1"):
        return await runtime.call_tool(tool or runtime.tools[0].name, arguments, "ptc-1")


def _value(result: CallToolResult) -> Any:
    return (result.meta or {})[FAST_AGENT_PTC_VALUE]


async def _wait_from_sandbox(runtime: ShellRuntime, process_id: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    while not values or values[-1]["status"] == "running":
        polled = await _call_from_sandbox(
            runtime,
            {"action": "wait", "process_id": process_id, "wait_sec": 10},
            tool="process",
        )
        values.append(_value(polled))
    return values


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile", "command_arguments"),
    [
        ("freeform_shell", {}),
        ("luna_exec", {}),
        ("grok_shell", {}),
        ("minimal_process", {"description": "test"}),
        ("native", {}),
    ],
)
async def test_sandbox_calls_get_separate_streams_and_exit_code_without_raising(
    profile: ShellToolProfile, command_arguments: dict[str, Any]
) -> None:
    runtime = _runtime(profile)
    try:
        result = await _call_from_sandbox(
            runtime,
            {"command": "printf '1.5\\n'; printf 'warn\\n' >&2; exit 3", **command_arguments},
        )
    finally:
        await runtime.close()

    value = (result.meta or {})[FAST_AGENT_PTC_VALUE]
    assert value["stdout"] == "1.5\n"
    assert value["stderr"] == "warn\n"
    assert value["exit_code"] == 3
    assert value["status"] == "failed"
    assert value["truncated"] is False
    # The model and display still see a failed command.
    assert result.is_error


@pytest.mark.asyncio
async def test_check_raises_for_failures_only() -> None:
    runtime = _runtime("freeform_shell")
    try:
        failed = await _call_from_sandbox(runtime, {"command": "exit 2", "check": True})
        passed = await _call_from_sandbox(runtime, {"command": "echo ok", "check": True})
        invalid = await _call_from_sandbox(runtime, {"command": "echo ok", "check": "yes"})
    finally:
        await runtime.close()

    assert failed.is_error and FAST_AGENT_PTC_VALUE not in (failed.meta or {})
    assert (passed.meta or {})[FAST_AGENT_PTC_VALUE]["stdout"] == "ok\n"
    assert invalid.is_error and FAST_AGENT_PTC_VALUE not in (invalid.meta or {})


@pytest.mark.asyncio
async def test_timeouts_are_values_and_model_calls_keep_their_wire_format() -> None:
    runtime = _runtime("freeform_shell")
    try:
        timed_out = await _call_from_sandbox(
            runtime, {"command": "echo started; sleep 5", "timeout": 1}
        )
        model_call = await runtime.call_tool("shell", {"input": "echo raw"}, "call-1")
        model_keywords = await runtime.call_tool("shell", {"command": "echo raw"}, "call-2")
    finally:
        await runtime.close()

    value = (timed_out.meta or {})[FAST_AGENT_PTC_VALUE]
    assert value["status"] == "timed_out"
    assert value["exit_code"] is None
    assert value["stdout"] == "started\n"
    assert not model_call.is_error and FAST_AGENT_PTC_VALUE not in (model_call.meta or {})
    assert model_keywords.is_error


def test_freeform_shell_is_a_keyword_function_in_the_sandbox() -> None:
    runtime = _runtime("freeform_shell")
    function = build_functions(runtime.tools)["shell"]

    assert function.signature().startswith(
        "shell(*, command: str, working_directory: str = ..., background: bool = ..., "
        "timeout: int = ..., check: bool = ...) -> dict[str, Any]"
    )
    assert "not JSON" not in function.stub()
    assert "exit_code: int | None," in function.stub()


def test_large_streams_keep_head_and_tail() -> None:
    streams = ShellStreamCapture()
    limit = streams.stdout.limit
    for index in range(limit // 10 + 100):
        streams.append(f"{index:09d}\n", is_stderr=False)

    rendered = streams.stdout.render()
    assert streams.stdout.truncated
    assert rendered.startswith("000000000\n")
    assert rendered.endswith(f"{limit // 10 + 99:09d}\n")
    assert "characters omitted" in rendered
    assert len(rendered) < limit + 100


@pytest.mark.asyncio
async def test_process_polls_return_only_output_the_sandbox_has_not_seen() -> None:
    runtime = _runtime(
        "luna_exec",
        idle_yield_seconds=0.05,
        foreground_yield_seconds=0.3,
        foreground_auto_await_max_seconds=0,
    )
    try:
        started = _value(
            await _call_from_sandbox(
                runtime, {"command": "echo first; sleep 1; echo second >&2; exit 3"}
            )
        )
        polls = await _wait_from_sandbox(runtime, started["process_id"])
        listed = _value(await _call_from_sandbox(runtime, {"action": "list"}, tool="process"))
        stopped = _value(
            await _call_from_sandbox(
                runtime, {"action": "stop", "process_id": started["process_id"]}, tool="process"
            )
        )
        unknown = await _call_from_sandbox(
            runtime, {"action": "status", "process_id": "process-404"}, tool="process"
        )
    finally:
        await runtime.close()

    assert started["status"] == "running" and started["exit_code"] is None
    assert started["stdout"] == "first\n"
    assert "".join(poll["stdout"] for poll in polls) == ""
    assert "".join(poll["stderr"] for poll in polls) == "second\n"
    assert (polls[-1]["status"], polls[-1]["exit_code"]) == ("failed", 3)
    assert [entry["process_id"] for entry in listed] == [started["process_id"]]
    assert stopped == {"process_id": started["process_id"], "status": "already_exited"}
    assert unknown.is_error and FAST_AGENT_PTC_VALUE not in (unknown.meta or {})


@pytest.mark.asyncio
async def test_processes_started_by_the_model_can_be_polled_from_the_sandbox() -> None:
    runtime = _runtime(
        "luna_exec",
        idle_yield_seconds=0.05,
        foreground_yield_seconds=0.3,
        foreground_auto_await_max_seconds=0,
    )
    try:
        started = await runtime.call_tool(
            "shell", {"command": "echo out; echo err >&2; sleep 1"}, "call-1"
        )
        metadata = process_result_metadata(started)
        assert metadata is not None
        process_id = metadata["process_id"]
        polls = await _wait_from_sandbox(runtime, process_id)
    finally:
        await runtime.close()

    assert "".join(poll["stdout"] for poll in polls) == "out\n"
    assert "".join(poll["stderr"] for poll in polls) == "err\n"
    assert polls[-1]["exit_code"] == 0


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="durable local processes require POSIX")
async def test_durable_background_processes_return_separate_streams(tmp_path: Path) -> None:
    runtime = _runtime(
        "luna_exec", working_directory=tmp_path, durable_process_root=tmp_path / "processes"
    )
    try:
        launched = _value(
            await _call_from_sandbox(
                runtime,
                {
                    "command": "echo ready; sleep 0.5; echo late >&2; exit 4",
                    "background": True,
                },
            )
        )
        await asyncio.sleep(0)
        polls = await _wait_from_sandbox(runtime, launched["process_id"])
    finally:
        await runtime.close()

    outputs = [launched, *polls]
    assert "".join(value["stdout"] for value in outputs) == "ready\n"
    assert "".join(value["stderr"] for value in outputs) == "late\n"
    assert (polls[-1]["status"], polls[-1]["exit_code"]) == ("failed", 4)
