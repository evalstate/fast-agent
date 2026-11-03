from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import sys
from typing import TYPE_CHECKING

import pytest
from mcp_types import CallToolResult, TextContent

if TYPE_CHECKING:
    from pathlib import Path

import fast_agent.tools.shell_runtime as shell_runtime_module
from fast_agent.config import Settings, ShellSettings, ShellToolProfile
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.tools.execution_environment import (
    ShellExecution,
    ShellExecutionCallbacks,
    ShellExecutionRequest,
    ShellRuntimeInfo,
)
from fast_agent.tools.shell_output import ShellOutputBuffer
from fast_agent.tools.shell_process import process_result_metadata
from fast_agent.tools.shell_profiles import resolve_shell_tool_profile
from fast_agent.tools.shell_runtime import ShellRuntime


class _ManagedEnvironment:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False
        self.requests: list[ShellExecutionRequest] = []

    async def open(self) -> None:
        return None

    @property
    def cwd(self) -> str:
        return "/workspace"

    def runtime_info(self) -> ShellRuntimeInfo:
        return ShellRuntimeInfo(name="pwsh", kind="remote", provider="test")

    async def execute(
        self,
        request: ShellExecutionRequest,
        *,
        callbacks: ShellExecutionCallbacks | None = None,
    ) -> ShellExecution:
        self.requests.append(request)
        if callbacks is not None:
            await callbacks.on_started(4321)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = request.terminate_on_cancel
            raise
        raise AssertionError("unreachable")

    async def close(self) -> None:
        return None


def _runtime(
    profile: ShellToolProfile,
    environment: _ManagedEnvironment | None = None,
) -> ShellRuntime:
    return ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("luna-exec-test"),
        shell_environment=environment,
        foreground_yield_seconds=0.001,
        config=Settings(shell_execution=ShellSettings(tool_profile=profile)),
    )


def _text(result: CallToolResult) -> str:
    assert result.content
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def test_luna_exec_profile_exposes_exec_and_unified_process() -> None:
    runtime = _runtime("luna_exec")

    assert [tool.name for tool in runtime.tools] == ["exec", "process"]
    execute = runtime.tools[0]
    assert set(execute.input_schema["properties"]) == {
        "command",
        "working_directory",
        "background",
        "timeout",
    }
    assert "default" not in execute.input_schema["properties"]["background"]
    assert "server or service" in execute.input_schema["properties"]["background"]["description"]
    assert "result or exit status matters" in (execute.description or "")

    process = runtime.tools[1]
    properties = process.input_schema["properties"]
    assert "read_output" in properties["action"]["enum"]
    assert set(properties) >= {"process_id", "action", "offset", "limit", "query"}
    assert "path" not in properties


@pytest.mark.asyncio
async def test_luna_exec_adopts_explicit_model_shell_tool_name() -> None:
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("luna-exec-test"),
        foreground_yield_seconds=0.001,
        model_shell_tool_name="shell",
        config=Settings(shell_execution=ShellSettings(tool_profile="luna_exec", show_bash=False)),
    )

    assert [tool.name for tool in runtime.tools] == ["shell", "process"]
    assert runtime.owns_tool("shell")
    assert not runtime.owns_tool("exec")
    assert set(runtime.tools[0].input_schema["properties"]) == {
        "command",
        "working_directory",
        "background",
        "timeout",
    }
    assert "returned by shell" in (runtime.tools[1].description or "")

    result = await runtime.call_tool("shell", {"command": "echo luna-shell-ok"})
    assert not result.is_error
    assert "luna-shell-ok" in _text(result)

    rejected = await runtime.call_tool("shell", {"command": "echo x", "run_in_background": True})
    assert rejected.is_error


def test_luna_exec_keeps_exec_name_without_model_shell_tool_name() -> None:
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("luna-exec-test"),
        model_shell_tool_name=None,
        config=Settings(shell_execution=ShellSettings(tool_profile="luna_exec")),
    )
    assert [tool.name for tool in runtime.tools] == ["exec", "process"]


@pytest.mark.parametrize("suffix", ["é", "€", "😀"])
@pytest.mark.parametrize("limit", [2048, 3000])
def test_preview_preserves_buffer_truncation_at_utf8_boundary(suffix: str, limit: int) -> None:
    buffer = ShellOutputBuffer(output_byte_limit=2048)
    output = "a" * 2047 + suffix
    buffer.append(output)

    preview = buffer.consume(limit)

    assert preview.split("\n[Output truncated:", 1)[0] == "a" * 2047
    assert f"showing 2047 of {len(output.encode('utf-8'))} bytes" in preview
    assert "\ufffd" not in preview
    assert buffer.consume() == ""


def test_preview_limit_does_not_change_subsequent_default_or_retention(tmp_path: Path) -> None:
    retained = tmp_path / "output"
    buffer = ShellOutputBuffer(
        output_byte_limit=2048,
        retained_output_path=retained,
        retained_output_max_bytes=8192,
    )
    buffer.append("é" * 800)
    assert "[Output truncated:" in buffer.consume(1000)
    buffer.append("z" * 1500)
    assert buffer.consume() == "z" * 1500
    assert buffer.output_byte_limit == 2048
    assert retained.read_text() == "é" * 800 + "z" * 1500


@pytest.mark.parametrize("profile", ["minimal_process", "grok_shell", "luna_exec"])
def test_process_guidance_distinguishes_preview_from_retained_reads(
    profile: ShellToolProfile,
) -> None:
    process = next(tool for tool in _runtime(profile).tools if tool.name == "process")

    guidance = process.input_schema["properties"]["limit"]["description"]
    assert "wait/status" in guidance
    assert "read_output" in guidance
    assert "Does not limit execution or retained output" in guidance


@pytest.mark.asyncio
# Pairwise cover of profile x action x limit x retain_output: every value pair appears
# at least once without spawning the full 32-process cartesian product.
@pytest.mark.parametrize(
    ("profile", "action", "limit", "retain_output"),
    [
        ("minimal_process", "wait", 1, True),
        ("luna_exec", "status", 1, False),
        ("minimal_process", "status", 20, False),
        ("luna_exec", "wait", 20, True),
        ("minimal_process", "wait", 999, False),
        ("luna_exec", "status", 999, True),
        ("minimal_process", "status", 1000, True),
        ("luna_exec", "wait", 1000, False),
    ],
)
async def test_wait_status_limit_preserves_process_and_allows_retained_read(
    tmp_path: Path,
    profile: ShellToolProfile,
    action: str,
    limit: int,
    retain_output: bool,
) -> None:
    gate = tmp_path / "finish"
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("process-output-test"),
        output_byte_limit=4096,
        config=Settings(
            shell_execution=ShellSettings(
                tool_profile=profile,
                show_bash=False,
                retain_truncated_output=retain_output,
                retained_output_max_bytes=4096,
                retained_output_temp_directory=tmp_path,
            )
        ),
    )
    script = (
        f"import pathlib,time; gate=pathlib.Path({str(gate)!r}); "
        "exec('while not gate.exists(): time.sleep(0.01)'); "
        "print('é' * 1000, flush=True); raise SystemExit(7)"
    )
    try:
        started = await runtime.call_tool(
            "exec" if profile == "luna_exec" else "bash",
            {
                "command": f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}",
                "background" if profile == "luna_exec" else "run_in_background": True,
            },
        )
        assert started.is_error is False
        metadata = process_result_metadata(started)
        assert metadata is not None
        process_id = metadata["process_id"]

        status = await runtime.call_tool(
            "process", {"process_id": process_id, "action": "status", "limit": limit}
        )
        status_metadata = process_result_metadata(status)
        assert status.is_error is False
        assert status_metadata is not None
        assert status_metadata["process_status"] == "running"

        gate.touch()
        if action == "status":
            # Wait for completion without consuming the process output.
            process = await runtime._get_managed_process(process_id)
            assert process is not None
            await asyncio.wait_for(asyncio.shield(process.task), timeout=10)
        waited = await runtime.call_tool(
            "process",
            {"process_id": process_id, "action": action, "wait_sec": 10, "limit": limit},
        )
        waited_metadata = process_result_metadata(waited)
        assert waited_metadata is not None
        assert waited_metadata["exit_code"] == 7
        assert "process exit code was 7" in _text(waited)
        preview = _text(waited).split("[Output truncated:", 1)[0].rstrip("\n")
        assert preview == "é" * (limit // 2)
        assert len(preview.encode("utf-8")) <= limit
        assert "[Output truncated:" in _text(waited)
        assert "\ufffd" not in preview
        if retain_output:
            assert "action='read_output'" in _text(waited)
        else:
            assert "action='read_output'" not in _text(waited)
            assert "Increase shell_execution.output_byte_limit" in _text(waited)
        again = await runtime.call_tool("process", {"process_id": process_id, "action": "status"})
        assert "[Output truncated:" not in _text(again)
        assert runtime.output_byte_limit == 4096

        output = await runtime.call_tool(
            "process", {"process_id": process_id, "action": "read_output", "limit": 20}
        )
        if not retain_output:
            assert output.is_error is True
            assert "retained_output: unavailable" in _text(output)
            return
        assert output.is_error is False
        payload = json.loads(_text(output))
        assert payload["content"] == "é" * 10
        assert payload["next_offset"] == 20
        assert payload["has_more"] is True
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["wait", "status", "stop"])
@pytest.mark.parametrize(("field", "value"), [("offset", 0), ("query", "error")])
async def test_process_lifecycle_rejects_read_only_arguments(
    action: str, field: str, value: object
) -> None:
    result = await _runtime("minimal_process").call_tool(
        "process", {"process_id": "process-1", "action": action, field: value}
    )

    assert result.is_error is True
    assert "require action='read_output'" in _text(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["wait", "status", "read_output"])
@pytest.mark.parametrize("limit", [True, False, 0, -1, 33001])
async def test_process_rejects_invalid_limit(action: str, limit: object) -> None:
    result = await _runtime("minimal_process").call_tool(
        "process", {"process_id": "process-1", "action": action, "limit": limit}
    )
    assert result.is_error is True
    assert "'limit' argument must be" in _text(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["list", "stop"])
async def test_process_rejects_unused_limit(action: str) -> None:
    arguments: dict[str, object] = {"action": action, "limit": 1000}
    if action == "stop":
        arguments["process_id"] = "process-1"
    result = await _runtime("minimal_process").call_tool("process", arguments)
    assert result.is_error is True
    assert "must be omitted" in _text(result)


@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-5.6-luna",
        "responses.gpt-5.6-luna",
        "codexresponses/gpt-5.6-luna",
        "openai/gpt-5.6-luna?reasoning=max",
        "GPT-5.6-Luna",
    ],
)
def test_auto_profile_selects_luna_exec(model_name: str) -> None:
    params = ModelDatabase.get_model_params(model_name)

    assert params is not None
    assert resolve_shell_tool_profile("auto", params.shell_tool_profile) == "luna_exec"


@pytest.mark.parametrize(
    "profile",
    ["native", "minimal_process", "grok_shell", "luna_exec"],
)
def test_explicit_profile_overrides_luna_auto_selection(profile: ShellToolProfile) -> None:
    assert resolve_shell_tool_profile(profile, "luna_exec") == profile


def test_luna_catalog_entry_selects_luna_exec() -> None:
    luna = ModelDatabase.get_model_params("gpt-5.6-luna")

    assert luna is not None
    assert luna.shell_tool_profile == "luna_exec"


@pytest.mark.asyncio
async def test_luna_exec_rejects_raw_detachment() -> None:
    runtime = _runtime("luna_exec")

    result = await runtime.call_tool(
        "exec",
        {"command": "python -m http.server 8765 &", "background": True},
    )

    assert result.is_error is True
    assert "Shell-level backgrounding was not executed" in _text(result)
    assert "background=true" in _text(result)


@pytest.mark.asyncio
async def test_luna_exec_rejects_background_with_timeout() -> None:
    runtime = _runtime("luna_exec")

    result = await runtime.call_tool(
        "exec",
        {"command": "server", "background": True, "timeout": 30},
    )

    assert result.is_error is True
    assert "cannot be combined" in _text(result)


@pytest.mark.asyncio
async def test_luna_exec_hard_timeout_uses_environment_cancellation() -> None:
    environment = _ManagedEnvironment()
    runtime = _runtime("luna_exec", environment)

    result = await runtime.call_tool(
        "exec",
        {"command": "build", "working_directory": "project", "timeout": 1},
    )

    metadata = process_result_metadata(result)
    assert result.is_error is True
    assert metadata is not None
    assert metadata["process_status"] == "timed_out"
    assert environment.requests[0].cwd == "/workspace/project"
    assert environment.cancelled is True


@pytest.mark.asyncio
async def test_luna_background_guidance_names_exec() -> None:
    environment = _ManagedEnvironment()
    runtime = _runtime("luna_exec", environment)

    result = await runtime.call_tool(
        "exec",
        {"command": "server", "background": True},
    )

    assert result.is_error is False
    assert "separate `exec` call" in _text(result)
    assert "separate `shell` call" not in _text(result)

    await runtime.close()


@pytest.mark.asyncio
async def test_process_read_output_paginates_owned_retained_output(
    tmp_path: Path,
) -> None:
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("process-output-test"),
        timeout_seconds=10,
        output_byte_limit=24,
        config=Settings(
            shell_execution=ShellSettings(
                tool_profile="minimal_process",
                show_bash=False,
                retain_truncated_output=True,
                retained_output_max_bytes=4096,
                retained_output_temp_directory=tmp_path,
            )
        ),
    )
    command = f"{sys.executable} -c \"print('0123456789' * 20)\""
    completed = await runtime.call_tool("bash", {"command": command})
    completed_metadata = process_result_metadata(completed)
    assert completed_metadata is not None

    first = await runtime.call_tool(
        "process",
        {
            "process_id": completed_metadata["process_id"],
            "action": "read_output",
            "offset": 0,
            "limit": 20,
        },
    )
    first_payload = json.loads(_text(first))
    first_metadata = process_result_metadata(first)

    assert first.is_error is False
    assert first_payload["content"] == "01234567890123456789"
    assert first_payload["next_offset"] == 20
    assert first_payload["has_more"] is True
    assert "path" not in first_payload
    assert first_metadata is not None
    assert first_metadata["output_read_offset"] == 0
    assert first_metadata["output_read_bytes"] == 20

    second = await runtime.call_tool(
        "process",
        {
            "process_id": completed_metadata["process_id"],
            "action": "read_output",
            "offset": 20,
            "limit": 20,
        },
    )
    second_payload = json.loads(_text(second))
    assert second_payload["content"] == "01234567890123456789"
    assert second_payload["next_offset"] == 40

    await runtime.close()


@pytest.mark.asyncio
async def test_process_read_output_searches_retained_lines(
    tmp_path: Path,
) -> None:
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("process-output-test"),
        timeout_seconds=10,
        output_byte_limit=16,
        config=Settings(
            shell_execution=ShellSettings(
                tool_profile="luna_exec",
                show_bash=False,
                retain_truncated_output=True,
                retained_output_max_bytes=4096,
                retained_output_temp_directory=tmp_path,
            )
        ),
    )
    script = "print('alpha'); print('FAILED one'); print('beta'); print('FAILED two')"
    completed = await runtime.call_tool(
        "exec",
        {"command": f'{sys.executable} -c "{script}"'},
    )
    completed_metadata = process_result_metadata(completed)
    assert completed_metadata is not None

    searched = await runtime.call_tool(
        "process",
        {
            "process_id": completed_metadata["process_id"],
            "action": "read_output",
            "query": "FAILED",
            "limit": 100,
        },
    )
    payload = json.loads(_text(searched))

    assert searched.is_error is False
    assert payload["match_count"] == 2
    assert payload["content"] == "FAILED one\nFAILED two\n"
    assert "alpha" not in payload["content"]

    await runtime.close()


@pytest.mark.asyncio
async def test_process_read_output_rejects_unretained_and_unknown_processes() -> None:
    runtime = _runtime("minimal_process")
    completed = await runtime.call_tool("bash", {"command": "printf short"})
    completed_metadata = process_result_metadata(completed)
    assert completed_metadata is not None

    unavailable = await runtime.call_tool(
        "process",
        {
            "process_id": completed_metadata["process_id"],
            "action": "read_output",
        },
    )
    missing = await runtime.call_tool(
        "process",
        {"process_id": "process-999", "action": "read_output"},
    )

    assert unavailable.is_error is True
    assert "retained_output: unavailable" in _text(unavailable)
    assert missing.is_error is True
    assert "was not found" in _text(missing)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            {"process_id": "process-1", "action": "status", "offset": 1},
            "require action='read_output'",
        ),
        (
            {"process_id": "process-1", "action": "read_output", "wait_sec": 10},
            "'wait_sec' must be omitted",
        ),
        (
            {"process_id": "process-1", "action": "read_output", "offset": -1},
            "'offset' argument must be a non-negative integer",
        ),
        (
            {"process_id": "process-1", "action": "read_output", "limit": 0},
            "'limit' argument must be a positive integer",
        ),
        (
            {"process_id": "process-1", "action": "read_output", "query": ""},
            "'query' argument is required",
        ),
    ],
)
async def test_process_read_output_validates_action_specific_arguments(
    arguments: dict[str, object],
    expected: str,
) -> None:
    runtime = _runtime("minimal_process")

    result = await runtime.call_tool("process", arguments)

    assert result.is_error is True
    assert expected in _text(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["minimal_process", "luna_exec"])
async def test_process_wait_sec_without_action_waits(
    tmp_path: Path, profile: ShellToolProfile
) -> None:
    gate = tmp_path / "finish"
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("process-wait-default-test"),
        config=Settings(shell_execution=ShellSettings(tool_profile=profile, show_bash=False)),
    )
    script = (
        f"import pathlib,time; gate=pathlib.Path({str(gate)!r}); "
        "exec('while not gate.exists(): time.sleep(0.01)'); print('finished')"
    )
    try:
        started = await runtime.call_tool(
            "exec" if profile == "luna_exec" else "bash",
            {
                "command": f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}",
                "background" if profile == "luna_exec" else "run_in_background": True,
            },
        )
        metadata = process_result_metadata(started)
        assert metadata is not None
        process_id = metadata["process_id"]

        asyncio.get_running_loop().call_later(0.3, gate.touch)
        waited = await runtime.call_tool("process", {"process_id": process_id, "wait_sec": 10})

        waited_metadata = process_result_metadata(waited)
        assert waited_metadata is not None
        assert waited_metadata["exit_code"] == 0
        assert "finished" in _text(waited)
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("lifecycle", ["session", "persistent"])
@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize(
    ("running_limit", "preview_limit", "output_limit"),
    [
        (6000, None, 16_000),
        (None, None, 16_000),
        (6000, 8000, 16_000),
        (6000, 1000, 16_000),
        (6000, None, 4000),
        (None, None, 4000),
    ],
)
async def test_running_output_policy_and_completion_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lifecycle: str,
    exit_code: int,
    running_limit: int | None,
    preview_limit: int | None,
    output_limit: int,
) -> None:
    if lifecycle == "persistent" and os.name != "posix":
        pytest.skip("durable local processes require POSIX")
    monkeypatch.setattr(shell_runtime_module, "BACKGROUND_LAUNCH_SETTLE_SECONDS", 0)
    start_gate = tmp_path / "start"
    finish_gate = tmp_path / "finish"
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("running-output-test"),
        output_byte_limit=output_limit,
        durable_process_root=tmp_path / "processes" if lifecycle == "persistent" else None,
        config=Settings(
            shell_execution=ShellSettings(
                tool_profile="minimal_process",
                show_bash=False,
                running_output_byte_limit=running_limit,
                retained_output_temp_directory=tmp_path,
            )
        ),
    )
    # Gate output until launch acknowledgement so neither lifecycle consumes it early.
    # 50-byte lines: 10,000 bytes while running, then 20,000 bytes before exiting.
    script = (
        "import pathlib,sys,time\n"
        f"while not pathlib.Path({str(start_gate)!r}).exists(): time.sleep(0.01)\n"
        "for i in range(200): print(f'line {i:04d} ' + 'x' * 39)\n"
        "sys.stdout.flush()\n"
        f"while not pathlib.Path({str(finish_gate)!r}).exists(): time.sleep(0.01)\n"
        "for i in range(200, 600): print(f'line {i:04d} ' + 'x' * 39)\n"
        f"sys.exit({exit_code})"
    )
    process_id: str | None = None
    try:
        started = await runtime.execute(
            {
                "command": f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}",
                "background": True,
                "lifecycle": lifecycle,
            },
        )
        metadata = process_result_metadata(started)
        assert metadata is not None
        process_id = metadata["process_id"]
        assert metadata["lifecycle"] == lifecycle
        start_gate.touch()
        store = runtime._durable_process_store
        process = await runtime._get_managed_process(process_id)
        async with asyncio.timeout(10):
            if lifecycle == "persistent":
                assert store is not None
                while (await asyncio.to_thread(store.get, process_id)).output_bytes < 10_000:
                    await asyncio.sleep(0.01)
            else:
                assert process is not None
                while process.output_state.total_output_bytes < 10_000:
                    await asyncio.sleep(0.01)

        running_result = await runtime.call_tool(
            "process",
            {"process_id": process_id, "action": "status", "limit": preview_limit},
        )
        running_metadata = process_result_metadata(running_result)
        assert running_metadata is not None
        assert running_metadata["process_status"] == "running"
        running = _text(running_result)
        first_output = "".join(f"line {i:04d} " + "x" * 39 + "\n" for i in range(200))
        if preview_limit is not None:
            assert first_output[:preview_limit].rstrip("\n") in running
            assert f"showing {preview_limit} of 10000 bytes" in running
            assert "mostly its latest output" not in running
            assert "line 0199" not in running
        elif running_limit is None and output_limit >= 10_000:
            assert first_output.rstrip("\n") in running
            assert "[Output truncated:" not in running
        else:
            budget = min(running_limit, output_limit) if running_limit else output_limit
            head_bytes = budget // (8 if running_limit else 2)
            tail_bytes = budget - head_bytes
            assert f"showing first {head_bytes} bytes and last {tail_bytes} bytes" in running
            assert first_output[:head_bytes].rstrip("\n") in running
            assert first_output[-tail_bytes:].rstrip("\n") in running
            assert ("mostly its latest output" in running) == (running_limit is not None)
            assert "line 0050" not in running

        # Status consumes the delta, but read_output must not lose the omitted middle.
        assert "line 0000" not in _text(
            await runtime.call_tool("process", {"process_id": process_id, "action": "status"})
        )
        finish_gate.touch()
        if lifecycle == "persistent":
            assert store is not None
            await asyncio.to_thread(store.wait, process_id, timeout_seconds=10)
        else:
            assert process is not None
            await asyncio.wait_for(asyncio.shield(process.task), timeout=10)
        done_result = await runtime.call_tool(
            "process", {"process_id": process_id, "action": "status"}
        )
        done_metadata = process_result_metadata(done_result)
        assert done_metadata is not None
        assert done_metadata["exit_code"] == exit_code
        assert done_metadata["process_status"] == ("failed" if exit_code else "completed")
        assert done_result.is_error == (exit_code != 0)
        done = _text(done_result)
        assert "still running, so mostly" not in done
        assert (
            f"showing first {output_limit // 2} bytes and last {output_limit // 2} bytes"
        ) in done
        assert "line 0200" in done and "line 0599" in done
        assert "line 0000" not in done

        readback = await runtime.call_tool(
            "process", {"process_id": process_id, "action": "read_output", "limit": 33_000}
        )
        retained = json.loads(_text(readback))["content"]
        final_output = "".join(f"line {i:04d} " + "x" * 39 + "\n" for i in range(200, 600))
        if lifecycle == "session" and running_limit is None and output_limit >= 10_000:
            # Session retention starts only when a preview is actually shortened.
            assert retained == final_output
        else:
            assert retained == first_output + final_output
    finally:
        start_gate.touch()
        finish_gate.touch()
        if process_id is not None:
            await runtime.terminate_process({"process_id": process_id})
        await runtime.close()


@pytest.mark.parametrize("output_limit", [1, 4000, 16_000])
def test_running_output_without_retention_explains_running_cap(output_limit: int) -> None:
    output = ShellOutputBuffer(output_byte_limit=output_limit)
    output.append("x" * 10_000)

    preview = output.consume(running_byte_limit=6000)

    budget = min(output_limit, 6000)
    head_bytes = max(budget // 8, 1)
    assert f"showing first {head_bytes} bytes and last {budget - head_bytes} bytes" in preview
    assert "shell_execution.running_output_byte_limit" in preview
    assert "set it to null" in preview
    assert "shell_execution.output_byte_limit still applies" in preview
    assert "Increase shell_execution.output_byte_limit to retain more." not in preview
    assert "read_output" not in preview
    assert output.consume() == ""


def test_shell_running_output_default_comes_from_settings() -> None:
    settings = ShellSettings()
    assert settings.running_output_byte_limit == 6000
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("running-output-test"),
    )
    assert runtime._running_output_byte_limit == settings.running_output_byte_limit
