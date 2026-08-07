from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import TYPE_CHECKING

import pytest
from mcp_types import CallToolResult, TextContent

if TYPE_CHECKING:
    from pathlib import Path

from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import Settings, ShellSettings, ShellToolProfile
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.tools.execution_environment import (
    ShellExecution,
    ShellExecutionCallbacks,
    ShellExecutionRequest,
    ShellRuntimeInfo,
)
from fast_agent.tools.shell_process import process_result_metadata
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
    assert (
        "verifier-persistent server or service"
        in (execute.input_schema["properties"]["background"]["description"])
    )
    assert "training" in (execute.description or "")

    process = runtime.tools[1]
    properties = process.input_schema["properties"]
    assert "read_output" in properties["action"]["enum"]
    assert set(properties) >= {"process_id", "action", "offset", "limit", "query"}
    assert "path" not in properties


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
    assert McpAgent._resolve_shell_tool_profile("auto", model_name) == "luna_exec"


@pytest.mark.parametrize(
    "profile",
    ["native", "minimal_process", "grok_shell", "luna_exec"],
)
def test_explicit_profile_overrides_luna_auto_selection(profile: ShellToolProfile) -> None:
    assert McpAgent._resolve_shell_tool_profile(profile, "gpt-5.6-luna") == profile


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
