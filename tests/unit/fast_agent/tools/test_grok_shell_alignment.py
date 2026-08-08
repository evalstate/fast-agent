from __future__ import annotations

import asyncio
import logging

import pytest
from pydantic import ValidationError

from fast_agent.config import Settings, ShellSettings, ShellToolProfile
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.tools.execution_environment import (
    ShellExecution,
    ShellExecutionCallbacks,
    ShellExecutionOptions,
    ShellExecutionRequest,
    ShellExecutionResult,
    ShellRuntimeInfo,
)
from fast_agent.tools.shell_process import process_result_metadata
from fast_agent.tools.shell_profiles import resolve_shell_tool_profile
from fast_agent.tools.shell_runtime import ShellRuntime


class _ManagedEnvironment:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
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
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = request.terminate_on_cancel
            raise
        return ShellExecution(
            result=ShellExecutionResult(stdout="done\n", stderr="", exit_code=0),
            options=ShellExecutionOptions(),
        )

    async def close(self) -> None:
        return None


def _runtime(
    profile: ShellToolProfile,
    environment: _ManagedEnvironment | None = None,
) -> ShellRuntime:
    return ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger("grok-tool-alignment-test"),
        shell_environment=environment,
        process_poll_default_wait_seconds=240,
        foreground_yield_seconds=0.001,
        config=Settings(shell_execution=ShellSettings(tool_profile=profile)),
    )


def test_grok_shell_profile_exposes_aligned_shell_and_unified_process() -> None:
    runtime = _runtime("grok_shell")

    assert [tool.name for tool in runtime.tools] == ["shell", "process"]
    shell = runtime.tools[0]
    assert set(shell.input_schema["properties"]) == {
        "command",
        "working_directory",
        "background",
        "timeout",
    }
    assert shell.input_schema["properties"]["timeout"]["maximum"] == 600


@pytest.mark.parametrize(
    "model_name",
    [
        "grok-4.5",
        "xai/grok-4.5",
        "xai.grok-4.3",
        "openrouter/x-ai/grok-4.5",
        "Grok 4.5",
    ],
)
def test_auto_profile_selects_aligned_shell_for_grok(model_name: str) -> None:
    params = ModelDatabase.get_model_params(model_name)

    assert ShellSettings().tool_profile == "auto"
    assert params is not None
    assert resolve_shell_tool_profile("auto", params.shell_tool_profile) == "grok_shell"


@pytest.mark.parametrize("model_name", [None, "gpt-5.6-sol", "claude-opus-4-6", "not-grok-4.5"])
def test_auto_profile_keeps_minimal_process_for_non_grok(model_name: str | None) -> None:
    params = ModelDatabase.get_model_params(model_name) if model_name is not None else None

    assert (
        resolve_shell_tool_profile(
            "auto",
            params.shell_tool_profile if params is not None else None,
        )
        == "minimal_process"
    )


def test_grok_catalog_entries_select_aligned_shell() -> None:
    grok_43 = ModelDatabase.get_model_params("grok-4.3")
    grok_45 = ModelDatabase.get_model_params("grok-4.5")

    assert grok_43 is not None
    assert grok_45 is not None
    assert grok_43.shell_tool_profile == "grok_shell"
    assert grok_45.shell_tool_profile == "grok_shell"


@pytest.mark.parametrize("profile", ["native", "minimal_process", "grok_shell"])
def test_explicit_profile_overrides_grok_auto_selection(profile: ShellToolProfile) -> None:
    assert resolve_shell_tool_profile(profile, "grok_shell") == profile


def test_dedicated_grok_process_profile_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ShellSettings(
            tool_profile="grok_process"  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.asyncio
async def test_grok_shell_explicit_timeout_suppresses_auto_yield() -> None:
    environment = _ManagedEnvironment()
    runtime = _runtime("grok_shell", environment)

    task = asyncio.create_task(
        runtime.call_tool(
            "shell",
            {"command": "build", "working_directory": "project", "timeout": 1},
        )
    )
    await environment.started.wait()
    await asyncio.sleep(0.01)
    environment.release.set()
    result = await task

    metadata = process_result_metadata(result)
    assert metadata is not None
    assert metadata["process_status"] == "completed"
    assert environment.requests[0].cwd == "/workspace/project"


@pytest.mark.asyncio
async def test_grok_shell_hard_timeout_uses_cross_platform_cancellation_contract() -> None:
    environment = _ManagedEnvironment()
    runtime = _runtime("grok_shell", environment)

    result = await runtime.call_tool(
        "shell",
        {"command": "build", "timeout": 1},
    )

    metadata = process_result_metadata(result)
    assert result.is_error is True
    assert metadata is not None
    assert metadata["process_status"] == "timed_out"
    assert environment.cancelled is True


@pytest.mark.asyncio
async def test_grok_shell_rejects_background_with_timeout() -> None:
    runtime = _runtime("grok_shell")

    result = await runtime.call_tool(
        "shell",
        {"command": "server", "background": True, "timeout": 30},
    )

    assert result.is_error is True
