from __future__ import annotations

from typing import Any

import pytest

from fast_agent.tools.huggingface_sandbox_environment import (
    HuggingFaceSandboxEnvironment,
    _SandboxCommandResult,
    _SandboxFiles,
)
from fast_agent.tools.huggingface_sandbox_environment import (
    _Sandbox as SandboxProtocol,
)
from fast_agent.tools.session_environment import ShellExecutionRequest


class _CommandResult(_SandboxCommandResult):
    stdout = "ok"
    stderr = ""
    exit_code = 0
    timed_out = False


class _Files(_SandboxFiles):
    def __init__(self) -> None:
        self.created: list[str] = []

    def mkdir(self, path: str) -> None:
        self.created.append(path)

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        return ""

    def write(self, path: str, data: str | bytes, mode: str | None = None) -> None:
        pass

    def exists(self, path: str) -> bool:
        return True

    def delete(self, path: str, recursive: bool = False) -> None:
        pass


class _Sandbox(SandboxProtocol):
    def __init__(self) -> None:
        self.test_files = _Files()
        self.files: _Files = self.test_files
        self.cwd: str | None = None

    def run(
        self,
        cmd: str | list[str],
        *,
        shell: bool | None = None,
        env: dict[str, Any] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> _CommandResult:
        self.cwd = cwd
        return _CommandResult()

    def kill(self) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_open_creates_configured_cwd() -> None:
    sandbox = _Sandbox()
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace/project")

    await environment.open()

    assert sandbox.test_files.created == ["/workspace/project"]


@pytest.mark.asyncio
async def test_execute_uses_created_default_cwd() -> None:
    sandbox = _Sandbox()
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace")
    await environment.open()

    await environment.execute(ShellExecutionRequest(command="pwd"))

    assert sandbox.cwd == "/workspace"
