from __future__ import annotations

from typing import Any

import pytest

from fast_agent.tools.execution_environment import ShellExecutionRequest
from fast_agent.tools.huggingface_sandbox_environment import (
    HuggingFaceSandboxEnvironment,
    _SandboxCommandResult,
    _SandboxFiles,
)
from fast_agent.tools.huggingface_sandbox_environment import (
    _Sandbox as SandboxProtocol,
)


class _CommandResult(_SandboxCommandResult):
    def __init__(
        self,
        *,
        stdout: str = "ok",
        stderr: str = "",
        exit_code: int | None = 0,
        timed_out: bool = False,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.timed_out = timed_out


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
        self.commands: list[str | list[str]] = []

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
        del shell, env, timeout, check
        self.commands.append(cmd)
        self.cwd = cwd
        if isinstance(cmd, list) and cmd[:2] == ["python3", "-c"]:
            return _CommandResult(
                stdout=(
                    "["
                    '{"path": "/workspace/skills/alpha", "name": "alpha", "kind": "directory"},'
                    '{"path": "/workspace/skills/readme.txt", "name": "readme.txt", "kind": "file"}'
                    "]"
                )
            )
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


@pytest.mark.asyncio
async def test_list_dir_returns_session_file_entries() -> None:
    sandbox = _Sandbox()
    environment = HuggingFaceSandboxEnvironment(sandbox=sandbox, cwd="/workspace")
    await environment.open()

    entries = await environment.list_dir("skills")

    assert [entry.name for entry in entries] == ["alpha", "readme.txt"]
    assert [entry.path for entry in entries] == [
        "/workspace/skills/alpha",
        "/workspace/skills/readme.txt",
    ]
    assert [entry.kind for entry in entries] == ["directory", "file"]
    assert isinstance(sandbox.commands[-1], list)
    assert sandbox.commands[-1][0] == "python3"
    assert sandbox.commands[-1][-1] == "/workspace/skills"
