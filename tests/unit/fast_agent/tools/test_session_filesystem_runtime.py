from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mcp.types import TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.context import Context
from fast_agent.tools.session_environment import (
    ShellExecution,
    ShellExecutionCallbacks,
    ShellExecutionOptions,
    ShellExecutionRequest,
    ShellExecutionResult,
    ShellRuntimeInfo,
)
from fast_agent.tools.session_filesystem_runtime import SessionFilesystemRuntime

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class FakeSessionEnvironment:
    def __init__(self, cwd: str = "/workspace") -> None:
        self._cwd = cwd
        self.files: dict[str, str] = {}
        self.removed: list[str] = []
        self.requests: list[ShellExecutionRequest] = []

    async def open(self) -> None:
        return None

    @property
    def cwd(self) -> str:
        return self._cwd

    def set_cwd(self, cwd: str | None) -> None:
        if cwd is not None:
            self._cwd = cwd

    def resolve_path(self, path: str) -> str:
        if path.startswith("/"):
            return path
        return f"{self._cwd.rstrip('/')}/{path}"

    def runtime_info(self) -> ShellRuntimeInfo:
        return ShellRuntimeInfo(name="bash", kind="remote", provider="fake")

    async def execute(
        self,
        request: ShellExecutionRequest,
        *,
        callbacks: ShellExecutionCallbacks | None = None,
    ) -> ShellExecution:
        del callbacks
        self.requests.append(request)
        return ShellExecution(
            result=ShellExecutionResult(stdout="remote", stderr="", exit_code=0),
            options=ShellExecutionOptions(),
        )

    async def execute_shell(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ShellExecutionResult:
        del env, timeout
        self.requests.append(ShellExecutionRequest(command=command, cwd=str(cwd) if cwd else None))
        return ShellExecutionResult(stdout="remote", stderr="", exit_code=0)

    async def read_text(self, path: str) -> str:
        return self.files[self.resolve_path(path)]

    async def write_text(self, path: str, content: str) -> None:
        self.files[self.resolve_path(path)] = content

    async def exists(self, path: str) -> bool:
        return self.resolve_path(path) in self.files

    async def mkdir(self, path: str) -> None:
        del path

    async def remove(self, path: str) -> None:
        resolved = self.resolve_path(path)
        self.removed.append(resolved)
        self.files.pop(resolved, None)

    async def close(self) -> None:
        return None


def _text(result) -> str:
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    return result.content[0].text


@pytest.mark.asyncio
async def test_session_filesystem_runtime_reads_and_writes_remote_files() -> None:
    env = FakeSessionEnvironment()
    runtime = SessionFilesystemRuntime(env, enable_read=True, enable_write=True)

    write = await runtime.call_tool(
        "write_text_file",
        {"path": "notes.txt", "content": "hello\nworld\n"},
    )
    read = await runtime.call_tool("read_text_file", {"path": "notes.txt", "line": 2})

    assert write.isError is False
    assert read.isError is False
    assert env.files["/workspace/notes.txt"] == "hello\nworld\n"
    assert _text(read) == "world"


@pytest.mark.asyncio
async def test_session_filesystem_runtime_applies_patch_to_remote_files() -> None:
    env = FakeSessionEnvironment()
    env.files["/workspace/notes.txt"] = "one\ntwo\n"
    runtime = SessionFilesystemRuntime(env, enable_read=True, enable_apply_patch=True)

    result = await runtime.call_tool(
        "apply_patch",
        {
            "input": (
                "*** Begin Patch\n"
                "*** Update File: notes.txt\n"
                "@@\n"
                "-one\n"
                "+ONE\n"
                " two\n"
                "*** End Patch\n"
            )
        },
    )

    assert result.isError is False
    assert env.files["/workspace/notes.txt"] == "ONE\ntwo\n"
    assert "M notes.txt" in _text(result)


@pytest.mark.asyncio
async def test_mcp_agent_routes_file_tools_to_injected_session_environment() -> None:
    env = FakeSessionEnvironment()
    env.files["/workspace/remote.txt"] = "remote file\n"
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model="gpt-5.4",
    )
    agent = McpAgent(config=config, context=Context(), shell_environment=env)

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "execute" in tool_names
    assert "read_text_file" in tool_names
    assert "apply_patch" in tool_names

    read = await agent.call_tool("read_text_file", {"path": "remote.txt"})
    shell = await agent.call_tool("execute", {"command": "pwd"})

    assert read.isError is False
    assert _text(read) == "remote file"
    assert shell.isError is False
    assert env.requests[-1].command == "pwd"

    await agent._aggregator.close()
