from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mcp.types import TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.context import Context
from fast_agent.skills.registry import SkillRegistry
from fast_agent.tools.session_environment import (
    SessionFileEntry,
    ShellExecution,
    ShellExecutionCallbacks,
    ShellExecutionOptions,
    ShellExecutionRequest,
    ShellExecutionResult,
    ShellRuntimeInfo,
)
from fast_agent.tools.session_filesystem_runtime import SessionFilesystemRuntime
from fast_agent.tools.skill_reader import READ_SKILL_TOOL_NAME

if TYPE_CHECKING:
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

    async def read_text(self, path: str) -> str:
        return self.files[self.resolve_path(path)]

    async def write_text(self, path: str, content: str) -> None:
        self.files[self.resolve_path(path)] = content

    async def exists(self, path: str) -> bool:
        return self.resolve_path(path) in self.files

    async def list_dir(self, path: str) -> list[SessionFileEntry]:
        resolved = self.resolve_path(path).rstrip("/")
        entries: list[SessionFileEntry] = []
        seen_directories: set[str] = set()
        for file_path in sorted(self.files):
            if not file_path.startswith(f"{resolved}/"):
                continue
            relative = file_path[len(resolved) + 1 :]
            name = relative.split("/", 1)[0]
            entry_path = f"{resolved}/{name}"
            if "/" in relative:
                if name in seen_directories:
                    continue
                seen_directories.add(name)
                entries.append(SessionFileEntry(path=entry_path, name=name, kind="directory"))
            else:
                entries.append(SessionFileEntry(path=entry_path, name=name, kind="file"))
        return entries

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
async def test_session_filesystem_runtime_preserves_full_file_content() -> None:
    env = FakeSessionEnvironment()
    env.files["/workspace/notes.txt"] = "hello\r\nworld\r\n"
    runtime = SessionFilesystemRuntime(env, enable_read=True, enable_write=True)

    read = await runtime.call_tool("read_text_file", {"path": "notes.txt"})

    assert read.isError is False
    assert _text(read) == "hello\r\nworld\r\n"


@pytest.mark.asyncio
async def test_fake_session_environment_lists_direct_children() -> None:
    env = FakeSessionEnvironment()
    env.files["/workspace/skills/alpha/SKILL.md"] = "alpha"
    env.files["/workspace/skills/beta/SKILL.md"] = "beta"
    env.files["/workspace/skills/readme.txt"] = "notes"

    entries = await env.list_dir("skills")

    assert entries == [
        SessionFileEntry(path="/workspace/skills/alpha", name="alpha", kind="directory"),
        SessionFileEntry(path="/workspace/skills/beta", name="beta", kind="directory"),
        SessionFileEntry(path="/workspace/skills/readme.txt", name="readme.txt", kind="file"),
    ]


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
    assert _text(read) == "remote file\n"
    assert shell.isError is False
    assert env.requests[-1].command == "pwd"

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_mcp_agent_keeps_host_skill_reader_with_injected_session_filesystem(
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "alpha"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Alpha skill\n---\nUse alpha.\n",
        encoding="utf-8",
    )
    manifests = SkillRegistry.load_directory(skills_root)
    env = FakeSessionEnvironment()
    config = AgentConfig(
        name="test",
        instruction="Skills:\n{{agentSkills}}",
        servers=[],
        shell=True,
        skills=skills_root,
    )
    config.skill_manifests = manifests
    agent = McpAgent(config=config, context=Context(), shell_environment=env)

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}

    assert "read_text_file" in tool_names
    assert READ_SKILL_TOOL_NAME in tool_names
    assert agent.skill_read_tool_name == READ_SKILL_TOOL_NAME

    await agent._aggregator.close()
