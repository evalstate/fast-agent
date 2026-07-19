from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path

import pytest
from mcp.types import ImageContent, TextContent
from PIL import Image

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import Settings, ShellSettings
from fast_agent.context import Context
from fast_agent.skills.registry import SkillRegistry
from fast_agent.tools.environment_filesystem_runtime import EnvironmentFilesystemRuntime
from fast_agent.tools.execution_environment import (
    EnvironmentFileEntry,
    ShellExecution,
    ShellExecutionCallbacks,
    ShellExecutionOptions,
    ShellExecutionRequest,
    ShellExecutionResult,
    ShellRuntimeInfo,
)
from fast_agent.tools.local_shell_executor import LocalEnvironment
from fast_agent.tools.skill_reader import READ_SKILL_TOOL_NAME

_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _image_bytes(image_format: str) -> bytes:
    output = BytesIO()
    Image.new("RGB", (1, 1), color="blue").save(output, format=image_format)
    return output.getvalue()


class FakeEnvironment:
    def __init__(self, cwd: str = "/workspace") -> None:
        self._cwd = cwd
        self.files: dict[str, str] = {}
        self.binary_files: dict[str, bytes] = {}
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

    async def read_bytes(self, path: str) -> bytes:
        resolved = self.resolve_path(path)
        binary_content = self.binary_files.get(resolved)
        if binary_content is not None:
            return binary_content
        return self.files[resolved].encode("utf-8")

    async def write_bytes(self, path: str, content: bytes) -> None:
        self.binary_files[self.resolve_path(path)] = content

    async def exists(self, path: str) -> bool:
        resolved = self.resolve_path(path)
        return resolved in self.files or resolved in self.binary_files

    async def list_dir(self, path: str) -> list[EnvironmentFileEntry]:
        resolved = self.resolve_path(path).rstrip("/")
        entries: list[EnvironmentFileEntry] = []
        seen_directories: set[str] = set()
        for file_path in sorted({*self.files, *self.binary_files}):
            if not file_path.startswith(f"{resolved}/"):
                continue
            relative = file_path[len(resolved) + 1 :]
            name = relative.split("/", 1)[0]
            entry_path = f"{resolved}/{name}"
            if "/" in relative:
                if name in seen_directories:
                    continue
                seen_directories.add(name)
                entries.append(EnvironmentFileEntry(path=entry_path, name=name, kind="directory"))
            else:
                entries.append(EnvironmentFileEntry(path=entry_path, name=name, kind="file"))
        return entries

    async def mkdir(self, path: str) -> None:
        del path

    async def remove(self, path: str) -> None:
        resolved = self.resolve_path(path)
        self.removed.append(resolved)
        self.files.pop(resolved, None)

    async def close(self) -> None:
        return None


class ShellOnlyEnvironment:
    def __init__(self, cwd: str = "/workspace") -> None:
        self._cwd = cwd
        self.requests: list[ShellExecutionRequest] = []

    async def open(self) -> None:
        return None

    @property
    def cwd(self) -> str:
        return self._cwd

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

    async def close(self) -> None:
        return None


def _text(result) -> str:
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    return result.content[0].text


@pytest.mark.asyncio
async def test_environment_filesystem_runtime_reads_and_writes_remote_files() -> None:
    env = FakeEnvironment()
    runtime = EnvironmentFilesystemRuntime(env, enable_read=True, enable_write=True)

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
async def test_environment_filesystem_runtime_preserves_full_file_content() -> None:
    env = FakeEnvironment()
    env.files["/workspace/notes.txt"] = "hello\r\nworld\r\n"
    runtime = EnvironmentFilesystemRuntime(env, enable_read=True, enable_write=True)

    read = await runtime.call_tool("read_text_file", {"path": "notes.txt"})

    assert read.isError is False
    assert _text(read) == "hello\r\nworld\r\n"


@pytest.mark.asyncio
async def test_environment_filesystem_runtime_attaches_environment_media() -> None:
    env = FakeEnvironment()
    env.binary_files["/workspace/image.png"] = _PNG_BYTES
    runtime = EnvironmentFilesystemRuntime(env, enable_attach_media="on")

    tool_names = {tool.name for tool in runtime.tools}
    result = await runtime.call_tool(
        "attach_media",
        {"source": "image.png", "mime_type": "image/png"},
    )
    pending = runtime.consume_pending_media_attachments()

    assert "attach_media" in tool_names
    assert result.isError is False
    assert "Staged image.png as embedded image/png media input" in _text(result)
    assert len(pending) == 1
    assert isinstance(pending[0], ImageContent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("data", "error"),
    [
        (
            b"not an image",
            "does not contain valid 'image/png' data",
        ),
        (
            b"\x89PNG\r\n\x1a\ntruncated",
            "does not contain valid 'image/png' data",
        ),
    ],
)
async def test_environment_filesystem_runtime_rejects_invalid_image_data(
    data: bytes,
    error: str,
) -> None:
    env = FakeEnvironment()
    env.binary_files["/workspace/image.png"] = data
    runtime = EnvironmentFilesystemRuntime(env, enable_attach_media="on")

    result = await runtime.call_tool(
        "attach_media",
        {"source": "image.png", "mime_type": "image/png"},
    )

    assert result.isError is True
    assert error in _text(result)
    assert runtime.consume_pending_media_attachments() == []


@pytest.mark.asyncio
async def test_environment_filesystem_runtime_converts_ppm_to_png() -> None:
    env = FakeEnvironment()
    env.binary_files["/workspace/screen.ppm"] = b"P6\n1 1\n255\n\x00\x00\x00"
    runtime = EnvironmentFilesystemRuntime(env, enable_attach_media="on")

    result = await runtime.call_tool(
        "attach_media",
        {"source": "screen.ppm", "mime_type": "image/png"},
    )
    pending = runtime.consume_pending_media_attachments()

    assert result.isError is False
    assert "Converted screen.ppm from image/x-portable-anymap to image/png" in _text(result)
    assert len(pending) == 1
    assert isinstance(pending[0], ImageContent)
    assert pending[0].mimeType == "image/png"
    assert base64.b64decode(pending[0].data).startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.asyncio
async def test_environment_filesystem_runtime_detects_pillow_image_without_known_mime() -> None:
    env = FakeEnvironment()
    env.binary_files["/workspace/screen.tga"] = _image_bytes("TGA")
    runtime = EnvironmentFilesystemRuntime(env, enable_attach_media="on")

    result = await runtime.call_tool("attach_media", {"source": "screen.tga"})
    pending = runtime.consume_pending_media_attachments()

    assert result.isError is False
    assert "Converted screen.tga from image/x-tga to image/png" in _text(result)
    assert len(pending) == 1
    assert isinstance(pending[0], ImageContent)
    assert pending[0].mimeType == "image/png"
    assert base64.b64decode(pending[0].data).startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.asyncio
async def test_environment_filesystem_runtime_hides_attach_media_without_byte_reads() -> None:
    class TextOnlyEnvironment:
        def __init__(self) -> None:
            self._inner = FakeEnvironment()

        @property
        def cwd(self) -> str:
            return self._inner.cwd

        def resolve_path(self, path: str) -> str:
            return self._inner.resolve_path(path)

        async def read_text(self, path: str) -> str:
            return await self._inner.read_text(path)

        async def write_text(self, path: str, content: str) -> None:
            await self._inner.write_text(path, content)

        async def exists(self, path: str) -> bool:
            return await self._inner.exists(path)

        async def list_dir(self, path: str) -> list[EnvironmentFileEntry]:
            return await self._inner.list_dir(path)

        async def mkdir(self, path: str) -> None:
            await self._inner.mkdir(path)

        async def remove(self, path: str) -> None:
            await self._inner.remove(path)

    runtime = EnvironmentFilesystemRuntime(TextOnlyEnvironment(), enable_attach_media="on")

    assert "attach_media" not in {tool.name for tool in runtime.tools}
    result = await runtime.call_tool("attach_media", {"source": "image.png"})
    assert result.isError is True


@pytest.mark.asyncio
async def test_fake_environment_lists_direct_children() -> None:
    env = FakeEnvironment()
    env.files["/workspace/skills/alpha/SKILL.md"] = "alpha"
    env.files["/workspace/skills/beta/SKILL.md"] = "beta"
    env.files["/workspace/skills/readme.txt"] = "notes"

    entries = await env.list_dir("skills")

    assert entries == [
        EnvironmentFileEntry(path="/workspace/skills/alpha", name="alpha", kind="directory"),
        EnvironmentFileEntry(path="/workspace/skills/beta", name="beta", kind="directory"),
        EnvironmentFileEntry(path="/workspace/skills/readme.txt", name="readme.txt", kind="file"),
    ]


@pytest.mark.asyncio
async def test_environment_filesystem_runtime_applies_patch_to_remote_files() -> None:
    env = FakeEnvironment()
    env.files["/workspace/notes.txt"] = "one\ntwo\n"
    runtime = EnvironmentFilesystemRuntime(env, enable_read=True, enable_apply_patch=True)

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
async def test_environment_filesystem_runtime_move_removes_source_file() -> None:
    env = FakeEnvironment()
    env.files["/workspace/a.py"] = "print('hi')\n"
    runtime = EnvironmentFilesystemRuntime(env, enable_read=True, enable_apply_patch=True)

    result = await runtime.call_tool(
        "apply_patch",
        {
            "input": (
                "*** Begin Patch\n"
                "*** Update File: a.py\n"
                "*** Move to: b.py\n"
                "@@\n"
                "-print('hi')\n"
                "+print('hello')\n"
                "*** End Patch\n"
            )
        },
    )

    assert result.isError is False
    assert env.files["/workspace/b.py"] == "print('hello')\n"
    assert "/workspace/a.py" not in env.files


@pytest.mark.asyncio
async def test_environment_filesystem_runtime_reports_edit_write_failure() -> None:
    class FailingWriteEnvironment(FakeEnvironment):
        async def write_text(self, path: str, content: str) -> None:
            del path, content
            raise OSError("disk full")

    env = FailingWriteEnvironment()
    env.files["/workspace/notes.txt"] = "hello world\n"
    runtime = EnvironmentFilesystemRuntime(env, enable_edit_file=True)

    result = await runtime.call_tool(
        "edit_file",
        {"path": "notes.txt", "old_string": "world", "new_string": "there"},
    )

    assert result.isError is True
    assert _text(result) == "Error writing file: disk full"


@pytest.mark.asyncio
async def test_mcp_agent_routes_file_tools_to_injected_execution_environment() -> None:
    env = FakeEnvironment()
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
async def test_mcp_agent_does_not_expose_host_file_tools_for_shell_only_environment() -> None:
    env = ShellOnlyEnvironment()
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
    assert "read_text_file" not in tool_names
    assert "write_text_file" not in tool_names
    assert "apply_patch" not in tool_names
    assert "edit_file" not in tool_names
    assert "attach_media" not in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_mcp_agent_stages_media_from_injected_execution_environment() -> None:
    env = FakeEnvironment()
    env.binary_files["/workspace/image.png"] = _PNG_BYTES
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model="gpt-5.4",
    )
    context = Context(config=Settings(shell_execution=ShellSettings(enable_attach_media="on")))
    agent = McpAgent(config=config, context=context, shell_environment=env)

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    result = await agent.call_tool(
        "attach_media",
        {"source": "image.png", "mime_type": "image/png"},
    )
    pending = agent._consume_pending_media_attachments()

    assert "attach_media" in tool_names
    assert result.isError is False
    assert len(pending) == 1
    assert isinstance(pending[0], ImageContent)

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_mcp_agent_uses_local_environment_filesystem_cwd(
    tmp_path: Path,
) -> None:
    (tmp_path / "local.txt").write_text("from local environment\n", encoding="utf-8")
    env = LocalEnvironment(logger=logging.getLogger("test-local-env"), working_directory=tmp_path)
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model="gpt-5.4",
    )
    context = Context(config=Settings(shell_execution=ShellSettings(enable_attach_media="on")))
    agent = McpAgent(config=config, context=context, shell_environment=env)

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    result = await agent.call_tool("read_text_file", {"path": "local.txt"})

    assert "read_text_file" in tool_names
    assert "attach_media" in tool_names
    assert result.isError is False
    assert _text(result) == "from local environment\n"

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_mcp_agent_reads_environment_skills_with_environment_read_tool() -> None:
    skill_markdown = "---\nname: alpha\ndescription: Alpha skill\n---\nUse alpha.\n"
    manifest_path = "/workspace/.fast-agent/skills/alpha/SKILL.md"
    env = FakeEnvironment()
    env.files[manifest_path] = skill_markdown
    manifest, error = SkillRegistry.parse_manifest_text(skill_markdown, path=Path(manifest_path))
    assert error is None and manifest is not None

    config = AgentConfig(
        name="test",
        instruction="Skills:\n{{agentSkills}}",
        servers=[],
        shell=True,
        skills=Path(manifest_path).parent,
    )
    config.skill_manifests = [manifest]
    agent = McpAgent(config=config, context=Context(), shell_environment=env)

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}

    # Environment-discovered skill paths are readable by the environment
    # read_text_file tool, so no host-side read_skill fallback is advertised.
    assert "read_text_file" in tool_names
    assert READ_SKILL_TOOL_NAME not in tool_names
    assert agent.skill_read_tool_name == "read_text_file"

    read = await agent.call_tool("read_text_file", {"path": str(manifest.path)})
    assert read.isError is False
    assert "Use alpha." in _text(read)

    await agent._aggregator.close()
