"""Experimental native local filesystem access from Monty."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from mcp_types import CallToolResult, TextContent, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import ProgrammaticToolCallingSettings, Settings, ShellSettings
from fast_agent.constants import FAST_AGENT_PTC_CALLS
from fast_agent.context import Context
from fast_agent.core.logging.logger import get_logger
from fast_agent.ptc import PTC_TOOL_NAME
from fast_agent.tools.filesystem_tool_definitions import build_read_text_file_tool
from fast_agent.tools.local_shell_executor import LocalEnvironment
from fast_agent.utils.tool_names import is_shell_command_tool_name

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

pytest.importorskip("pydantic_monty")


@asynccontextmanager
async def _agent(tmp_path: Path) -> AsyncGenerator[tuple[McpAgent, str]]:
    """Yield a code-only PTC agent working in ``tmp_path`` and its sandbox shell function."""
    environment = LocalEnvironment(logger=get_logger(__name__), working_directory=tmp_path)
    agent = McpAgent(
        AgentConfig(name="ptc-files", servers=[], shell=True),
        shell_environment=environment,
        context=Context(
            config=Settings(
                ptc=ProgrammaticToolCallingSettings(enabled=True),
                shell_execution=ShellSettings(show_bash=False),
            )
        ),
    )
    try:
        await agent.list_tools()
        runtime = agent._programmatic_tool_runtime()
        assert runtime is not None
        shell = next(
            name
            for name, function in runtime.functions.items()
            if is_shell_command_tool_name(function.tool_name)
        )
        yield agent, shell
    finally:
        await agent.shutdown()
        await environment.close()


async def _run(agent: McpAgent, code: str) -> CallToolResult:
    return await agent.call_tool(PTC_TOOL_NAME, {"code": code}, "call-1")


def _text(result: CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def _value(result: CallToolResult) -> Any:
    assert not result.is_error, _text(result)
    return json.loads(_text(result).split("Result:\n", 1)[1])


@pytest.mark.asyncio
async def test_python_file_apis_reach_the_tools_working_directory(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.c").write_text("int main(){}\n")
    (tmp_path / "blob.bin").write_bytes(bytes([0, 1, 254, 255]))
    code = """
import os
from pathlib import Path
out = {"cwd": os.getcwd(), "listdir": sorted(os.listdir("src"))}
out["checks"] = [os.path.exists("src/main.c"), os.path.isdir("src"), os.path.isfile("nope")]
with open("src/main.c") as f:
    out["read"] = f.read()
with open("notes.txt", "w") as f:
    for i in range(3):
        f.write(f"line {i}\\n")
with open("notes.txt", "a") as f:
    f.write("tail\\n")
out["lines"] = Path("notes.txt").read_text().splitlines()
data = Path("blob.bin").read_bytes()
Path("copy.bin").write_bytes(data + data)
out["sizes"] = [len(data), os.stat("copy.bin").st_size]
os.makedirs("build/obj", exist_ok=True)
out["walk"] = sorted(top for top, _, _ in os.walk("build"))
out["glob"] = sorted(str(p) for p in Path(".").rglob("*.c"))
os.rename("copy.bin", "build/moved.bin")
os.remove("notes.txt")
errors = []
for action in (lambda: open("missing.txt"), lambda: os.mkdir("src"), lambda: os.rmdir("build")):
    try:
        action()
    except OSError as exc:
        errors.append(type(exc).__name__)
out["errors"] = errors
out
"""
    async with _agent(tmp_path) as (agent, _shell):
        value = _value(await _run(agent, code))

    assert value == {
        "cwd": tmp_path.as_posix(),
        "listdir": ["main.c"],
        "checks": [True, True, False],
        "read": "int main(){}\n",
        "lines": ["line 0", "line 1", "line 2", "tail"],
        "sizes": [4, 8],
        "walk": ["build", "build/obj"],
        "glob": ["src/main.c"],
        "errors": ["FileNotFoundError", "FileExistsError", "OSError"],
    }
    assert (tmp_path / "build" / "moved.bin").read_bytes() == bytes([0, 1, 254, 255]) * 2
    assert not (tmp_path / "notes.txt").exists()


@pytest.mark.asyncio
async def test_native_writes_are_visible_to_tools_without_nested_filesystem_calls(tmp_path) -> None:
    async with _agent(tmp_path) as (agent, shell):
        code = (
            "with open('log.txt', 'w') as f:\n"
            "    for i in range(20):\n"
            "        f.write(f'{i}\\n')\n"
            f"r = await {shell}(command='wc -l < log.txt')\n"
            "int(r['stdout'])"
        )
        result = await _run(agent, code)
        tools = [call["tool"] for call in (result.meta or {})[FAST_AGENT_PTC_CALLS]]

    assert _value(result) == 20
    assert len(tools) == 1
    assert is_shell_command_tool_name(tools[0])


@pytest.mark.asyncio
async def test_reads_over_the_size_limit_fail_before_loading_the_file(tmp_path) -> None:
    (tmp_path / "small.bin").write_bytes(b"\x00" * (500 * 1024))
    (tmp_path / "large.iso").write_bytes(b"\x00" * (600 * 1024))
    (tmp_path / "large.txt").write_text("x" * (5 * 1024 * 1024))
    code = """
from pathlib import Path
out = {"small": len(open("small.bin", "rb").read())}
for name, read in (
    ("binary", lambda: open("large.iso", "rb").read(16)),
    ("text", lambda: Path("large.txt").read_text()),
):
    try:
        read()
    except OSError as exc:
        out[name] = [type(exc).__name__, str(exc)]
out
"""
    async with _agent(tmp_path) as (agent, _shell):
        value = _value(await _run(agent, code))

    assert value["small"] == 500 * 1024
    for name, size, limit in (("binary", "600 KB", "512 KB"), ("text", "5 MB", "4 MB")):
        kind, message = value[name]
        assert kind == "OSError"
        assert f"File too large to read in the sandbox ({size}; the limit is {limit}" in message
        assert "head -c" in message


@pytest.mark.asyncio
async def test_rename_follows_rename_semantics(tmp_path) -> None:
    (tmp_path / "empty").mkdir()
    (tmp_path / "target_dir").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "target_dir")
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x").write_text("x")
    (tmp_path / "src").mkdir()
    (tmp_path / "file.txt").write_text("data")
    os.link(tmp_path / "file.txt", tmp_path / "hardlink.txt")
    (tmp_path / "alias").symlink_to(tmp_path / "file.txt")
    code = """
import os
os.rename("empty", "empty")
os.rename("alias", os.path.abspath("alias"))
os.rename("file.txt", "hardlink.txt")
os.rename("hardlink.txt", "link")
errors = []
for source, target in (("src", "full"), ("src", "link"), ("empty", "file.txt")):
    try:
        os.rename(source, target)
    except OSError as exc:
        errors.append(str(exc))
os.rename("src", "empty")
errors
"""
    async with _agent(tmp_path) as (agent, _shell):
        errors = _value(await _run(agent, code))

    assert (tmp_path / "file.txt").read_text() == "data"
    assert (tmp_path / "alias").is_symlink()
    assert not (tmp_path / "link").is_symlink() and (tmp_path / "link").read_text() == "data"
    assert list((tmp_path / "target_dir").iterdir()) == []
    assert (tmp_path / "empty").is_dir() and not (tmp_path / "src").exists()
    assert ["Directory not empty" in errors[0], "Not a directory" in errors[1]] == [True, True]
    assert "Not a directory" in errors[2]


class _RecordingFilesystem:
    """An injected filesystem runtime (as ACP provides) for an agent without a shell."""

    def __init__(self) -> None:
        self.reads: list[str] = []

    @property
    def tools(self) -> list[Tool]:
        return [build_read_text_file_tool()]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        request_params: Any = None,
    ) -> CallToolResult:
        self.reads.append(str((arguments or {})["path"]))
        return CallToolResult(content=[TextContent(type="text", text="notes")])

    def metadata(self) -> dict[str, Any]:
        return {}


@pytest.mark.asyncio
async def test_relative_paths_resolve_against_the_agent_directory_without_a_shell(
    tmp_path,
) -> None:
    from fast_agent.tools.local_filesystem_runtime import LocalFilesystemRuntime

    (tmp_path / "notes.txt").write_text("notes")
    filesystem = LocalFilesystemRuntime(get_logger(__name__), tmp_path, enable_write=False)
    agent = McpAgent(
        AgentConfig(name="ptc-no-shell", servers=[], cwd=tmp_path),
        context=Context(config=Settings(ptc=ProgrammaticToolCallingSettings(enabled=True))),
    )
    agent.set_filesystem_runtime(filesystem)
    try:
        await agent.list_tools()
        result = await _run(agent, "from pathlib import Path\nPath('notes.txt').read_text()")
        denied = await _run(agent, "from pathlib import Path\nPath('out.txt').write_text('no')")
    finally:
        await agent.shutdown()

    assert _value_text(result) == "notes"
    assert denied.is_error and "PermissionError" in _text(denied)
    assert not (tmp_path / "out.txt").exists()


def _value_text(result: CallToolResult) -> str:
    assert not result.is_error, _text(result)
    return _text(result).split("Result:\n", 1)[1]


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs POSIX permissions")
async def test_a_failed_rename_keeps_the_target_it_would_replace(tmp_path) -> None:
    (tmp_path / "locked").mkdir()
    (tmp_path / "locked" / "source").write_text("source")
    (tmp_path / "locked" / "source_dir").mkdir()
    (tmp_path / "target_file").write_text("target")
    (tmp_path / "link").symlink_to(tmp_path / "target_file")
    (tmp_path / "empty").mkdir()
    (tmp_path / "locked").chmod(0o555)
    code = """
import os
errors = []
for source, target in (("locked/source", "link"), ("locked/source_dir", "empty")):
    try:
        os.rename(source, target)
    except OSError as exc:
        errors.append(type(exc).__name__)
errors
"""
    try:
        async with _agent(tmp_path) as (agent, _shell):
            errors = _value(await _run(agent, code))
    finally:
        (tmp_path / "locked").chmod(0o755)

    assert errors == ["PermissionError", "PermissionError"]
    assert (tmp_path / "link").is_symlink() and (tmp_path / "link").read_text() == "target"
    assert (tmp_path / "empty").is_dir()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "empty",
        "link",
        "locked",
        "target_file",
    ]


@pytest.mark.asyncio
async def test_reads_after_a_change_see_the_new_contents(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("old a")
    (tmp_path / "b.txt").write_text("old b")
    (tmp_path / "c.txt").write_text("c")
    code = """
import os
from pathlib import Path
fa = open("a.txt")
Path("a.txt").write_text("new a")
fb = open("b.txt")
os.rename("c.txt", "b.txt")
[fa.read(), fb.read()]
"""
    async with _agent(tmp_path) as (agent, _shell):
        assert _value(await _run(agent, code)) == ["new a", "c"]


@pytest.mark.asyncio
async def test_a_relative_agent_directory_resolves_for_the_sandbox(tmp_path, monkeypatch) -> None:
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path)
    agent = McpAgent(
        AgentConfig(name="ptc-relative", servers=[], shell=True, cwd=Path("work")),
        context=Context(
            config=Settings(
                ptc=ProgrammaticToolCallingSettings(enabled=True),
                shell_execution=ShellSettings(show_bash=False),
            )
        ),
    )
    try:
        await agent.list_tools()
        runtime = agent._programmatic_tool_runtime()
        assert runtime is not None
        shell = next(
            name
            for name, function in runtime.functions.items()
            if is_shell_command_tool_name(function.tool_name)
        )
        written = await _run(
            agent,
            "open('out.txt', 'w').write('hi')\nimport os\n"
            f"[os.getcwd(), (await {shell}(command='pwd'))['stdout'].strip()]",
        )
    finally:
        await agent.shutdown()

    work = (tmp_path / "work").resolve().as_posix()
    assert _value(written) == [work, work]
    assert (tmp_path / "work" / "out.txt").read_text() == "hi"


@pytest.mark.asyncio
async def test_odd_entries_list_and_stat_like_python(tmp_path) -> None:
    (tmp_path / "dangling").symlink_to(tmp_path / "missing")
    (tmp_path / "names").mkdir()
    (tmp_path / "names" / "one\ntwo").write_text("x")
    (tmp_path / "names" / "two").write_text("y")
    code = """
import os
from pathlib import Path
try:
    os.stat("dangling")
    stat = "no error"
except OSError as exc:
    stat = type(exc).__name__
[
    stat,
    sorted(os.listdir("names")),
    sorted(p.name for p in Path("names").glob("*")),
    sorted(f for _, _, files in os.walk("names") for f in files),
]
"""
    async with _agent(tmp_path) as (agent, _shell):
        stat, *listings = _value(await _run(agent, code))

    assert stat == "FileNotFoundError"
    assert listings == [["one\ntwo", "two"]] * 3


@pytest.mark.asyncio
async def test_writes_made_before_a_script_error_are_kept(tmp_path) -> None:
    async with _agent(tmp_path) as (agent, _shell):
        result = await _run(agent, "open('partial.txt', 'w').write('done')\n1 / 0")

    assert result.is_error
    assert "ZeroDivisionError" in _text(result)
    assert (tmp_path / "partial.txt").read_text() == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("shell", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
async def test_custom_and_mixed_filesystems_require_an_adapter(tmp_path, shell, enabled) -> None:
    from fast_agent.core.exceptions import AgentConfigError

    filesystem = _RecordingFilesystem()
    agent = McpAgent(
        AgentConfig(name="ptc-custom", servers=[], shell=shell, cwd=tmp_path),
        context=Context(config=Settings(ptc=ProgrammaticToolCallingSettings(enabled=enabled))),
    )
    agent.set_filesystem_runtime(filesystem)
    try:
        if enabled:
            with pytest.raises(AgentConfigError, match="local POSIX"):
                await agent.list_tools()
            with pytest.raises(AgentConfigError, match="local POSIX"):
                await _run(agent, "open('out.txt', 'w').write('wrong workspace')")
        else:
            tools = await agent.list_tools()
            assert PTC_TOOL_NAME not in {tool.name for tool in tools.tools}
            assert not (await agent.call_tool("read_text_file", {"path": "notes.txt"})).is_error
            assert filesystem.reads == ["notes.txt"]
    finally:
        await agent.shutdown()
    assert not (tmp_path / "out.txt").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docker", "custom-local"])
async def test_nonlocal_and_unknown_environments_are_rejected(tmp_path, kind) -> None:
    from fast_agent.core.exceptions import AgentConfigError
    from fast_agent.tools.docker_shell_environment import DockerShellEnvironment

    class CustomEnvironment(LocalEnvironment):
        pass

    environment = (
        DockerShellEnvironment(container="unused")
        if kind == "docker"
        else CustomEnvironment(logger=get_logger(__name__), working_directory=tmp_path)
    )
    agent = McpAgent(
        AgentConfig(name="ptc-environment", servers=[], shell=True),
        shell_environment=environment,
        context=Context(config=Settings(ptc=ProgrammaticToolCallingSettings(enabled=True))),
    )
    try:
        with pytest.raises(AgentConfigError, match="local POSIX"):
            await agent.__aenter__()
    finally:
        await agent.shutdown()
        await environment.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("context_only", [False, True])
async def test_acp_terminal_requires_an_adapter(context_only) -> None:
    from unittest.mock import Mock

    from acp import AgentSideConnection

    from fast_agent.acp.acp_context import ACPContext
    from fast_agent.acp.terminal_runtime import ACPTerminalRuntime
    from fast_agent.core.exceptions import AgentConfigError

    connection = Mock(spec=AgentSideConnection)
    acp = ACPContext(connection=connection, session_id="ptc")
    terminal = ACPTerminalRuntime(connection, "ptc", "test")
    agent = McpAgent(
        AgentConfig(name="ptc-acp", servers=[]),
        context=Context(
            config=Settings(ptc=ProgrammaticToolCallingSettings(enabled=True)), acp=acp
        ),
    )
    if context_only:
        acp.set_terminal_runtime(terminal)
    else:
        agent.set_external_runtime(terminal)
    try:
        with pytest.raises(AgentConfigError, match="local POSIX"):
            await agent.list_tools()
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_named_mcp_tools_do_not_grant_host_filesystem_access(tmp_path) -> None:
    from fast_agent.ptc.runtime import ProgrammaticToolRuntime

    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings(enabled=True))
    runtime.model_tools(
        [
            build_read_text_file_tool(),
            Tool(name="remote__execute", input_schema={"type": "object"}),
        ]
    )
    (tmp_path / "secret").write_text("not available")

    async def call(_name, _arguments, _call_id) -> CallToolResult:
        pytest.fail("Native file access must not dispatch MCP tools")

    try:
        result = await runtime.execute(
            {"code": "open('secret').read()"}, call, cwd=tmp_path.as_posix()
        )
    finally:
        await runtime.close()
    assert result.is_error and "PermissionError" in _text(result)
    assert "not available" not in _text(result)


@pytest.mark.asyncio
async def test_open_does_not_read_or_keep_file_descriptors(tmp_path, monkeypatch) -> None:
    import fast_agent.ptc.files as files

    path = tmp_path / "data.txt"
    path.write_text("old")
    descriptors: list[int] = []
    native_open = files.os.open

    def record_open(*args, **kwargs):
        fd = native_open(*args, **kwargs)
        descriptors.append(fd)
        return fd

    def unexpected_read(*args):
        pytest.fail("open must not read file contents")

    monkeypatch.setattr(files.os, "open", record_open)
    monkeypatch.setattr(files, "_read", unexpected_read)
    adapter = files.SandboxFilesystem(read=True, write=True)
    for _ in range(20):
        await adapter(name="open", args=(path, "r"), kwargs={}, is_async=True)
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.asyncio
async def test_cancellation_settles_a_started_native_write(tmp_path, monkeypatch) -> None:
    import asyncio
    import threading

    from fast_agent.ptc.files import SandboxFilesystem

    started = threading.Event()
    release = threading.Event()
    native_write = SandboxFilesystem.path_write_text

    def slow_write(self, path, data):
        started.set()
        assert release.wait(timeout=5)
        return native_write(self, path, data)

    monkeypatch.setattr(SandboxFilesystem, "path_write_text", slow_write)
    async with _agent(tmp_path) as (agent, _shell):
        task = asyncio.create_task(
            _run(agent, "from pathlib import Path\nPath('out').write_text('ok')")
        )
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert (tmp_path / "out").read_text() == "ok"


@pytest.mark.asyncio
async def test_native_errors_and_text_semantics(tmp_path) -> None:
    (tmp_path / "text").write_bytes("café\r\nline\rend".encode())
    (tmp_path / "directory").mkdir()
    os.mkfifo(tmp_path / "fifo")
    code = """
from pathlib import Path
errors = []
for action in (
    lambda: open('directory'),
    lambda: open('fifo'),
    lambda: Path('missing/child').write_text('no'),
):
    try:
        action()
    except OSError as exc:
        errors.append(type(exc).__name__)
[Path('text').read_text(), errors]
"""
    async with _agent(tmp_path) as (agent, _shell):
        assert _value(await _run(agent, code)) == [
            "café\nline\nend",
            ["IsADirectoryError", "OSError", "FileNotFoundError"],
        ]
    assert not (tmp_path / "missing").exists()


@pytest.mark.asyncio
async def test_environment_changes_cannot_reuse_a_local_ptc_runtime(tmp_path) -> None:
    from fast_agent.core.exceptions import AgentConfigError

    settings = Settings(ptc=ProgrammaticToolCallingSettings(enabled=True))
    agent = McpAgent(
        AgentConfig(name="ptc-switch", servers=[], shell=True, cwd=tmp_path),
        context=Context(config=settings),
    )
    try:
        await agent.list_tools()
        agent.set_filesystem_runtime(_RecordingFilesystem())
        with pytest.raises(AgentConfigError, match="local POSIX"):
            await _run(agent, "open('wrong', 'w').write('no')")
        settings.ptc.enabled = False
        tools = await agent.list_tools()
        assert PTC_TOOL_NAME not in {tool.name for tool in tools.tools}
    finally:
        await agent.shutdown()
    assert not (tmp_path / "wrong").exists()


@pytest.mark.asyncio
async def test_local_environment_filesystem_cwd_without_a_shell(tmp_path) -> None:
    from fast_agent.tools.environment_filesystem_runtime import EnvironmentFilesystemRuntime

    (tmp_path / "notes").write_text("local environment")
    environment = LocalEnvironment(logger=get_logger(__name__), working_directory=tmp_path)
    agent = McpAgent(
        AgentConfig(name="ptc-local-files", servers=[]),
        shell_environment=environment,
        context=Context(config=Settings(ptc=ProgrammaticToolCallingSettings(enabled=True))),
    )
    agent.set_filesystem_runtime(EnvironmentFilesystemRuntime(environment, enable_write=False))
    try:
        await agent.list_tools()
        assert _value_text(await _run(agent, "open('notes').read()")) == "local environment"
    finally:
        await agent.shutdown()
        await environment.close()
