"""Sandbox file I/O (open(), pathlib, os) served by the agent's file and shell tools."""

from __future__ import annotations

import errno
import json
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest
from mcp_types import CallToolResult, TextContent, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import ProgrammaticToolCallingSettings, Settings, ShellSettings
from fast_agent.constants import FAST_AGENT_PTC_CALLS, FAST_AGENT_PTC_VALUE
from fast_agent.context import Context
from fast_agent.core.logging.logger import get_logger
from fast_agent.ptc import PTC_TOOL_NAME
from fast_agent.ptc.files import SandboxFilesystem, SandboxFileTools
from fast_agent.tools.filesystem_tool_definitions import build_read_text_file_tool
from fast_agent.tools.local_shell_executor import LocalEnvironment
from fast_agent.utils.tool_names import is_shell_command_tool_name

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

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
async def test_buffered_writes_land_in_one_call_before_a_tool_reads_the_file(tmp_path) -> None:
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
    assert tools[0] == "write_text_file"
    assert len(tools) == 2


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
async def test_file_operations_still_running_at_the_shell_limit_are_stopped() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def call(tool_name: str, arguments: dict[str, Any]) -> CallToolResult:
        calls.append((tool_name, arguments))
        value = {"stdout": "", "exit_code": None, "status": "running", "process_id": "process-7"}
        return CallToolResult(content=[], meta={FAST_AGENT_PTC_VALUE: value})

    shell = Tool(
        name="execute",
        input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
    )
    process = Tool(
        name="process",
        input_schema={
            "type": "object",
            "properties": {"action": {"type": "string"}, "process_id": {"type": "string"}},
        },
    )
    files = SandboxFilesystem(SandboxFileTools.from_tools([shell, process]), call, guidance="")

    with pytest.raises(OSError) as raised:
        await files.handler(name="Path.read_bytes", args=("/slow.bin",), kwargs={}, is_async=True)

    assert raised.value.errno == errno.ETIMEDOUT
    assert calls[-1] == ("process", {"action": "stop", "process_id": "process-7"})


@pytest.mark.asyncio
async def test_file_operations_display_as_one_line_without_their_transport(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "blob.bin").write_bytes(bytes(range(256)))
    shown: list[tuple[str, str]] = []
    async with _agent(tmp_path) as (agent, shell):
        monkeypatch.setattr(
            agent.display,
            "show_tool_call",
            lambda tool_name, *_args, **_kwargs: shown.append(("call", tool_name)),
        )
        monkeypatch.setattr(
            agent.display,
            "show_sandbox_file_operation",
            lambda description, **_kwargs: shown.append(("file", description)),
        )
        code = f"data = open('blob.bin', 'rb').read()\nawait {shell}(command='true')\nlen(data)"
        assert _value(await _run(agent, code)) == 256

    assert shown == [("file", f"read {(tmp_path / 'blob.bin').as_posix()}"), ("call", shell)]


@pytest.mark.asyncio
async def test_file_handles_passed_as_paths_are_resolved_to_their_path() -> None:
    from pydantic_monty import MontyFileHandle

    calls: list[tuple[str, dict[str, Any]]] = []

    async def call(tool_name: str, arguments: dict[str, Any]) -> CallToolResult:
        calls.append((tool_name, arguments))
        return CallToolResult(content=[TextContent(type="text", text="ok")])

    write = Tool(
        name="write_text_file",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        },
    )
    files = SandboxFilesystem(SandboxFileTools.from_tools([write]), call, guidance="")
    handle = MontyFileHandle("/work/out.txt", "w")

    await files.handler(name="Path.write_text", args=(handle, "data"), kwargs={}, is_async=True)
    await files.finish()

    assert calls == [("write_text_file", {"path": "/work/out.txt", "content": "data"})]


@pytest.mark.asyncio
async def test_buffered_writes_are_committed_early_past_the_size_bound(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("fast_agent.ptc.files._MAX_BUFFERED_WRITE_BYTES", 16)
    code = """
from pathlib import Path
with open("log.txt", "w") as f:
    for i in range(20):
        f.write(f"line {i}\\n")
with open("log.txt", "a") as f:
    f.write("tail\\n")
for name in ("a.bin", "b.bin"):
    Path(name).write_bytes(b"0123456789" * 4)
Path("a.bin").write_bytes(b"replaced")
"""
    async with _agent(tmp_path) as (agent, _shell):
        result = await _run(agent, code)
        commits = len((result.meta or {})[FAST_AGENT_PTC_CALLS])

    assert not result.is_error, _text(result)
    assert (tmp_path / "log.txt").read_text() == "".join(
        f"line {i}\n" for i in range(20)
    ) + "tail\n"
    assert (tmp_path / "a.bin").read_bytes() == b"replaced"
    assert (tmp_path / "b.bin").read_bytes() == b"0123456789" * 4
    assert commits > 4, commits  # committed while writing, not once at the end


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
    filesystem = _RecordingFilesystem()
    agent = McpAgent(
        AgentConfig(name="ptc-no-shell", servers=[], cwd=tmp_path),
        context=Context(config=Settings(ptc=ProgrammaticToolCallingSettings(enabled=True))),
    )
    agent.set_filesystem_runtime(filesystem)
    try:
        await agent.list_tools()
        result = await _run(agent, "from pathlib import Path\nPath('notes.txt').read_text()")
    finally:
        await agent.shutdown()

    assert _value_text(result) == "notes"
    assert filesystem.reads == [(tmp_path / "notes.txt").as_posix()]


def _value_text(result: CallToolResult) -> str:
    assert not result.is_error, _text(result)
    return _text(result).split("Result:\n", 1)[1]


@pytest.mark.asyncio
async def test_writes_made_before_a_script_error_are_kept(tmp_path) -> None:
    async with _agent(tmp_path) as (agent, _shell):
        result = await _run(agent, "open('partial.txt', 'w').write('done')\n1 / 0")

    assert result.is_error
    assert "ZeroDivisionError" in _text(result)
    assert (tmp_path / "partial.txt").read_text() == "done"
