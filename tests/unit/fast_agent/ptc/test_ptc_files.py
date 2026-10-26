"""Sandbox file I/O (open(), pathlib, os) served by the agent's file and shell tools."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest
from mcp_types import CallToolResult, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import ProgrammaticToolCallingSettings, Settings, ShellSettings
from fast_agent.constants import FAST_AGENT_PTC_CALLS
from fast_agent.context import Context
from fast_agent.core.logging.logger import get_logger
from fast_agent.ptc import PTC_TOOL_NAME
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
async def test_writes_made_before_a_script_error_are_kept(tmp_path) -> None:
    async with _agent(tmp_path) as (agent, _shell):
        result = await _run(agent, "open('partial.txt', 'w').write('done')\n1 / 0")

    assert result.is_error
    assert "ZeroDivisionError" in _text(result)
    assert (tmp_path / "partial.txt").read_text() == "done"
