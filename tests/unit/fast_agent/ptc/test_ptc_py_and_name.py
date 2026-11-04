"""py() runs the machine's Python from sandbox code; ptc.tool_name renames the code tool."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pytest
from mcp_types import CallToolResult, TextContent, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import ProgrammaticToolCallingSettings, Settings, ShellSettings
from fast_agent.context import Context
from fast_agent.mcp_server_registry import ServerRegistry
from fast_agent.ptc.runtime import ProgrammaticToolRuntime
from fast_agent.tools.shell_runtime import ShellRuntime

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.config import ShellToolProfile

pytest.importorskip("pydantic_monty")


def _text(result: CallToolResult) -> str:
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


def _shell(tmp_path: Path, profile: ShellToolProfile) -> ShellRuntime:
    return ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        model_shell_tool_name="shell",
        config=Settings(shell_execution=ShellSettings(tool_profile=profile, show_bash=False)),
        working_directory=tmp_path,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["minimal_process", "freeform_shell", "native"])
async def test_py_runs_cpython_without_shell_quoting(
    tmp_path: Path, profile: ShellToolProfile
) -> None:
    shell = _shell(tmp_path, profile)
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings())
    runtime.model_tools(shell.tools, output_guidance="off")

    async def dispatch(name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
        return await shell.call_tool(name, arguments, call_id)

    code = (
        'r = await py("""\n'
        "import difflib\n"
        'print(difflib.SequenceMatcher(None, "abcd", "abxd").ratio())\n'
        'print("it\'s $HOME `x`")\n'
        '""")\n'
        "print(r['exit_code'], r['stdout'])"
    )
    try:
        result = await runtime.execute({"code": code}, dispatch)
        text = _text(result)
        assert not result.is_error, text
        assert "0 0.75\nit's $HOME `x`" in text
        failed = await runtime.execute(
            {"code": 'await py("raise SystemExit(3)", check=True)'}, dispatch
        )
        assert failed.is_error
    finally:
        await runtime.close()
        await shell.close()


def test_py_is_described_only_with_a_local_shell(tmp_path: Path) -> None:
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings())
    with_shell = runtime.model_tools(_shell(tmp_path, "minimal_process").tools)[0].description or ""
    assert "await py(" in with_shell and "not the machine's Python" in with_shell
    assert "For full CPython or packages" not in with_shell
    mcp_only = (
        runtime.model_tools(
            [
                Tool(name="lookup", input_schema={"type": "object"}),
                Tool(name="py", input_schema={"type": "object"}),
            ]
        )[0].description
        or ""
    )
    assert "await py(" not in mcp_only
    # Without a local shell, a tool named py keeps its name.
    assert "py" in runtime.functions


@pytest.mark.asyncio
async def test_tool_name_setting_renames_the_code_tool_end_to_end() -> None:
    agent = McpAgent(
        AgentConfig(name="ptc-name", servers=[], shell=True),
        context=Context(
            config=Settings(
                ptc=ProgrammaticToolCallingSettings(enabled=True, tool_name="exec_sandbox"),
                shell_execution=ShellSettings(show_bash=False),
            ),
            server_registry=ServerRegistry(),
        ),
    )
    try:
        await agent.initialize()
        tools = (await agent.list_tools()).tools
        assert [tool.name for tool in tools] == ["exec_sandbox"]
        assert "`exec_sandbox` invocation" in agent.instruction
        assert "execute_python" not in agent.instruction
        result = await agent.call_tool("exec_sandbox", {"code": "print(6 * 7)"})
        assert not result.is_error and "42" in _text(result)
    finally:
        await agent.shutdown()
