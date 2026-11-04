"""The code tool's name and what its description claims about the sandbox."""

from __future__ import annotations

from typing import Any

import pytest
from mcp_types import CallToolResult, TextContent, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import ProgrammaticToolCallingSettings, Settings, ShellSettings
from fast_agent.context import Context
from fast_agent.mcp_server_registry import ServerRegistry
from fast_agent.ptc import PTC_TOOL_NAME
from fast_agent.ptc.runtime import ProgrammaticToolRuntime

pytest.importorskip("pydantic_monty")


@pytest.mark.asyncio
async def test_code_tool_is_exec_sandbox_end_to_end() -> None:
    assert PTC_TOOL_NAME == "exec_sandbox"
    agent = McpAgent(
        AgentConfig(name="ptc-name", servers=[], shell=True),
        context=Context(
            config=Settings(
                ptc=ProgrammaticToolCallingSettings(enabled=True),
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
        result = await agent.call_tool("exec_sandbox", {"code": "print(6 * 7)"})
        assert not result.is_error
        assert "42" in "\n".join(b.text for b in result.content if isinstance(b, TextContent))
    finally:
        await agent.shutdown()


def _text(result: CallToolResult) -> str:
    return "\n".join(b.text for b in result.content if isinstance(b, TextContent))


async def _no_tools(name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
    raise AssertionError(name)


_SHELL = Tool(
    name="bash",
    input_schema={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "hinted"),
    [
        ("print(str.maketrans('ab', 'ba'))", True),
        ("print((5).bit_length())", True),
        ("import difflib", True),
        ("x = {'a': 1}\nprint(x['b'])", False),
    ],
)
async def test_sandbox_gaps_point_at_the_system_python(code: str, hinted: bool) -> None:
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings())
    runtime.model_tools([_SHELL])
    try:
        result = await runtime.execute({"code": code}, _no_tools)
        assert result.is_error
        text = _text(result)
        assert ("the sandbox's Python lacks this" in text) is hinted
        if hinted:
            assert 'await bash(command="python3 script.py")' in text
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_gather_bracket_typo_gets_a_hint() -> None:
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings())
    runtime.model_tools([_SHELL])
    try:
        typo = await runtime.execute(
            {"code": 'r = await asyncio.gather(*[bash(command=c) for c in ["a", "b"])'},
            _no_tools,
        )
        assert typo.is_error and "the comprehension needs its own `]`" in _text(typo)
        other = await runtime.execute({"code": "x = [1, 2)"}, _no_tools)
        assert other.is_error and "comprehension needs" not in _text(other)
    finally:
        await runtime.close()
