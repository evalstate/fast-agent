"""Shell output guidance and the sandbox helpers it describes, run in the real Monty sandbox."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from mcp_types import CallToolResult, TextContent, Tool
from pydantic import ValidationError

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import (
    ProgrammaticToolCallingSettings,
    Settings,
    ShellSettings,
    ShellToolProfile,
)
from fast_agent.context import Context
from fast_agent.mcp.tool_result_truncation import bound_tool_result_for_llm
from fast_agent.ptc import PTC_TOOL_NAME
from fast_agent.ptc.runtime import ProgrammaticToolRuntime
from fast_agent.tools.shell_runtime import ShellRuntime
from fast_agent.utils.tool_names import is_shell_command_tool_name

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from fast_agent.ptc.output import PtcOutputGuidance
    from fast_agent.ptc.runtime import PtcToolFormat
    from fast_agent.ptc.stubs import PtcFunction

pytest.importorskip("pydantic_monty")

PRINT_MARKER = "Don't `print(r)` or return `r`"
HELPERS_MARKER = "Print `show(r)` instead"


def _text(result: CallToolResult) -> str:
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


def _shell(
    tmp_path: Path, profile: ShellToolProfile = "minimal_process", **kwargs: Any
) -> ShellRuntime:
    return ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        model_shell_tool_name="shell",
        config=Settings(shell_execution=ShellSettings(tool_profile=profile, show_bash=False)),
        working_directory=tmp_path,
        **kwargs,
    )


def _shell_function(runtime: ProgrammaticToolRuntime) -> PtcFunction:
    return next(
        function
        for function in runtime.functions.values()
        if is_shell_command_tool_name(function.tool_name)
    )


def _description(runtime: ProgrammaticToolRuntime, tools: list[Tool], **kwargs: Any) -> str:
    description = runtime.model_tools(tools, **kwargs)[0].description
    assert description is not None
    return description


async def _no_tools(name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
    raise AssertionError(f"Unexpected tool call: {name}")


@pytest.mark.parametrize("guidance", ["auto", "off", "print", "helpers"])
def test_output_guidance_setting(guidance: str) -> None:
    assert ProgrammaticToolCallingSettings().output_guidance == "auto"
    settings = Settings.model_validate({"ptc": {"output_guidance": guidance}})
    assert settings.ptc.output_guidance == guidance
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from fast_agent.config import Settings; print(Settings().ptc.output_guidance)",
        ],
        env={**os.environ, "PTC__OUTPUT_GUIDANCE": guidance},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == guidance
    with pytest.raises(ValidationError):
        ProgrammaticToolCallingSettings.model_validate({"output_guidance": "unknown"})


@pytest.mark.parametrize(
    ("setting", "model_default", "expected"),
    [
        ("auto", None, "print"),
        ("auto", "helpers", "helpers"),
        ("off", "helpers", "off"),
        ("helpers", None, "helpers"),
        ("print", "helpers", "print"),
    ],
)
def test_output_guidance_resolution(
    setting: Any, model_default: PtcOutputGuidance | None, expected: str
) -> None:
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings(output_guidance=setting))
    assert runtime.resolve_output_guidance(model_default) == expected


@pytest.mark.parametrize("profile", ["minimal_process", "luna_exec", "native"])
@pytest.mark.parametrize("guidance", ["print", "helpers"])
@pytest.mark.parametrize("tool_format", ["json", "freeform"])
def test_guidance_only_adds_its_note(
    tmp_path: Path,
    profile: ShellToolProfile,
    guidance: PtcOutputGuidance,
    tool_format: PtcToolFormat,
) -> None:
    tools = _shell(tmp_path, profile).tools
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings())
    off = _description(runtime, tools, tool_format=tool_format, output_guidance="off")
    on = _description(
        runtime, tools, tool_format=tool_format, output_guidance=guidance, result_byte_limit=33_000
    )
    shell = _shell_function(runtime)
    note = next(line for line in on.splitlines() if line.startswith("- Shell"))
    # Turning guidance off restores the description byte for byte.
    assert on.replace(note + "\n", "", 1) == off
    assert "- Shell res" not in off and "- Shell output" not in off
    assert (PRINT_MARKER in note) is (guidance == "print")
    assert (HELPERS_MARKER in note) is (guidance == "helpers")
    assert f"`r = {shell.example_call()}`" in note
    assert "about 33 KB" in note
    if profile == "native":
        assert "poll_process(process_id=r['process_id'])" in note
        assert "read_output" not in note
    else:
        assert "process(action='wait', process_id=r['process_id'])" in note
        assert "process(action='read_output', process_id=..., query='error')" in note


def test_guidance_needs_a_local_shell_in_code(tmp_path: Path) -> None:
    lookup = Tool(name="lookup", input_schema={"type": "object"})
    mcp_bash = Tool(
        name="server__bash",
        input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
    )
    direct = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings(direct_tools=["shell"]))
    for runtime, tools in [
        (ProgrammaticToolRuntime(ProgrammaticToolCallingSettings()), [lookup, mcp_bash]),
        (direct, _shell(tmp_path, "luna_exec").tools),
    ]:
        guided = _description(runtime, tools, output_guidance="helpers")
        assert guided == _description(runtime, tools, output_guidance="off")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "settings", "marker"),
    [
        ("gpt-6-astra", {}, HELPERS_MARKER),
        ("gpt-5.6", {}, PRINT_MARKER),
        ("gpt-6-astra", {"output_guidance": "print"}, PRINT_MARKER),
        ("gpt-5.6", {"output_guidance": "off"}, None),
    ],
)
async def test_agent_guidance_follows_model_default_and_setting(
    model: str, settings: dict[str, Any], marker: str | None
) -> None:
    agent = McpAgent(
        AgentConfig(name="ptc-output", servers=[], shell=True),
        context=Context(
            config=Settings(
                default_model=model,
                ptc=ProgrammaticToolCallingSettings(enabled=True, **settings),
                shell_execution=ShellSettings(show_bash=False),
            )
        ),
    )
    try:
        tools = (await agent.list_tools()).tools
        description = next(tool for tool in tools if tool.name == PTC_TOOL_NAME).description or ""
        for candidate in (PRINT_MARKER, HELPERS_MARKER):
            assert (candidate in description) is (candidate == marker)
    finally:
        await agent.shutdown()


@pytest_asyncio.fixture
async def helpers(tmp_path: Path) -> AsyncIterator[ProgrammaticToolRuntime]:
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings())
    runtime.model_tools(_shell(tmp_path).tools, output_guidance="print")
    try:
        yield runtime
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("tail('a\\nb\\nc\\n', 2)", "b\nc"),
        ("head('a\\nb\\nc\\n', 2)", "a\nb"),
        ("tail('a\\nb', 0)", ""),
        ("head('a\\nb', -1)", ""),
        ("tail('')", ""),
        ("tail('\\n'.join(str(i) for i in range(45)))", "\n".join(map(str, range(5, 45)))),
        ("head('\\n'.join(str(i) for i in range(45)))", "\n".join(map(str, range(40)))),
        (
            "grep('a\\nError one\\nb\\nFAIL two\\nc\\nerror three', 'error|fail', 1, 2)",
            "1: a\n2: Error one\n3: b\n4: FAIL two\n5: c\n[1 more matches]",
        ),
        ("grep('a\\nB\\nc', 'b', 20)", "1: a\n2: B\n3: c"),
        ("grep('a\\nb', 'missing')", ""),
        ("grep('x\\nx', 'x', limit=0)", "[2 more matches]"),
        ("grep('a\\nB\\nc', 'b', context=-1)", "2: B"),
        (
            (
                "show({'stdout': 'a\\nb\\nc\\n', 'stderr': 'warn\\nbad\\n', "
                "'exit_code': 1, 'status': 'failed', 'process_id': 'p1'}, n=1)"
            ),
            "exit_code=1 status=failed process_id=p1\n--- stderr\nbad\n--- stdout\nc",
        ),
        (
            (
                "show({'stdout': 'ok\\nERROR bad\\n', 'stderr': '', "
                "'exit_code': None, 'status': 'running', 'process_id': 'p2'}, pattern='error')"
            ),
            "status=running process_id=p2\n--- stdout\n2: ERROR bad",
        ),
        (
            "show({'stdout': '', 'exit_code': 0, 'status': 'completed', 'process_id': None})",
            "exit_code=0 status=completed\n--- stdout\n",
        ),
    ],
)
async def test_helpers_in_monty(
    helpers: ProgrammaticToolRuntime, expression: str, expected: str
) -> None:
    result = await helpers.execute(
        {"code": f"print('BEGIN'); print({expression}); print('END')"}, _no_tools
    )
    assert not result.is_error, _text(result)
    assert _text(result).endswith(f"Output:\nBEGIN\n{expected}\nEND")


@pytest.mark.asyncio
async def test_helpers_are_only_defined_with_guidance(tmp_path: Path) -> None:
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings())
    runtime.model_tools(_shell(tmp_path).tools, output_guidance="off")
    try:
        result = await runtime.execute({"code": "print(show({}))"}, _no_tools)
        assert result.is_error
        assert "Type check failed" in _text(result)
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_helper_names_win_over_tools(tmp_path: Path) -> None:
    names = ("tail", "head", "grep", "show")
    tools = [
        *_shell(tmp_path).tools,
        *(Tool(name=n, input_schema={"type": "object"}) for n in names),
    ]
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings())

    async def dispatch(name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
        return CallToolResult(content=[TextContent(type="text", text=name)])

    try:
        runtime.model_tools(tools, output_guidance="off")
        assert {f.python_name for f in runtime.functions.values()} >= set(names)
        runtime.model_tools(tools, output_guidance="helpers")
        for name in names:
            function = next(f for f in runtime.functions.values() if f.tool_name == name)
            assert function.python_name == f"{name}_tool"
            result = await runtime.execute(
                {"code": f"print(await {function.python_name}())\nprint(tail('a\\nb', 1))"},
                dispatch,
            )
            assert not result.is_error, _text(result)
            assert _text(result).endswith(f"{name}\nb")
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["minimal_process", "luna_exec", "native"])
async def test_described_calls_run(tmp_path: Path, profile: ShellToolProfile) -> None:
    """The calls the guidance shows type-check and run against the real shell tools."""
    shell = _shell(tmp_path, profile)
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings())
    runtime.model_tools(shell.tools, output_guidance="helpers")
    function = _shell_function(runtime)
    process = "poll_process" if profile == "native" else "process"
    wait = (
        "(process_id=r['process_id'])"
        if profile == "native"
        else ("(action='wait', process_id=r['process_id'])")
    )

    async def dispatch(name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
        return await shell.call_tool(name, arguments, call_id)

    try:
        result = await runtime.execute(
            {
                "code": f"r = await {function.python_name}(command='seq 1 500; echo boom >&2; exit 3')\n"
                "print(show(r, n=3))\n"
                f"print(show(r, pattern='^25$'))\n"
                f"if r['status'] == 'running':\n    print(show(await {process}{wait}))"
            },
            dispatch,
        )
        text = _text(result)
        assert not result.is_error, text
        assert "exit_code=3 status=failed" in text
        assert "--- stderr\nboom" in text
        assert "498\n499\n500" in text
        assert "25: 25" in text
    finally:
        await runtime.close()
        await shell.close()


@pytest.mark.asyncio
async def test_unprinted_output_can_be_read_back(tmp_path: Path) -> None:
    """Output that a long print lost in the middle is still retrievable, as the note says."""
    shell = _shell(tmp_path, "luna_exec", output_byte_limit=1024)
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings())
    runtime.model_tools(shell.tools, output_guidance="print")
    function = _shell_function(runtime)

    async def dispatch(name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
        return await shell.call_tool(name, arguments, call_id)

    async def retain() -> str | None:
        return None

    try:
        result = await runtime.execute(
            {
                "code": f'r = await {function.python_name}(command="seq 1 10000; '
                'echo error-middle; seq 1 10000")\n'
                "await store('pid', r['process_id'])\nprint(r['stdout'])"
            },
            dispatch,
        )
        assert not result.is_error, _text(result)
        preview = await bound_tool_result_for_llm(result, byte_limit=16_000, retain=retain)
        assert "error-middle" not in _text(preview)
        readback = await runtime.execute(
            {
                "code": "r = await process(action='read_output', process_id=await load('pid'), "
                "query='error-middle')\nprint(r['content'])"
            },
            dispatch,
        )
        assert not readback.is_error, _text(readback)
        assert "error-middle" in _text(readback)
        assert len(_text(readback)) < 200
    finally:
        await runtime.close()
        await shell.close()
