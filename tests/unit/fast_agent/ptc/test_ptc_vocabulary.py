"""Offline contracts: catalog-independent PTC, unchanged direct profiles, real dispatch."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest
import pytest_asyncio
from mcp_types import CallToolResult, TextContent, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.subagent_tool import install_subagent_tool
from fast_agent.config import ProgrammaticToolCallingSettings, Settings, ShellSettings
from fast_agent.constants import FAST_AGENT_PTC_CALLS
from fast_agent.context import Context
from fast_agent.core.harness_tools import set_harness_tools
from fast_agent.core.logging.transport import AsyncEventBus
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.mcp_server_registry import ServerRegistry
from fast_agent.ptc.stubs import build_functions
from fast_agent.ptc.vocabulary import (
    LEGACY_FUNCTIONS,
    PTC_BUILTIN_FUNCTIONS,
    PTC_VOCABULARY_META,
    PTC_VOCABULARY_VERSION,
)
from fast_agent.tools.shell_tool_definitions import (
    MAX_ALIGNED_SHELL_TIMEOUT_SECONDS,
    build_ptc_shell_tool,
    parse_luna_exec_arguments,
)
from fast_agent.tools.tool_surface import ToolExposure

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fast_agent.core.logging.events import Event
    from fast_agent.tools.shell_profiles import ShellToolProfile


class _WarningTransport:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def send_event(self, event: Event) -> None:
        if event.type == "warning":
            self.events.append(event)


@pytest_asyncio.fixture
async def warnings_recorded() -> AsyncIterator[list[Event]]:
    AsyncEventBus.reset()
    transport = _WarningTransport()
    bus = AsyncEventBus.get(transport=transport)
    await bus.start()
    yield transport.events
    await bus.stop()
    AsyncEventBus.reset()


# Intentional product contract, not a per-model implementation table.
SIGNATURES = {
    "shell": "shell(*, command: str, working_directory: str = ..., background: bool = ..., timeout: int = ..., check: bool = ...) -> dict[str, Any]",
    "process": 'process(*, process_id: str = ..., action: Literal["list", "status", "wait", "stop", "read_output"] = ..., wait_sec: int = ..., offset: int = ..., limit: int = ..., query: str = ...) -> Any',
    "read_text_file": "read_text_file(*, path: str, line: int = ..., limit: int = ...) -> Any",
    "write_text_file": "write_text_file(*, path: str, content: str) -> Any",
    "edit_file": "edit_file(*, path: str, new_string: str, old_string: str = ..., replace_all: bool = ...) -> Any",
    "attach_media": "attach_media(*, source: str, mime_type: str = ..., name: str = ..., description: str = ...) -> Any",
    "subagent": "subagent(*, message: str, model: str | None = ..., label: str | None = ..., include_user_message: bool = ...) -> dict[str, Any]",
    "get_resource": "get_resource(*, uri: str, server_name: str | None = ...) -> Any",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", sorted(ModelDatabase.MODELS))
@pytest.mark.parametrize(
    "profile", ["auto", "minimal_process", "grok_shell", "luna_exec", "freeform_shell", "native"]
)
async def test_catalog_ptc_contract(model: str, profile: ShellToolProfile) -> None:
    agent = McpAgent(
        AgentConfig(name="contract", servers=[], shell=True, model=model, subagents=True),
        context=Context(
            config=Settings(
                shell_execution=ShellSettings(tool_profile=profile, enable_attach_media="on")
            ),
            ptc_enabled=True,
        ),
    )
    try:
        install_subagent_tool(agent)
        set_harness_tools(agent, True)
        surface = await agent.tool_surface()
        functions = {t.python_name: t for t in surface.exposed(ToolExposure.PROGRAMMATIC)}
        assert functions.keys() == PTC_BUILTIN_FUNCTIONS
        assert {name: f.signature for name, f in functions.items()} == SIGNATURES
        assert all(t.python_name == t.tool.name for t in functions.values())
        assert surface.programmatic is not None
        assert surface.programmatic.vocabulary == PTC_VOCABULARY_VERSION == 1
        instructions = surface.programmatic.instructions
        assert ".- " not in instructions
        assert all(
            line.startswith("- ")
            for line in instructions[instructions.index("- Sandbox") :].splitlines()
            if line.strip()
        )
        # Ignore f/g placeholders in the generic gather example.
        called = set(re.findall(r"await (\w+)\(", surface.programmatic.instructions))
        assert called <= functions.keys() | {"name", "store", "load", "stored", "settle"}
        assert "returned by shell" in (functions["process"].stub or "")
    finally:
        await agent.shutdown()


def test_canonical_timeout_schema_matches_enforced_limit() -> None:
    tool = build_ptc_shell_tool(shell_name="bash")
    timeout = tool.input_schema["properties"]["timeout"]
    assert timeout["minimum"] == 1
    assert timeout["maximum"] == MAX_ALIGNED_SHELL_TIMEOUT_SECONDS
    arguments = {"command": "true", "timeout": timeout["maximum"]}
    assert (
        parse_luna_exec_arguments(arguments, tool_name=tool.name).hard_timeout_seconds
        == timeout["maximum"]
    )
    arguments["timeout"] += 1
    with pytest.raises(ValueError, match=rf"timeout.*at most {timeout['maximum']}\b"):
        parse_luna_exec_arguments(arguments, tool_name=tool.name)


@pytest.mark.asyncio
async def test_canonical_stubs_expose_schema_bounds_and_enums() -> None:
    agent = McpAgent(
        AgentConfig(name="bounds", servers=[], shell=True, model="passthrough", subagents=True),
        context=Context(config=Settings(), ptc_enabled=True),
    )
    try:
        install_subagent_tool(agent)
        set_harness_tools(agent, True)
        surface = await agent.tool_surface()
        for entry in surface.exposed(ToolExposure.PROGRAMMATIC):
            assert entry.stub is not None
            for name, schema in entry.tool.input_schema.get("properties", {}).items():
                bounds = [
                    value
                    for key, value in schema.items()
                    if key in {"minimum", "maximum", "minLength", "maxLength"}
                ]
                if bounds:
                    argument_doc = next(
                        line
                        for line in entry.stub.splitlines()
                        if line.strip().startswith(f"{name}:")
                    )
                    assert all(f": {value}." in argument_doc for value in bounds)
                if "enum" in schema:
                    assert entry.signature is not None
                    assert "Literal[" in entry.signature
                    assert all(json.dumps(value) in entry.signature for value in schema["enum"])

        command = surface.find("shell")
        assert command is not None and command.stub is not None
        assert f"Minimum: 1. Maximum: {MAX_ALIGNED_SHELL_TIMEOUT_SECONDS}." in command.stub
        assert "For longer work, omit timeout" in command.stub
        assert "process(action='wait')" in command.stub
        assert "background=true for servers or services" in command.stub
        result = await agent.call_tool(
            "shell", {"command": "true", "timeout": MAX_ALIGNED_SHELL_TIMEOUT_SECONDS + 1}
        )
        assert result.is_error
        assert f"at most {MAX_ALIGNED_SHELL_TIMEOUT_SECONDS}" in str(result.content)
    finally:
        await agent.shutdown()


DIRECT_BASELINE = json.loads(Path(__file__).with_name("direct_names_v0.json").read_text())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "baseline", DIRECT_BASELINE, ids=lambda row: f"{row['model']}-{row['profile']}"
)
async def test_direct_names_captured_before_change(baseline: dict) -> None:
    agent = McpAgent(
        AgentConfig(name="direct", servers=[], shell=True, model=baseline["model"]),
        context=Context(
            config=Settings(
                shell_execution=ShellSettings(
                    tool_profile=baseline["profile"], enable_attach_media="on"
                )
            ),
            ptc_enabled=False,
        ),
    )
    try:
        assert [t.name for t in (await agent.list_tools()).tools] == baseline["names"]
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["canonical", "profile"])
async def test_temporary_switch_and_profile_diagnostics(
    mode: Literal["canonical", "profile"],
) -> None:
    agent = McpAgent(
        AgentConfig(name="switch", servers=[], shell=True, model="codexresponses.gpt-5.6-luna"),
        context=Context(
            config=Settings(ptc=ProgrammaticToolCallingSettings(enabled=True, function_names=mode))
        ),
    )
    try:
        surface = await agent.tool_surface()
        name = "shell" if mode == "canonical" else "exec_tool"
        command = surface.find(name)
        assert command is not None
        assert command.python_name == name
        process = surface.find("process")
        assert process is not None
        # ptc-vocab-v0: the comparison arm deliberately retains parent prose.
        description_name = name if mode == "canonical" else "exec"
        assert f"returned by {description_name}" in (process.stub or "")
        assert ("Tool `exec`" not in (command.stub or "")) == (mode == "canonical")
        result = await agent.call_tool(
            "exec_sandbox",
            {
                "code": (
                    f"await {name}(command='printf done')\n"
                    "await read_text_file(path='/nonexistent-ptc-vocabulary-file')"
                )
            },
        )
        text = "\n".join(b.text for b in result.content if isinstance(b, TextContent))
        assert result.is_error
        assert f"side effects): {name}" in text
        assert result.meta is not None
        assert result.meta[PTC_VOCABULARY_META] == (1 if mode == "canonical" else 0)
        record = result.meta[FAST_AGENT_PTC_CALLS][0]
        assert record["tool"] == ("shell" if mode == "canonical" else "exec")
        assert record["function"] == name
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("old", [name for name in LEGACY_FUNCTIONS if name != "exec"])
async def test_legacy_names_fail_before_side_effects_with_hint(old: str, tmp_path: Path) -> None:
    agent = McpAgent(
        AgentConfig(name="history", servers=[], shell=True, model="passthrough"),
        context=Context(config=Settings(), ptc_enabled=True),
    )
    try:
        await agent.list_tools()
        marker = tmp_path / "should-not-exist"
        result = await agent.call_tool(
            "exec_sandbox",
            {
                "code": (
                    f"await write_text_file(path={str(marker)!r}, content='wrong')\n"
                    f"await {old}(command='true')"
                )
            },
        )
        assert result.is_error
        text = "\n".join(b.text for b in result.content if isinstance(b, TextContent))
        assert f"`{old}` is `{LEGACY_FUNCTIONS[old]}` here" in text
        assert not marker.exists()
        assert result.meta is not None and result.meta[PTC_VOCABULARY_META] == 1
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", ["shell", "process", "read_text_file", "subagent", "get_resource", "exec_sandbox"]
)
async def test_user_collision_preserves_builtin_and_user_dispatch(
    name: str, warnings_recorded: list[Event]
) -> None:
    from fast_agent.tools.function_tool_loader import build_default_function_tool

    def user_tool() -> str:
        return "user function survived"

    agent = McpAgent(
        AgentConfig(name="collision", servers=[], shell=True, subagents=True, model="passthrough"),
        tools=[build_default_function_tool(user_tool, name=name)],
        context=Context(config=Settings(), ptc_enabled=True),
    )
    try:
        install_subagent_tool(agent)
        set_harness_tools(agent, True)
        surface = await agent.tool_surface()
        assert surface.find(name) is not None
        assert surface.find(f"{name}_tool") is not None
        # Let the actual asynchronous logger deliver the scheduled event.
        await asyncio.sleep(0)
        assert any(f"exposed as '{name}_tool'" in event.message for event in warnings_recorded)
        result = await agent.call_tool("exec_sandbox", {"code": f"await {name}_tool()"})
        assert not result.is_error
        assert "user function survived" in str(result.content)
        # The built-in command still really executes, not the user function.
        result = await agent.call_tool(
            "exec_sandbox", {"code": "await shell(command='printf builtin')"}
        )
        assert not result.is_error and "builtin" in str(result.content)
    finally:
        await agent.shutdown()


def test_mcp_sanitized_collisions_are_order_independent() -> None:
    tools = [
        Tool(name=n, input_schema={"type": "object"})
        for n in ("my-server__search", "my_server__search")
    ]
    forward = build_functions(tools)
    reverse = build_functions(list(reversed(tools)))
    assert {n: f.tool_name for n, f in forward.items()} == {
        n: f.tool_name for n, f in reverse.items()
    }
    assert len(forward) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("direct_name", ["shell", "bash"])
@pytest.mark.parametrize("model", ["astra", "deepseek", "kimi"])
async def test_portable_hybrid_shell(direct_name: str, model: str) -> None:
    agent = McpAgent(
        AgentConfig(name="hybrid", servers=[], shell=True, model=model),
        context=Context(
            config=Settings(
                ptc=ProgrammaticToolCallingSettings(enabled=True, direct_tools=[direct_name])
            )
        ),
    )
    try:
        tools = {t.name: t for t in (await agent.list_tools()).tools}
        assert tools.keys() == {"exec_sandbox", "shell"}
        assert set(tools["shell"].input_schema["properties"]) == {
            "command",
            "working_directory",
            "background",
            "timeout",
        }
        result = await agent.call_tool("shell", {"command": "printf hybrid"})
        assert not result.is_error and "hybrid" in str(result.content)
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["auto", "on", "apply_patch", "edit_file", "off"])
async def test_ptc_file_policy_only_off_applies(mode: str) -> None:
    agent = McpAgent(
        AgentConfig(name="files", servers=[], shell=True, model="codexresponses.gpt-5.5"),
        context=Context(
            config=Settings(shell_execution=ShellSettings(write_text_file_mode=mode)),
            ptc_enabled=True,
        ),
    )
    try:
        surface = await agent.tool_surface()
        assert surface.find("read_text_file")
        assert surface.find("apply_patch") is None
        assert (surface.find("write_text_file") is None) == (mode == "off")
        assert (surface.find("edit_file") is None) == (mode == "off")
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_model_switch_rebuilds_canonical_and_direct_tools() -> None:
    agent = McpAgent(
        AgentConfig(name="switch", servers=[], shell=True, model="passthrough"),
        context=Context(config=Settings(), server_registry=ServerRegistry()),
    )
    try:
        await agent.initialize()
        for enabled in (True, False, True):
            await agent.attach_llm(
                ModelFactory.create_factory(f"passthrough?ptc={str(enabled).lower()}")
            )
            surface = await agent.tool_surface()
            assert (surface.find("shell") is not None) is enabled
            assert (surface.find("bash") is not None) is not enabled
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_file_values_are_deliberately_unchanged(tmp_path: Path) -> None:
    agent = McpAgent(
        AgentConfig(name="values", servers=[], shell=True, model="opus55"),
        context=Context(config=Settings(), ptc_enabled=True),
    )
    try:
        await agent.list_tools()
        result = await agent.call_tool(
            "exec_sandbox",
            {
                "code": (
                    f"written = await write_text_file(path={str(tmp_path / 'data.json')!r}, content={json.dumps({'a': 1})!r})\n"
                    f"read = await read_text_file(path={str(tmp_path / 'data.json')!r})\n"
                    "print(written)\nprint(read['a'])"
                )
            },
        )
        assert not result.is_error
        assert "Successfully wrote" in str(result.content)
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    [
        """await shell(command='bash -c "exit 3"', check=True)""",
        "raise ValueError('could not execute step')",
        """raise ValueError('ptc.py:1:7: error[unresolved-reference] Name `bash` used when not defined')""",
        "print('bash')\nawait missing_function()",
        "await shell(command=123) # bash",
        "await exec(command='true')",  # Python builtin, not an undefined identifier.
    ],
)
async def test_legacy_hints_ignore_command_text_and_unrelated_errors(code: str) -> None:
    agent = McpAgent(
        AgentConfig(name="hints", servers=[], shell=True, model="passthrough"),
        context=Context(config=Settings(), ptc_enabled=True),
    )
    try:
        await agent.list_tools()
        result = await agent.call_tool("exec_sandbox", {"code": code})
        assert result.is_error
        text = "\n".join(b.text for b in result.content if isinstance(b, TextContent))
        assert "is `shell` here" not in text
        if code.startswith("raise"):
            assert "ValueError:" in text and "Type check failed" not in text
        if "exit 3" in code:
            assert "exit code 3" in text and "Type check failed" not in text
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_route_names_override_config_and_rebuild_on_switch() -> None:
    agent = McpAgent(
        AgentConfig(
            name="route", servers=[], shell=True, model="passthrough?ptc=on&ptc_names=profile"
        ),
        context=Context(config=Settings(), server_registry=ServerRegistry()),
    )
    try:
        surface = await agent.tool_surface()
        assert surface.programmatic is not None and surface.programmatic.vocabulary == 0
        assert surface.find("bash") is not None
        await agent.initialize()
        for mode, version, shell_name in [("canonical", 1, "shell"), ("profile", 0, "bash")]:
            await agent.attach_llm(
                ModelFactory.create_factory(f"passthrough?ptc=on&ptc_names={mode}")
            )
            surface = await agent.tool_surface()
            assert surface.programmatic is not None and surface.programmatic.vocabulary == version
            assert surface.find(shell_name) is not None
        await agent.attach_llm(
            ModelFactory.create_factory("passthrough?ptc=off&ptc_names=canonical")
        )
        surface = await agent.tool_surface()
        assert surface.programmatic is None and surface.find("bash") is not None
    finally:
        await agent.shutdown()


# Full definitions, instruction prose, function order, signatures and stubs captured
# offline from git archive 146a72df8, using the same interpreter/resources. Only the
# additive vocabulary marker is excluded; no names, descriptions or schemas are normalized.
PARENT_SURFACES = [
    (mode, row)
    for mode in ("direct", "profile")
    for row in json.loads(Path(__file__).with_name(f"{mode}_surface_v0.json").read_text())["rows"]
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "baseline"), PARENT_SURFACES)
async def test_full_surface_matches_parent(mode: str, baseline: dict) -> None:
    enabled = mode == "profile"
    agent = McpAgent(
        AgentConfig(
            name="t",
            instruction="x",
            servers=[],
            shell=True,
            model=baseline["model"],
            subagents=baseline["extras"],
        ),
        context=Context(
            config=Settings(
                shell_execution=ShellSettings(
                    tool_profile=baseline["profile"], enable_attach_media="on"
                ),
                ptc=ProgrammaticToolCallingSettings(function_names="profile"),
            ),
            ptc_enabled=enabled,
        ),
    )
    try:
        if baseline["extras"]:
            install_subagent_tool(agent)
            set_harness_tools(agent, True)
        tools = [
            tool.model_dump(mode="json", by_alias=True, exclude_none=True)
            for tool in (await agent.list_tools()).tools
        ]
        for tool in tools:
            tool.get("_meta", {}).pop(PTC_VOCABULARY_META, None)
        record = {"tools": tools}
        if enabled:
            surface = await agent.tool_surface()
            assert surface.programmatic is not None
            programmatic = asdict(surface.programmatic)
            programmatic.pop("vocabulary")
            record["programmatic"] = programmatic
            record["functions"] = [
                (entry.python_name, entry.tool.name, entry.signature, entry.stub)
                for entry in surface.exposed(ToolExposure.PROGRAMMATIC)
            ]
        digest = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert digest == baseline["sha256"], (mode, baseline)
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_mcp_collision_dispatches_alias_back_to_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = McpAgent(
        AgentConfig(name="mcp-collision", servers=[], shell=True, model="passthrough"),
        context=Context(config=Settings(), ptc_enabled=True),
    )
    seen: list[str] = []

    async def list_tools() -> list[Tool]:
        # Adversarial prefiltered boundary: normal MCP names include server__ and
        # cannot collide with these built-ins. Verify the alias route if one does.
        return [Tool(name="shell", input_schema={"type": "object"})]

    async def call_tool(name: str, arguments: dict, *args, **kwargs) -> CallToolResult:
        seen.append(name)
        return CallToolResult(content=[TextContent(type="text", text="from MCP")])

    monkeypatch.setattr(agent, "_get_filtered_mcp_tools", list_tools)
    monkeypatch.setattr(agent._aggregator, "call_tool", call_tool)
    try:
        await agent.list_tools()
        result = await agent.call_tool("exec_sandbox", {"code": "await shell_tool()"})
        assert not result.is_error and "from MCP" in str(result.content)
        assert seen == ["shell"]
        result = await agent.call_tool(
            "exec_sandbox", {"code": "await shell(command='printf local')"}
        )
        assert not result.is_error and "local" in str(result.content)
        assert seen == ["shell"]
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_profile_keeps_parent_reserved_name_rejection() -> None:
    from fast_agent.core.exceptions import AgentConfigError
    from fast_agent.tools.function_tool_loader import build_default_function_tool

    def user_tool() -> str:
        return "user"

    agent = McpAgent(
        AgentConfig(
            name="profile-collision", servers=[], model="passthrough?ptc=on&ptc_names=profile"
        ),
        tools=[build_default_function_tool(user_tool, name="get_resource")],
        context=Context(config=Settings()),
    )
    try:
        with pytest.raises(AgentConfigError, match="reserved"):
            set_harness_tools(agent, True)
    finally:
        await agent.shutdown()
