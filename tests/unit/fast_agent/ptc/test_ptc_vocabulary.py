"""Offline contracts: catalog-independent PTC, unchanged direct profiles, real dispatch."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest
import pytest_asyncio
from mcp_types import TextContent, Tool

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
        # Ignore f/g placeholders in the generic gather example.
        called = set(re.findall(r"await (\w+)\(", surface.programmatic.instructions))
        assert called <= functions.keys() | {"name", "store", "load", "stored", "settle"}
        assert "returned by shell" in (functions["process"].stub or "")
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
        assert f"returned by {name}" in (process.stub or "")
        assert "Tool `exec`" not in (command.stub or "")
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
@pytest.mark.parametrize("old", LEGACY_FUNCTIONS)
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
