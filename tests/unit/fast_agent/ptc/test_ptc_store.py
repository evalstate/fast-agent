"""The store: the only state sandbox runs share, and what sessions persist for them."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from mcp_types import CallToolResult, TextContent, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import ProgrammaticToolCallingSettings, Settings
from fast_agent.context import Context
from fast_agent.ptc import PTC_TOOL_NAME, ProgrammaticToolRuntime
from fast_agent.ptc.store import SandboxStore
from fast_agent.session import SessionHydrator
from fast_agent.session.session_manager import SessionManager

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("pydantic_monty")

VALUES = (
    "{'text': 'é', 'bytes': b'\\x00\\xff', 'int': 2**100, 'float': 1.5, 'inf': float('inf'), "
    "'flag': True, 'none': None, 'list': [1, [2]], 'tuple': (1, 'a'), 'set': {1, 2}, "
    "'frozen': frozenset({3}), 'int_keys': {1: 'one', (2, 3): 'pair'}, '$tagged': {'$set': 1}}"
)


async def _dispatch(name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=name)])


async def _run(runtime: ProgrammaticToolRuntime, code: str) -> str:
    result = await runtime.execute({"code": code}, _dispatch)
    block = result.content[0]
    assert isinstance(block, TextContent)
    return ("ERROR " if result.is_error else "") + block.text


@pytest_asyncio.fixture
async def runtime():
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings(enabled=True))
    runtime.model_tools([])
    yield runtime
    await runtime.close()


@pytest.mark.asyncio
async def test_values_round_trip_between_runs_exactly(runtime) -> None:
    stored = await _run(runtime, f"v = {VALUES}\nawait store('v', v)\nawait load('v') == v")
    later = await _run(runtime, f"(await load('v')) == {VALUES}")

    assert stored.endswith("Result:\ntrue"), stored
    assert later.endswith("Result:\ntrue"), later


@pytest.mark.asyncio
async def test_store_errors_are_specific_and_catchable(runtime) -> None:
    code = """
from dataclasses import dataclass
@dataclass
class Point:
    x: int
out = []
try:
    await store("p", Point(1))
except TypeError as exc:
    out.append(str(exc))
try:
    await load("nope")
except KeyError as exc:
    out.append(str(exc))
await store("kept", 1)
out.append(await load("nope", None))
out.append(await stored())
out
"""
    text = await _run(runtime, code)
    errors = json.loads(text.split("Result:\n", 1)[1])

    assert "a MontyClassProxy value cannot be stored; store plain data" in errors[0]
    assert "'nope' was not stored; stored keys: nothing" in errors[1]
    assert errors[2:] == [None, {"kept": 1}]


@pytest.mark.asyncio
async def test_the_store_is_size_limited(runtime, monkeypatch) -> None:
    monkeypatch.setattr("fast_agent.ptc.store.MAX_STORE_BYTES", 100)

    full = await _run(runtime, "await store('a', 'x' * 60)\nawait store('b', 'y' * 60)")
    replaced = await _run(runtime, "await store('a', 'z' * 60)\nawait stored()")

    assert full.startswith("ERROR") and "over its 0 MB limit" in full
    assert replaced.endswith('Result:\n{"a": 62}'), replaced


@pytest.mark.asyncio
async def test_state_round_trips_for_session_persistence() -> None:
    original = SandboxStore()
    await original.functions()["store"]("rows", [(1, b"\\x01"), {2: {3}}])
    revision = original.revision

    restored = SandboxStore()
    restored.restore(original.to_state())

    assert await restored.functions()["load"]("rows") == [(1, b"\\x01"), {2: {3}}]
    assert await restored.functions()["stored"]() == await original.functions()["stored"]()
    assert revision > 0 and restored.revision > 0


@pytest.mark.asyncio
async def test_tools_named_like_store_functions_do_not_shadow_them() -> None:
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings(enabled=True))
    runtime.model_tools([Tool(name="store", input_schema={"type": "object"})])
    try:
        text = await _run(runtime, "await store('k', 1)\n[await load('k'), await store_tool()]")
    finally:
        await runtime.close()

    assert text.endswith('Result:\n[1, "store"]'), text


def _agent(*, ptc: bool = True) -> McpAgent:
    settings = Settings(ptc=ProgrammaticToolCallingSettings(enabled=ptc))
    return McpAgent(AgentConfig(name="coder", servers=[]), context=Context(config=settings))


@pytest.mark.asyncio
async def test_stored_values_are_saved_forked_and_resumed_with_the_session(tmp_path: Path) -> None:
    manager = SessionManager(
        cwd=tmp_path, home_override=tmp_path / ".fast-agent", respect_env_override=False
    )
    session = manager.create_session()
    original = _agent()
    try:
        await original.list_tools()
        await original.call_tool(PTC_TOOL_NAME, {"code": "await store('rows', [(1, b'x')])"})
        await session.save_history(original)
    finally:
        await original.shutdown()

    filename = "ptc_store_coder.json"
    assert session.load_snapshot().continuation.agents["coder"].ptc_store_file == filename
    fork = manager.fork_session(session.info.name)
    assert fork is not None and (fork.directory / filename).is_file()

    resumed, without_ptc = _agent(), _agent(ptc=False)
    try:
        await resumed.list_tools()
        restored = await SessionHydrator().hydrate_session(
            session=fork, agents={"coder": resumed}, fallback_agent_name="coder"
        )
        loaded = await resumed.call_tool(
            PTC_TOOL_NAME, {"code": "await load('rows') == [(1, b'x')]"}
        )
        skipped = await SessionHydrator().hydrate_session(
            session=fork, agents={"coder": without_ptc}, fallback_agent_name="coder"
        )
    finally:
        await resumed.shutdown()
        await without_ptc.shutdown()

    assert restored.warnings == []
    block = loaded.content[0]
    assert isinstance(block, TextContent) and block.text.endswith("Result:\ntrue"), block
    assert [warning.code for warning in skipped.warnings] == ["ptc-store-restore-skipped"]
