"""The store: the only state sandbox runs share, and what sessions persist for them."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
import pytest_asyncio
from acp import AgentSideConnection
from mcp_types import CallToolResult, TextContent, Tool

from fast_agent.acp.terminal_runtime import ACPTerminalRuntime
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import ProgrammaticToolCallingSettings, Settings
from fast_agent.context import Context
from fast_agent.hooks.hook_context import HookContext
from fast_agent.hooks.session_history import save_session_history
from fast_agent.mcp.prompt import Prompt
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


def _agent(*, ptc: bool = True, session_manager: SessionManager | None = None) -> McpAgent:
    settings = Settings(ptc=ProgrammaticToolCallingSettings(enabled=ptc))
    context = Context(config=settings, session_manager=session_manager)
    return McpAgent(AgentConfig(name="coder", servers=[]), context=context)


@pytest.mark.asyncio
async def test_stored_values_are_saved_forked_and_resumed_with_the_session(tmp_path: Path) -> None:
    manager = SessionManager(
        cwd=tmp_path, home_override=tmp_path / ".fast-agent", respect_env_override=False
    )
    session = manager.create_session()
    original = _agent(session_manager=manager)
    try:
        await original.list_tools()
        await original.call_tool(PTC_TOOL_NAME, {"code": "await store('rows', [(1, b'x')])"})
        # Save the way a turn end does: the hook passes a HistoryAgent wrapper.
        original.load_message_history([Prompt.user("keep the rows")])
        message = original.message_history[-1]
        await save_session_history(
            HookContext(
                runner=SimpleNamespace(iteration=1, request_params=None),
                agent=original,
                message=message,
                hook_type="after_turn_complete",
            )
        )
    finally:
        await original.shutdown()

    filename = "ptc_store_coder.json"
    saved = session.load_snapshot().continuation.agents["coder"]
    assert saved.ptc_store_file == filename
    assert saved.ptc_enabled is True
    fork = manager.fork_session(session.info.name)
    assert fork is not None and (fork.directory / filename).is_file()

    # A resumed session keeps its own PTC setting, not the one the agent started with.
    resumed, started_without_ptc, unservable = _agent(), _agent(ptc=False), _agent(ptc=False)
    unservable.set_external_runtime(ACPTerminalRuntime(Mock(spec=AgentSideConnection), "p", "t"))
    loaded: list[CallToolResult] = []
    try:
        hydrations = []
        for agent in (resumed, started_without_ptc, unservable):
            hydrations.append(
                await SessionHydrator().hydrate_session(
                    session=fork, agents={"coder": agent}, fallback_agent_name="coder"
                )
            )
        for agent in (resumed, started_without_ptc):
            loaded.append(
                await agent.call_tool(PTC_TOOL_NAME, {"code": "await load('rows') == [(1, b'x')]"})
            )
    finally:
        for agent in (resumed, started_without_ptc, unservable):
            await agent.shutdown()

    assert [[warning.code for warning in h.warnings] for h in hydrations] == [
        [],
        [],
        ["ptc-restore-fallback", "ptc-store-restore-skipped"],
    ]
    assert not unservable.ptc_enabled
    for result in loaded:
        block = result.content[0]
        assert isinstance(block, TextContent) and block.text.endswith("Result:\ntrue"), block


@pytest.mark.asyncio
async def test_retain_output_evicts_oldest_auto_entries_only_and_survives_restore() -> None:
    store = SandboxStore()
    await store.functions()["store"]("user_key", "keep-me")

    keys = [store.retain_output(f"payload-{i}", max_entries=3) for i in range(1, 6)]
    assert keys == [
        "truncated_output_1",
        "truncated_output_2",
        "truncated_output_3",
        "truncated_output_4",
        "truncated_output_5",
    ]
    stored_keys = await store.functions()["stored"]()
    assert set(stored_keys) == {
        "user_key",
        "truncated_output_3",
        "truncated_output_4",
        "truncated_output_5",
    }
    assert await store.functions()["load"]("user_key") == "keep-me"
    assert await store.functions()["load"]("truncated_output_5") == "payload-5"

    restored = SandboxStore()
    restored.restore(store.to_state())
    next_key = restored.retain_output("payload-6", max_entries=3)
    assert next_key == "truncated_output_6"
    assert set(await restored.functions()["stored"]()) == {
        "user_key",
        "truncated_output_4",
        "truncated_output_5",
        "truncated_output_6",
    }


@pytest.mark.asyncio
async def test_retain_output_respects_store_cap_without_mutating_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fast_agent.ptc.store.MAX_STORE_BYTES", 120)
    store = SandboxStore()
    await store.functions()["store"]("user_key", "u" * 60)
    first_key = store.retain_output("a" * 10, max_bytes=100, max_entries=4)
    assert first_key == "truncated_output_1"
    revision_before = store.revision

    # Too large to fit alongside user_key even if truncated_output_1 were evicted.
    assert store.retain_output("b" * 80, max_bytes=100, max_entries=4) is None
    assert store.revision == revision_before
    assert set(await store.functions()["stored"]()) == {"user_key", "truncated_output_1"}


@pytest.mark.asyncio
async def test_retain_output_counts_utf8_bytes_for_multibyte_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.mcp import tool_result_spool

    store = SandboxStore()
    emoji_payload = "🚀" * 16  # 16 chars, 64 UTF-8 bytes (+2 quotes = 66 bytes)
    assert store.retain_output(emoji_payload, max_bytes=18) is None

    monkeypatch.setattr(tool_result_spool, "PTC_STORE_RESULT_MAX_BYTES", 18)
    result = CallToolResult(content=[TextContent(type="text", text=emoji_payload)])
    notice = tool_result_spool.store_tool_result(
        store, result, tool_name="exec_sandbox", ptc_tool_name=PTC_TOOL_NAME
    )
    assert notice is not None
    assert "The first 4 characters" in notice
    assert await store.functions()["load"]("truncated_output_1") == "🚀" * 4
    assert (await store.functions()["stored"]())["truncated_output_1"] == 18


@pytest.mark.asyncio
async def test_retain_output_never_evicts_user_created_or_replaced_prefixed_keys() -> None:
    store = SandboxStore()
    await store.functions()["store"]("truncated_output_1", "user-created")

    k2 = store.retain_output("auto-2", max_entries=2)
    assert k2 == "truncated_output_2"
    # User overwrites an auto-issued key: it becomes user-owned and must not be evicted.
    await store.functions()["store"]("truncated_output_2", "user-replaced")

    k3 = store.retain_output("auto-3", max_entries=2)
    k4 = store.retain_output("auto-4", max_entries=2)
    k5 = store.retain_output("auto-5", max_entries=2)
    assert (k3, k4, k5) == ("truncated_output_3", "truncated_output_4", "truncated_output_5")

    restored = SandboxStore()
    restored.restore(store.to_state())
    k6 = restored.retain_output("auto-6", max_entries=2)
    assert k6 == "truncated_output_6"
    assert set(await restored.functions()["stored"]()) == {
        "truncated_output_1",
        "truncated_output_2",
        "truncated_output_5",
        "truncated_output_6",
    }
    assert await restored.functions()["load"]("truncated_output_1") == "user-created"
    assert await restored.functions()["load"]("truncated_output_2") == "user-replaced"
