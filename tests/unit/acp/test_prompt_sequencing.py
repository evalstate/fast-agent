"""Tests that AgentACPServer serializes overlapping prompts per session.

ACP session/update notifications are only correlated by sessionId, so fast-agent must not
process multiple prompts concurrently for the same session.

This test verifies that a second session/prompt call blocks (queues) behind the first.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pytest
from acp.schema import TextContentBlock

from fast_agent.acp.server.agent_acp_server import ACPSessionState, AgentACPServer
from fast_agent.core.agent_app import AgentApp
from fast_agent.core.fastagent import AgentInstance

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


@dataclass
class DummyResult:
    text: str
    stop_reason: str = "end_turn"

    def last_text(self) -> str:
        return self.text


class DummyAgent:
    """Minimal agent implementing the subset used by AgentACPServer.prompt()."""

    def __init__(self, started_evt: asyncio.Event, proceed_evt: asyncio.Event, text: str):
        self._started_evt = started_evt
        self._proceed_evt = proceed_evt
        self._text = text
        self.usage_accumulator = None  # Avoid status line branch

    async def generate(self, prompt_message: Any, request_params: Any = None) -> DummyResult:
        self._started_evt.set()
        await self._proceed_evt.wait()
        return DummyResult(self._text)


class DummyApp:
    """Placeholder for AgentInstance.app (unused by AgentACPServer.prompt path)."""


@pytest.mark.asyncio
async def test_overlapping_prompts_are_serialized() -> None:
    started1 = asyncio.Event()
    started2 = asyncio.Event()
    proceed1 = asyncio.Event()
    proceed2 = asyncio.Event()

    agent = DummyAgent(started_evt=started1, proceed_evt=proceed1, text="first")

    # Build a minimal AgentInstance.
    # AgentInstance is strongly typed (AgentApp + AgentProtocol), but the ACP server
    # code path under test only uses instance.agents[...].generate(...).
    agents: dict[str, "AgentProtocol"] = {"default": cast("AgentProtocol", agent)}
    instance = AgentInstance(app=AgentApp(agents), agents=agents, registry_version=0)

    async def create_instance() -> AgentInstance:
        return instance

    async def dispose_instance(_instance: AgentInstance) -> None:
        return None

    server = AgentACPServer(
        primary_instance=instance,
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        instance_scope="shared",
        server_name="test",
        permissions_enabled=False,
    )

    session_id = "s-1"
    server.sessions[session_id] = instance
    server._session_state[session_id] = ACPSessionState(session_id=session_id, instance=instance)

    # Prompt 1: start immediately but block inside DummyAgent.generate
    t1 = asyncio.create_task(
        server.prompt(prompt=[TextContentBlock(type="text", text="p1")], session_id=session_id)
    )

    await asyncio.wait_for(started1.wait(), timeout=1.0)

    # Swap in a second agent instance for the second prompt to detect when it actually starts.
    # (We keep it under the same agent name.)
    server.sessions[session_id].agents["default"] = cast(
        "AgentProtocol",
        DummyAgent(started_evt=started2, proceed_evt=proceed2, text="second"),
    )

    # Prompt 2: should queue behind prompt 1 (i.e., should not start yet)
    t2 = asyncio.create_task(
        server.prompt(prompt=[TextContentBlock(type="text", text="p2")], session_id=session_id)
    )

    # Give the event loop a moment; if prompts were concurrent, started2 would be set.
    await asyncio.sleep(0.05)
    assert not started2.is_set(), "Second prompt started while first was still running"

    # Finish prompt 1
    proceed1.set()
    r1 = await asyncio.wait_for(t1, timeout=1.0)
    assert r1.stop_reason in {"end_turn", "end_turn"}  # tolerate mapping

    # Now prompt 2 should start and complete
    await asyncio.wait_for(started2.wait(), timeout=1.0)
    proceed2.set()
    r2 = await asyncio.wait_for(t2, timeout=1.0)
    assert r2.stop_reason in {"end_turn", "end_turn"}