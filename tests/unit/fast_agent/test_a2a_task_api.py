from __future__ import annotations

import json
from typing import Any, cast

import pytest
from a2a.types import Message, Part, Role

from fast_agent.a2a.server import _attach_a2a_task_tools
from fast_agent.a2a.task_api import (
    _reset_current_task,
    _set_current_task,
    current_task,
    return_artifact,
    start_task,
)
from fast_agent.core.harness import AgentHarness


class _ToolCapableAgent:
    def __init__(self) -> None:
        self.tools: list[object] = []

    def add_tool(self, tool: object, *, replace: bool = True) -> None:
        del replace
        self.tools.append(tool)


class _FakeUpdater:
    task_id = "task-1"
    context_id = "ctx-1"

    def __init__(self) -> None:
        self.started: list[Message] = []
        self.artifacts: list[dict[str, object]] = []

    def new_agent_message(self, *, parts: list[Part]) -> Message:
        return Message(role=Role.ROLE_AGENT, message_id="msg-1", parts=parts)

    async def start_work(self, *, message: Message) -> None:
        self.started.append(message)

    async def add_artifact(
        self,
        *,
        parts: list[Part],
        artifact_id: str | None = None,
        name: str | None = None,
        append: bool = False,
        last_chunk: bool = False,
    ) -> None:
        self.artifacts.append(
            {
                "parts": parts,
                "artifact_id": artifact_id,
                "name": name,
                "append": append,
                "last_chunk": last_chunk,
            }
        )


@pytest.mark.asyncio
async def test_a2a_task_api_publishes_status_and_artifact() -> None:
    updater = _FakeUpdater()
    token = _set_current_task(cast("Any", updater))
    try:
        handle = await start_task("Working on it")
        artifact_handle = await return_artifact(
            "partial answer",
            name="draft",
            artifact_id="draft-1",
            append=True,
            last_chunk=False,
        )
    finally:
        _reset_current_task(token)

    assert handle.task_id == "task-1"
    assert artifact_handle.context_id == "ctx-1"
    assert updater.started[0].parts[0].text == "Working on it"
    assert updater.artifacts == [
        {
            "parts": [Part(text="partial answer")],
            "artifact_id": "draft-1",
            "name": "draft",
            "append": True,
            "last_chunk": False,
        }
    ]


@pytest.mark.asyncio
async def test_a2a_task_api_requires_active_task_context() -> None:
    assert current_task() is None
    with pytest.raises(RuntimeError, match="only available"):
        await start_task()


@pytest.mark.asyncio
async def test_attach_a2a_task_tools_adds_runnable_model_tools() -> None:
    agent = _ToolCapableAgent()

    _attach_a2a_task_tools(cast("Any", agent))

    assert [getattr(tool, "name", None) for tool in agent.tools] == [
        "start_task",
        "return_artifact",
    ]
    updater = _FakeUpdater()
    token = _set_current_task(cast("Any", updater))
    try:
        start_result = await cast("Any", agent.tools[0]).run({"message": "Tool work"})
        artifact_result = await cast("Any", agent.tools[1]).run(
            {"text": "Tool artifact", "name": "tool"}
        )
    finally:
        _reset_current_task(token)

    assert json.loads(start_result.content[0].text) == {
        "task_id": "task-1",
        "context_id": "ctx-1",
    }
    assert json.loads(artifact_result.content[0].text) == {
        "task_id": "task-1",
        "context_id": "ctx-1",
    }
    assert updater.started[0].parts[0].text == "Tool work"
    assert updater.artifacts[0]["name"] == "tool"


@pytest.mark.asyncio
async def test_agent_harness_proxies_a2a_task_api() -> None:
    updater = _FakeUpdater()
    harness = object.__new__(AgentHarness)
    token = _set_current_task(cast("Any", updater))
    try:
        handle = await harness.start_task("Harness work")
        artifact_handle = await harness.return_artifact("Harness artifact", name="harness")
    finally:
        _reset_current_task(token)

    assert handle.task_id == "task-1"
    assert artifact_handle.context_id == "ctx-1"
    assert updater.started[0].parts[0].text == "Harness work"
    assert updater.artifacts[0]["name"] == "harness"
