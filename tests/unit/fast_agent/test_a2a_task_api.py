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
    return_message,
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


class _FakeEventQueue:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def enqueue_event(self, event: object) -> None:
        self.events.append(event)


def _set_fake_task(updater: _FakeUpdater, event_queue: _FakeEventQueue):
    return _set_current_task(
        cast("Any", updater),
        event_queue=cast("Any", event_queue),
        request_message=Message(
            role=Role.ROLE_USER,
            message_id="request-1",
            parts=[Part(text="request")],
        ),
    )


@pytest.mark.asyncio
async def test_a2a_task_api_publishes_status_and_artifact() -> None:
    updater = _FakeUpdater()
    event_queue = _FakeEventQueue()
    token = _set_fake_task(updater, event_queue)
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
    assert len(event_queue.events) == 1
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
async def test_a2a_task_api_can_return_standalone_message_before_task_starts() -> None:
    updater = _FakeUpdater()
    event_queue = _FakeEventQueue()
    token = _set_fake_task(updater, event_queue)
    try:
        handle = await return_message("Please clarify the research goal.")
    finally:
        _reset_current_task(token)

    assert handle.task_id == "task-1"
    assert len(event_queue.events) == 1
    message = cast("Message", event_queue.events[0])
    assert message.role == Role.ROLE_AGENT
    assert message.context_id == "ctx-1"
    assert message.task_id == ""
    assert message.parts[0].text == "Please clarify the research goal."
    assert updater.started == []
    assert updater.artifacts == []


@pytest.mark.asyncio
async def test_a2a_task_api_rejects_standalone_message_after_task_starts() -> None:
    updater = _FakeUpdater()
    event_queue = _FakeEventQueue()
    token = _set_fake_task(updater, event_queue)
    try:
        await start_task("Research started")
        with pytest.raises(RuntimeError, match="after starting a task"):
            await return_message("Too late")
    finally:
        _reset_current_task(token)


@pytest.mark.asyncio
async def test_attach_a2a_task_tools_adds_runnable_model_tools() -> None:
    agent = _ToolCapableAgent()

    _attach_a2a_task_tools(cast("Any", agent))

    assert [getattr(tool, "name", None) for tool in agent.tools] == [
        "start_task",
        "return_artifact",
        "return_message",
    ]
    updater = _FakeUpdater()
    event_queue = _FakeEventQueue()
    token = _set_fake_task(updater, event_queue)
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
    event_queue = _FakeEventQueue()
    harness = object.__new__(AgentHarness)
    token = _set_fake_task(updater, event_queue)
    try:
        handle = await harness.start_task("Harness work")
        artifact_handle = await harness.return_artifact("Harness artifact", name="harness")
    finally:
        _reset_current_task(token)

    assert handle.task_id == "task-1"
    assert artifact_handle.context_id == "ctx-1"
    assert updater.started[0].parts[0].text == "Harness work"
    assert updater.artifacts[0]["name"] == "harness"


@pytest.mark.asyncio
async def test_agent_harness_proxies_a2a_message_api() -> None:
    updater = _FakeUpdater()
    event_queue = _FakeEventQueue()
    harness = object.__new__(AgentHarness)
    token = _set_fake_task(updater, event_queue)
    try:
        handle = await harness.return_message("Refine this request.")
    finally:
        _reset_current_task(token)

    assert handle.context_id == "ctx-1"
    message = cast("Message", event_queue.events[0])
    assert message.parts[0].text == "Refine this request."
