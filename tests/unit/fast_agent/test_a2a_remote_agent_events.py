from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import pytest
from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
)
from google.protobuf.json_format import MessageToDict
from mcp.types import EmbeddedResource, TextContent, TextResourceContents
from pydantic import AnyUrl

from fast_agent.a2a.config import A2AAgentConfig
from fast_agent.a2a.remote_agent import A2ARemoteAgent, _parts_from_messages
from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.types import PromptMessageExtended

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from fast_agent.llm.stream_types import StreamChunk


async def _events(*events: StreamResponse) -> AsyncIterator[StreamResponse]:
    for event in events:
        yield event


def _remote_agent() -> A2ARemoteAgent:
    return A2ARemoteAgent(
        config=AgentConfig(name="remote", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url="http://127.0.0.1:41242"),
    )


class _FakeStreamHandle:
    def __init__(self, *, preserve: bool) -> None:
        self.preserve = preserve
        self.chunks: list[str] = []
        self.finalized = False

    def update_chunk(self, chunk: StreamChunk) -> None:
        self.chunks.append(chunk.text)

    async def wait_for_drain(self) -> None:
        return

    def preserve_final_frame(self) -> bool:
        return self.preserve and bool(self.chunks)

    def finalize(self, message: PromptMessageExtended) -> None:
        del message
        self.finalized = True


class _FakeDisplay:
    def __init__(self, *, preserve: bool = True) -> None:
        self.handle = _FakeStreamHandle(preserve=preserve)
        self.assistant_messages: list[PromptMessageExtended] = []

    def show_user_message(self, *_args: object, **_kwargs: object) -> None:
        return

    @contextmanager
    def streaming_assistant_message(self, **_kwargs: object) -> Iterator[_FakeStreamHandle]:
        yield self.handle

    async def show_assistant_message(
        self,
        message: PromptMessageExtended,
        **_kwargs: object,
    ) -> None:
        self.assistant_messages.append(message)


class _FakeClient:
    def __init__(self, events: list[StreamResponse]) -> None:
        self.events = events
        self.requests: list[object] = []

    def send_message(self, request: object) -> AsyncIterator[StreamResponse]:
        self.requests.append(request)
        return _events(*self.events)


def _artifact_update(
    text: str,
    *,
    append: bool = False,
    last_chunk: bool = False,
) -> StreamResponse:
    return StreamResponse(
        artifact_update=TaskArtifactUpdateEvent(
            task_id="task-1",
            context_id="ctx-1",
            artifact=Artifact(name="response", parts=[Part(text=text)]),
            append=append,
            last_chunk=last_chunk,
        )
    )


def test_a2a_remote_agent_starts_without_client_generated_context_id() -> None:
    agent = _remote_agent()

    assert agent.context_id is None
    assert agent.current_task_id is None
    assert agent.task_status_summary().total == 0
    assert agent.prompt_status_line() is None


@pytest.mark.asyncio
async def test_a2a_remote_agent_clears_task_id_for_terminal_full_task_event() -> None:
    agent = _remote_agent()
    agent.current_task_id = "previous-task"

    result = await agent._consume_events(
        _events(
            StreamResponse(
                task=Task(
                    id="terminal-task",
                    context_id="ctx-1",
                    status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                    artifacts=[Artifact(name="response", parts=[Part(text="done")])],
                )
            )
        )
    )

    assert result.text == "done"
    assert result.state == "TASK_STATE_COMPLETED"
    assert agent.context_id == "ctx-1"
    assert agent.last_task_state == "TASK_STATE_COMPLETED"
    assert agent.current_task_id is None
    assert agent.task_status_summary().finished == 1
    assert agent.task_status_summary().pending == 0


@pytest.mark.asyncio
async def test_a2a_remote_agent_keeps_task_id_for_input_required_full_task_event() -> None:
    agent = _remote_agent()

    result = await agent._consume_events(
        _events(
            StreamResponse(
                task=Task(
                    id="input-task",
                    context_id="ctx-2",
                    status=TaskStatus(state=TaskState.TASK_STATE_INPUT_REQUIRED),
                )
            )
        )
    )

    assert result.state == "TASK_STATE_INPUT_REQUIRED"
    assert agent.context_id == "ctx-2"
    assert agent.last_task_state == "TASK_STATE_INPUT_REQUIRED"
    assert agent.current_task_id == "input-task"
    assert agent.task_status_summary().finished == 0
    assert agent.task_status_summary().pending == 1


def test_a2a_remote_agent_formats_prompt_status_line() -> None:
    agent = _remote_agent()
    agent.context_id = "context-1234567890"
    agent.current_task_id = "task-pending"
    agent.last_task_state = "TASK_STATE_INPUT_REQUIRED"
    agent.task_states = {
        "task-done": "TASK_STATE_COMPLETED",
        "task-failed": "TASK_STATE_FAILED",
        "task-pending": "TASK_STATE_INPUT_REQUIRED",
    }

    assert (
        agent.prompt_status_line()
        == "(a2a) - Context ID: cont...7890. Tasks: 2 finished, 1 pending. /tasks for info"
    )


def test_a2a_remote_agent_clears_context_for_no_history_completed_turns() -> None:
    agent = _remote_agent()
    agent.context_id = "ctx-completed"
    agent.current_task_id = None
    agent.last_task_state = "TASK_STATE_COMPLETED"

    agent._prepare_turn_state(use_history=False)

    assert agent.context_id is None
    assert agent.current_task_id is None
    assert agent.last_task_state is None


def test_a2a_remote_agent_keeps_input_required_task_for_no_history_follow_up() -> None:
    agent = _remote_agent()
    agent.context_id = "ctx-input"
    agent.current_task_id = "task-input"
    agent.last_task_state = "TASK_STATE_INPUT_REQUIRED"

    agent._prepare_turn_state(use_history=False)

    assert agent.context_id == "ctx-input"
    assert agent.current_task_id == "task-input"
    assert agent.last_task_state == "TASK_STATE_INPUT_REQUIRED"


@pytest.mark.asyncio
async def test_a2a_remote_agent_first_request_omits_context_and_task_ids() -> None:
    agent = _remote_agent()
    display = _FakeDisplay()
    fake_client = _FakeClient(
        [
            StreamResponse(
                task=Task(
                    id="task-server",
                    context_id="ctx-server",
                    status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                )
            )
        ]
    )
    agent.display = cast("Any", display)
    agent._client = fake_client

    await agent.generate_impl(
        [PromptMessageExtended(role="user", content=[TextContent(type="text", text="hello")])]
    )

    request = cast("Any", fake_client.requests[0])
    assert request.message.context_id == ""
    assert request.message.task_id == ""
    assert agent.context_id == "ctx-server"


@pytest.mark.asyncio
async def test_a2a_remote_agent_message_only_response_updates_context_without_task_state() -> None:
    agent = _remote_agent()

    result = await agent._consume_events(
        _events(
            StreamResponse(
                message=Message(
                    role=Role.ROLE_AGENT,
                    message_id="message-only",
                    context_id="ctx-message",
                    parts=[Part(text="hello")],
                )
            )
        )
    )

    assert result.text == "hello"
    assert result.state is None
    assert agent.context_id == "ctx-message"
    assert agent.current_task_id is None
    assert agent.last_task_state is None
    assert agent.last_event_kind == "message"


def test_a2a_remote_agent_keeps_message_only_context_for_no_history_follow_up() -> None:
    agent = _remote_agent()
    agent.context_id = "ctx-refinement"
    agent.last_event_kind = "message"

    agent._prepare_turn_state(use_history=False)

    assert agent.context_id == "ctx-refinement"
    assert agent.current_task_id is None
    assert agent.last_task_state is None
    assert agent.last_event_kind == "message"


@pytest.mark.asyncio
async def test_a2a_remote_agent_aggregates_artifact_updates_without_live_stream() -> None:
    agent = _remote_agent()
    display = _FakeDisplay(preserve=True)
    agent.display = cast("Any", display)
    agent._client = _FakeClient(
        [
            _artifact_update("one "),
            _artifact_update("two", append=True, last_chunk=True),
            StreamResponse(
                task=Task(
                    id="task-1",
                    context_id="ctx-1",
                    status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                )
            ),
        ]
    )

    response = await agent.generate_impl(
        [PromptMessageExtended(role="user", content=[TextContent(type="text", text="stream")])]
    )

    assert response.all_text() == "one two"
    assert display.handle.chunks == []
    assert display.handle.finalized
    assert [message.all_text() for message in display.assistant_messages] == ["one two"]


@pytest.mark.asyncio
async def test_a2a_remote_agent_renders_final_message_when_live_display_cannot_preserve() -> None:
    agent = _remote_agent()
    display = _FakeDisplay(preserve=False)
    agent.display = cast("Any", display)
    agent._client = _FakeClient([_artifact_update("final", last_chunk=True)])

    response = await agent.generate_impl(
        [PromptMessageExtended(role="user", content=[TextContent(type="text", text="stream")])]
    )

    assert response.all_text() == "final"
    assert display.handle.chunks == []
    assert [message.all_text() for message in display.assistant_messages] == ["final"]


def test_a2a_remote_agent_sends_json_text_resources_as_data_parts() -> None:
    parts = _parts_from_messages(
        [
            PromptMessageExtended(
                role="user",
                content=[
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri=AnyUrl("resource:///query.json"),
                            mimeType="application/json",
                            text='{"format": "markdown", "limit": 5}',
                        ),
                    )
                ],
            )
        ]
    )

    assert len(parts) == 1
    assert parts[0].HasField("data")
    assert parts[0].media_type == "application/json"
    assert MessageToDict(parts[0])["data"] == {"format": "markdown", "limit": 5.0}
