from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest_asyncio
import uvicorn
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
)
from fastapi import FastAPI
from google.protobuf.json_format import ParseDict

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from a2a.server.agent_execution.context import RequestContext
    from a2a.server.events.event_queue import EventQueue

def _data_part(value: dict[str, object]) -> Part:
    part = Part()
    ParseDict(value, part.data)
    return part


LONG_STREAM_CHUNKS = [
    "Starting the remote analysis task.\n\n",
    "Step 1 — Reading the request and identifying the goal.\n",
    "Step 2 — Checking the available A2A task context.\n",
    "Step 3 — Building a concise response plan.\n",
    "Step 4 — Verifying the streamed artifact updates are ordered.\n",
    "Step 5 — Preparing the final summary.\n\n",
    "Remote analysis complete.",
]


FAKE_A2A_HELP = """Fake A2A server commands:
- hello: echo a normal response
- please stream: emit two short streaming artifact updates
- please long stream: emit a longer multi-step streaming artifact
- respond with files: return text, URL, data, and raw byte parts
- artifact append: replace and append updates on the same artifact
- need input: enter an INPUT_REQUIRED task; reply with a value such as blue
- help: show this menu"""


def _is_help_query(query: str) -> bool:
    normalized = query.strip().lower()
    return normalized in {"help", "?", "commands", "menu"} or "what can you do" in normalized


def _agent_message(*, text: str, context_id: str | None) -> Message:
    message = Message(
        role=Role.ROLE_AGENT,
        message_id=str(uuid.uuid4()),
        parts=[Part(text=text)],
    )
    if context_id:
        message.context_id = context_id
    return message


@dataclass(frozen=True)
class A2ATestServer:
    base_url: str
    card: AgentCard
    executor: EchoAgentExecutor


class EchoAgentExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.seen_queries: list[str] = []
        self.seen_part_kinds: list[list[str]] = []
        self.pending_input_tasks: set[str] = set()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id or "",
            context_id=context.context_id or "",
        )
        await updater.cancel()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.message:
            return
        query = context.get_user_input()
        self.seen_queries.append(query)
        self.seen_part_kinds.append(
            [part.WhichOneof("content") or "unknown" for part in context.message.parts]
        )

        if _is_help_query(query) and context.task_id not in self.pending_input_tasks:
            await event_queue.enqueue_event(
                _agent_message(text=FAKE_A2A_HELP, context_id=context.context_id)
            )
            return

        normalized_query = query.lower()
        taskless_query = not any(
            marker in normalized_query
            for marker in [
                "long stream",
                "stream",
                "respond with files",
                "artifact append",
                "need input",
            ]
        )
        if context.task_id not in self.pending_input_tasks and taskless_query:
            await asyncio.sleep(0.01)
            summary = ",".join(self.seen_part_kinds[-1])
            await event_queue.enqueue_event(
                _agent_message(text=f"echo: {query} [{summary}]", context_id=context.context_id)
            )
            return

        if not context.task_id or not context.context_id:
            return

        await event_queue.enqueue_event(
            Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[context.message],
            )
        )

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        )
        await updater.start_work(
            message=updater.new_agent_message(parts=[Part(text="working")])
        )

        if _is_help_query(query):
            if context.task_id in self.pending_input_tasks:
                await updater.update_status(
                    TaskState.TASK_STATE_INPUT_REQUIRED,
                    message=updater.new_agent_message(
                        parts=[
                            Part(
                                text=(
                                    f"{FAKE_A2A_HELP}\n\n"
                                    "Current task is still waiting for input."
                                )
                            )
                        ]
                    ),
                )
                return

        if context.task_id in self.pending_input_tasks:
            self.pending_input_tasks.remove(context.task_id)
            await updater.add_artifact(
                parts=[Part(text=f"input received: {query}")],
                name="input-response",
                last_chunk=True,
            )
            await updater.complete()
            return

        if "long stream" in query:
            for index, chunk in enumerate(LONG_STREAM_CHUNKS, start=1):
                await updater.add_artifact(
                    parts=[Part(text=chunk)],
                    name="long-stream",
                    last_chunk=index == len(LONG_STREAM_CHUNKS),
                )
                await asyncio.sleep(0.01)
            await updater.complete()
            return

        if "stream" in query:
            await updater.add_artifact(
                parts=[Part(text="stream chunk one")],
                name="stream",
                last_chunk=False,
            )
            await asyncio.sleep(0.01)
            await updater.add_artifact(
                parts=[Part(text="stream chunk two")],
                name="stream",
                last_chunk=True,
            )
            await updater.complete()
            return

        if "respond with files" in query:
            await updater.add_artifact(
                parts=[
                    Part(text="file response"),
                    Part(
                        url="https://example.com/report.pdf",
                        media_type="application/pdf",
                        filename="report.pdf",
                    ),
                    _data_part({"ok": True, "count": 2}),
                    Part(raw=b"abc", media_type="text/plain", filename="note.txt"),
                ],
                name="files",
                last_chunk=True,
            )
            await updater.complete()
            return

        if "artifact append" in query:
            artifact_id = "append-contract"
            await updater.add_artifact(
                parts=[Part(text="draft")],
                name="append-contract",
                artifact_id=artifact_id,
                append=False,
                last_chunk=False,
            )
            await updater.add_artifact(
                parts=[Part(text="final")],
                name="append-contract",
                artifact_id=artifact_id,
                append=False,
                last_chunk=False,
            )
            await updater.add_artifact(
                parts=[Part(text="\nrepeat")],
                name="append-contract",
                artifact_id=artifact_id,
                append=True,
                last_chunk=False,
            )
            await updater.add_artifact(
                parts=[Part(text="\nrepeat")],
                name="append-contract",
                artifact_id=artifact_id,
                append=True,
                last_chunk=True,
            )
            await updater.complete()
            return

        if "need input" in query:
            self.pending_input_tasks.add(context.task_id)
            await updater.update_status(
                TaskState.TASK_STATE_INPUT_REQUIRED,
                message=updater.new_agent_message(
                    parts=[Part(text="Please provide the missing value.")]
                ),
            )
            return



@pytest_asyncio.fixture
async def a2a_test_server(unused_tcp_port: int, wait_for_port) -> AsyncIterator[A2ATestServer]:
    host = "127.0.0.1"
    port = unused_tcp_port
    base_url = f"http://{host}:{port}"
    card = AgentCard(
        name="fast-agent test A2A server",
        description="Deterministic A2A test server.",
        provider=AgentProvider(organization="fast-agent", url="https://fast-agent.ai"),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="echo",
                name="Echo",
                description="Echo user input.",
                tags=["test"],
                examples=["hello"],
                input_modes=["text/plain"],
                output_modes=["text/plain"],
            )
        ],
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                url=f"{base_url}/a2a/jsonrpc",
            ),
            AgentInterface(
                protocol_binding="HTTP+JSON",
                protocol_version="1.0",
                url=f"{base_url}/a2a/rest",
            ),
        ],
    )
    executor = EchoAgentExecutor()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    app = FastAPI()
    app.routes.extend(create_agent_card_routes(agent_card=card))
    app.routes.extend(
        create_jsonrpc_routes(request_handler=request_handler, rpc_url="/a2a/jsonrpc")
    )
    app.routes.extend(create_rest_routes(request_handler=request_handler, path_prefix="/a2a/rest"))

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    await wait_for_port(host, port, timeout=5.0)

    try:
        yield A2ATestServer(base_url=base_url, card=card, executor=executor)
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5.0)
