"""Deterministic A2A server for fast-agent CLI/TUI smoke tests.

Run:
    uv run python tests/integration/a2a/fake_server.py --port 41242

Useful prompts:
    help
    hello
    please stream
    please long stream
    respond with files
    need input
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from typing import TYPE_CHECKING

import uvicorn
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes, create_rest_routes
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
    from a2a.server.agent_execution.context import RequestContext
    from a2a.server.events.event_queue import EventQueue

logger = logging.getLogger(__name__)

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


class FakeAgentExecutor(AgentExecutor):
    def __init__(self) -> None:
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
        if _is_help_query(query) and context.task_id not in self.pending_input_tasks:
            await event_queue.enqueue_event(
                _agent_message(text=FAKE_A2A_HELP, context_id=context.context_id)
            )
            return

        taskless_query = not any(
            marker in query.lower()
            for marker in ["long stream", "stream", "files", "need input"]
        )
        if context.task_id not in self.pending_input_tasks and taskless_query:
            kinds = ",".join(
                part.WhichOneof("content") or "unknown" for part in context.message.parts
            )
            await event_queue.enqueue_event(
                _agent_message(
                    text=f"fake echo: {query} [{kinds}]",
                    context_id=context.context_id,
                )
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
        await updater.start_work(message=updater.new_agent_message(parts=[Part(text="working")]))

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

        if "long stream" in query.lower():
            for index, chunk in enumerate(LONG_STREAM_CHUNKS, start=1):
                await updater.add_artifact(
                    parts=[Part(text=chunk)],
                    name="long-stream",
                    last_chunk=index == len(LONG_STREAM_CHUNKS),
                )
                await asyncio.sleep(0.35)
            await updater.complete()
            return

        if "stream" in query.lower():
            await updater.add_artifact(parts=[Part(text="stream chunk one")], name="stream")
            await asyncio.sleep(0.4)
            await updater.add_artifact(
                parts=[Part(text="stream chunk two")], name="stream", last_chunk=True
            )
            await updater.complete()
            return

        if "files" in query.lower():
            data_part = Part()
            ParseDict({"ok": True, "source": "fake-a2a-server"}, data_part.data)
            await updater.add_artifact(
                parts=[
                    Part(text="file response"),
                    Part(
                        url="https://example.com/report.pdf",
                        media_type="application/pdf",
                        filename="report.pdf",
                    ),
                    data_part,
                    Part(raw=b"abc", media_type="text/plain", filename="note.txt"),
                ],
                name="files",
                last_chunk=True,
            )
            await updater.complete()
            return

        if "need input" in query.lower():
            self.pending_input_tasks.add(context.task_id)
            await updater.update_status(
                TaskState.TASK_STATE_INPUT_REQUIRED,
                message=updater.new_agent_message(
                    parts=[Part(text="Please provide the missing value.")]
                ),
            )
            return



def build_app(host: str, port: int) -> FastAPI:
    base_url = f"http://{host}:{port}"
    card = AgentCard(
        name="fast-agent fake A2A server",
        description="Deterministic server for fast-agent A2A demos and tests.",
        provider=AgentProvider(organization="fast-agent", url="https://fast-agent.ai"),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text/plain", "application/octet-stream", "image/*"],
        default_output_modes=["text/plain", "application/json", "application/octet-stream"],
        skills=[
            AgentSkill(
                id="fake_echo_stream_files",
                name="Fake echo/stream/files",
                description="Echoes text, streams short/long chunks, and returns URL/data/raw parts.",
                tags=["test", "streaming", "files"],
                examples=["hello", "please stream", "please long stream", "respond with files"],
                input_modes=["text/plain", "application/octet-stream", "image/*"],
                output_modes=["text/plain", "application/json", "application/octet-stream"],
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
    request_handler = DefaultRequestHandler(
        agent_executor=FakeAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    app = FastAPI()
    app.routes.extend(create_agent_card_routes(agent_card=card))
    app.routes.extend(create_jsonrpc_routes(request_handler=request_handler, rpc_url="/a2a/jsonrpc"))
    app.routes.extend(create_rest_routes(request_handler=request_handler, path_prefix="/a2a/rest"))
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="fast-agent fake A2A server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41242)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("Agent card: http://%s:%s/.well-known/agent-card.json", args.host, args.port)
    uvicorn.run(build_app(args.host, args.port), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
