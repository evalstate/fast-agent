from __future__ import annotations

import asyncio
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
    Part,
    Task,
    TaskState,
    TaskStatus,
)
from fastapi import FastAPI

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from a2a.server.agent_execution.context import RequestContext
    from a2a.server.events.event_queue import EventQueue

@dataclass(frozen=True)
class A2ATestServer:
    base_url: str
    card: AgentCard


class EchoAgentExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.seen_queries: list[str] = []

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id or "",
            context_id=context.context_id or "",
        )
        await updater.cancel()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.message or not context.task_id or not context.context_id:
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

        query = context.get_user_input()
        self.seen_queries.append(query)
        await asyncio.sleep(0.01)
        await updater.add_artifact(
            parts=[Part(text=f"echo: {query}")],
            name="response",
            last_chunk=True,
        )
        await updater.complete()


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
        default_input_modes=["text"],
        default_output_modes=["text", "task-status"],
        skills=[
            AgentSkill(
                id="echo",
                name="Echo",
                description="Echo user input.",
                tags=["test"],
                examples=["hello"],
                input_modes=["text"],
                output_modes=["text", "task-status"],
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
        agent_executor=EchoAgentExecutor(),
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
        yield A2ATestServer(base_url=base_url, card=card)
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5.0)
