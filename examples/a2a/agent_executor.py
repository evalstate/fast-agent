from __future__ import annotations

from typing import TYPE_CHECKING

from a2a.server.agent_execution import AgentExecutor
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Part,
)

from fast_agent import AgentHarness, AgentRequest, FastAgent

if TYPE_CHECKING:
    from a2a.server.agent_execution import RequestContext
    from a2a.server.events import EventQueue

DEFAULT_AGENT_NAME = "helper"

fast = FastAgent(
    "A2A fast-agent Demo",
    parse_cli_args=False,
    quiet=True,
)


@fast.agent(
    name=DEFAULT_AGENT_NAME,
    instruction="You are a helpful AI agent answering incoming A2A messages.",
    default=True,
)
async def helper() -> None:
    """Default agent registered with FastAgent."""
    pass


class A2AHarnessAdapter:
    """Translate A2A request context into the protocol-neutral harness API."""

    def __init__(
        self,
        harness: AgentHarness,
        *,
        default_agent_name: str = DEFAULT_AGENT_NAME,
    ) -> None:
        self._harness = harness
        self._default_agent_name = default_agent_name

    async def invoke(self, context: RequestContext) -> str:
        request = AgentRequest.text(
            context.get_user_input().strip(),
            agent=self._default_agent_name,
            session_id=context.context_id,
            metadata={"a2a_task_id": context.task_id or ""},
        )
        response = await self._harness.invoke(request)
        return response.text_content()


class FastAgentExecutor(AgentExecutor):
    """A2A AgentExecutor that proxies requests to a FastAgent harness."""

    def __init__(self, adapter: A2AHarnessAdapter) -> None:
        self._adapter = adapter

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        if not context.message or not context.task_id or not context.context_id:
            return

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        )
        await updater.start_work(
            message=updater.new_agent_message(parts=[Part(text="fast-agent is working")])
        )
        response_text = await self._adapter.invoke(context)
        await updater.add_artifact(
            parts=[Part(text=response_text)],
            name="response",
            append=False,
            last_chunk=True,
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.task_id or not context.context_id:
            return
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        )
        await updater.cancel()


def agent_card(*, host: str, port: int) -> AgentCard:
    base_url = f"http://{_url_host(host)}:{port}"
    return AgentCard(
        name="fast-agent A2A harness demo",
        description="A fast-agent harness exposed through an explicit A2A adapter.",
        provider=AgentProvider(organization="fast-agent", url="https://fast-agent.ai"),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id=DEFAULT_AGENT_NAME,
                name=DEFAULT_AGENT_NAME,
                description="Send a message to the helper fast-agent agent.",
                tags=["fast-agent", "helper"],
                examples=["Hello"],
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


def _url_host(bind_host: str) -> str:
    if bind_host in {"0.0.0.0", "::", ""}:
        return "localhost"
    if ":" in bind_host and not bind_host.startswith("["):
        return f"[{bind_host}]"
    return bind_host
