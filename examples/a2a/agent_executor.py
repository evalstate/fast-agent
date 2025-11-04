from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, cast

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.utils import new_agent_text_message

from fast_agent import FastAgent

if TYPE_CHECKING:
    from a2a.server.events import EventQueue
    from a2a.types import AgentCard

    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import AgentProtocol

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


class FastAgentExecutor(AgentExecutor):
    """AgentExecutor that proxies requests to a FastAgent runtime."""

    def __init__(self, default_agent_name: str = DEFAULT_AGENT_NAME) -> None:
        self._stack = AsyncExitStack()
        self._agents: AgentApp | None = None
        self._default_agent_name = default_agent_name

    async def _agents_app(self) -> AgentApp:
        """Ensure the FastAgent runtime is running and return the AgentApp."""
        if self._agents is None:
            self._agents = await self._stack.enter_async_context(fast.run())
        return self._agents

    async def _agent(self, agent_name: str | None = None) -> AgentProtocol:
        """Return the requested agent or fall back to the default agent."""
        app = await self._agents_app()
        agents_map = cast("dict[str, AgentProtocol]", getattr(app, "_agents"))

        if agent_name and agent_name in agents_map:
            return agents_map[agent_name]

        if self._default_agent_name and self._default_agent_name in agents_map:
            return agents_map[self._default_agent_name]

        for candidate in agents_map.values():
            config = getattr(candidate, "config", None)
            if config is not None and getattr(config, "default", False):
                return candidate

        return next(iter(agents_map.values()))

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        message = context.get_user_input().strip()
        if not message:
            return

        agent = await self._agent()
        response = await agent.send(message)
        await event_queue.enqueue_event(new_agent_text_message(response))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")

    async def agent_card(self, agent_name: str | None = None) -> AgentCard:
        """Return the FastAgent-provided AgentCard for the given agent."""
        agent = await self._agent(agent_name)
        return await agent.agent_card()

    async def shutdown(self) -> None:
        """Close the FastAgent runtime."""
        await self._stack.aclose()
        self._agents = None
