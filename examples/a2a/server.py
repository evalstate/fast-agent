import asyncio

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, TransportProtocol
from agent_executor import FastAgentExecutor
from uvicorn import Config, Server

HOST = "0.0.0.0"
PORT = 9999


def _with_server_url(card: AgentCard) -> AgentCard:
    """Inject JSON-RPC transport details while preserving FastAgent metadata."""
    base_url = f"http://localhost:{PORT}/"
    return card.model_copy(
        update={
            "url": base_url,
            "preferred_transport": TransportProtocol.jsonrpc,
            "additional_interfaces": [],
            "supports_authenticated_extended_card": False,
            "default_input_modes": ["text"],
            "default_output_modes": ["text"],
        }
    )


async def main() -> None:
    executor = FastAgentExecutor()
    agent_card = _with_server_url(await executor.agent_card())

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    config = Config(server.build(), host=HOST, port=PORT)
    uvicorn_server = Server(config)

    try:
        await uvicorn_server.serve()
    finally:
        await executor.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
