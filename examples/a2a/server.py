import asyncio

import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes, create_rest_routes
from a2a.server.routes.common import DefaultServerCallContextBuilder
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from agent_executor import A2AHarnessAdapter, FastAgentExecutor, agent_card, fast
from fastapi import FastAPI

HOST = "0.0.0.0"
PORT = 9999


async def main() -> None:
    card = agent_card(host=HOST, port=PORT)
    async with fast.harness() as harness:
        request_handler = DefaultRequestHandler(
            agent_executor=FastAgentExecutor(A2AHarnessAdapter(harness)),
            task_store=InMemoryTaskStore(),
            agent_card=card,
        )
        context_builder = DefaultServerCallContextBuilder()
        app = FastAPI(title=card.name)
        app.routes.extend(create_agent_card_routes(agent_card=card))
        app.routes.extend(
            create_jsonrpc_routes(
                request_handler=request_handler,
                rpc_url="/a2a/jsonrpc",
                context_builder=context_builder,
            )
        )
        app.routes.extend(
            create_rest_routes(
                request_handler=request_handler,
                path_prefix="/a2a/rest",
                context_builder=context_builder,
            )
        )

        server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PORT, log_level="warning"))
        await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
