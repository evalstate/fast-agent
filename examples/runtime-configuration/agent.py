import asyncio

from fast_agent import FastAgent
from fast_agent.core import Core
from fast_agent.mcp_server_registry import ServerRegistry


def change_env_of_server(server_name: str, env_name: str, new_env: str):
    def change_server_registry(server_registry: ServerRegistry):
        server_config = server_registry.registry[server_name]
        if server_config.env is None:
            server_config.env = {}
        server_config.env[env_name] = new_env
        return server_registry

    def inner(core: Core):
        if core._context is None:
            raise ValueError("Context is not initialized")
        registry = core._context.server_registry
        if registry is None:
            raise ValueError("Server registry is not initialized")
        registry = change_server_registry(registry)
        core._context.server_registry = registry
        return core

    return inner


fast = FastAgent("RAG Application")


@fast.agent(servers=["env_get_server"])
async def main(token: str) -> None:
    change_fn = change_env_of_server("env_get_server", "TOKEN", token)
    async with fast.run(core_modifiers=[change_fn]) as agent:
        await agent.default.generate("What is the value of 'TOKEN' env variable?")


if __name__ == "__main__":
    asyncio.run(main("CUSTOM_TOKEN_VALUE_1"))
    asyncio.run(main("CUSTOM_TOKEN_VALUE_2"))
