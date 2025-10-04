import pytest

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


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.xfail(reason="Environment variables are not set when we initialize the agent app")
async def test_runtime_configuration_no_change(fast_agent):
    """Test if environment variables are correctly set when we do nothing"""
    fast = fast_agent

    @fast.agent(servers=["env_get_server"])
    async def agent_no_change():
        async with fast.run():
            env = fast.app.server_registry.registry["env_get_server"].env
            assert env == {"TOKEN": "DEFAULT_TOKEN"}

    await agent_no_change()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_runtime_configuration_change_env(fast_agent):
    """Test if environment variables are correctly set when we change them at runtime"""
    fast = fast_agent

    @fast.agent(servers=["env_get_server"])
    async def agent_change_env():
        new_value = "NEW_TOKEN"
        change_fn = change_env_of_server("env_get_server", "TOKEN", new_value)

        async with fast.run(core_modifiers=[change_fn]):
            env = fast.app.server_registry.registry["env_get_server"].env
            assert env == {"TOKEN": new_value}

    await agent_change_env()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.xfail(reason="Environment variables are not set when we initialize the agent app")
async def test_runtime_configuration_add_env(fast_agent):
    """Test if environment variables are correctly set when we add a new one"""
    fast = fast_agent

    @fast.agent(servers=["env_get_server"])
    async def agent_add_env():
        new_value = "NEW_TOKEN"
        change_fn = change_env_of_server("env_get_server", "OTHER_TOKEN", new_value)

        async with fast.run(core_modifiers=[change_fn]):
            env = fast.app.server_registry.registry["env_get_server"].env
            assert env == {"TOKEN": "DEFAULT_TOKEN", "OTHER_TOKEN": new_value}

    await agent_add_env()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_runtime_configuration_many_apps(fast_agent):
    """Test if environment variables are correctly set when we have more than one application"""
    fast = fast_agent

    @fast.agent(servers=["env_get_server"])
    async def agent_app(env_name, env_value):
        change_fn = change_env_of_server("env_get_server", env_name, env_value)

        async with fast.run(core_modifiers=[change_fn]):
            env = fast.app.server_registry.registry["env_get_server"].env
            expected_env = {**env, env_name: env_value}
            assert env == expected_env

    await agent_app("NEW_ENV1", "VALUE_ONE")
    await agent_app("NEW_ENV2", "VALUE_TWO")
