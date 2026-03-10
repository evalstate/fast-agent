import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Protocol, cast

from mcp import types
from mcp.server.fastmcp import Context as MCPContext
from mcp.server.fastmcp.server import RequestContext

from fast_agent.core.agent_app import AgentApp
from fast_agent.core.fastagent import AgentInstance
from fast_agent.mcp.server.agent_server import AgentMCPServer

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


class _DummyLLM:
    def __init__(self) -> None:
        self.message_history: list = []


class _DummyAgent:
    def __init__(self) -> None:
        self._llm = _DummyLLM()

    async def send(self, message: str) -> str:
        return message

    async def shutdown(self) -> None:
        return None


class _SessionWithExitStack(Protocol):
    _exit_stack: AsyncExitStack


def _mcp_context(session: _SessionWithExitStack) -> MCPContext:
    """Create a minimal typed MCP context for server instance-scoping tests."""
    request = SimpleNamespace(headers={})
    request_context = RequestContext(
        request_id="test-request",
        meta=None,
        session=cast("Any", session),
        lifespan_context=None,
        request=request,
    )
    return MCPContext(request_context=request_context)


def _assert_prompts_enabled(server: AgentMCPServer) -> None:
    handlers = server.mcp_server._mcp_server.request_handlers
    assert types.ListPromptsRequest in handlers
    assert types.GetPromptRequest in handlers


def _assert_prompts_disabled(server: AgentMCPServer) -> None:
    handlers = server.mcp_server._mcp_server.request_handlers
    assert types.ListPromptsRequest not in handlers
    assert types.GetPromptRequest not in handlers


def _assert_resources_disabled(server: AgentMCPServer) -> None:
    handlers = server.mcp_server._mcp_server.request_handlers
    assert types.ListResourcesRequest not in handlers
    assert types.ReadResourceRequest not in handlers
    assert types.ListResourceTemplatesRequest not in handlers


def test_tool_description_supports_agent_placeholder():
    async def create_instance() -> AgentInstance:
        agent = cast("AgentProtocol", _DummyAgent())
        app = AgentApp({"worker": agent})
        return AgentInstance(app=app, agents={"worker": agent})

    async def dispose_instance(instance: AgentInstance) -> None:
        await instance.shutdown()

    primary_instance = asyncio.run(create_instance())

    server = AgentMCPServer(
        primary_instance=primary_instance,
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        instance_scope="shared",
        tool_description="Use {agent}",
    )

    _assert_prompts_enabled(server)
    _assert_resources_disabled(server)

    tool = server.mcp_server._tool_manager._tools["worker"]
    assert tool.description == "Use worker"


def test_tool_description_defaults_when_not_provided():
    async def create_instance() -> AgentInstance:
        agent = cast("AgentProtocol", _DummyAgent())
        app = AgentApp({"writer": agent})
        return AgentInstance(app=app, agents={"writer": agent})

    async def dispose_instance(instance: AgentInstance) -> None:
        await instance.shutdown()

    primary_instance = asyncio.run(create_instance())

    server = AgentMCPServer(
        primary_instance=primary_instance,
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        instance_scope="shared",
        tool_description="Custom text",
    )

    _assert_prompts_enabled(server)
    _assert_resources_disabled(server)

    tool = server.mcp_server._tool_manager._tools["writer"]
    assert tool.description == "Custom text"


def test_tool_name_template_overrides_default():
    async def create_instance() -> AgentInstance:
        agent = cast("AgentProtocol", _DummyAgent())
        app = AgentApp({"worker": agent})
        return AgentInstance(app=app, agents={"worker": agent})

    async def dispose_instance(instance: AgentInstance) -> None:
        await instance.shutdown()

    primary_instance = asyncio.run(create_instance())

    server = AgentMCPServer(
        primary_instance=primary_instance,
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        instance_scope="shared",
        tool_name_template="{agent}_send",
    )

    _assert_prompts_enabled(server)
    _assert_resources_disabled(server)

    assert "worker_send" in server.mcp_server._tool_manager._tools


def test_request_scope_creates_ephemeral_instances():
    asyncio.run(_exercise_request_scope())


async def _exercise_request_scope():
    create_count = 0
    dispose_count = 0

    async def create_instance() -> AgentInstance:
        nonlocal create_count
        create_count += 1
        agent = cast("AgentProtocol", _DummyAgent())
        app = AgentApp({"worker": agent})
        return AgentInstance(app=app, agents={"worker": agent})

    async def dispose_instance(instance: AgentInstance) -> None:
        nonlocal dispose_count
        dispose_count += 1
        await instance.shutdown()

    primary_instance = await create_instance()
    server = AgentMCPServer(
        primary_instance=primary_instance,
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        instance_scope="request",
        tool_description="Request scoped",
    )

    _assert_prompts_disabled(server)
    _assert_resources_disabled(server)

    inst_one = await server._acquire_instance(None)
    await server._release_instance(None, inst_one)
    inst_two = await server._acquire_instance(None)
    await server._release_instance(None, inst_two)

    assert inst_one is not inst_two
    assert create_count == 3  # primary + two ephemeral instances
    assert dispose_count == 2


def test_connection_scope_reuses_instance_until_session_close():
    asyncio.run(_exercise_connection_scope())


async def _exercise_connection_scope():
    create_count = 0
    dispose_count = 0

    async def create_instance() -> AgentInstance:
        nonlocal create_count
        create_count += 1
        agent = cast("AgentProtocol", _DummyAgent())
        app = AgentApp({"worker": agent})
        return AgentInstance(app=app, agents={"worker": agent})

    async def dispose_instance(instance: AgentInstance) -> None:
        nonlocal dispose_count
        dispose_count += 1
        await instance.shutdown()

    primary_instance = await create_instance()
    server = AgentMCPServer(
        primary_instance=primary_instance,
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        instance_scope="connection",
        tool_description="Connection scoped",
    )

    _assert_prompts_enabled(server)
    _assert_resources_disabled(server)

    class _DummySession:
        def __init__(self) -> None:
            self._exit_stack = AsyncExitStack()

    session = _DummySession()
    await session._exit_stack.__aenter__()

    ctx = _mcp_context(session)

    inst_one = await server._acquire_instance(ctx)
    inst_two = await server._acquire_instance(ctx)

    assert inst_one is inst_two
    assert create_count == 2  # primary + connection instance
    await server._release_instance(ctx, inst_one)
    assert dispose_count == 0

    await session._exit_stack.aclose()
    assert dispose_count == 1
