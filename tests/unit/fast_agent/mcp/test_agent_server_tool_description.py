import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Protocol, cast

from mcp.types import BlobResourceContents, EmbeddedResource, ImageContent, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.core.agent_app import AgentApp
from fast_agent.core.fastagent import AgentInstance
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.server.agent_server import AgentMCPServer, _history_to_fastmcp_messages

if TYPE_CHECKING:
    from fastmcp.tools import FunctionTool

    from fast_agent.interfaces import AgentProtocol


class _DummyLLM:
    def __init__(self) -> None:
        self.message_history: list = []


class _DummyAgent:
    def __init__(self) -> None:
        self._llm = _DummyLLM()

    @property
    def message_history(self) -> list:
        return self._llm.message_history

    async def send(self, message: str) -> str:
        return message

    async def shutdown(self) -> None:
        return None


class _DescribedAgent(_DummyAgent):
    def __init__(self) -> None:
        super().__init__()
        self.config = AgentConfig("worker", description="Config description")


class _SessionWithExitStack(Protocol):
    _exit_stack: AsyncExitStack


def _mcp_context(session: _SessionWithExitStack) -> Any:
    """Create a minimal MCP-like context for instance-scoping tests."""
    request_context = SimpleNamespace(
        request_id="test-request",
        meta=None,
        session=cast("object", session),
        request=SimpleNamespace(headers={}),
    )
    return SimpleNamespace(session=session, request_context=request_context)


def _tool(server: AgentMCPServer, name: str):
    return cast("FunctionTool", asyncio.run(server.mcp_server.get_tool(name)))


def _prompt_names(server: AgentMCPServer) -> set[str]:
    return {prompt.name for prompt in asyncio.run(server.mcp_server.list_prompts())}


async def _prompt_names_async(server: AgentMCPServer) -> set[str]:
    return {prompt.name for prompt in await server.mcp_server.list_prompts()}


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

    assert "worker_history" in _prompt_names(server)
    assert _tool(server, "worker").description == "Use worker"


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

    assert "writer_history" in _prompt_names(server)
    assert _tool(server, "writer").description == "Custom text"


def test_tool_description_uses_agent_config_description_when_available():
    async def create_instance() -> AgentInstance:
        agent = cast("AgentProtocol", _DescribedAgent())
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
    )

    assert _tool(server, "worker").description == "Config description"


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

    assert _tool(server, "worker_send").name == "worker_send"


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

    assert await _prompt_names_async(server) == set()

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

    assert "worker_history" in await _prompt_names_async(server)

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


def test_history_prompt_wraps_images_as_embedded_resources():
    history = [
        PromptMessageExtended(
            role="user",
            content=[
                TextContent(type="text", text="Describe this image."),
                ImageContent(type="image", data="base64-image", mimeType="image/png"),
            ],
        )
    ]

    messages = _history_to_fastmcp_messages(history)

    assert isinstance(messages[0].content, TextContent)
    assert messages[0].content.text == "Describe this image."
    assert isinstance(messages[1].content, EmbeddedResource)
    assert isinstance(messages[1].content.resource, BlobResourceContents)
    assert messages[1].content.resource.blob == "base64-image"
    assert messages[1].content.resource.mimeType == "image/png"


def test_shutdown_continues_after_dispose_failure():
    asyncio.run(_exercise_shutdown_failure_resilience())


async def _exercise_shutdown_failure_resilience():
    dispose_attempts: list[str] = []
    instance_labels: dict[int, str] = {}

    async def create_instance(name: str) -> AgentInstance:
        agent = cast("AgentProtocol", _DummyAgent())
        app = AgentApp({"worker": agent})
        instance = AgentInstance(app=app, agents={"worker": agent})
        instance_labels[id(instance)] = name
        return instance

    async def dispose_instance(instance: AgentInstance) -> None:
        label = instance_labels[id(instance)]
        dispose_attempts.append(label)
        if label in {"connection", "primary"}:
            raise RuntimeError(f"dispose failed for {label}")
        await instance.shutdown()

    primary_instance = await create_instance("primary")
    server = AgentMCPServer(
        primary_instance=primary_instance,
        create_instance=lambda: create_instance("ephemeral"),
        dispose_instance=dispose_instance,
        instance_scope="shared",
        tool_description="Shared scope",
    )

    server._connection_instances[1] = await create_instance("connection")
    server._stale_instances.append(await create_instance("stale"))

    await server.shutdown()

    assert dispose_attempts == ["connection", "primary", "stale"]
