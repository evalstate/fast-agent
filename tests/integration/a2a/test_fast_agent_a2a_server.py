from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import pytest
import pytest_asyncio
import uvicorn
from mcp.types import TextContent

from fast_agent.a2a.config import A2AAgentConfig
from fast_agent.a2a.remote_agent import A2ARemoteAgent
from fast_agent.a2a.server import AgentA2AServer
from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.core.agent_app import AgentApp
from fast_agent.core.fastagent import AgentInstance
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.types import LlmStopReason, PromptMessageExtended

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fast_agent.interfaces import AgentProtocol


@dataclass
class RecordingAgent:
    name: str = "worker"
    agent_type: AgentType = AgentType.BASIC
    message_history: list[PromptMessageExtended] = field(default_factory=list)
    received: list[PromptMessageExtended] = field(default_factory=list)
    config: AgentConfig = field(init=False)

    def __post_init__(self) -> None:
        self.config = AgentConfig(
            name=self.name,
            agent_type=self.agent_type,
            default=True,
            use_history=True,
        )

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def set_model(self, model: str | None) -> None:
        del model

    def clear(self, *, clear_prompts: bool = False) -> None:
        del clear_prompts
        self.message_history.clear()

    def pop_last_message(self) -> PromptMessageExtended | None:
        return self.message_history.pop() if self.message_history else None

    async def __call__(self, message: Any) -> str:
        return await self.send(message)

    async def send(self, message: Any, request_params: Any = None) -> str:
        response = await self.generate(message, request_params=request_params)
        return response.all_text()

    async def generate(self, messages: Any, request_params: Any = None) -> PromptMessageExtended:
        del request_params
        if isinstance(messages, PromptMessageExtended):
            prompt = messages
        else:
            prompt = PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text=str(messages))],
            )
        self.received.append(prompt)
        self.message_history.append(prompt)
        response = PromptMessageExtended(
            role="assistant",
            content=[
                TextContent(
                    type="text",
                    text=f"server saw {len(self.message_history)}: {prompt.all_text()}",
                )
            ],
        )
        self.message_history.append(response)
        return response

    async def structured(self, messages: Any, model: type, request_params: Any = None) -> tuple:
        del model
        return None, await self.generate(messages, request_params=request_params)


class StreamingRecordingAgent(RecordingAgent):
    def __init__(self, name: str = "worker") -> None:
        super().__init__(name=name)
        self._stream_listeners: list[Any] = []

    def add_stream_listener(self, listener: Any) -> Any:
        self._stream_listeners.append(listener)

        def remove_listener() -> None:
            if listener in self._stream_listeners:
                self._stream_listeners.remove(listener)

        return remove_listener

    def add_tool_stream_listener(self, listener: Any) -> Any:
        del listener

        def remove_listener() -> None:
            return None

        return remove_listener

    async def generate(self, messages: Any, request_params: Any = None) -> PromptMessageExtended:
        del request_params
        if isinstance(messages, PromptMessageExtended):
            prompt = messages
        else:
            prompt = PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text=str(messages))],
            )
        self.received.append(prompt)
        self.message_history.append(prompt)
        for text in ("stream ", "from ", "server"):
            for listener in list(self._stream_listeners):
                listener(StreamChunk(text=text))
            await asyncio.sleep(0.05)
        response = PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="stream from server")],
        )
        self.message_history.append(response)
        return response


class InputRequiredRecordingAgent(RecordingAgent):
    waiting_for_input: bool = False

    async def generate(self, messages: Any, request_params: Any = None) -> PromptMessageExtended:
        del request_params
        if isinstance(messages, PromptMessageExtended):
            prompt = messages
        else:
            prompt = PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text=str(messages))],
            )
        self.received.append(prompt)
        self.message_history.append(prompt)
        if not self.waiting_for_input:
            self.waiting_for_input = True
            response = PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="Please provide the missing value.")],
                stop_reason=LlmStopReason.PAUSE,
            )
            self.message_history.append(response)
            return response

        self.waiting_for_input = False
        response = PromptMessageExtended(
            role="assistant",
            content=[
                TextContent(
                    type="text",
                    text=f"input received: {prompt.all_text()}",
                )
            ],
            stop_reason=LlmStopReason.END_TURN,
        )
        self.message_history.append(response)
        return response


@dataclass(frozen=True)
class RunningFastAgentA2AServer:
    base_url: str
    server: AgentA2AServer
    created_agents: list[RecordingAgent]


def _instance(agent: RecordingAgent) -> AgentInstance:
    protocol_agent = cast("AgentProtocol", agent)
    return AgentInstance(
        app=AgentApp({agent.name: protocol_agent}),
        agents={agent.name: protocol_agent},
    )


@pytest_asyncio.fixture
async def fast_agent_a2a_server(
    unused_tcp_port: int,
    wait_for_port,
) -> AsyncIterator[RunningFastAgentA2AServer]:
    host = "127.0.0.1"
    port = unused_tcp_port
    created_agents: list[RecordingAgent] = []
    disposed: list[AgentInstance] = []

    async def create_instance() -> AgentInstance:
        agent = RecordingAgent(name="worker")
        created_agents.append(agent)
        return _instance(agent)

    async def dispose_instance(instance: AgentInstance) -> None:
        disposed.append(instance)
        await instance.shutdown()

    bootstrap = _instance(RecordingAgent(name="worker"))
    server = AgentA2AServer(
        primary_instance=bootstrap,
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        server_name="fast-agent test server",
        host=host,
        port=port,
    )
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(server.asgi_app(), host=host, port=port, log_level="warning")
    )
    task = asyncio.create_task(uvicorn_server.serve())
    await wait_for_port(host, port, timeout=5.0)

    try:
        yield RunningFastAgentA2AServer(
            base_url=f"http://{host}:{port}",
            server=server,
            created_agents=created_agents,
        )
    finally:
        uvicorn_server.should_exit = True
        await asyncio.wait_for(task, timeout=5.0)
        await server.executor.shutdown()
        assert disposed


@pytest_asyncio.fixture
async def streaming_fast_agent_a2a_server(
    unused_tcp_port: int,
    wait_for_port,
) -> AsyncIterator[RunningFastAgentA2AServer]:
    host = "127.0.0.1"
    port = unused_tcp_port
    created_agents: list[RecordingAgent] = []
    disposed: list[AgentInstance] = []

    async def create_instance() -> AgentInstance:
        agent = StreamingRecordingAgent(name="worker")
        created_agents.append(agent)
        return _instance(agent)

    async def dispose_instance(instance: AgentInstance) -> None:
        disposed.append(instance)
        await instance.shutdown()

    bootstrap = _instance(StreamingRecordingAgent(name="worker"))
    server = AgentA2AServer(
        primary_instance=bootstrap,
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        server_name="fast-agent streaming test server",
        host=host,
        port=port,
    )
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(server.asgi_app(), host=host, port=port, log_level="warning")
    )
    task = asyncio.create_task(uvicorn_server.serve())
    await wait_for_port(host, port, timeout=5.0)

    try:
        yield RunningFastAgentA2AServer(
            base_url=f"http://{host}:{port}",
            server=server,
            created_agents=created_agents,
        )
    finally:
        uvicorn_server.should_exit = True
        await asyncio.wait_for(task, timeout=5.0)
        await server.executor.shutdown()
        assert disposed


@pytest_asyncio.fixture
async def input_required_fast_agent_a2a_server(
    unused_tcp_port: int,
    wait_for_port,
) -> AsyncIterator[RunningFastAgentA2AServer]:
    host = "127.0.0.1"
    port = unused_tcp_port
    created_agents: list[RecordingAgent] = []
    disposed: list[AgentInstance] = []

    async def create_instance() -> AgentInstance:
        agent = InputRequiredRecordingAgent(name="worker")
        created_agents.append(agent)
        return _instance(agent)

    async def dispose_instance(instance: AgentInstance) -> None:
        disposed.append(instance)
        await instance.shutdown()

    bootstrap = _instance(InputRequiredRecordingAgent(name="worker"))
    server = AgentA2AServer(
        primary_instance=bootstrap,
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        server_name="fast-agent input required test server",
        host=host,
        port=port,
    )
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(server.asgi_app(), host=host, port=port, log_level="warning")
    )
    task = asyncio.create_task(uvicorn_server.serve())
    await wait_for_port(host, port, timeout=5.0)

    try:
        yield RunningFastAgentA2AServer(
            base_url=f"http://{host}:{port}",
            server=server,
            created_agents=created_agents,
        )
    finally:
        uvicorn_server.should_exit = True
        await asyncio.wait_for(task, timeout=5.0)
        await server.executor.shutdown()
        assert disposed


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fast_agent_a2a_server_serves_jsonrpc_agent_with_context_sessions(
    fast_agent_a2a_server: RunningFastAgentA2AServer,
) -> None:
    client = A2ARemoteAgent(
        config=AgentConfig(name="remote", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url=fast_agent_a2a_server.base_url, transport="JSONRPC"),
    )
    await client.initialize()
    try:
        first = await client.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text="first")],
                )
            ]
        )
        second = await client.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text="second")],
                )
            ]
        )
    finally:
        await client.shutdown()

    assert first.all_text() == "server saw 1: first"
    assert second.all_text() == "server saw 3: second"
    assert len(fast_agent_a2a_server.created_agents) == 1
    assert fast_agent_a2a_server.server.agent_card.name == "fast-agent test server"
    assert {
        interface.protocol_binding
        for interface in fast_agent_a2a_server.server.agent_card.supported_interfaces
    } == {"JSONRPC", "HTTP+JSON"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fast_agent_a2a_server_serves_http_json_transport(
    fast_agent_a2a_server: RunningFastAgentA2AServer,
) -> None:
    client = A2ARemoteAgent(
        config=AgentConfig(name="remote_http", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url=fast_agent_a2a_server.base_url, transport="HTTP+JSON"),
    )
    await client.initialize()
    try:
        response = await client.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text="over rest")],
                )
            ]
        )
    finally:
        await client.shutdown()

    assert response.all_text() == "server saw 1: over rest"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fast_agent_a2a_server_streams_live_artifact_updates_to_client(
    streaming_fast_agent_a2a_server: RunningFastAgentA2AServer,
) -> None:
    client = A2ARemoteAgent(
        config=AgentConfig(name="remote_stream", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(
            url=streaming_fast_agent_a2a_server.base_url,
            transport="JSONRPC",
        ),
    )
    chunks: list[str] = []
    client.add_stream_listener(lambda chunk: chunks.append(chunk.text))
    await client.initialize()
    try:
        response = await client.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text="stream please")],
                )
            ]
        )
    finally:
        await client.shutdown()

    assert chunks == ["stream ", "from ", "server"]
    assert response.all_text() == "stream from server"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fast_agent_a2a_server_preserves_input_required_task_for_follow_up(
    input_required_fast_agent_a2a_server: RunningFastAgentA2AServer,
) -> None:
    client = A2ARemoteAgent(
        config=AgentConfig(name="remote_input", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(
            url=input_required_fast_agent_a2a_server.base_url,
            transport="JSONRPC",
        ),
    )
    await client.initialize()
    try:
        first = await client.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text="need input")],
                )
            ]
        )
        input_task_id = client.current_task_id
        second = await client.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text="blue")],
                )
            ]
        )
    finally:
        await client.shutdown()

    assert first.all_text() == "A2A task TASK_STATE_INPUT_REQUIRED: Please provide the missing value."
    assert input_task_id
    assert "input received: blue" in second.all_text()
    assert client.current_task_id is None
    assert client.last_task_state == "TASK_STATE_COMPLETED"
    assert len(input_required_fast_agent_a2a_server.created_agents) == 1
