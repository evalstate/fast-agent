from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
import pytest_asyncio
import uvicorn
from a2a.client import ClientConfig, create_client
from a2a.types import (
    CancelTaskRequest,
    GetTaskRequest,
    ListTasksRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
)
from fastapi.testclient import TestClient
from fastmcp.server.auth import AccessToken
from google.protobuf.json_format import MessageToDict
from mcp.types import (
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)
from pydantic import AnyUrl

from fast_agent.a2a.config import A2AAgentConfig
from fast_agent.a2a.remote_agent import A2ARemoteAgent
from fast_agent.a2a.server import (
    AgentA2AServer,
    _bearer_token_from_call_context,
    _parts_from_prompt_message,
    _prompt_from_a2a_message,
)
from fast_agent.a2a.task_api import (
    return_artifact as return_a2a_artifact,
)
from fast_agent.a2a.task_api import (
    return_message as return_a2a_message,
)
from fast_agent.a2a.task_api import (
    start_task as start_a2a_task,
)
from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.core.agent_app import AgentApp
from fast_agent.core.fastagent import AgentInstance
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.mcp.auth.context import request_bearer_token
from fast_agent.mcp.auth.presence import HuggingFaceTokenVerifier
from fast_agent.types import LlmStopReason, PromptMessageExtended

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fast_agent.interfaces import AgentProtocol


def _patch_hf_token_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    async def verify_token(self: HuggingFaceTokenVerifier, token: str) -> AccessToken | None:
        del self
        if token == "invalid-token":
            return None
        return AccessToken(token=token, client_id="test-client", scopes=["access"])

    monkeypatch.setattr(HuggingFaceTokenVerifier, "verify_token", verify_token)


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


class NoHistoryRecordingAgent(RecordingAgent):
    def __post_init__(self) -> None:
        self.config = AgentConfig(
            name=self.name,
            agent_type=self.agent_type,
            default=True,
            use_history=False,
        )

    async def generate(self, messages: Any, request_params: Any = None) -> PromptMessageExtended:
        use_history = request_params.use_history if request_params is not None else self.config.use_history
        if isinstance(messages, PromptMessageExtended):
            prompt = messages
        else:
            prompt = PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text=str(messages))],
            )
        self.received.append(prompt)
        history_len = len(self.message_history)
        response = PromptMessageExtended(
            role="assistant",
            content=[
                TextContent(
                    type="text",
                    text=f"server history {history_len}: {prompt.all_text()}",
                )
            ],
        )
        if use_history:
            self.message_history.append(prompt)
            self.message_history.append(response)
        return response


class NamedResponseAgent(RecordingAgent):
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
            content=[TextContent(type="text", text=f"{self.name} handled: {prompt.all_text()}")],
        )
        self.message_history.append(response)
        return response


class CancellableRecordingAgent(RecordingAgent):
    def __init__(self, name: str = "worker") -> None:
        super().__init__(name=name)
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def generate(self, messages: Any, request_params: Any = None) -> PromptMessageExtended:
        del messages, request_params
        self.started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="not cancelled")],
        )


class TokenEchoAgent(RecordingAgent):
    async def generate(self, messages: Any, request_params: Any = None) -> PromptMessageExtended:
        del messages, request_params
        return PromptMessageExtended(
            role="assistant",
            content=[
                TextContent(
                    type="text",
                    text=request_bearer_token.get() or "missing",
                )
            ],
        )


class ResearchIntakeAgent(RecordingAgent):
    a2a_defer_task_start = True

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
        text = prompt.all_text()
        if "unclear" in text:
            await return_a2a_message(
                "Please clarify the research goal, audience, and desired output format."
            )
            return PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="refinement requested")],
            )

        handle = await start_a2a_task("Research task accepted")
        await return_a2a_artifact(
            "Scoping sources\n",
            name="progress",
            artifact_id=f"{handle.task_id}:progress",
            last_chunk=False,
        )
        await return_a2a_artifact(
            "Reading primary references\n",
            name="progress",
            artifact_id=f"{handle.task_id}:progress",
            append=True,
            last_chunk=False,
        )
        return PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text=f"Research complete: {text}")],
        )


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


def _multi_agent_instance(*agents: RecordingAgent) -> AgentInstance:
    protocol_agents = {agent.name: cast("AgentProtocol", agent) for agent in agents}
    return AgentInstance(
        app=AgentApp(protocol_agents),
        agents=protocol_agents,
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
        config=AgentConfig(name="remote", agent_type=AgentType.A2A, use_history=True),
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
    skills = {skill.id: skill for skill in fast_agent_a2a_server.server.agent_card.skills}
    assert set(skills) == {"worker"}
    assert skills["worker"].name == "worker"
    assert skills["worker"].description == "Send a message to the worker fast-agent agent."
    assert list(skills["worker"].tags) == ["fast-agent", "basic"]
    assert list(fast_agent_a2a_server.server.agent_card.default_input_modes) == [
        "text/plain",
        "application/json",
        "application/octet-stream",
        "image/*",
    ]
    assert list(fast_agent_a2a_server.server.agent_card.default_output_modes) == [
        "text/plain",
        "application/json",
        "application/octet-stream",
        "image/*",
    ]
    assert list(skills["worker"].input_modes) == [
        "text/plain",
        "application/json",
        "application/octet-stream",
        "image/*",
    ]
    assert list(skills["worker"].output_modes) == [
        "text/plain",
        "application/json",
        "application/octet-stream",
        "image/*",
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a2a_remote_agent_without_history_uses_fresh_server_contexts(
    fast_agent_a2a_server: RunningFastAgentA2AServer,
) -> None:
    client = A2ARemoteAgent(
        config=AgentConfig(name="remote_no_history", agent_type=AgentType.A2A, use_history=False),
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
        first_context_id = client.context_id
        assert first_context_id
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
    assert second.all_text() == "server saw 1: second"
    assert client.context_id != first_context_id
    assert len(fast_agent_a2a_server.created_agents) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fast_agent_a2a_server_shared_instance_scope_reuses_primary_instance(
    unused_tcp_port: int,
    wait_for_port,
) -> None:
    host = "127.0.0.1"
    port = unused_tcp_port
    created_agents: list[RecordingAgent] = []

    async def create_instance() -> AgentInstance:
        agent = RecordingAgent(name="worker")
        created_agents.append(agent)
        return _instance(agent)

    async def dispose_instance(instance: AgentInstance) -> None:
        await instance.shutdown()

    server = AgentA2AServer(
        primary_instance=_instance(RecordingAgent(name="worker")),
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        server_name="fast-agent shared scope test server",
        host=host,
        port=port,
        instance_scope="shared",
    )
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(server.asgi_app(), host=host, port=port, log_level="warning")
    )
    task = asyncio.create_task(uvicorn_server.serve())
    await wait_for_port(host, port, timeout=5.0)

    client = A2ARemoteAgent(
        config=AgentConfig(name="remote_shared", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url=f"http://{host}:{port}", transport="JSONRPC"),
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
        uvicorn_server.should_exit = True
        await asyncio.wait_for(task, timeout=5.0)
        await server.executor.shutdown()

    assert first.all_text() == "server saw 1: first"
    assert second.all_text() == "server saw 3: second"
    assert not created_agents


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fast_agent_a2a_server_request_instance_scope_disposes_each_turn(
    unused_tcp_port: int,
    wait_for_port,
) -> None:
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

    server = AgentA2AServer(
        primary_instance=_instance(RecordingAgent(name="worker")),
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        server_name="fast-agent request scope test server",
        host=host,
        port=port,
        instance_scope="request",
    )
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(server.asgi_app(), host=host, port=port, log_level="warning")
    )
    task = asyncio.create_task(uvicorn_server.serve())
    await wait_for_port(host, port, timeout=5.0)

    client = A2ARemoteAgent(
        config=AgentConfig(name="remote_request", agent_type=AgentType.A2A, use_history=True),
        a2a_config=A2AAgentConfig(url=f"http://{host}:{port}", transport="JSONRPC"),
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
        uvicorn_server.should_exit = True
        await asyncio.wait_for(task, timeout=5.0)
        await server.executor.shutdown()

    assert first.all_text() == "server saw 1: first"
    assert second.all_text() == "server saw 1: second"
    assert len(created_agents) == 2
    assert len(disposed) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fast_agent_a2a_server_routes_to_agent_skill_named_in_metadata(
    unused_tcp_port: int,
    wait_for_port,
) -> None:
    host = "127.0.0.1"
    port = unused_tcp_port
    created_agents: list[tuple[NamedResponseAgent, NamedResponseAgent]] = []
    disposed: list[AgentInstance] = []

    def agent_pair() -> tuple[NamedResponseAgent, NamedResponseAgent]:
        primary = NamedResponseAgent(name="primary")
        primary.config.default = True
        specialist = NamedResponseAgent(name="specialist")
        specialist.config.default = False
        specialist.config.description = "Handle specialist work."
        return primary, specialist

    async def create_instance() -> AgentInstance:
        primary, specialist = agent_pair()
        created_agents.append((primary, specialist))
        return _multi_agent_instance(primary, specialist)

    async def dispose_instance(instance: AgentInstance) -> None:
        disposed.append(instance)
        await instance.shutdown()

    bootstrap_primary, bootstrap_specialist = agent_pair()
    server = AgentA2AServer(
        primary_instance=_multi_agent_instance(bootstrap_primary, bootstrap_specialist),
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        server_name="fast-agent routing test server",
        host=host,
        port=port,
    )
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(server.asgi_app(), host=host, port=port, log_level="warning")
    )
    task = asyncio.create_task(uvicorn_server.serve())
    await wait_for_port(host, port, timeout=5.0)

    http_client = httpx.AsyncClient()
    client = await create_client(
        f"http://{host}:{port}",
        client_config=ClientConfig(
            httpx_client=http_client,
            supported_protocol_bindings=["JSONRPC"],
        ),
    )
    response_text: str | None = None
    try:
        async for event in client.send_message(
            SendMessageRequest(
                message=Message(
                    role=Role.ROLE_USER,
                    message_id="target-specialist",
                    parts=[Part(text="route this")],
                    metadata={"agent": "specialist"},
                )
            )
        ):
            if event.HasField("artifact_update"):
                artifact_parts = event.artifact_update.artifact.parts
                if artifact_parts and artifact_parts[0].HasField("text"):
                    response_text = artifact_parts[0].text
    finally:
        await client.close()
        await http_client.aclose()
        uvicorn_server.should_exit = True
        await asyncio.wait_for(task, timeout=5.0)
        await server.executor.shutdown()

    skills = {skill.id: skill for skill in server.agent_card.skills}
    assert set(skills) == {"primary", "specialist"}
    assert skills["specialist"].description == "Handle specialist work."
    assert response_text == "specialist handled: route this"
    assert created_agents
    primary, specialist = created_agents[0]
    assert not primary.received
    assert len(specialist.received) == 1
    assert disposed


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fast_agent_a2a_server_research_intake_can_refine_or_start_task(
    unused_tcp_port: int,
    wait_for_port,
) -> None:
    host = "127.0.0.1"
    port = unused_tcp_port

    async def create_instance() -> AgentInstance:
        return _instance(ResearchIntakeAgent(name="research"))

    async def dispose_instance(instance: AgentInstance) -> None:
        await instance.shutdown()

    server = AgentA2AServer(
        primary_instance=_instance(ResearchIntakeAgent(name="research")),
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        server_name="fast-agent research intake test server",
        host=host,
        port=port,
    )
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(server.asgi_app(), host=host, port=port, log_level="warning")
    )
    task = asyncio.create_task(uvicorn_server.serve())
    await wait_for_port(host, port, timeout=5.0)

    http_client = httpx.AsyncClient()
    client = await create_client(
        f"http://{host}:{port}",
        client_config=ClientConfig(
            httpx_client=http_client,
            supported_protocol_bindings=["JSONRPC"],
        ),
    )
    try:
        refinement_events = [
            event
            async for event in client.send_message(
                SendMessageRequest(
                    message=Message(
                        role=Role.ROLE_USER,
                        message_id="refine-research",
                        parts=[Part(text="unclear topic")],
                    )
                )
            )
        ]
        research_events = [
            event
            async for event in client.send_message(
                SendMessageRequest(
                    message=Message(
                        role=Role.ROLE_USER,
                        message_id="start-research",
                        parts=[Part(text="Research A2A task lifecycle for developers")],
                    )
                )
            )
        ]
    finally:
        await client.close()
        await http_client.aclose()
        uvicorn_server.should_exit = True
        await asyncio.wait_for(task, timeout=5.0)
        await server.executor.shutdown()

    assert len(refinement_events) == 1
    refinement = refinement_events[0]
    assert refinement.HasField("message")
    assert refinement.message.context_id
    assert refinement.message.task_id == ""
    assert refinement.message.parts[0].text == (
        "Please clarify the research goal, audience, and desired output format."
    )

    assert any(event.HasField("task") for event in research_events)
    status_states = [
        event.status_update.status.state
        for event in research_events
        if event.HasField("status_update")
    ]
    assert TaskState.TASK_STATE_WORKING in status_states
    assert status_states[-1] == TaskState.TASK_STATE_COMPLETED
    artifact_text = "\n".join(
        part.text
        for event in research_events
        if event.HasField("artifact_update")
        for part in event.artifact_update.artifact.parts
        if part.HasField("text")
    )
    assert "Scoping sources" in artifact_text
    assert "Reading primary references" in artifact_text
    assert "Research complete: Research A2A task lifecycle for developers" in artifact_text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fast_agent_a2a_server_context_does_not_force_agent_history(
    unused_tcp_port: int,
    wait_for_port,
) -> None:
    host = "127.0.0.1"
    port = unused_tcp_port
    created_agents: list[NoHistoryRecordingAgent] = []
    disposed: list[AgentInstance] = []

    async def create_instance() -> AgentInstance:
        agent = NoHistoryRecordingAgent(name="worker")
        created_agents.append(agent)
        return _instance(agent)

    async def dispose_instance(instance: AgentInstance) -> None:
        disposed.append(instance)
        await instance.shutdown()

    server = AgentA2AServer(
        primary_instance=_instance(NoHistoryRecordingAgent(name="worker")),
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        server_name="fast-agent no-history test server",
        host=host,
        port=port,
    )
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(server.asgi_app(), host=host, port=port, log_level="warning")
    )
    task = asyncio.create_task(uvicorn_server.serve())
    await wait_for_port(host, port, timeout=5.0)

    client = A2ARemoteAgent(
        config=AgentConfig(name="remote", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url=f"http://{host}:{port}", transport="JSONRPC"),
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
        uvicorn_server.should_exit = True
        await asyncio.wait_for(task, timeout=5.0)
        await server.executor.shutdown()

    assert first.all_text() == "server history 0: first"
    assert second.all_text() == "server history 0: second"
    assert len(created_agents) == 2
    assert disposed


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fast_agent_a2a_server_cancel_task_cancels_running_agent(
    unused_tcp_port: int,
    wait_for_port,
) -> None:
    host = "127.0.0.1"
    port = unused_tcp_port
    created_agents: list[CancellableRecordingAgent] = []
    disposed: list[AgentInstance] = []

    async def create_instance() -> AgentInstance:
        agent = CancellableRecordingAgent(name="worker")
        created_agents.append(agent)
        return _instance(agent)

    async def dispose_instance(instance: AgentInstance) -> None:
        disposed.append(instance)
        await instance.shutdown()

    server = AgentA2AServer(
        primary_instance=_instance(CancellableRecordingAgent(name="worker")),
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        server_name="fast-agent cancellation test server",
        host=host,
        port=port,
    )
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(server.asgi_app(), host=host, port=port, log_level="warning")
    )
    server_task = asyncio.create_task(uvicorn_server.serve())
    await wait_for_port(host, port, timeout=5.0)

    http_client = httpx.AsyncClient()
    client = await create_client(
        f"http://{host}:{port}",
        client_config=ClientConfig(
            httpx_client=http_client,
            supported_protocol_bindings=["JSONRPC"],
        ),
    )
    events: list[Any] = []
    stream_error: BaseException | None = None

    async def consume_stream() -> None:
        nonlocal stream_error
        try:
            async for event in client.send_message(
                SendMessageRequest(
                    message=Message(
                        role=Role.ROLE_USER,
                        message_id="cancel-me",
                        parts=[Part(text="please wait")],
                    )
                )
            ):
                events.append(event)
        except BaseException as exc:
            stream_error = exc

    stream_task = asyncio.create_task(consume_stream())
    try:
        deadline = asyncio.get_running_loop().time() + 5
        while not created_agents and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        assert created_agents
        await asyncio.wait_for(created_agents[0].started.wait(), timeout=5)
        task_id = next(event.task.id for event in events if event.HasField("task"))

        cancelled = await client.cancel_task(CancelTaskRequest(id=task_id))

        assert cancelled.status.state == TaskState.TASK_STATE_CANCELED
        await asyncio.wait_for(created_agents[0].cancelled.wait(), timeout=5)
        fetched = await client.get_task(GetTaskRequest(id=task_id))
        listed = await client.list_tasks(ListTasksRequest())
    finally:
        stream_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stream_task
        await client.close()
        await http_client.aclose()
        uvicorn_server.should_exit = True
        await asyncio.wait_for(server_task, timeout=5.0)
        await server.executor.shutdown()

    assert stream_error is None or isinstance(stream_error, asyncio.CancelledError)
    assert fetched.status.state == TaskState.TASK_STATE_CANCELED
    assert any(task.id == task_id and task.status.state == TaskState.TASK_STATE_CANCELED for task in listed.tasks)
    assert disposed


@pytest.mark.integration
def test_fast_agent_a2a_server_does_not_advertise_wildcard_bind_host() -> None:
    agent = RecordingAgent(name="worker")

    async def create_instance() -> AgentInstance:
        return _instance(RecordingAgent(name="worker"))

    async def dispose_instance(instance: AgentInstance) -> None:
        del instance

    server = AgentA2AServer(
        primary_instance=_instance(agent),
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        server_name="fast-agent wildcard test server",
        host="0.0.0.0",
        port=41241,
    )

    static_urls = {interface.url for interface in server.agent_card.supported_interfaces}
    assert static_urls == {
        "http://localhost:41241/a2a/jsonrpc",
        "http://localhost:41241/a2a/rest",
    }

    client = TestClient(server.asgi_app(), base_url="http://agent.example:41241")
    response = client.get("/.well-known/agent-card.json")
    response.raise_for_status()

    urls = {interface["url"] for interface in response.json()["supportedInterfaces"]}
    assert urls == {
        "http://agent.example:41241/a2a/jsonrpc",
        "http://agent.example:41241/a2a/rest",
    }


@pytest.mark.integration
def test_fast_agent_a2a_server_uses_public_url_env_for_dynamic_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = RecordingAgent(name="worker")

    async def create_instance() -> AgentInstance:
        return _instance(RecordingAgent(name="worker"))

    async def dispose_instance(instance: AgentInstance) -> None:
        del instance

    monkeypatch.setenv("FAST_AGENT_PUBLIC_URL", "https://agent.example")
    server = AgentA2AServer(
        primary_instance=_instance(agent),
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        server_name="fast-agent public URL test server",
        host="0.0.0.0",
        port=41241,
    )

    client = TestClient(server.asgi_app(), base_url="http://internal.example:41241")
    response = client.get("/.well-known/agent-card.json")
    response.raise_for_status()

    urls = {interface["url"] for interface in response.json()["supportedInterfaces"]}
    assert urls == {
        "https://agent.example/a2a/jsonrpc",
        "https://agent.example/a2a/rest",
    }


@pytest.mark.integration
def test_fast_agent_a2a_server_preserves_raw_file_input_parts() -> None:
    prompt = _prompt_from_a2a_message(
        Message(
            role=Role.ROLE_USER,
            message_id="file-input",
            parts=[
                Part(
                    raw=b"%PDF test bytes",
                    media_type="application/pdf",
                    filename="report.pdf",
                )
            ],
        )
    )

    assert len(prompt.content) == 1
    content = prompt.content[0]
    assert isinstance(content, EmbeddedResource)
    assert isinstance(content.resource, BlobResourceContents)
    assert str(content.resource.uri) == "attachment:///report.pdf"
    assert content.resource.mimeType == "application/pdf"
    assert content.resource.blob == "JVBERiB0ZXN0IGJ5dGVz"


@pytest.mark.integration
def test_fast_agent_a2a_server_maps_raw_image_input_parts() -> None:
    prompt = _prompt_from_a2a_message(
        Message(
            role=Role.ROLE_USER,
            message_id="image-input",
            parts=[
                Part(
                    raw=b"image bytes",
                    media_type="image/png",
                    filename="chart.png",
                )
            ],
        )
    )

    assert len(prompt.content) == 1
    content = prompt.content[0]
    assert isinstance(content, ImageContent)
    assert content.mimeType == "image/png"
    assert content.data == "aW1hZ2UgYnl0ZXM="


@pytest.mark.integration
def test_fast_agent_a2a_server_preserves_raw_audio_as_blob_resource() -> None:
    prompt = _prompt_from_a2a_message(
        Message(
            role=Role.ROLE_USER,
            message_id="audio-input",
            parts=[
                Part(
                    raw=b"audio bytes",
                    media_type="audio/wav",
                    filename="clip.wav",
                )
            ],
        )
    )

    assert len(prompt.content) == 1
    content = prompt.content[0]
    assert isinstance(content, EmbeddedResource)
    assert isinstance(content.resource, BlobResourceContents)
    assert str(content.resource.uri) == "attachment:///clip.wav"
    assert content.resource.mimeType == "audio/wav"
    assert content.resource.blob == "YXVkaW8gYnl0ZXM="


@pytest.mark.integration
def test_fast_agent_a2a_server_emits_blob_resources_as_raw_file_parts() -> None:
    parts = _parts_from_prompt_message(
        PromptMessageExtended(
            role="assistant",
            content=[
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri=AnyUrl("attachment:///report.pdf"),
                        mimeType="application/pdf",
                        blob="JVBERiB0ZXN0IGJ5dGVz",
                    ),
                )
            ],
        )
    )

    assert len(parts) == 1
    assert parts[0].raw == b"%PDF test bytes"
    assert parts[0].media_type == "application/pdf"
    assert parts[0].filename == "report.pdf"


@pytest.mark.integration
def test_fast_agent_a2a_server_emits_json_text_resources_as_data_parts() -> None:
    parts = _parts_from_prompt_message(
        PromptMessageExtended(
            role="assistant",
            content=[
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=AnyUrl("resource:///tickets.json"),
                        mimeType="application/json",
                        text='{"tickets": [{"id": "REQ123", "status": "open"}]}',
                    ),
                )
            ],
        )
    )

    assert len(parts) == 1
    assert parts[0].HasField("data")
    assert parts[0].media_type == "application/json"
    assert MessageToDict(parts[0])["data"] == {
        "tickets": [{"id": "REQ123", "status": "open"}]
    }


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
async def test_fast_agent_a2a_server_aggregates_live_artifact_updates_without_client_stream(
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

    assert chunks == []
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
    assert first.stop_reason == LlmStopReason.PAUSE
    assert input_task_id
    assert "input received: blue" in second.all_text()
    assert second.stop_reason == LlmStopReason.END_TURN
    assert client.current_task_id is None
    assert client.last_task_state == "TASK_STATE_COMPLETED"
    assert len(input_required_fast_agent_a2a_server.created_agents) == 1


@pytest.mark.integration
def test_fast_agent_a2a_server_hf_auth_card_and_rejection(monkeypatch) -> None:
    _patch_hf_token_verifier(monkeypatch)
    monkeypatch.setenv("FAST_AGENT_SERVE_OAUTH", "huggingface")
    monkeypatch.setenv("FAST_AGENT_OAUTH_RESOURCE_URL", "http://testserver")
    server = AgentA2AServer(
        primary_instance=_instance(TokenEchoAgent(name="worker")),
        create_instance=lambda: _async_instance(TokenEchoAgent(name="worker")),
        dispose_instance=_async_dispose_instance,
        server_name="fast-agent auth test server",
        host="127.0.0.1",
        port=41241,
    )
    client = TestClient(server.asgi_app(), base_url="http://testserver")

    card_response = client.get("/.well-known/agent-card.json")
    card_response.raise_for_status()
    payload = card_response.json()

    assert "hf_bearer" in payload["securitySchemes"]
    assert payload["securityRequirements"] == [{"schemes": {"hf_bearer": {}}}]
    assert payload["skills"][0]["securityRequirements"] == [{"schemes": {"hf_bearer": {}}}]

    rejected = client.post("/a2a/jsonrpc", json={})
    assert rejected.status_code == 401
    assert rejected.headers["www-authenticate"].startswith("Bearer ")

    invalid = client.post(
        "/a2a/jsonrpc",
        headers={"Authorization": "Bearer invalid-token"},
        json={},
    )
    assert invalid.status_code == 401
    assert invalid.headers["www-authenticate"].startswith("Bearer ")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Authorization": "Bearer request-token"}, "request-token"),
        ({"X-HF-Authorization": "Bearer hf-space-token"}, "hf-space-token"),
    ],
)
async def test_fast_agent_a2a_server_passes_bearer_token_to_request_context(
    monkeypatch,
    unused_tcp_port: int,
    wait_for_port,
    headers: dict[str, str],
    expected: str,
) -> None:
    _patch_hf_token_verifier(monkeypatch)
    monkeypatch.setenv("FAST_AGENT_SERVE_OAUTH", "huggingface")
    monkeypatch.setenv("FAST_AGENT_OAUTH_RESOURCE_URL", "http://127.0.0.1")
    host = "127.0.0.1"
    port = unused_tcp_port
    disposed: list[AgentInstance] = []

    async def create_instance() -> AgentInstance:
        return _instance(TokenEchoAgent(name="worker"))

    async def dispose_instance(instance: AgentInstance) -> None:
        disposed.append(instance)
        await instance.shutdown()

    server = AgentA2AServer(
        primary_instance=_instance(TokenEchoAgent(name="worker")),
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        server_name="fast-agent auth propagation test server",
        host=host,
        port=port,
    )
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(server.asgi_app(), host=host, port=port, log_level="warning")
    )
    server_task = asyncio.create_task(uvicorn_server.serve())
    await wait_for_port(host, port, timeout=5.0)

    client = A2ARemoteAgent(
        config=AgentConfig(name="remote_auth", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(
            url=f"http://{host}:{port}",
            transport="JSONRPC",
            headers=headers,
        ),
    )
    await client.initialize()
    try:
        response = await client.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text="who am i")],
                )
            ]
        )
    finally:
        await client.shutdown()
        uvicorn_server.should_exit = True
        await asyncio.wait_for(server_task, timeout=5.0)
        await server.executor.shutdown()

    assert response.all_text() == expected
    assert disposed


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fast_agent_a2a_server_sets_bearer_token_before_instance_creation(
    monkeypatch,
    unused_tcp_port: int,
    wait_for_port,
) -> None:
    _patch_hf_token_verifier(monkeypatch)
    monkeypatch.setenv("FAST_AGENT_SERVE_OAUTH", "huggingface")
    monkeypatch.setenv("FAST_AGENT_OAUTH_RESOURCE_URL", "http://127.0.0.1")
    host = "127.0.0.1"
    port = unused_tcp_port
    tokens_seen: list[str | None] = []

    async def create_instance() -> AgentInstance:
        tokens_seen.append(request_bearer_token.get())
        return _instance(TokenEchoAgent(name="worker"))

    async def dispose_instance(instance: AgentInstance) -> None:
        await instance.shutdown()

    server = AgentA2AServer(
        primary_instance=_instance(TokenEchoAgent(name="worker")),
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        server_name="fast-agent auth early propagation test server",
        host=host,
        port=port,
        instance_scope="request",
    )
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(server.asgi_app(), host=host, port=port, log_level="warning")
    )
    server_task = asyncio.create_task(uvicorn_server.serve())
    await wait_for_port(host, port, timeout=5.0)

    client = A2ARemoteAgent(
        config=AgentConfig(name="remote_auth", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(
            url=f"http://{host}:{port}",
            transport="JSONRPC",
            headers={"Authorization": "Bearer request-token"},
        ),
    )
    await client.initialize()
    try:
        response = await client.generate_impl(
            [
                PromptMessageExtended(
                    role="user",
                    content=[TextContent(type="text", text="who am i")],
                )
            ]
        )
    finally:
        await client.shutdown()
        uvicorn_server.should_exit = True
        await asyncio.wait_for(server_task, timeout=5.0)
        await server.executor.shutdown()

    assert response.all_text() == "request-token"
    assert tokens_seen == ["request-token"]


def test_bearer_token_from_call_context_prefers_saved_request_state() -> None:
    context = SimpleNamespace(
        call_context=SimpleNamespace(
            state={
                "fast_agent_bearer_token": "saved-token",
                "headers": {"authorization": "Bearer header-token"},
            }
        )
    )

    assert _bearer_token_from_call_context(cast("Any", context)) == "saved-token"


async def _async_instance(agent: RecordingAgent) -> AgentInstance:
    return _instance(agent)


async def _async_dispose_instance(instance: AgentInstance) -> None:
    await instance.shutdown()
