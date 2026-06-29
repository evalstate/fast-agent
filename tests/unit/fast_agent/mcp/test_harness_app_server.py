from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from fastmcp import FastMCP
from mcp.types import TextContent

from fast_agent.core.agent_app import AgentApp
from fast_agent.core.fastagent import AgentInstance
from fast_agent.core.harness_app import AppOpenRequest
from fast_agent.mcp.server.common import get_oauth_config, normalize_serve_oauth_provider
from fast_agent.mcp.server.harness_app_server import (
    HarnessMCPAppRuntimeOptions,
    HarnessMCPAppServer,
    HarnessMCPAppServerOptions,
    ManagedAgentToolSpec,
    create_harness_mcp_app_runtime,
)
from fast_agent.tools.session_environment import ShellExecutionResult
from fast_agent.types import AgentRequest, AgentResponse, PromptMessageExtended

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from fastmcp import Context as MCPContext

    from fast_agent.core.agent_instance_factory import AgentInstanceFactory
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.tools.session_environment import ShellExecutor


class RecordingAppSession:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    @property
    def agent_app(self) -> object:
        return object()

    @property
    def env(self) -> object:
        return object()

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        await request.report("working", progress=1, total=2)
        return AgentResponse.text(f"{request.session_id}:{request.agent}:{request.message.all_text()}")


class RecordingApp:
    def __init__(self) -> None:
        self.session = RecordingAppSession()
        self.opened: list[AppOpenRequest] = []

    @asynccontextmanager
    async def open(
        self,
        request: AppOpenRequest | None = None,
    ) -> "AsyncIterator[RecordingAppSession]":
        resolved = request or AppOpenRequest()
        self.opened.append(resolved)
        yield self.session


class RuntimeFakeAgent:
    name = "agent"

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.shutdown_count = 0

    async def generate(
        self,
        messages: Any,
        request_params: Any = None,
    ) -> PromptMessageExtended:
        del request_params
        text = messages.all_text()
        self.messages.append(text)
        return PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text=f"runtime:{text}")],
        )

    async def shutdown(self) -> None:
        self.shutdown_count += 1


class RuntimeInstanceFactory:
    def __init__(self) -> None:
        self.agent = RuntimeFakeAgent()
        self.created: list[AgentInstance] = []
        self.disposed: list[AgentInstance] = []

    async def create_instance(self) -> AgentInstance:
        wrapped = cast("AgentProtocol", self.agent)
        instance = AgentInstance(AgentApp({"agent": wrapped}), {"agent": wrapped})
        self.created.append(instance)
        return instance

    async def dispose_instance(self, instance: AgentInstance) -> None:
        self.disposed.append(instance)
        await instance.shutdown()


class RuntimeShellExecutor:
    async def execute_shell(
        self,
        command: str,
        *,
        cwd: "str | Path | None" = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ShellExecutionResult:
        del command, cwd, env, timeout
        return ShellExecutionResult(stdout="", stderr="", exit_code=0)


def mcp_context_with_session(
    session_id: str,
    progress_events: list[tuple[float, float | None, str | None]] | None = None,
) -> "MCPContext":
    headers = {"mcp-session-id": session_id}
    request = SimpleNamespace(headers=headers)
    request_context = SimpleNamespace(request=request)

    async def report_progress(
        progress: float,
        total: float | None = None,
        message: str | None = None,
    ) -> None:
        if progress_events is not None:
            progress_events.append((progress, total, message))

    return cast(
        "MCPContext",
        SimpleNamespace(request_context=request_context, report_progress=report_progress),
    )


@pytest.mark.unit
def test_serve_oauth_provider_normalizes_huggingface_aliases() -> None:
    assert normalize_serve_oauth_provider(" HF ") == "huggingface"
    assert normalize_serve_oauth_provider(" HuggingFace ") == "huggingface"


@pytest.mark.unit
def test_serve_oauth_provider_treats_blank_as_disabled() -> None:
    assert normalize_serve_oauth_provider("   ") is None
    assert normalize_serve_oauth_provider(None) is None


@pytest.mark.unit
def test_get_oauth_config_normalizes_provider_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAST_AGENT_SERVE_OAUTH", " HF ")

    oauth_provider, _scopes, _resource_url = get_oauth_config()

    assert oauth_provider == "huggingface"


@pytest.mark.unit
def test_get_oauth_config_accepts_space_and_comma_separated_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAST_AGENT_OAUTH_SCOPES", "openid profile,inference-api")

    _provider, scopes, _resource_url = get_oauth_config()

    assert scopes == ["openid", "profile", "inference-api"]


@pytest.mark.unit
def test_get_oauth_config_defaults_to_spaces_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAST_AGENT_SERVE_OAUTH", "huggingface")
    monkeypatch.delenv("FAST_AGENT_OAUTH_RESOURCE_URL", raising=False)
    monkeypatch.delenv("FAST_AGENT_OAUTH_SCOPES", raising=False)
    monkeypatch.setenv("OAUTH_SCOPES", "openid profile jobs")
    monkeypatch.setenv("SPACE_HOST", "demo-space.hf.space")

    _provider, scopes, resource_url = get_oauth_config()

    assert scopes == ["openid", "profile", "jobs"]
    assert resource_url == "https://demo-space.hf.space"


@pytest.mark.asyncio
async def test_harness_mcp_adapter_uses_mcp_session_and_default_agent() -> None:
    app = RecordingApp()
    progress_events: list[tuple[float, float | None, str | None]] = []
    server = HarnessMCPAppServer(
        app,
        HarnessMCPAppServerOptions(
            server_name="test",
            default_agent="support",
        ),
    )

    response = await server.adapter.invoke_agent(
        ctx=mcp_context_with_session("mcp-123", progress_events),
        message="hello",
        session_id=None,
        agent=None,
    )

    assert response.text_content() == "mcp-123:support:hello"
    assert app.opened[0].session_id == "mcp-123"
    assert app.opened[0].agent == "support"
    assert app.opened[0].metadata["mcp_session_id"] == "mcp-123"
    assert app.session.requests[0].session_id == "mcp-123"
    assert app.session.requests[0].agent == "support"
    assert app.session.requests[0].metadata["harness_session_id"] == "mcp-123"
    assert app.session.requests[0].params is not None
    assert app.session.requests[0].progress is not None
    assert progress_events == [(1, 2, "working")]


@pytest.mark.asyncio
async def test_harness_mcp_app_server_does_not_register_send_by_default() -> None:
    server = HarnessMCPAppServer(
        RecordingApp(),
        HarnessMCPAppServerOptions(server_name="test", default_agent="support"),
    )

    assert await server.mcp_server.list_tools() == []


@pytest.mark.asyncio
async def test_harness_mcp_app_server_registers_managed_agent_tools() -> None:
    app = RecordingApp()
    server = HarnessMCPAppServer(
        app,
        HarnessMCPAppServerOptions(
            server_name="test",
            default_agent="support",
            managed_agent_tools=(
                ManagedAgentToolSpec(
                    name="support",
                    agent="support",
                    description="Support agent.",
                ),
            ),
        ),
    )

    tools = await server.mcp_server.list_tools()
    result = await server.mcp_server.call_tool("support", {"message": "hello"})

    assert [tool.name for tool in tools] == ["support"]
    assert tools[0].description == "Support agent."
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "None:support:hello"


@pytest.mark.asyncio
async def test_harness_mcp_adapter_allows_explicit_session_and_agent() -> None:
    app = RecordingApp()
    server = HarnessMCPAppServer(
        app,
        HarnessMCPAppServerOptions(server_name="test", default_agent="support"),
    )

    response = await server.adapter.invoke_agent(
        ctx=mcp_context_with_session("mcp-123"),
        message="hello",
        session_id="explicit",
        agent="reviewer",
    )

    assert response.text_content() == "explicit:reviewer:hello"
    assert app.opened[0].session_id == "explicit"
    assert app.opened[0].agent == "reviewer"
    assert app.opened[0].metadata["mcp_session_id"] == "mcp-123"
    assert app.opened[0].metadata["requested_session_id"] == "explicit"


@pytest.mark.asyncio
async def test_harness_mcp_request_scope_aligns_open_and_request_session_ids() -> None:
    app = RecordingApp()
    cleaned: list[str] = []

    async def cleanup(session_id: str) -> None:
        cleaned.append(session_id)

    server = HarnessMCPAppServer(
        app,
        HarnessMCPAppServerOptions(
            server_name="test",
            default_agent="support",
            session_scope="request",
            cleanup_session=cleanup,
        ),
    )

    await server.adapter.invoke_agent(
        ctx=mcp_context_with_session("mcp-123"),
        message="hello",
        session_id="explicit",
        agent="reviewer",
    )

    open_request = app.opened[0]
    agent_request = app.session.requests[0]
    assert open_request.session_id is not None
    assert open_request.session_id.startswith("request-")
    assert agent_request.session_id == open_request.session_id
    assert open_request.metadata["mcp_session_id"] == "mcp-123"
    assert open_request.metadata["requested_session_id"] == "explicit"
    assert open_request.metadata["harness_session_id"] == open_request.session_id
    assert agent_request.metadata["harness_session_id"] == open_request.session_id
    assert cleaned == [open_request.session_id]


@pytest.mark.asyncio
async def test_harness_mcp_adapter_invokes_agent_with_structured_arguments() -> None:
    app = RecordingApp()
    server = HarnessMCPAppServer(
        app,
        HarnessMCPAppServerOptions(server_name="test", default_agent="researcher"),
    )

    response = await server.adapter.invoke_agent(
        ctx=mcp_context_with_session("mcp-123"),
        agent="researcher",
        arguments={"repo": "fast-agent-ai/fast-agent", "depth": "quick"},
    )

    assert response.text_content() == (
        'mcp-123:researcher:{"depth": "quick", "repo": "fast-agent-ai/fast-agent"}'
    )
    assert app.session.requests[0].state["mcp_arguments"] == {
        "repo": "fast-agent-ai/fast-agent",
        "depth": "quick",
    }


@pytest.mark.asyncio
async def test_harness_mcp_adapter_registers_explicit_agent_tool_with_template() -> None:
    app = RecordingApp()
    server = HarnessMCPAppServer(
        app,
        HarnessMCPAppServerOptions(server_name="test", default_agent="researcher"),
    )
    mcp = FastMCP("test")

    server.adapter.register_agent_tool(
        mcp,
        name="research",
        agent="researcher",
        description="Research a topic.",
        input_schema={
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "depth": {"type": "string", "enum": ["quick", "deep"]},
            },
            "required": ["topic"],
        },
        render_arguments="Research {{topic}}.\nDepth: {{depth}}",
    )

    result = await mcp.call_tool("research", {"topic": "MCP adapters", "depth": "quick"})

    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "None:researcher:Research MCP adapters.\nDepth: quick"
    request = app.session.requests[0]
    assert request.state["mcp_arguments"] == {"topic": "MCP adapters", "depth": "quick"}


@pytest.mark.asyncio
async def test_harness_mcp_adapter_registered_tool_defaults_to_message_schema() -> None:
    app = RecordingApp()
    server = HarnessMCPAppServer(
        app,
        HarnessMCPAppServerOptions(server_name="test", default_agent="support"),
    )
    mcp = FastMCP("test")

    server.adapter.register_agent_tool(mcp, name="chat", agent="support")

    tools = await mcp.list_tools()
    result = await mcp.call_tool("chat", {"message": "hello"})

    assert tools[0].parameters["required"] == ["message"]
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "None:support:hello"


@pytest.mark.asyncio
async def test_harness_mcp_adapter_requires_exactly_one_input_shape() -> None:
    app = RecordingApp()
    server = HarnessMCPAppServer(app, HarnessMCPAppServerOptions(server_name="test"))

    with pytest.raises(ValueError, match="Exactly one of message or arguments"):
        await server.adapter.invoke_agent(
            ctx=mcp_context_with_session("mcp-123"),
            message="hello",
            arguments={"message": "hello"},
        )


@pytest.mark.asyncio
async def test_harness_mcp_runtime_builds_app_sessions_and_closes_owned_instances() -> None:
    factory = RuntimeInstanceFactory()
    runtime = create_harness_mcp_app_runtime(
        instance_factory=cast("AgentInstanceFactory", factory),
        shell_executor=cast("ShellExecutor", RuntimeShellExecutor()),
        settings=None,
        options=HarnessMCPAppRuntimeOptions(
            server_name="test",
            default_agent="agent",
            managed_agent_tools=(
                ManagedAgentToolSpec(
                    name="agent",
                    agent="agent",
                    description="Send a message to agent.",
                ),
            ),
        ),
    )

    result = await runtime.server.mcp_server.call_tool("agent", {"message": "hello runtime"})
    await runtime.close()

    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "runtime:hello runtime"
    assert factory.agent.messages == ["hello runtime"]
    assert len(factory.created) == 1
    assert factory.disposed == factory.created
    assert factory.agent.shutdown_count == 1


@pytest.mark.asyncio
async def test_harness_mcp_runtime_request_scope_disposes_each_call() -> None:
    factory = RuntimeInstanceFactory()
    runtime = create_harness_mcp_app_runtime(
        instance_factory=cast("AgentInstanceFactory", factory),
        shell_executor=cast("ShellExecutor", RuntimeShellExecutor()),
        settings=None,
        options=HarnessMCPAppRuntimeOptions(
            server_name="test",
            default_agent="agent",
            instance_scope="request",
            managed_agent_tools=(
                ManagedAgentToolSpec(
                    name="agent",
                    agent="agent",
                    description="Send a message to agent.",
                ),
            ),
        ),
    )

    first_response = await runtime.server.mcp_server.call_tool("agent", {"message": "first"})
    second_response = await runtime.server.mcp_server.call_tool("agent", {"message": "second"})
    await runtime.close()

    assert isinstance(first_response.content[0], TextContent)
    assert isinstance(second_response.content[0], TextContent)
    assert first_response.content[0].text == "runtime:first"
    assert second_response.content[0].text == "runtime:second"
    assert len(factory.created) == 2
    assert factory.disposed == factory.created
