from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.core.agent_app import AgentApp
from fast_agent.core.fastagent import AgentInstance
from fast_agent.mcp.server.agent_server import AgentMCPServer

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


class CapturingAgent:
    def __init__(self) -> None:
        self.received_request_params = []

    async def send(self, message: str, request_params=None) -> str:
        self.received_request_params.append(request_params)
        return f"echo:{message}"

    async def shutdown(self) -> None:
        return None


class _NoopNotificationSession:
    async def send_notification(self, *_args, **_kwargs) -> None:
        return None


def _build_test_context() -> object:
    request_context = SimpleNamespace(
        meta=None,
        request=SimpleNamespace(headers={}),
        request_id="req-1",
        session=_NoopNotificationSession(),
    )
    return SimpleNamespace(session=object(), request_context=request_context)


async def _build_server(agent: CapturingAgent) -> AgentMCPServer:
    async def create_instance() -> AgentInstance:
        wrapped = cast("AgentProtocol", agent)
        app = AgentApp({"worker": wrapped})
        return AgentInstance(app=app, agents={"worker": wrapped})

    async def dispose_instance(instance: AgentInstance) -> None:
        await instance.shutdown()

    primary = await create_instance()
    return AgentMCPServer(
        primary_instance=primary,
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        instance_scope="shared",
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_send_tool_schema_includes_response_mode_enum() -> None:
    server = await _build_server(CapturingAgent())

    tool = server.mcp_server._tool_manager._tools["worker"]
    properties = tool.parameters.get("properties", {})
    response_mode_schema = properties.get("response_mode")

    assert response_mode_schema is not None
    assert response_mode_schema.get("enum") == ["inherit", "postprocess", "passthrough"]
    assert response_mode_schema.get("default") == "inherit"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_response_mode_inherit_does_not_override_passthrough_flag() -> None:
    agent = CapturingAgent()
    server = await _build_server(agent)

    ctx = _build_test_context()
    tool = server.mcp_server._tool_manager._tools["worker"]

    await tool.fn(message="hello", ctx=ctx, response_mode="inherit")

    request_params = agent.received_request_params[-1]
    assert request_params is not None
    assert "tool_result_passthrough" not in request_params.model_fields_set


@pytest.mark.unit
@pytest.mark.asyncio
async def test_response_mode_explicit_values_override_passthrough_flag() -> None:
    agent = CapturingAgent()
    server = await _build_server(agent)

    ctx = _build_test_context()
    tool = server.mcp_server._tool_manager._tools["worker"]

    await tool.fn(message="hello", ctx=ctx, response_mode="postprocess")
    await tool.fn(message="hello", ctx=ctx, response_mode="passthrough")

    postprocess_params = agent.received_request_params[-2]
    passthrough_params = agent.received_request_params[-1]

    assert postprocess_params is not None
    assert passthrough_params is not None

    assert postprocess_params.tool_result_passthrough is False
    assert passthrough_params.tool_result_passthrough is True
    assert "tool_result_passthrough" in postprocess_params.model_fields_set
    assert "tool_result_passthrough" in passthrough_params.model_fields_set
