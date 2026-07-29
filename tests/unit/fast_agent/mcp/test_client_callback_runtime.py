from __future__ import annotations

import asyncio
from typing import Any

import pytest
from mcp_types import (
    CreateMessageRequestParams,
    ElicitRequestURLParams,
    ElicitResult,
    ListRootsResult,
    SamplingMessage,
    TextContent,
    ToolListChangedNotification,
)

from fast_agent.config import (
    MCPElicitationSettings,
    MCPRootSettings,
    MCPSamplingSettings,
    MCPServerSettings,
    Settings,
)
from fast_agent.context import Context
from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
from fast_agent.mcp.mcp_aggregator import MCPAggregator


def _context(*, auto_sampling: bool = False, elicitation_mode: str = "forms") -> Context:
    return Context(
        config=Settings.model_validate(
            {
                "mcp": {"client": {"auto_sampling": auto_sampling}},
                "elicitation": {"mode": elicitation_mode},
            }
        )
    )


@pytest.mark.asyncio
async def test_runtime_exposes_roots_without_reading_sdk_session() -> None:
    runtime = MCPClientCallbackRuntime(
        server_name="filesystem",
        server_config=MCPServerSettings(
            roots=[
                MCPRootSettings(
                    uri="file:///workspace",
                    server_uri_alias="file:///presented-workspace",
                    name="workspace",
                )
            ]
        ),
        context=_context(),
    )

    assert runtime.list_roots_callback is not None
    request_context: Any = None
    result = await runtime.list_roots_callback(request_context)

    assert isinstance(result, ListRootsResult)
    assert len(result.roots) == 1
    assert str(result.roots[0].uri) == "file:///presented-workspace"
    assert result.roots[0].name == "workspace"
    assert runtime.sampling_callback is None
    assert runtime.sampling_capabilities is None


@pytest.mark.asyncio
async def test_runtime_sampling_callback_captures_fast_agent_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_sample(
        params: CreateMessageRequestParams,
        *,
        server_name: str,
        server_config: MCPServerSettings | None,
        agent_model: str | None,
        api_key: str | None,
        app_context: Context | None,
    ) -> object:
        captured.update(
            params=params,
            server_name=server_name,
            server_config=server_config,
            agent_model=agent_model,
            api_key=api_key,
            app_context=app_context,
        )
        return "sampled"

    monkeypatch.setattr("fast_agent.mcp.client_callback_runtime.sample", fake_sample)
    app_context = _context()
    server_config = MCPServerSettings(sampling=MCPSamplingSettings(model="configured-model"))
    runtime = MCPClientCallbackRuntime(
        server_name="sampling-server",
        server_config=server_config,
        agent_model="agent-model",
        api_key="agent-key",
        context=app_context,
    )
    params = CreateMessageRequestParams(
        max_tokens=64,
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text="Hello"))],
    )

    assert runtime.sampling_callback is not None
    request_context: Any = None
    result = await runtime.sampling_callback(request_context, params)

    assert result == "sampled"
    assert captured == {
        "params": params,
        "server_name": "sampling-server",
        "server_config": server_config,
        "agent_model": "agent-model",
        "api_key": "agent-key",
        "app_context": app_context,
    }
    assert runtime.sampling_capabilities is not None


@pytest.mark.asyncio
async def test_runtime_forms_callback_queues_url_elicitation() -> None:
    runtime = MCPClientCallbackRuntime(
        server_name="identity",
        server_config=MCPServerSettings(
            command="identity-server",
            elicitation=MCPElicitationSettings(mode="forms"),
        ),
        agent_name="researcher",
        context=_context(),
    )
    params = ElicitRequestURLParams(
        mode="url",
        message="Authenticate to continue",
        url="https://example.com/auth",
        elicitation_id="url-1",
    )

    assert runtime.elicitation_callback is not None
    request_context: Any = None
    result = await runtime.elicitation_callback(request_context, params)

    assert isinstance(result, ElicitResult)
    assert result.action == "accept"
    queued = runtime.consume_pending_url_elicitations()
    assert len(queued) == 1
    assert queued[0].message == "Authenticate to continue"
    assert queued[0].url == "https://example.com/auth"
    assert queued[0].elicitation_id == "url-1"


@pytest.mark.asyncio
async def test_runtime_forwards_server_notifications_to_aggregator() -> None:
    received: list[tuple[str, object]] = []

    async def notify(server_name: str, notification: object) -> None:
        received.append((server_name, notification))

    aggregator = object.__new__(MCPAggregator)
    aggregator.server_notification_callback = notify

    runtime = MCPClientCallbackRuntime(
        server_name="notifier",
        server_config=None,
        aggregator=aggregator,
        context=_context(),
    )
    notification = ToolListChangedNotification()

    await runtime.message_handler(notification)
    await asyncio.sleep(0)

    assert received == [("notifier", notification)]
