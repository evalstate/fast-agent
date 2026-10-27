from __future__ import annotations

import asyncio
import contextvars
from typing import Any

import pytest
from mcp_types import (
    CreateMessageRequestParams,
    ElicitRequestURLParams,
    ElicitResult,
    ListRootsResult,
    ProgressNotification,
    ProgressNotificationParams,
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
from fast_agent.context import Context, cleanup_context, get_initialized_context
from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
from fast_agent.mcp.mcp_aggregator import MCPAggregator
from fast_agent.mcp.mcp_connection_manager import _transport_notification_handler
from fast_agent.mcp.transport_tracking import TransportChannelMetrics


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
async def test_runtime_without_context_does_not_initialize_global_application() -> None:
    await cleanup_context()

    runtime = MCPClientCallbackRuntime(
        server_name="oauth",
        server_config=MCPServerSettings(),
    )

    assert runtime.context is None
    assert get_initialized_context() is None


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
    current_model = "bootstrap-model"
    runtime = MCPClientCallbackRuntime(
        server_name="sampling-server",
        server_config=server_config,
        agent_model="initial-model",
        agent_model_resolver=lambda: current_model,
        api_key="agent-key",
        context=app_context,
    )
    params = CreateMessageRequestParams(
        max_tokens=64,
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text="Hello"))],
    )

    assert runtime.sampling_callback is not None
    current_model = "restored-model"
    request_context: Any = None
    result = await runtime.sampling_callback(request_context, params)

    assert result == "sampled"
    assert captured == {
        "params": params,
        "server_name": "sampling-server",
        "server_config": server_config,
        "agent_model": "restored-model",
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


@pytest.mark.asyncio
async def test_runtime_dispatches_legacy_tool_list_notification() -> None:
    refreshed = asyncio.Event()

    async def refresh(server_name: str) -> None:
        assert server_name == "dynamic"
        refreshed.set()

    runtime = MCPClientCallbackRuntime(
        server_name="dynamic",
        server_config=None,
        tool_list_changed_callback=refresh,
        context=_context(),
    )

    await runtime.message_handler(ToolListChangedNotification())
    await asyncio.wait_for(refreshed.wait(), timeout=1)


@pytest.mark.parametrize(
    ("transport", "snapshot_name"),
    [
        ("http", "post_sse"),
        ("sse", "get"),
        ("stdio", "stdio"),
    ],
)
@pytest.mark.asyncio
async def test_runtime_records_progress_on_bound_transport_channel(
    transport: str,
    snapshot_name: str,
) -> None:
    metrics = TransportChannelMetrics()
    config = MCPServerSettings.model_validate(
        {
            "name": "progress",
            "transport": transport,
            "url": "https://example.com/mcp" if transport != "stdio" else None,
            "command": "progress-server" if transport == "stdio" else None,
        }
    )
    runtime = MCPClientCallbackRuntime(
        server_name="progress",
        server_config=config,
        transport_notification_handler=_transport_notification_handler(config, metrics),
        context=_context(),
    )

    await runtime.message_handler(
        ProgressNotification(
            params=ProgressNotificationParams(
                progress_token="tool-1",
                progress=1,
                total=2,
                message="halfway",
            )
        )
    )

    snapshot = metrics.snapshot()
    channel = getattr(snapshot, snapshot_name)
    assert channel is not None
    assert channel.notification_count == 1
    assert channel.last_message_summary == "notify notifications/progress"


def _queue(runtime: MCPClientCallbackRuntime, url: str) -> None:
    runtime.queue_url_elicitation(message="Sign in", url=url, elicitation_id=url)


@pytest.mark.asyncio
async def test_concurrent_requests_keep_their_own_url_elicitations() -> None:
    runtime = MCPClientCallbackRuntime(server_name="identity", server_config=None)
    a_queued = asyncio.Event()
    b_failed = asyncio.Event()

    async def request_a() -> list[str]:
        with runtime.url_elicitation_scope() as scope:
            _queue(runtime, "https://a.test")
            a_queued.set()
            await b_failed.wait()
            return [item.url for item in runtime.consume_url_elicitations(scope)]

    async def request_b_fails() -> None:
        await a_queued.wait()
        try:
            with runtime.url_elicitation_scope():
                _queue(runtime, "https://b.test")
                raise RuntimeError("request failed")
        except RuntimeError:
            b_failed.set()

    a_urls, _ = await asyncio.gather(request_a(), request_b_fails())

    assert a_urls == ["https://a.test"]


@pytest.mark.asyncio
async def test_uncorrelated_url_elicitations_attach_to_next_completed_request() -> None:
    runtime = MCPClientCallbackRuntime(server_name="identity", server_config=None)
    _queue(runtime, "https://stale.test")

    with runtime.url_elicitation_scope() as first:
        with runtime.url_elicitation_scope() as second:
            # Legacy server requests arrive on the receive loop, outside any request scope.
            async def receive_loop() -> None:
                _queue(runtime, "https://x.test")

            await asyncio.get_running_loop().create_task(
                receive_loop(), context=contextvars.Context()
            )
            first_urls = [item.url for item in runtime.consume_url_elicitations(first)]
        second_urls = [item.url for item in runtime.consume_url_elicitations(second)]

    assert first_urls == ["https://x.test"]
    assert second_urls == []
