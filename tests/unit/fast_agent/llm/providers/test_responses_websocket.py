from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from aiohttp import WSMsgType
from mcp.types import TextContent

from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider.openai.responses_websocket import (
    ManagedWebSocketConnection,
    ResponsesWebSocketError,
    WebSocketConnectionManager,
    WebSocketResponsesStream,
    build_ws_headers,
    resolve_responses_ws_url,
    send_response_create,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.request_params import RequestParams

if TYPE_CHECKING:
    from mcp import Tool


class _FakeSession:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeWebSocket:
    def __init__(
        self,
        messages: list[SimpleNamespace] | None = None,
        *,
        fail_send_times: int = 0,
    ) -> None:
        self.closed = False
        self._messages = messages or []
        self._fail_send_times = fail_send_times
        self.sent_payloads: list[str] = []
        self._exception: BaseException | None = None

    async def receive(self, timeout: float | None = None) -> SimpleNamespace:
        del timeout
        if self._messages:
            return self._messages.pop(0)
        return SimpleNamespace(type=WSMsgType.CLOSED, data=None)

    async def send_str(self, payload: str, compress: int | None = None) -> None:
        del compress
        if self._fail_send_times > 0:
            self._fail_send_times -= 1
            raise RuntimeError("socket closed")
        self.sent_payloads.append(payload)

    async def close(self) -> None:
        self.closed = True

    def exception(self) -> BaseException | None:
        return self._exception


class _FakeResponsesClient:
    async def __aenter__(self) -> _FakeResponsesClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb


class _ReleaseTrackingConnectionManager:
    def __init__(self, connection: ManagedWebSocketConnection) -> None:
        self.connection = connection
        self.release_keep_values: list[bool] = []

    async def acquire(self, create_connection: Any) -> tuple[ManagedWebSocketConnection, bool]:
        del create_connection
        self.connection.busy = True
        return self.connection, True

    async def release(
        self,
        connection: ManagedWebSocketConnection,
        *,
        reusable: bool,
        keep: bool,
    ) -> None:
        del reusable
        connection.busy = False
        self.release_keep_values.append(keep)


class _SequenceConnectionManager:
    def __init__(self, connections: list[ManagedWebSocketConnection]) -> None:
        self._connections = connections
        self.acquire_calls = 0
        self.release_keep_values: list[bool] = []

    async def acquire(self, create_connection: Any) -> tuple[ManagedWebSocketConnection, bool]:
        self.acquire_calls += 1
        if self._connections:
            connection = self._connections.pop(0)
        else:
            connection = await create_connection()
        connection.busy = True
        return connection, True

    async def release(
        self,
        connection: ManagedWebSocketConnection,
        *,
        reusable: bool,
        keep: bool,
    ) -> None:
        del reusable
        connection.busy = False
        self.release_keep_values.append(keep)


class _CapturingLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.info_data: list[dict[str, Any] | None] = []

    def info(self, message: str, data: dict[str, Any] | None = None) -> None:
        self.info_messages.append(message)
        self.info_data.append(data)

    def debug(self, message: str, data: dict[str, Any] | None = None) -> None:
        del message, data

    def warning(self, message: str, data: dict[str, Any] | None = None) -> None:
        del message, data

    def error(self, message: str, data: dict[str, Any] | None = None, exc_info: Any = None) -> None:
        del message, data, exc_info


class _CapturingDisplay:
    def __init__(self) -> None:
        self.status_messages: list[str] = []

    def show_status_message(self, content: Any) -> None:
        self.status_messages.append(getattr(content, "plain", str(content)))


@pytest.mark.asyncio
async def test_send_response_create_envelope() -> None:
    websocket = _FakeWebSocket()

    await send_response_create(
        websocket,
        {"model": "gpt-5.3-codex", "input": [], "store": False},
    )

    assert len(websocket.sent_payloads) == 1
    payload = json.loads(websocket.sent_payloads[0])
    assert payload["type"] == "response.create"
    assert payload["stream"] is True
    assert payload["model"] == "gpt-5.3-codex"


def test_resolve_responses_ws_url() -> None:
    assert resolve_responses_ws_url("https://chatgpt.com/backend-api/codex") == (
        "wss://chatgpt.com/backend-api/codex/responses"
    )
    assert resolve_responses_ws_url("http://localhost:8080/v1") == "ws://localhost:8080/v1/responses"
    assert resolve_responses_ws_url("https://api.openai.com/v1/responses") == (
        "wss://api.openai.com/v1/responses"
    )


def test_build_ws_headers() -> None:
    headers = build_ws_headers(
        api_key="token-123",
        default_headers={"originator": "fast-agent"},
        extra_headers={"chatgpt-account-id": "acct_abc"},
    )

    assert headers["Authorization"] == "Bearer token-123"
    assert headers["OpenAI-Beta"] == "responses_websockets=2026-02-06"
    assert headers["originator"] == "fast-agent"
    assert headers["chatgpt-account-id"] == "acct_abc"


@pytest.mark.asyncio
async def test_websocket_stream_terminal_events_and_final_response() -> None:
    messages = [
        SimpleNamespace(
            type=WSMsgType.TEXT,
            data=json.dumps({"type": "response.output_text.delta", "delta": "hello"}),
        ),
        SimpleNamespace(
            type=WSMsgType.TEXT,
            data=json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "output_text": "hello",
                        "output": [],
                    },
                }
            ),
        ),
    ]
    websocket = _FakeWebSocket(messages)
    stream = WebSocketResponsesStream(websocket)

    collected: list[Any] = []
    async for event in stream:
        collected.append(event)

    assert len(collected) == 2
    assert getattr(collected[0], "type", None) == "response.output_text.delta"
    assert getattr(collected[0], "delta", None) == "hello"
    assert stream.stream_started

    final_response = await stream.get_final_response()
    assert getattr(final_response, "status", None) == "completed"
    assert getattr(final_response, "output_text", None) == "hello"


@pytest.mark.asyncio
async def test_websocket_stream_close_before_completion_raises() -> None:
    websocket = _FakeWebSocket([SimpleNamespace(type=WSMsgType.CLOSED, data=None)])
    stream = WebSocketResponsesStream(websocket)

    with pytest.raises(ResponsesWebSocketError) as excinfo:
        await stream.__anext__()

    assert not excinfo.value.stream_started


@dataclass
class _ConnectionFactory:
    created: list[ManagedWebSocketConnection]

    async def __call__(self) -> ManagedWebSocketConnection:
        connection = ManagedWebSocketConnection(session=_FakeSession(), websocket=_FakeWebSocket())
        self.created.append(connection)
        return connection


@pytest.mark.asyncio
async def test_websocket_connection_manager_reuses_idle_socket() -> None:
    manager = WebSocketConnectionManager(idle_timeout_seconds=60.0)
    factory = _ConnectionFactory(created=[])

    first, first_reusable = await manager.acquire(factory)
    assert first_reusable
    await manager.release(first, reusable=first_reusable, keep=True)

    second, second_reusable = await manager.acquire(factory)
    assert second_reusable
    assert second is first


@pytest.mark.asyncio
async def test_websocket_connection_manager_busy_uses_temporary_socket() -> None:
    manager = WebSocketConnectionManager(idle_timeout_seconds=60.0)
    factory = _ConnectionFactory(created=[])

    reusable, reusable_flag = await manager.acquire(factory)
    assert reusable_flag

    temp, temp_flag = await manager.acquire(factory)
    assert not temp_flag
    assert temp is not reusable

    await manager.release(temp, reusable=temp_flag, keep=True)
    assert temp.websocket.closed
    assert temp.session.closed

    await manager.release(reusable, reusable=reusable_flag, keep=True)

    reused_again, reused_again_flag = await manager.acquire(factory)
    assert reused_again_flag
    assert reused_again is reusable


@pytest.mark.asyncio
async def test_websocket_connection_manager_invalidation_on_error() -> None:
    manager = WebSocketConnectionManager(idle_timeout_seconds=60.0)
    factory = _ConnectionFactory(created=[])

    first, first_reusable = await manager.acquire(factory)
    await manager.release(first, reusable=first_reusable, keep=False)
    assert first.websocket.closed
    assert first.session.closed

    second, second_reusable = await manager.acquire(factory)
    assert second_reusable
    assert second is not first


class _TransportHarness(ResponsesLLM):
    def __init__(self, **kwargs: Any) -> None:
        self.ws_error: ResponsesWebSocketError | None = None
        self.sse_calls = 0
        self.ws_calls = 0
        super().__init__(provider=Provider.CODEX_RESPONSES, model="gpt-5.3-codex", **kwargs)

    def _supports_websocket_transport(self) -> bool:
        return True

    async def _responses_completion_sse(
        self,
        *,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None,
        model_name: str,
    ) -> tuple[Any, list[str], list[dict[str, Any]]]:
        self.sse_calls += 1
        response = SimpleNamespace(
            status="completed",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="sse")],
                )
            ],
            usage=None,
        )
        return response, [], input_items

    async def _responses_completion_ws(
        self,
        *,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None,
        model_name: str,
    ) -> tuple[Any, list[str], list[dict[str, Any]]]:
        del request_params, tools, model_name
        self.ws_calls += 1
        if self.ws_error:
            raise self.ws_error
        response = SimpleNamespace(
            status="completed",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="ws")],
                )
            ],
            usage=None,
        )
        return response, [], input_items


class _ConnectionLifecycleHarness(ResponsesLLM):
    def __init__(self) -> None:
        super().__init__(provider=Provider.CODEX_RESPONSES, model="gpt-5.3-codex", transport="websocket")
        connection = ManagedWebSocketConnection(session=_FakeSession(), websocket=_FakeWebSocket())
        self._release_manager = _ReleaseTrackingConnectionManager(connection)
        self._ws_connections = self._release_manager
        self._capturing_logger = _CapturingLogger()
        self.logger = cast("Any", self._capturing_logger)
        self._capturing_display = _CapturingDisplay()
        self.display = cast("Any", self._capturing_display)

    def _supports_websocket_transport(self) -> bool:
        return True

    def _responses_client(self) -> Any:
        return _FakeResponsesClient()

    async def _normalize_input_files(
        self,
        client: Any,
        input_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        del client
        return input_items

    def _build_response_args(
        self,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None,
    ) -> dict[str, Any]:
        del tools
        model_name = request_params.model or "gpt-5.3-codex"
        return {
            "model": model_name,
            "input": input_items,
            "store": False,
        }

    def _base_responses_url(self) -> str:
        return "https://api.openai.com/v1"

    def _build_websocket_headers(self) -> dict[str, str]:
        return {}

    async def _process_stream(
        self,
        stream: Any,
        model: str,
        capture_filename: Any,
    ) -> tuple[Any, list[str]]:
        del stream, model, capture_filename
        response = SimpleNamespace(
            status="completed",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="ws")],
                )
            ],
            usage=None,
        )
        return response, []


class _TimeoutLifecycleHarness(_ConnectionLifecycleHarness):
    async def _process_stream(
        self,
        stream: Any,
        model: str,
        capture_filename: Any,
    ) -> tuple[Any, list[str]]:
        del stream, model, capture_filename
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_auto_transport_falls_back_to_sse_before_stream_start() -> None:
    harness = _TransportHarness(name="transport-harness", transport="auto")
    harness.ws_error = ResponsesWebSocketError("connect failed", stream_started=False)
    params = RequestParams(model="gpt-5.3-codex")

    result = await harness._responses_completion(
        input_items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        request_params=params,
    )

    assert harness.ws_calls == 1
    assert harness.sse_calls == 1
    assert harness.active_transport == "sse"
    assert result.content == [TextContent(type="text", text="sse")]


@pytest.mark.asyncio
async def test_auto_transport_does_not_fallback_after_stream_start() -> None:
    harness = _TransportHarness(name="transport-harness", transport="auto")
    harness.ws_error = ResponsesWebSocketError("stream failed", stream_started=True)
    params = RequestParams(model="gpt-5.3-codex")

    with pytest.raises(ResponsesWebSocketError):
        await harness._responses_completion(
            input_items=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
            request_params=params,
        )

    assert harness.ws_calls == 1
    assert harness.sse_calls == 0


@pytest.mark.asyncio
async def test_websocket_transport_falls_back_before_stream_start() -> None:
    harness = _TransportHarness(name="transport-harness", transport="websocket")
    harness.ws_error = ResponsesWebSocketError("connect failed", stream_started=False)
    params = RequestParams(model="gpt-5.3-codex")

    result = await harness._responses_completion(
        input_items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        request_params=params,
    )

    assert harness.ws_calls == 1
    assert harness.sse_calls == 1
    assert harness.active_transport == "sse"
    assert result.content == [TextContent(type="text", text="sse")]


@pytest.mark.asyncio
async def test_websocket_transport_raises_after_stream_start() -> None:
    harness = _TransportHarness(name="transport-harness", transport="websocket")
    harness.ws_error = ResponsesWebSocketError("stream failed", stream_started=True)
    params = RequestParams(model="gpt-5.3-codex")

    with pytest.raises(ResponsesWebSocketError):
        await harness._responses_completion(
            input_items=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
            request_params=params,
        )

    assert harness.ws_calls == 1
    assert harness.sse_calls == 0


@pytest.mark.asyncio
async def test_websocket_transport_sets_active_transport_marker() -> None:
    harness = _TransportHarness(name="transport-harness", transport="websocket")
    params = RequestParams(model="gpt-5.3-codex")

    result = await harness._responses_completion(
        input_items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        request_params=params,
    )

    assert harness.active_transport == "websocket"
    assert result.content == [TextContent(type="text", text="ws")]


@pytest.mark.asyncio
async def test_websocket_success_keeps_connection_for_reuse() -> None:
    harness = _ConnectionLifecycleHarness()
    params = RequestParams(model="gpt-5.3-codex")
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]

    response, streamed_summary, normalized_input = await harness._responses_completion_ws(
        input_items=input_items,
        request_params=params,
        tools=None,
        model_name="gpt-5.3-codex",
    )

    assert getattr(response, "status", None) == "completed"
    assert streamed_summary == []
    assert normalized_input == input_items
    assert harness._release_manager.release_keep_values == [True]


@pytest.mark.asyncio
async def test_websocket_streaming_timeout_releases_reusable_connection() -> None:
    harness = _TimeoutLifecycleHarness()
    params = RequestParams(model="gpt-5.3-codex", streaming_timeout=0.01)
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]

    with pytest.raises(TimeoutError, match="Streaming did not complete within"):
        await harness._responses_completion_ws(
            input_items=input_items,
            request_params=params,
            tools=None,
            model_name="gpt-5.3-codex",
        )

    assert harness._release_manager.release_keep_values == [False]


@pytest.mark.asyncio
async def test_websocket_reused_connection_shows_status_message() -> None:
    harness = _ConnectionLifecycleHarness()
    harness._release_manager.connection.last_used_monotonic = 42.0
    params = RequestParams(model="gpt-5.3-codex")
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]

    response, streamed_summary, normalized_input = await harness._responses_completion_ws(
        input_items=input_items,
        request_params=params,
        tools=None,
        model_name="gpt-5.3-codex",
    )

    assert getattr(response, "status", None) == "completed"
    assert streamed_summary == []
    assert normalized_input == input_items
    assert "WebSocket reused" in harness._capturing_display.status_messages


@pytest.mark.asyncio
async def test_websocket_reestablishes_stale_reused_socket_once() -> None:
    harness = _ConnectionLifecycleHarness()
    stale_reused = ManagedWebSocketConnection(
        session=_FakeSession(),
        websocket=_FakeWebSocket(fail_send_times=1),
        last_used_monotonic=123.0,
    )
    fresh = ManagedWebSocketConnection(session=_FakeSession(), websocket=_FakeWebSocket())
    sequence_manager = _SequenceConnectionManager([stale_reused, fresh])
    harness._ws_connections = sequence_manager
    params = RequestParams(model="gpt-5.3-codex")
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]

    response, streamed_summary, normalized_input = await harness._responses_completion_ws(
        input_items=input_items,
        request_params=params,
        tools=None,
        model_name="gpt-5.3-codex",
    )

    assert getattr(response, "status", None) == "completed"
    assert streamed_summary == []
    assert normalized_input == input_items
    assert sequence_manager.acquire_calls == 2
    assert sequence_manager.release_keep_values == [False, True]
    assert any(
        "re-establishing connection" in message
        for message in harness._capturing_logger.info_messages
    )
    reconnect_log_data = next(
        (
            payload
            for payload in harness._capturing_logger.info_data
            if payload and payload.get("error")
        ),
        None,
    )
    assert reconnect_log_data is not None
    assert reconnect_log_data.get("stream_started") is False
    assert reconnect_log_data.get("websocket_closed") is False
    assert "WebSocket reconnected" in harness._capturing_display.status_messages
