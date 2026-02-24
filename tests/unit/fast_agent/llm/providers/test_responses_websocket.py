from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from aiohttp import WSMsgType
from mcp.types import TextContent

from fast_agent.llm.provider.openai.codex_responses import CodexResponsesLLM
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider.openai.responses_websocket import (
    RESPONSES_CREATE_EVENT_TYPE,
    ManagedWebSocketConnection,
    PlannedWsRequest,
    ResponsesWebSocketError,
    StatefulContinuationResponsesWsPlanner,
    StatelessResponsesWsPlanner,
    WebSocketConnectionManager,
    WebSocketResponsesStream,
    build_ws_headers,
    resolve_responses_ws_url,
    send_response_create,
    send_response_request,
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


class _CloseTrackingConnectionManager:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


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
    assert "stream" not in payload
    assert payload["model"] == "gpt-5.3-codex"


def _build_ws_arguments(input_items: list[dict[str, Any]], *, temperature: float = 0.0) -> dict[str, Any]:
    return {
        "model": "gpt-5.3-codex",
        "input": input_items,
        "store": False,
        "temperature": temperature,
    }


def _build_input_message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }


def test_stateless_planner_always_create() -> None:
    planner = StatelessResponsesWsPlanner()
    first = planner.plan(_build_ws_arguments([_build_input_message("one")]))
    second = planner.plan(_build_ws_arguments([_build_input_message("one"), _build_input_message("two")]))
    assert first.event_type == RESPONSES_CREATE_EVENT_TYPE
    assert second.event_type == RESPONSES_CREATE_EVENT_TYPE


def test_continuation_planner_first_request_create() -> None:
    planner = StatefulContinuationResponsesWsPlanner()
    planned = planner.plan(_build_ws_arguments([_build_input_message("one")]))
    assert planned.event_type == RESPONSES_CREATE_EVENT_TYPE


def test_continuation_planner_prefix_extension_uses_previous_response_id() -> None:
    planner = StatefulContinuationResponsesWsPlanner()
    first_arguments = _build_ws_arguments([_build_input_message("one")])
    planner.commit(first_arguments, planner.plan(first_arguments), {"id": "resp_1"})

    second_arguments = _build_ws_arguments(
        [_build_input_message("one"), _build_input_message("two")]
    )
    planned = planner.plan(second_arguments)

    assert planned.event_type == RESPONSES_CREATE_EVENT_TYPE
    assert planned.arguments["previous_response_id"] == "resp_1"
    assert planned.arguments["input"] == [_build_input_message("two")]
    assert planned.arguments["model"] == "gpt-5.3-codex"


def test_continuation_planner_non_prefix_forces_create() -> None:
    planner = StatefulContinuationResponsesWsPlanner()
    first_arguments = _build_ws_arguments([_build_input_message("one")])
    planner.commit(first_arguments, planner.plan(first_arguments), {"id": "resp_1"})

    non_prefix = _build_ws_arguments(
        [_build_input_message("different"), _build_input_message("two")]
    )
    planned = planner.plan(non_prefix)
    assert planned.event_type == RESPONSES_CREATE_EVENT_TYPE
    assert "previous_response_id" not in planned.arguments


def test_continuation_planner_equal_or_shorter_input_forces_create() -> None:
    planner = StatefulContinuationResponsesWsPlanner()
    baseline = _build_ws_arguments([_build_input_message("one"), _build_input_message("two")])
    planner.commit(baseline, planner.plan(baseline), {"id": "resp_1"})

    equal = planner.plan(_build_ws_arguments([_build_input_message("one"), _build_input_message("two")]))
    shorter = planner.plan(_build_ws_arguments([_build_input_message("one")]))

    assert equal.event_type == RESPONSES_CREATE_EVENT_TYPE
    assert shorter.event_type == RESPONSES_CREATE_EVENT_TYPE
    assert "previous_response_id" not in equal.arguments
    assert "previous_response_id" not in shorter.arguments


def test_continuation_planner_signature_change_forces_create() -> None:
    planner = StatefulContinuationResponsesWsPlanner()
    baseline = _build_ws_arguments([_build_input_message("one")], temperature=0.0)
    planner.commit(baseline, planner.plan(baseline), {"id": "resp_1"})

    changed_signature = _build_ws_arguments(
        [_build_input_message("one"), _build_input_message("two")],
        temperature=0.9,
    )
    planned = planner.plan(changed_signature)
    assert planned.event_type == RESPONSES_CREATE_EVENT_TYPE
    assert "previous_response_id" not in planned.arguments


def test_continuation_planner_missing_response_id_forces_fresh_create() -> None:
    planner = StatefulContinuationResponsesWsPlanner()
    baseline = _build_ws_arguments([_build_input_message("one")])
    planner.commit(baseline, planner.plan(baseline), {"status": "completed"})

    extended = _build_ws_arguments([_build_input_message("one"), _build_input_message("two")])
    planned = planner.plan(extended)
    assert planned.event_type == RESPONSES_CREATE_EVENT_TYPE
    assert "previous_response_id" not in planned.arguments


def test_continuation_planner_rollback_resets_state() -> None:
    planner = StatefulContinuationResponsesWsPlanner()
    baseline = _build_ws_arguments([_build_input_message("one")])
    planner.commit(baseline, planner.plan(baseline), {"id": "resp_1"})

    planner.rollback(RuntimeError("boom"), stream_started=False)

    extended = _build_ws_arguments([_build_input_message("one"), _build_input_message("two")])
    planned = planner.plan(extended)
    assert planned.event_type == RESPONSES_CREATE_EVENT_TYPE


@pytest.mark.asyncio
async def test_send_response_request_create_envelope() -> None:
    websocket = _FakeWebSocket()
    planned = PlannedWsRequest(
        event_type=RESPONSES_CREATE_EVENT_TYPE,
        arguments={"model": "gpt-5.3-codex", "input": [_build_input_message("one")]},
    )

    await send_response_request(websocket, planned)

    assert len(websocket.sent_payloads) == 1
    payload = json.loads(websocket.sent_payloads[0])
    assert payload["type"] == RESPONSES_CREATE_EVENT_TYPE
    assert payload["model"] == "gpt-5.3-codex"


@pytest.mark.asyncio
async def test_send_response_request_continuation_envelope() -> None:
    websocket = _FakeWebSocket()
    planned = PlannedWsRequest(
        event_type=RESPONSES_CREATE_EVENT_TYPE,
        arguments={
            "model": "gpt-5.3-codex",
            "previous_response_id": "resp_1",
            "input": [_build_input_message("two")],
        },
    )

    await send_response_request(websocket, planned)

    assert len(websocket.sent_payloads) == 1
    payload = json.loads(websocket.sent_payloads[0])
    assert payload["type"] == RESPONSES_CREATE_EVENT_TYPE
    assert payload["input"] == [_build_input_message("two")]
    assert payload["model"] == "gpt-5.3-codex"
    assert payload["previous_response_id"] == "resp_1"


@pytest.mark.asyncio
async def test_send_response_create_delegates_to_generic_sender() -> None:
    websocket = _FakeWebSocket()

    await send_response_create(
        websocket,
        {"model": "gpt-5.3-codex", "input": [_build_input_message("one")]},
    )

    assert len(websocket.sent_payloads) == 1
    payload = json.loads(websocket.sent_payloads[0])
    assert payload["type"] == RESPONSES_CREATE_EVENT_TYPE


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


@pytest.mark.asyncio
async def test_websocket_stream_error_payload_exposes_error_details() -> None:
    websocket = _FakeWebSocket(
        [
            SimpleNamespace(
                type=WSMsgType.TEXT,
                data=json.dumps(
                    {
                        "type": "error",
                        "status": 400,
                        "error": {
                            "code": "previous_response_not_found",
                            "message": "Previous response with id 'resp_abc' not found.",
                            "param": "previous_response_id",
                        },
                    }
                ),
            )
        ]
    )
    stream = WebSocketResponsesStream(websocket)

    with pytest.raises(ResponsesWebSocketError) as excinfo:
        await stream.__anext__()

    assert excinfo.value.error_code == "previous_response_not_found"
    assert excinfo.value.status == 400
    assert excinfo.value.error_param == "previous_response_id"


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


@pytest.mark.asyncio
async def test_responses_llm_close_closes_websocket_manager() -> None:
    harness = _TransportHarness(transport="websocket")
    close_manager = _CloseTrackingConnectionManager()
    harness._ws_connections = cast("Any", close_manager)

    await harness.close()

    assert close_manager.close_calls == 1


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
        self._response_counter = 0

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
        self._response_counter += 1
        response = SimpleNamespace(
            id=f"resp_{self._response_counter}",
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


class _ContinuationConnectionLifecycleHarness(CodexResponsesLLM):
    def __init__(self) -> None:
        super().__init__(provider=Provider.CODEX_RESPONSES, model="gpt-5.3-codex", transport="websocket")
        self.connection = ManagedWebSocketConnection(session=_FakeSession(), websocket=_FakeWebSocket())
        self._release_manager = _ReleaseTrackingConnectionManager(self.connection)
        self._ws_connections = self._release_manager
        self._capturing_logger = _CapturingLogger()
        self.logger = cast("Any", self._capturing_logger)
        self._capturing_display = _CapturingDisplay()
        self.display = cast("Any", self._capturing_display)
        self.raise_stream_error: Exception | None = None
        self.raise_stream_error_once: Exception | None = None
        self._response_counter = 0

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
            "metadata": {"stable": True},
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
        if self.raise_stream_error_once is not None:
            error = self.raise_stream_error_once
            self.raise_stream_error_once = None
            raise error
        if self.raise_stream_error:
            raise self.raise_stream_error
        self._response_counter += 1
        response = SimpleNamespace(
            id=f"resp_{self._response_counter}",
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


class _TimeoutContinuationConnectionLifecycleHarness(_ContinuationConnectionLifecycleHarness):
    def __init__(self) -> None:
        super().__init__()
        self.hang_stream = False

    async def _process_stream(
        self,
        stream: Any,
        model: str,
        capture_filename: Any,
    ) -> tuple[Any, list[str]]:
        if self.hang_stream:
            del stream, model, capture_filename
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
        return await super()._process_stream(stream, model, capture_filename)


class _PlannedAcquireConnectionManager:
    def __init__(self, planned_connections: list[tuple[ManagedWebSocketConnection, bool]]) -> None:
        self._planned_connections = planned_connections
        self.release_keep_values: list[bool] = []

    async def acquire(self, create_connection: Any) -> tuple[ManagedWebSocketConnection, bool]:
        del create_connection
        if not self._planned_connections:
            raise AssertionError("no planned connections left")
        connection, reusable = self._planned_connections.pop(0)
        connection.busy = True
        return connection, reusable

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


def _ws_input_items(*texts: str) -> list[dict[str, Any]]:
    return [_build_input_message(text) for text in texts]


def _sent_payloads(connection: ManagedWebSocketConnection) -> list[str]:
    websocket = connection.websocket
    assert isinstance(websocket, _FakeWebSocket)
    return websocket.sent_payloads


def test_codex_responses_llm_uses_continuation_ws_planner() -> None:
    llm = CodexResponsesLLM(provider=Provider.CODEX_RESPONSES, model="gpt-5.3-codex")
    planner = llm._new_ws_request_planner()
    assert isinstance(planner, StatefulContinuationResponsesWsPlanner)


def test_responses_llm_uses_continuation_ws_planner() -> None:
    llm = ResponsesLLM(provider=Provider.RESPONSES, model="gpt-5.3-codex")
    planner = llm._new_ws_request_planner()
    assert isinstance(planner, StatefulContinuationResponsesWsPlanner)


@pytest.mark.asyncio
async def test_websocket_completion_ws_uses_create_on_first_turn() -> None:
    harness = _ContinuationConnectionLifecycleHarness()
    params = RequestParams(model="gpt-5.3-codex")

    await harness._responses_completion_ws(
        input_items=_ws_input_items("hello"),
        request_params=params,
        tools=None,
        model_name="gpt-5.3-codex",
    )

    payloads = _sent_payloads(harness.connection)
    assert len(payloads) == 1
    first_payload = json.loads(payloads[0])
    assert first_payload["type"] == RESPONSES_CREATE_EVENT_TYPE


@pytest.mark.asyncio
async def test_websocket_completion_ws_uses_previous_response_id_on_second_turn() -> None:
    harness = _ContinuationConnectionLifecycleHarness()
    params = RequestParams(model="gpt-5.3-codex")

    await harness._responses_completion_ws(
        input_items=_ws_input_items("hello"),
        request_params=params,
        tools=None,
        model_name="gpt-5.3-codex",
    )
    await harness._responses_completion_ws(
        input_items=_ws_input_items("hello", "next"),
        request_params=params,
        tools=None,
        model_name="gpt-5.3-codex",
    )

    payloads = _sent_payloads(harness.connection)
    first_payload = json.loads(payloads[0])
    second_payload = json.loads(payloads[1])
    assert first_payload["type"] == RESPONSES_CREATE_EVENT_TYPE
    assert second_payload["type"] == RESPONSES_CREATE_EVENT_TYPE
    assert second_payload["previous_response_id"] == "resp_1"
    assert second_payload["input"] == _ws_input_items("next")


@pytest.mark.asyncio
async def test_websocket_debug_status_reports_continuation_efficiency() -> None:
    harness = _ContinuationConnectionLifecycleHarness()
    harness._ws_debug_inline = True
    params = RequestParams(model="gpt-5.3-codex")

    await harness._responses_completion_ws(
        input_items=_ws_input_items("hello"),
        request_params=params,
        tools=None,
        model_name="gpt-5.3-codex",
    )
    await harness._responses_completion_ws(
        input_items=_ws_input_items("hello", "next"),
        request_params=params,
        tools=None,
        model_name="gpt-5.3-codex",
    )

    assert any(
        "WS create 1/1 items" in message for message in harness._capturing_display.status_messages
    )
    assert any(
        "WS continuation 1/2 items" in message
        for message in harness._capturing_display.status_messages
    )
    assert any(
        "WS continuation" in message and "% saved" in message
        for message in harness._capturing_display.status_messages
    )


@pytest.mark.asyncio
async def test_websocket_completion_ws_rolls_back_planner_on_send_failure() -> None:
    harness = _ContinuationConnectionLifecycleHarness()
    params = RequestParams(model="gpt-5.3-codex")

    await harness._responses_completion_ws(
        input_items=_ws_input_items("hello"),
        request_params=params,
        tools=None,
        model_name="gpt-5.3-codex",
    )

    websocket = harness.connection.websocket
    assert isinstance(websocket, _FakeWebSocket)
    websocket._fail_send_times = 1
    with pytest.raises(ResponsesWebSocketError):
        await harness._responses_completion_ws(
            input_items=_ws_input_items("hello", "next"),
            request_params=params,
            tools=None,
            model_name="gpt-5.3-codex",
        )

    await harness._responses_completion_ws(
        input_items=_ws_input_items("hello", "next", "third"),
        request_params=params,
        tools=None,
        model_name="gpt-5.3-codex",
    )

    payloads = _sent_payloads(harness.connection)
    third_payload = json.loads(payloads[1])
    assert third_payload["type"] == RESPONSES_CREATE_EVENT_TYPE


@pytest.mark.asyncio
async def test_websocket_completion_ws_rolls_back_planner_on_stream_failure() -> None:
    harness = _ContinuationConnectionLifecycleHarness()
    params = RequestParams(model="gpt-5.3-codex")

    await harness._responses_completion_ws(
        input_items=_ws_input_items("hello"),
        request_params=params,
        tools=None,
        model_name="gpt-5.3-codex",
    )

    harness.raise_stream_error = ResponsesWebSocketError("stream failed", stream_started=True)
    with pytest.raises(ResponsesWebSocketError):
        await harness._responses_completion_ws(
            input_items=_ws_input_items("hello", "next"),
            request_params=params,
            tools=None,
            model_name="gpt-5.3-codex",
        )

    harness.raise_stream_error = None
    await harness._responses_completion_ws(
        input_items=_ws_input_items("hello", "next", "third"),
        request_params=params,
        tools=None,
        model_name="gpt-5.3-codex",
    )

    payloads = _sent_payloads(harness.connection)
    third_payload = json.loads(payloads[2])
    assert third_payload["type"] == RESPONSES_CREATE_EVENT_TYPE


@pytest.mark.asyncio
async def test_websocket_completion_ws_rolls_back_planner_on_timeout() -> None:
    harness = _TimeoutContinuationConnectionLifecycleHarness()
    params = RequestParams(model="gpt-5.3-codex")

    await harness._responses_completion_ws(
        input_items=_ws_input_items("hello"),
        request_params=params,
        tools=None,
        model_name="gpt-5.3-codex",
    )

    harness.hang_stream = True

    with pytest.raises(TimeoutError):
        await harness._responses_completion_ws(
            input_items=_ws_input_items("hello", "next"),
            request_params=RequestParams(model="gpt-5.3-codex", streaming_timeout=0.01),
            tools=None,
            model_name="gpt-5.3-codex",
        )

    harness.hang_stream = False
    await harness._responses_completion_ws(
        input_items=_ws_input_items("hello", "next", "third"),
        request_params=params,
        tools=None,
        model_name="gpt-5.3-codex",
    )

    payloads = _sent_payloads(harness.connection)
    third_payload = json.loads(payloads[2])
    assert third_payload["type"] == RESPONSES_CREATE_EVENT_TYPE


@pytest.mark.asyncio
async def test_temporary_connection_has_isolated_planner_state() -> None:
    harness = _ContinuationConnectionLifecycleHarness()
    reusable_connection = harness.connection
    temporary_connection = ManagedWebSocketConnection(session=_FakeSession(), websocket=_FakeWebSocket())
    manager = _PlannedAcquireConnectionManager(
        [(temporary_connection, False), (reusable_connection, True)]
    )
    harness._ws_connections = manager
    params = RequestParams(model="gpt-5.3-codex")

    await harness._responses_completion_ws(
        input_items=_ws_input_items("hello"),
        request_params=params,
        tools=None,
        model_name="gpt-5.3-codex",
    )
    await harness._responses_completion_ws(
        input_items=_ws_input_items("hello", "next"),
        request_params=params,
        tools=None,
        model_name="gpt-5.3-codex",
    )

    temp_payload = json.loads(_sent_payloads(temporary_connection)[0])
    reusable_payload = json.loads(_sent_payloads(reusable_connection)[0])
    assert temp_payload["type"] == RESPONSES_CREATE_EVENT_TYPE
    assert reusable_payload["type"] == RESPONSES_CREATE_EVENT_TYPE


@pytest.mark.asyncio
async def test_planner_state_resets_when_connection_not_kept() -> None:
    harness = _ContinuationConnectionLifecycleHarness()
    params = RequestParams(model="gpt-5.3-codex")

    await harness._responses_completion_ws(
        input_items=_ws_input_items("hello"),
        request_params=params,
        tools=None,
        model_name="gpt-5.3-codex",
    )

    assert harness.connection.session_state.request_planner is not None

    harness.raise_stream_error = ResponsesWebSocketError("stream failed", stream_started=True)
    with pytest.raises(ResponsesWebSocketError):
        await harness._responses_completion_ws(
            input_items=_ws_input_items("hello", "next"),
            request_params=params,
            tools=None,
            model_name="gpt-5.3-codex",
        )

    assert harness.connection.session_state.request_planner is None


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
async def test_websocket_transport_raises_before_stream_start() -> None:
    harness = _TransportHarness(name="transport-harness", transport="websocket")
    harness.ws_error = ResponsesWebSocketError("connect failed", stream_started=False)
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_code",
    [
        "previous_response_not_found",
        "websocket_connection_limit_reached",
    ],
)
async def test_websocket_retries_on_recoverable_server_error_codes(error_code: str) -> None:
    harness = _ContinuationConnectionLifecycleHarness()
    first_connection = ManagedWebSocketConnection(session=_FakeSession(), websocket=_FakeWebSocket())
    second_connection = ManagedWebSocketConnection(session=_FakeSession(), websocket=_FakeWebSocket())
    manager = _PlannedAcquireConnectionManager(
        planned_connections=[
            (first_connection, False),
            (second_connection, False),
        ]
    )
    harness._ws_connections = manager
    harness.raise_stream_error_once = ResponsesWebSocketError(
        "recoverable websocket error",
        stream_started=False,
        error_code=error_code,
        status=400,
    )

    params = RequestParams(model="gpt-5.3-codex")
    response, streamed_summary, normalized_input = await harness._responses_completion_ws(
        input_items=_ws_input_items("hello"),
        request_params=params,
        tools=None,
        model_name="gpt-5.3-codex",
    )

    assert getattr(response, "status", None) == "completed"
    assert streamed_summary == []
    assert normalized_input == _ws_input_items("hello")
    assert manager.release_keep_values == [False, True]

    first_payload = json.loads(_sent_payloads(first_connection)[0])
    second_payload = json.loads(_sent_payloads(second_connection)[0])
    assert first_payload["type"] == RESPONSES_CREATE_EVENT_TYPE
    assert second_payload["type"] == RESPONSES_CREATE_EVENT_TYPE
    assert "previous_response_id" not in second_payload
