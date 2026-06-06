from __future__ import annotations

import asyncio
import copy
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Protocol, cast
from urllib.parse import urlparse, urlunparse

import aiohttp
from aiohttp import WSMsgType, WSServerHandshakeError

from fast_agent.llm.provider.openai.responses_events import (
    is_responses_failure_event,
    is_responses_terminal_event,
)
from fast_agent.llm.provider.openai.tool_event_helpers import (
    item_type_is_responses_function_tool_call,
)
from fast_agent.utils.numeric import int_or_none
from fast_agent.utils.text import strip_str_to_none

RESPONSES_WEBSOCKET_BETA_HEADER = "responses_websockets=2026-02-06"
RESPONSES_WEBSOCKET_BETA_HEADER_NAME = "OpenAI-Beta"
RESPONSES_CREATE_EVENT_TYPE = "response.create"
_STREAM_START_EVENT_TYPES = {
    "response.output_item.added",
    "response.function_call_arguments.delta",
    "response.reasoning_summary_text.delta",
    "response.reasoning_summary.delta",
    "response.reasoning.delta",
    "response.reasoning_text.delta",
    "response.output_text.delta",
    "response.text.delta",
}


class ResponsesWebSocketError(RuntimeError):
    """Raised for WebSocket transport failures.

    Attributes:
        stream_started: Whether any meaningful streaming output/tool event was observed.
    """

    def __init__(
        self,
        message: str,
        *,
        stream_started: bool = False,
        error_code: str | None = None,
        status: int | None = None,
        error_param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stream_started = stream_started
        self.error_code = error_code
        self.status = status
        self.error_param = error_param


class _AttrObjectView:
    """Tiny adapter that exposes dictionary keys as attributes recursively."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = {key: _to_attr_object(value) for key, value in data.items()}

    def __getattr__(self, key: str) -> Any:
        if key in self._data:
            return self._data[key]
        raise AttributeError(key)

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return _to_plain_data(self._data)

    def __repr__(self) -> str:
        return f"_AttrObjectView({self._data!r})"


def _to_plain_data(value: Any) -> Any:
    if isinstance(value, _AttrObjectView):
        return {key: _to_plain_data(item) for key, item in value._data.items()}
    if isinstance(value, Mapping):
        return {key: _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    return value


def _to_attr_object(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _AttrObjectView(value)
    if isinstance(value, list):
        return [_to_attr_object(item) for item in value]
    return value


def _stream_event_started(event_type: str | None) -> bool:
    if not event_type:
        return False
    if event_type in _STREAM_START_EVENT_TYPES:
        return True
    if event_type.startswith("response.output_text"):
        return True
    return event_type.startswith("response.text")


def _non_empty_string(value: Any) -> str | None:
    return strip_str_to_none(value)


def resolve_responses_ws_url(base_url: str) -> str:
    """Build a WebSocket URL matching the Responses endpoint path."""

    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid base URL for websocket transport: '{base_url}'")

    path = (parsed.path or "").rstrip("/")
    if not path.endswith("/responses"):
        path = f"{path}/responses" if path else "/responses"

    if parsed.scheme == "https":
        scheme = "wss"
    elif parsed.scheme == "http":
        scheme = "ws"
    else:
        scheme = parsed.scheme

    return urlunparse(
        (
            scheme,
            parsed.netloc,
            path,
            "",
            parsed.query,
            parsed.fragment,
        )
    )


def build_ws_headers(
    *,
    api_key: str,
    default_headers: Mapping[str, str] | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build headers for Responses websocket requests."""

    headers = dict(default_headers or {})
    headers.setdefault("Authorization", f"Bearer {api_key}")
    headers[RESPONSES_WEBSOCKET_BETA_HEADER_NAME] = RESPONSES_WEBSOCKET_BETA_HEADER
    if extra_headers:
        headers.update(extra_headers)
    return headers


class WebSocketLike(Protocol):
    closed: bool

    async def receive(self, timeout: float | None = None) -> Any: ...

    async def send_str(self, payload: str, compress: int | None = None) -> Any: ...

    async def close(self) -> Any: ...

    def exception(self) -> BaseException | None: ...


class ClientSessionLike(Protocol):
    closed: bool

    async def close(self) -> Any: ...


@dataclass(frozen=True)
class PlannedWsRequest:
    """Planner-produced request envelope for websocket Responses events."""

    event_type: str
    arguments: dict[str, Any]


class ResponsesWsRequestPlanner(Protocol):
    """Policy object for choosing websocket request type/payload."""

    def plan(self, full_arguments: Mapping[str, Any]) -> PlannedWsRequest: ...

    def commit(
        self,
        full_arguments: Mapping[str, Any],
        planned: PlannedWsRequest,
        final_response: Any | None = None,
    ) -> None: ...

    def rollback(self, error: BaseException, *, stream_started: bool) -> None: ...

    def reset(self) -> None: ...


class StatelessResponsesWsPlanner:
    """Always emits ``response.create`` and does not retain state."""

    def plan(self, full_arguments: Mapping[str, Any]) -> PlannedWsRequest:
        return PlannedWsRequest(
            event_type=RESPONSES_CREATE_EVENT_TYPE,
            arguments=_copy_arguments(full_arguments),
        )

    def commit(
        self,
        full_arguments: Mapping[str, Any],
        planned: PlannedWsRequest,
        final_response: Any | None = None,
    ) -> None:
        del full_arguments, planned, final_response

    def rollback(self, error: BaseException, *, stream_started: bool) -> None:
        del error, stream_started

    def reset(self) -> None:
        return None


class StatefulContinuationResponsesWsPlanner:
    """Emit ``response.create`` with ``previous_response_id`` for safe continuations."""

    def __init__(self) -> None:
        self._last_signature: dict[str, Any] | None = None
        self._last_input_fingerprints: list[str] | None = None
        self._last_response_id: str | None = None

    def plan(self, full_arguments: Mapping[str, Any]) -> PlannedWsRequest:
        signature = _sanitize_request_signature(full_arguments)
        input_items = _extract_input_items(full_arguments)
        input_fingerprints = _fingerprint_input_items(input_items)

        if (
            signature is None
            or input_items is None
            or input_fingerprints is None
            or self._last_signature is None
            or self._last_input_fingerprints is None
            or self._last_response_id is None
        ):
            return self._planned_create(full_arguments)

        prior_input = self._last_input_fingerprints
        if (
            signature != self._last_signature
            or len(input_fingerprints) <= len(prior_input)
            or input_fingerprints[: len(prior_input)] != prior_input
        ):
            return self._planned_create(full_arguments)

        incremental_items = copy.deepcopy(input_items[len(prior_input) :])
        incremental_items = _strip_replayed_response_items(incremental_items)
        if not incremental_items:
            return self._planned_create(full_arguments)

        continuation_arguments = _copy_arguments(full_arguments)
        continuation_arguments["input"] = incremental_items
        continuation_arguments["previous_response_id"] = self._last_response_id

        return PlannedWsRequest(
            event_type=RESPONSES_CREATE_EVENT_TYPE,
            arguments=continuation_arguments,
        )

    def commit(
        self,
        full_arguments: Mapping[str, Any],
        planned: PlannedWsRequest,
        final_response: Any | None = None,
    ) -> None:
        del planned
        signature = _sanitize_request_signature(full_arguments)
        input_items = _extract_input_items(full_arguments)
        input_fingerprints = _fingerprint_input_items(input_items)
        response_id = _extract_response_id(final_response)
        if signature is None or input_fingerprints is None or response_id is None:
            self.reset()
            return
        self._last_signature = signature
        self._last_input_fingerprints = input_fingerprints
        self._last_response_id = response_id

    def rollback(self, error: BaseException, *, stream_started: bool) -> None:
        del error, stream_started
        self.reset()

    def reset(self) -> None:
        self._last_signature = None
        self._last_input_fingerprints = None
        self._last_response_id = None

    @staticmethod
    def _planned_create(full_arguments: Mapping[str, Any]) -> PlannedWsRequest:
        create_arguments = _copy_arguments(full_arguments)
        create_arguments.pop("previous_response_id", None)
        return PlannedWsRequest(
            event_type=RESPONSES_CREATE_EVENT_TYPE,
            arguments=create_arguments,
        )


# Backward-compatibility alias for older imports.
# New code should use StatefulContinuationResponsesWsPlanner directly.
StatefulAppendResponsesWsPlanner = StatefulContinuationResponsesWsPlanner


@dataclass
class WebSocketSessionState:
    """Connection-local state for websocket request planning."""

    request_planner: ResponsesWsRequestPlanner | None = None


@dataclass
class ManagedWebSocketConnection:
    """A websocket + owning HTTP session."""

    session: ClientSessionLike
    websocket: WebSocketLike
    busy: bool = False
    last_used_monotonic: float = 0.0
    session_state: WebSocketSessionState = field(default_factory=WebSocketSessionState)


async def connect_websocket(
    *,
    url: str,
    headers: Mapping[str, str],
    timeout_seconds: float | None = None,
) -> ManagedWebSocketConnection:
    # Stream idleness is enforced while iterating websocket events, so avoid a
    # second websocket/session timeout here. A separate wall-clock timeout would
    # race the idle timeout logic and prematurely kill healthy long-running
    # streams that are still producing events.
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
    try:
        if timeout_seconds is None:
            websocket = await session.ws_connect(url, headers=dict(headers), autoping=True)
        else:
            async with asyncio.timeout(timeout_seconds):
                websocket = await session.ws_connect(url, headers=dict(headers), autoping=True)
    except WSServerHandshakeError as exc:
        await session.close()
        raise ResponsesWebSocketError(
            str(exc),
            status=exc.status,
        ) from exc
    except Exception:
        await session.close()
        raise
    return ManagedWebSocketConnection(
        session=cast("ClientSessionLike", session),
        websocket=cast("WebSocketLike", websocket),
    )


async def close_websocket_connection(connection: ManagedWebSocketConnection) -> None:
    if not connection.websocket.closed:
        with suppress(Exception):
            await connection.websocket.close()
    if not connection.session.closed:
        await connection.session.close()


async def send_response_create(
    websocket: WebSocketLike,
    arguments: Mapping[str, Any],
) -> None:
    await send_response_request(
        websocket,
        PlannedWsRequest(
            event_type=RESPONSES_CREATE_EVENT_TYPE,
            arguments=_copy_arguments(arguments),
        ),
    )


async def send_response_request(
    websocket: WebSocketLike,
    planned_request: PlannedWsRequest,
) -> None:
    # WebSocket mode expects the Responses create payload shape plus event envelope.
    # Transport-specific flags like `stream`/`background` are intentionally omitted.
    payload = _copy_arguments(planned_request.arguments)
    payload["type"] = planned_request.event_type
    await websocket.send_str(json.dumps(payload))


def _copy_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(arguments))


def _sanitize_request_signature(arguments: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        comparable = _copy_arguments(arguments)
    except Exception:
        return None
    comparable.pop("input", None)
    comparable.pop("type", None)
    comparable.pop("stream", None)
    return comparable


def _extract_input_items(arguments: Mapping[str, Any]) -> list[Any] | None:
    input_items = arguments.get("input")
    if not isinstance(input_items, list):
        return None
    return input_items


def _strip_replayed_response_items(items: list[Any]) -> list[Any]:
    """Drop replayed assistant output from the front of continuation input.

    When using ``previous_response_id`` the service already has prior model output
    in the response chain. Re-sending assistant output items (reasoning,
    function-call items, assistant messages) can trigger duplicate item-id
    errors, so continuation payloads should start from the first new client item.
    """
    first_new_index = 0
    for item in items:
        if _is_replayed_response_item(item):
            first_new_index += 1
            continue
        break
    return items[first_new_index:]


def _is_replayed_response_item(item: Any) -> bool:
    if not isinstance(item, Mapping):
        return False
    item_type = item.get("type")
    if item_type == "reasoning":
        return True
    if item_type_is_responses_function_tool_call(item_type):
        return True
    if item_type == "message":
        return item.get("role") == "assistant"
    return False


def _fingerprint_input_items(input_items: list[Any] | None) -> list[str] | None:
    if input_items is None:
        return None
    fingerprints: list[str] = []
    for item in input_items:
        try:
            fingerprints.append(
                json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
        except (TypeError, ValueError):
            return None
    return fingerprints


def _extract_response_id(final_response: Any | None) -> str | None:
    if final_response is None:
        return None
    response_id = getattr(final_response, "id", None)
    if isinstance(response_id, str) and response_id:
        return response_id
    if isinstance(final_response, Mapping):
        mapped_id = final_response.get("id")
        if isinstance(mapped_id, str) and mapped_id:
            return mapped_id
    return None


def _merge_completed_output_into_response(
    response: Any,
    completed_output_items: list[tuple[int | None, int, Any]],
) -> Any:
    if not completed_output_items:
        return response

    response_data = _to_plain_data(response)
    if not isinstance(response_data, Mapping):
        return response

    existing_output = response_data.get("output")
    if isinstance(existing_output, list) and existing_output:
        return response_data

    merged_output = [
        _to_plain_data(item)
        for _output_index, _sequence_number, item in sorted(
            completed_output_items,
            key=lambda entry: (
                entry[0] is None,
                -1 if entry[0] is None else entry[0],
                entry[1],
            ),
        )
    ]
    if not merged_output:
        return response_data

    merged_response = dict(response_data)
    merged_response["output"] = merged_output
    return merged_response


class WebSocketResponsesStream:
    """Adapter exposing websocket payloads through the Responses stream interface."""

    def __init__(self, websocket: WebSocketLike) -> None:
        self._websocket = websocket
        self._stream_started = False
        self._saw_terminal_event = False
        self._stop_after_next = False
        self._final_response: Any | None = None
        self.first_event_monotonic: float | None = None
        self._completed_output_items: list[tuple[int | None, int, Any]] = []
        self._events_seen = 0
        self._last_frame_preview: str | None = None

    @property
    def stream_started(self) -> bool:
        return self._stream_started

    def __aiter__(self) -> WebSocketResponsesStream:
        return self

    async def __anext__(self) -> Any:
        if self._stop_after_next:
            raise StopAsyncIteration

        while True:
            message = await self._websocket.receive()
            should_return, event = self._event_from_message(message)
            if should_return:
                return event

    def _event_from_message(self, message: Any) -> tuple[bool, Any | None]:
        self._raise_if_closed_message(message)
        self._raise_if_error_message(message)
        if message.type not in {WSMsgType.TEXT, WSMsgType.BINARY}:
            return False, None

        raw_data = self._raw_message_data(message)
        payload = self._payload_from_raw_data(raw_data)
        event_type = payload.get("type")
        self._raise_top_level_payload_error(payload, event_type)
        self._record_completed_output_item(payload, event_type)
        payload = self._merge_response_payload(payload)
        event = _to_attr_object(payload)
        self._record_event_state(payload, event_type)
        self._raise_event_error(payload, event_type)
        return True, event

    def _raise_if_closed_message(self, message: Any) -> None:
        if message.type not in {WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED}:
            return
        if self._saw_terminal_event:
            raise StopAsyncIteration

        close_code = getattr(message, "data", None)
        if close_code is None:
            close_code = getattr(self._websocket, "close_code", None)
        close_reason = getattr(message, "extra", None)
        diagnostics = self._close_diagnostics(close_code, close_reason)
        detail = "; ".join(diagnostics)
        raise ResponsesWebSocketError(
            "WebSocket stream closed before completion event" + (f" ({detail})" if detail else ""),
            stream_started=self._stream_started,
        )

    def _close_diagnostics(self, close_code: Any, close_reason: Any) -> list[str]:
        diagnostics = []
        if close_code is not None:
            diagnostics.append(f"close_code={close_code}")
        if close_reason:
            diagnostics.append(f"reason={close_reason}")
        diagnostics.append(f"events_seen={self._events_seen}")
        if self._last_frame_preview:
            diagnostics.append(f"last_frame={self._last_frame_preview}")
        if close_code == 1008:
            diagnostics.append(
                "hint=policy_violation (account/feature may not permit Responses websocket beta)"
            )
        return diagnostics

    def _raise_if_error_message(self, message: Any) -> None:
        if message.type != WSMsgType.ERROR:
            return
        ws_error = self._websocket.exception()
        detail = str(ws_error) if ws_error else "unknown websocket error"
        raise ResponsesWebSocketError(
            f"WebSocket transport error: {detail}",
            stream_started=self._stream_started,
        )

    @staticmethod
    def _raw_message_data(message: Any) -> str:
        if message.type == WSMsgType.BINARY:
            return message.data.decode("utf-8", errors="replace")
        return str(message.data)

    def _payload_from_raw_data(self, raw_data: str) -> dict[str, Any]:
        self._last_frame_preview = _preview_text(raw_data)
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            raise ResponsesWebSocketError(
                "Received non-JSON websocket message"
                + (f" ({self._last_frame_preview})" if self._last_frame_preview else ""),
                stream_started=self._stream_started,
            ) from exc

        if not isinstance(payload, dict):
            raise ResponsesWebSocketError(
                "Received unexpected websocket payload"
                + (f" ({self._last_frame_preview})" if self._last_frame_preview else ""),
                stream_started=self._stream_started,
            )
        return payload

    def _raise_top_level_payload_error(
        self,
        payload: Mapping[str, Any],
        event_type: Any,
    ) -> None:
        if isinstance(event_type, str) or "error" not in payload:
            return
        self._raise_payload_error(payload)

    def _record_completed_output_item(
        self,
        payload: Mapping[str, Any],
        event_type: Any,
    ) -> None:
        if event_type != "response.output_item.done":
            return
        item = payload.get("item")
        if item is None:
            return
        output_index = payload.get("output_index")
        sequence_number = payload.get("sequence_number")
        sequence_index = int_or_none(sequence_number)
        self._completed_output_items.append(
            (
                int_or_none(output_index),
                sequence_index if sequence_index is not None else self._events_seen,
                item,
            )
        )

    def _merge_response_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "response" not in payload:
            return payload
        merged_payload = dict(payload)
        merged_payload["response"] = _merge_completed_output_into_response(
            payload["response"],
            self._completed_output_items,
        )
        return merged_payload

    def _record_event_state(
        self,
        payload: Mapping[str, Any],
        event_type: Any,
    ) -> None:
        if isinstance(event_type, str) and _stream_event_started(event_type):
            self._stream_started = True
        if "response" in payload:
            self._final_response = _to_attr_object(payload["response"])
        if is_responses_terminal_event(event_type):
            self._saw_terminal_event = True
            self._stop_after_next = True
        if self.first_event_monotonic is None:
            self.first_event_monotonic = time.perf_counter()
        self._events_seen += 1

    def _raise_event_error(
        self,
        payload: Mapping[str, Any],
        event_type: Any,
    ) -> None:
        if not is_responses_failure_event(event_type):
            return
        self._raise_payload_error(payload)

    def _raise_payload_error(self, payload: Mapping[str, Any]) -> None:
        error_message, error_code, error_status, error_param = self._extract_error_details(payload)
        raise ResponsesWebSocketError(
            error_message,
            stream_started=self._stream_started,
            error_code=error_code,
            status=error_status,
            error_param=error_param,
        )

    async def get_final_response(self) -> Any:
        if self._final_response is None:
            raise ResponsesWebSocketError(
                "WebSocket stream did not provide a final response payload.",
                stream_started=self._stream_started,
            )
        return self._final_response

    @staticmethod
    def _extract_error_details(
        payload: Mapping[str, Any],
    ) -> tuple[str, str | None, int | None, str | None]:
        error_status = int_or_none(payload.get("status"))
        error_code: str | None = None
        error_param: str | None = None

        message = payload.get("message")
        top_level_message = _non_empty_string(message)

        error = payload.get("error")
        if (error_text := _non_empty_string(error)) is not None:
            return error_text, None, error_status, None
        if isinstance(error, Mapping):
            code_value = error.get("code")
            if (code_value := _non_empty_string(code_value)) is not None:
                error_code = code_value

            status_value = error.get("status")
            if (status_value := int_or_none(status_value)) is not None:
                error_status = status_value

            param_value = error.get("param")
            if (param_value := _non_empty_string(param_value)) is not None:
                error_param = param_value

            error_message = error.get("message")
            if (error_message := _non_empty_string(error_message)) is not None:
                return error_message, error_code, error_status, error_param

        if top_level_message:
            return top_level_message, error_code, error_status, error_param

        return "WebSocket Responses request failed.", error_code, error_status, error_param


def _preview_text(raw_data: str, *, limit: int = 240) -> str:
    compact = " ".join(raw_data.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


class WebSocketConnectionManager:
    """Maintain one reusable websocket, with temporary sockets for concurrent calls."""

    def __init__(
        self,
        *,
        idle_timeout_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._mutex = asyncio.Lock()
        self._reusable_connection: ManagedWebSocketConnection | None = None
        self._idle_timeout_seconds = idle_timeout_seconds
        self._clock = clock

    async def acquire(
        self,
        create_connection: Callable[[], Awaitable[ManagedWebSocketConnection]],
    ) -> tuple[ManagedWebSocketConnection, bool]:
        async with self._mutex:
            await self._expire_idle_locked()
            reusable = self._reusable_connection
            if reusable and self._is_open(reusable) and not reusable.busy:
                reusable.busy = True
                return reusable, True

            if reusable and reusable.busy and self._is_open(reusable):
                temp = await create_connection()
                temp.busy = True
                return temp, False

            if reusable:
                await close_websocket_connection(reusable)
                self._reusable_connection = None

            fresh = await create_connection()
            fresh.busy = True
            self._reusable_connection = fresh
            return fresh, True

    async def release(
        self,
        connection: ManagedWebSocketConnection,
        *,
        reusable: bool,
        keep: bool,
    ) -> None:
        async with self._mutex:
            if reusable and self._reusable_connection is connection:
                if keep and self._is_open(connection):
                    connection.busy = False
                    connection.last_used_monotonic = self._clock()
                    return
                await close_websocket_connection(connection)
                self._reusable_connection = None
                return

            await close_websocket_connection(connection)

    async def close(self) -> None:
        async with self._mutex:
            reusable = self._reusable_connection
            self._reusable_connection = None
            if reusable:
                await close_websocket_connection(reusable)

    async def _expire_idle_locked(self) -> None:
        reusable = self._reusable_connection
        if not reusable:
            return
        if reusable.busy:
            return
        if self._idle_timeout_seconds <= 0:
            return
        elapsed = self._clock() - reusable.last_used_monotonic
        if elapsed < self._idle_timeout_seconds:
            return
        await close_websocket_connection(reusable)
        self._reusable_connection = None

    @staticmethod
    def _is_open(connection: ManagedWebSocketConnection) -> bool:
        return not connection.session.closed and not connection.websocket.closed
