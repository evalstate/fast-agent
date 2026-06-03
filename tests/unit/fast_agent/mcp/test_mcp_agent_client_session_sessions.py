from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from mcp import ClientSession, ServerNotification
from mcp.types import (
    LATEST_PROTOCOL_VERSION,
    CallToolRequest,
    CallToolRequestParams,
    ClientCapabilities,
    ClientRequest,
    Implementation,
    InitializeRequest,
    InitializeRequestParams,
    ProgressNotification,
    ProgressNotificationParams,
    RequestParams,
    ResourceUpdatedNotification,
    ResourceUpdatedNotificationParams,
)

from fast_agent.config import MCPServerSettings
from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

if TYPE_CHECKING:
    from datetime import timedelta

    from mcp.shared.message import MessageMetadata
    from mcp.shared.session import ProgressFnT, ReceiveResultT


class _SessionPayload:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = dict(payload)

    def model_dump(
        self,
        *,
        by_alias: bool = False,
        exclude_none: bool = False,
    ) -> dict[str, Any]:
        del by_alias, exclude_none
        return dict(self._payload)


def _new_session() -> MCPAgentClientSession:
    session = object.__new__(MCPAgentClientSession)
    session._experimental_session_supported = False
    session._experimental_session_features = ()
    session._experimental_session_cookie = None
    session.agent_name = "demo-agent"
    session.session_server_name = "demo-server"
    session.server_config = None
    return session


def _initialize_request() -> ClientRequest:
    return ClientRequest(
        InitializeRequest(
            params=InitializeRequestParams(
                protocolVersion=LATEST_PROTOCOL_VERSION,
                capabilities=ClientCapabilities(),
                clientInfo=Implementation(name="test-client", version="1.0.0"),
            )
        )
    )


def test_capture_experimental_session_capability_data_layer_sessions() -> None:
    session = _new_session()

    session._capture_experimental_session_capability({"sessions": {}})

    assert session.experimental_session_supported is True
    assert session.experimental_session_features == ("create", "delete")


def test_capture_experimental_session_capability_requires_sessions_capability() -> None:
    session = _new_session()

    session._capture_experimental_session_capability(
        {
            "experimental": {
                "session": {
                    "features": ["create", "list", "delete"],
                }
            }
        }
    )

    assert session.experimental_session_supported is False
    assert session.experimental_session_features == ()


def test_merge_experimental_session_meta_preserves_input() -> None:
    session = _new_session()
    session._experimental_session_cookie = {
        "sessionId": "sess-123",
        "state": "token-123",
    }

    source: dict[str, Any] = {"custom": {"value": "x"}}
    merged = session._merge_experimental_session_meta(source)

    assert merged == {
        "custom": {"value": "x"},
        "io.modelcontextprotocol/session": {
            "sessionId": "sess-123",
            "state": "token-123",
        },
    }
    assert source == {"custom": {"value": "x"}}


def test_update_experimental_session_cookie_from_meta_and_revocation() -> None:
    session = _new_session()

    session._update_experimental_session_cookie(
        {
            "io.modelcontextprotocol/session": {
                "sessionId": "sess-xyz",
                "state": "state-1",
                "expiresAt": "2026-03-01T00:00:00Z",
            }
        }
    )

    assert session.experimental_session_cookie == {
        "sessionId": "sess-xyz",
        "state": "state-1",
        "expiresAt": "2026-03-01T00:00:00Z",
    }
    assert session.experimental_session_id == "sess-xyz"

    session._update_experimental_session_cookie(
        {"io.modelcontextprotocol/session": None}
    )
    assert session.experimental_session_cookie is None
    assert session.experimental_session_id is None


def test_update_experimental_session_cookie_without_state_keeps_session_only() -> None:
    session = _new_session()

    session._update_experimental_session_cookie(
        {
            "io.modelcontextprotocol/session": {
                "sessionId": "sess-no-state",
                "expiresAt": "2026-03-01T00:00:00Z",
            }
        }
    )

    assert session.experimental_session_cookie == {
        "sessionId": "sess-no-state",
        "expiresAt": "2026-03-01T00:00:00Z",
    }


def test_update_experimental_session_cookie_without_state_does_not_mutate_existing_state() -> None:
    session = _new_session()
    session._experimental_session_cookie = {
        "sessionId": "sess-xyz",
        "state": "state-1",
        "expiresAt": "2026-03-01T00:00:00Z",
    }

    session._update_experimental_session_cookie(
        {
            "io.modelcontextprotocol/session": {
                "sessionId": "sess-xyz",
                "expiresAt": "2026-03-01T12:00:00Z",
            }
        }
    )

    assert session.experimental_session_cookie == {
        "sessionId": "sess-xyz",
        "state": "state-1",
        "expiresAt": "2026-03-01T12:00:00Z",
    }


def test_extract_request_session_id_accepts_model_dump_params() -> None:
    session_meta: dict[str, object] = {
        "io.modelcontextprotocol/session": {
            "sessionId": "sess-from-params",
        }
    }
    request = ClientRequest(
        CallToolRequest(
            params=CallToolRequestParams(
                name="demo-tool",
                _meta=RequestParams.Meta.model_validate(session_meta),
            ),
        )
    )

    assert MCPAgentClientSession._extract_request_session_id(request) == "sess-from-params"


@pytest.mark.parametrize("method", [" PING ", "notifications/PING", "rpc.PING"])
def test_is_ping_request_normalizes_method_case_and_padding(method: str) -> None:
    request = SimpleNamespace(root=SimpleNamespace(method=method))

    assert MCPAgentClientSession._is_ping_request(cast("ClientRequest", request)) is True


def test_experimental_session_title_uses_trimmed_direct_title() -> None:
    session = _new_session()
    session._experimental_session_cookie = {
        "sessionId": "sess-xyz",
        "title": "  Sprint review  ",
        "data": {"label": "fallback"},
    }

    assert session.experimental_session_title == "Sprint review"


def test_experimental_session_title_falls_back_to_nested_label() -> None:
    session = _new_session()
    session._experimental_session_cookie = {
        "sessionId": "sess-xyz",
        "title": "   ",
        "data": {"label": "  Fallback label  "},
    }

    assert session.experimental_session_title == "Fallback label"


def test_maybe_advertise_experimental_session_capability_disabled_by_default() -> None:
    session = _new_session()
    session.server_config = MCPServerSettings(name="demo", transport="http", url="http://example.com")

    request = _initialize_request()
    updated = session._maybe_advertise_experimental_session_capability(request)

    root = getattr(updated, "root", None)
    assert isinstance(root, InitializeRequest)
    params = root.params
    assert params is not None
    assert params.capabilities.experimental is None


def test_maybe_advertise_experimental_session_capability_in_initialize_request() -> None:
    session = _new_session()
    session.server_config = MCPServerSettings(
        name="demo",
        transport="http",
        url="http://example.com",
        experimental_session_advertise=True,
    )

    request = _initialize_request()
    updated = session._maybe_advertise_experimental_session_capability(request)

    root = getattr(updated, "root", None)
    assert isinstance(root, InitializeRequest)
    params = root.params
    assert params is not None
    experimental = params.capabilities.experimental
    assert isinstance(experimental, dict)
    session_payload = experimental.get("experimental/sessions")
    assert session_payload == {}


def test_maybe_advertise_experimental_session_capability_preserves_existing_session_payload() -> None:
    session = _new_session()
    session.server_config = MCPServerSettings(
        name="demo",
        transport="http",
        url="http://example.com",
        experimental_session_advertise=True,
    )

    request = ClientRequest(
        InitializeRequest(
            params=InitializeRequestParams(
                protocolVersion=LATEST_PROTOCOL_VERSION,
                capabilities=ClientCapabilities(
                    experimental={"experimental/sessions": {"mode": "custom"}}
                ),
                clientInfo=Implementation(name="test-client", version="1.0.0"),
            )
        )
    )

    updated = session._maybe_advertise_experimental_session_capability(request)
    root = getattr(updated, "root", None)
    assert isinstance(root, InitializeRequest)
    params = root.params
    assert params is not None
    experimental = params.capabilities.experimental
    assert isinstance(experimental, dict)
    assert experimental.get("experimental/sessions") == {"mode": "custom"}


@pytest.mark.asyncio
async def test_maybe_establish_experimental_session_sends_create_request() -> None:
    class _RecordingSession(MCPAgentClientSession):
        async def send_request(
            self,
            request: ClientRequest,
            result_type: type[ReceiveResultT],
            request_read_timeout_seconds: timedelta | None = None,
            metadata: MessageMetadata | None = None,
            progress_callback: ProgressFnT | None = None,
        ) -> ReceiveResultT:
            del result_type, request_read_timeout_seconds, metadata, progress_callback
            self.recorded_request = request
            result = SimpleNamespace(
                session=_SessionPayload(
                    {
                        "sessionId": "sess-created",
                        "state": "state-created",
                    }
                )
            )
            return cast(
                "ReceiveResultT",
                result,
            )

    session = object.__new__(_RecordingSession)
    session._experimental_session_supported = True
    session._experimental_session_features = ("create", "delete")
    session._experimental_session_cookie = None
    session.agent_name = "demo-agent"
    session.session_server_name = "demo-server"
    session.recorded_request = None

    await session._maybe_establish_experimental_session()

    assert session.recorded_request is not None
    assert getattr(session.recorded_request, "method", None) == "sessions/create"
    assert getattr(session.recorded_request, "params", None) is None
    assert session.experimental_session_cookie == {
        "sessionId": "sess-created",
        "state": "state-created",
    }


def test_set_experimental_session_cookie_requires_session_id() -> None:
    session = _new_session()

    session.set_experimental_session_cookie({"sessionId": "sess-manual"})
    assert session.experimental_session_cookie == {
        "sessionId": "sess-manual",
    }

    session.set_experimental_session_cookie(None)
    assert session.experimental_session_cookie is None


@pytest.mark.asyncio
async def test_experimental_session_list_returns_active_session_snapshot() -> None:
    session = _new_session()
    session._experimental_session_cookie = {
        "sessionId": "sess-current",
        "state": "state-current",
    }

    listed = await session.experimental_session_list()

    assert listed == [
        {
            "sessionId": "sess-current",
            "state": "state-current",
        }
    ]


@pytest.mark.asyncio
async def test_experimental_session_delete_includes_session_meta() -> None:
    class _RecordingSession(MCPAgentClientSession):
        async def send_request(
            self,
            request: ClientRequest,
            result_type: type[ReceiveResultT],
            request_read_timeout_seconds: timedelta | None = None,
            metadata: MessageMetadata | None = None,
            progress_callback: ProgressFnT | None = None,
        ) -> ReceiveResultT:
            del result_type, request_read_timeout_seconds, metadata, progress_callback
            method = getattr(request, "method", None)
            if method == "sessions/delete":
                params = getattr(request, "params", None)
                assert params is not None
                dumped = params.model_dump(by_alias=True, exclude_none=True)
                assert dumped.get("_meta") == {
                    "io.modelcontextprotocol/session": {
                        "sessionId": "sess-current"
                    }
                }
                return cast("ReceiveResultT", SimpleNamespace(deleted=True))
            raise AssertionError(f"Unexpected method: {method}")

    session = object.__new__(_RecordingSession)
    session._experimental_session_cookie = {"sessionId": "sess-current"}
    session._experimental_session_supported = True
    session._experimental_session_features = ("delete",)
    session.agent_name = "demo-agent"
    session.session_server_name = "demo-server"

    deleted = await session.experimental_session_delete()

    assert deleted is True
    assert session.experimental_session_cookie is None


@pytest.mark.asyncio
async def test_experimental_session_create_replaces_existing_cookie() -> None:
    class _RecordingSession(MCPAgentClientSession):
        async def send_request(
            self,
            request: ClientRequest,
            result_type: type[ReceiveResultT],
            request_read_timeout_seconds: timedelta | None = None,
            metadata: MessageMetadata | None = None,
            progress_callback: ProgressFnT | None = None,
        ) -> ReceiveResultT:
            del request, result_type, request_read_timeout_seconds, metadata, progress_callback
            result = SimpleNamespace(
                session=_SessionPayload(
                    {
                        "sessionId": "sess-new",
                        "expiresAt": "2026-02-24T12:00:00Z",
                        "state": "state-new",
                    }
                )
            )
            return cast(
                "ReceiveResultT",
                result,
            )

    session = object.__new__(_RecordingSession)
    session._experimental_session_cookie = {
        "sessionId": "sess-old",
        "state": "state-old",
    }
    session._experimental_session_supported = True
    session._experimental_session_features = ("create",)
    session.agent_name = "demo-agent"
    session.session_server_name = "demo-server"

    cookie = await session.experimental_session_create(title="ignored-by-test")

    assert cookie == {
        "sessionId": "sess-new",
        "expiresAt": "2026-02-24T12:00:00Z",
        "state": "state-new",
    }


@pytest.mark.asyncio
async def test_received_notification_schedules_server_notification_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _noop_parent_notification(
        self: ClientSession, notification: ServerNotification
    ) -> None:
        del self, notification

    received: list[ServerNotification] = []
    started = asyncio.Event()
    release = asyncio.Event()
    notification_tasks: list[asyncio.Task[None]] = []
    real_create_task = asyncio.create_task

    async def _record_notification(
        self: MCPAgentClientSession, notification: ServerNotification
    ) -> None:
        del self
        started.set()
        await release.wait()
        received.append(notification)

    def _capture_create_task(coro: Any) -> asyncio.Task[None]:
        task = real_create_task(coro)
        if getattr(getattr(coro, "cr_code", None), "co_name", None) == "_record_notification":
            notification_tasks.append(task)
        return task

    monkeypatch.setattr(ClientSession, "_received_notification", _noop_parent_notification)
    monkeypatch.setattr(
        MCPAgentClientSession,
        "_handle_server_notification",
        _record_notification,
    )
    monkeypatch.setattr(asyncio, "create_task", _capture_create_task)

    session = _new_session()
    session._aggregator = SimpleNamespace(server_notification_callback=object())
    session._tool_list_changed_callback = None

    notification = ServerNotification(
        ResourceUpdatedNotification(
            params=ResourceUpdatedNotificationParams(
                uri=cast("Any", "file:///demo.txt")
            )
        )
    )

    await session._received_notification(notification)

    assert len(notification_tasks) == 1
    await started.wait()
    assert received == []
    release.set()
    await notification_tasks[0]
    assert received == [notification]


@pytest.mark.asyncio
async def test_received_notification_skips_progress_for_server_notification_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _noop_parent_notification(
        self: ClientSession, notification: ServerNotification
    ) -> None:
        del self, notification

    received: list[ServerNotification] = []

    async def _record_notification(
        self: MCPAgentClientSession, notification: ServerNotification
    ) -> None:
        del self
        received.append(notification)

    monkeypatch.setattr(ClientSession, "_received_notification", _noop_parent_notification)
    monkeypatch.setattr(
        MCPAgentClientSession,
        "_handle_server_notification",
        _record_notification,
    )

    session = _new_session()
    session._aggregator = SimpleNamespace(server_notification_callback=object())
    session._tool_list_changed_callback = None

    notification = ServerNotification(
        ProgressNotification(
            params=ProgressNotificationParams(
                progressToken="tool-call-1",
                progress=1,
                total=10,
                message="step 1",
            )
        )
    )

    await session._received_notification(notification)

    assert received == []
