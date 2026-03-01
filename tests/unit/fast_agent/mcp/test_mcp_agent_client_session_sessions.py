from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from mcp.types import (
    LATEST_PROTOCOL_VERSION,
    ClientCapabilities,
    ClientRequest,
    Implementation,
    InitializeRequest,
    InitializeRequestParams,
)

from fast_agent.config import MCPServerSettings
from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession


class _SessionPayload:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = dict(payload)

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        del exclude_none
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
        async def send_request(self, request, result_type, **kwargs):  # type: ignore[override]
            del result_type, kwargs
            self.recorded_request = request
            return SimpleNamespace(
                session=_SessionPayload(
                    {
                        "sessionId": "sess-created",
                        "state": "state-created",
                    }
                )
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
        async def send_request(self, request, result_type, **kwargs):  # type: ignore[override]
            del result_type, kwargs
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
                return SimpleNamespace(deleted=True)
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
        async def send_request(self, request, result_type, **kwargs):  # type: ignore[override]
            del request, result_type, kwargs
            return SimpleNamespace(
                session=_SessionPayload(
                    {
                        "sessionId": "sess-new",
                        "expiresAt": "2026-02-24T12:00:00Z",
                        "state": "state-new",
                    }
                )
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
