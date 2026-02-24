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


def test_capture_experimental_session_capability() -> None:
    session = _new_session()

    session._capture_experimental_session_capability(
        {
            "session": {
                "version": 2,
                "features": ["create", "list", "delete"],
            }
        }
    )

    assert session.experimental_session_supported is True
    assert session.experimental_session_features == ("create", "delete", "list")


def test_merge_experimental_session_meta_preserves_input() -> None:
    session = _new_session()
    session._experimental_session_cookie = {"id": "sess-123"}

    source: dict[str, Any] = {"custom": {"value": "x"}}
    merged = session._merge_experimental_session_meta(source)

    assert merged == {
        "custom": {"value": "x"},
        "mcp/session": {"id": "sess-123"},
    }
    assert source == {"custom": {"value": "x"}}


def test_update_experimental_session_cookie_from_meta_and_revocation() -> None:
    session = _new_session()

    session._update_experimental_session_cookie(
        {
            "mcp/session": {
                "id": "sess-xyz",
                "data": {"title": "Session Title"},
            }
        }
    )

    assert session.experimental_session_cookie == {
        "id": "sess-xyz",
        "data": {"title": "Session Title"},
    }
    assert session.experimental_session_id == "sess-xyz"
    assert session.experimental_session_title == "Session Title"

    session._update_experimental_session_cookie({"mcp/session": None})
    assert session.experimental_session_cookie is None
    assert session.experimental_session_id is None
    assert session.experimental_session_title is None


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
    session_payload = experimental.get("session")
    assert isinstance(session_payload, dict)
    assert session_payload.get("version") == 2
    assert "features" not in session_payload


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
                    experimental={"session": {"version": 99, "features": ["custom"]}}
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
    assert experimental.get("session") == {"version": 99, "features": ["custom"]}


@pytest.mark.asyncio
async def test_maybe_establish_experimental_session_sends_create_request() -> None:
    class _RecordingSession(MCPAgentClientSession):
        async def send_request(self, request, result_type, **kwargs):  # type: ignore[override]
            del result_type, kwargs
            self.recorded_request = request
            return SimpleNamespace(id="sess-created", expiry=None, data={"title": "demo"})

    session = object.__new__(_RecordingSession)
    session._experimental_session_supported = True
    session._experimental_session_features = ("create", "list")
    session._experimental_session_cookie = None
    session.agent_name = "demo-agent"
    session.session_server_name = "demo-server"
    session.recorded_request = None

    await session._maybe_establish_experimental_session()

    assert session.recorded_request is not None
    assert getattr(session.recorded_request, "method", None) == "session/create"
    params = getattr(session.recorded_request, "params", None)
    hints = getattr(params, "hints", None)
    assert getattr(hints, "label", None) == "demo-agent · demo-server"
    assert getattr(hints, "data", None) == {"title": "demo-agent · demo-server"}
    assert session.experimental_session_cookie == {
        "id": "sess-created",
        "data": {"title": "demo"},
    }


def test_set_experimental_session_cookie_accepts_manual_override() -> None:
    session = _new_session()

    session.set_experimental_session_cookie({"id": "sess-manual", "data": {"title": "Manual"}})
    assert session.experimental_session_cookie == {
        "id": "sess-manual",
        "data": {"title": "Manual"},
    }

    session.set_experimental_session_cookie(None)
    assert session.experimental_session_cookie is None


@pytest.mark.asyncio
async def test_experimental_session_list_and_delete_requests() -> None:
    class _RecordingSession(MCPAgentClientSession):
        async def send_request(self, request, result_type, **kwargs):  # type: ignore[override]
            del result_type, kwargs
            method = getattr(request, "method", None)
            if method == "session/list":
                return SimpleNamespace(sessions=[{"id": "sess-a"}, {"id": "sess-b"}])
            if method == "session/delete":
                return SimpleNamespace(deleted=True)
            raise AssertionError(f"Unexpected method: {method}")

    session = object.__new__(_RecordingSession)
    session._experimental_session_cookie = {"id": "sess-current"}
    session._experimental_session_supported = True
    session._experimental_session_features = ("list", "delete")
    session.agent_name = "demo-agent"
    session.session_server_name = "demo-server"

    listed = await session.experimental_session_list()
    deleted = await session.experimental_session_delete("sess-a")

    assert listed == [{"id": "sess-a"}, {"id": "sess-b"}]
    assert deleted is True


@pytest.mark.asyncio
async def test_experimental_session_delete_includes_cookie_meta() -> None:
    class _RecordingSession(MCPAgentClientSession):
        async def send_request(self, request, result_type, **kwargs):  # type: ignore[override]
            del result_type, kwargs
            method = getattr(request, "method", None)
            if method == "session/delete":
                params = getattr(request, "params", None)
                assert params is not None
                dumped = params.model_dump(by_alias=True, exclude_none=True)
                assert dumped.get("_meta") == {"mcp/session": {"id": "sess-current"}}
                return SimpleNamespace(deleted=True)
            raise AssertionError(f"Unexpected method: {method}")

    session = object.__new__(_RecordingSession)
    session._experimental_session_cookie = {"id": "sess-current"}
    session._experimental_session_supported = True
    session._experimental_session_features = ("delete",)
    session.agent_name = "demo-agent"
    session.session_server_name = "demo-server"

    deleted = await session.experimental_session_delete()

    assert deleted is True


@pytest.mark.asyncio
async def test_experimental_session_create_replaces_existing_cookie() -> None:
    class _RecordingSession(MCPAgentClientSession):
        async def send_request(self, request, result_type, **kwargs):  # type: ignore[override]
            del request, result_type, kwargs
            return SimpleNamespace(
                id="sess-new",
                expiry="2026-02-24T12:00:00Z",
                data={"title": "fresh"},
            )

    session = object.__new__(_RecordingSession)
    session._experimental_session_cookie = {"id": "sess-old", "data": {"title": "stale"}}
    session._experimental_session_supported = True
    session._experimental_session_features = ("create",)
    session.agent_name = "demo-agent"
    session.session_server_name = "demo-server"

    cookie = await session.experimental_session_create(title="ignored-by-test")

    assert cookie == {
        "id": "sess-new",
        "expiry": "2026-02-24T12:00:00Z",
        "data": {"title": "fresh"},
    }
