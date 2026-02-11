from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession


def _new_session() -> MCPAgentClientSession:
    session = object.__new__(MCPAgentClientSession)
    session._experimental_session_supported = False
    session._experimental_session_features = ()
    session._experimental_session_cookie = None
    session.agent_name = "demo-agent"
    session.session_server_name = "demo-server"
    return session


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


def test_build_experimental_session_title_includes_server_name() -> None:
    session = _new_session()
    assert session._build_experimental_session_title() == "demo-agent · demo-server"


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
