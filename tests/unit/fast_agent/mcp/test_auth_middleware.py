from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from fast_agent.mcp.auth.middleware import HFAuthHeaderMiddleware, _header_name_matches

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from starlette.types import Receive, Scope, Send


async def _noop_receive() -> MutableMapping[str, Any]:
    return {}


async def _noop_send(_message: object) -> None:
    return None


@pytest.mark.unit
def test_header_name_matches_bytes_case_insensitively() -> None:
    assert _header_name_matches(b"X-HF-Authorization", b"x-hf-authorization")
    assert not _header_name_matches(b"x-other-header", b"x-hf-authorization")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hf_auth_header_middleware_copies_hf_header_to_authorization() -> None:
    captured_scope: Scope | None = None

    async def app(scope: Scope, _receive: Receive, _send: Send) -> None:
        nonlocal captured_scope
        captured_scope = scope

    middleware = HFAuthHeaderMiddleware(app)
    scope: Scope = {
        "type": "http",
        "headers": [(b"X-HF-Authorization", b"Bearer hf-token")],
    }

    await middleware(scope, _noop_receive, _noop_send)

    assert captured_scope is not None
    assert (b"authorization", b"Bearer hf-token") in captured_scope["headers"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hf_auth_header_middleware_preserves_existing_authorization() -> None:
    captured_scope: Scope | None = None

    async def app(scope: Scope, _receive: Receive, _send: Send) -> None:
        nonlocal captured_scope
        captured_scope = scope

    middleware = HFAuthHeaderMiddleware(app)
    scope: Scope = {
        "type": "http",
        "headers": [
            (b"Authorization", b"Bearer user-token"),
            (b"X-HF-Authorization", b"Bearer hf-token"),
        ],
    }

    await middleware(scope, _noop_receive, _noop_send)

    assert captured_scope is not None
    assert captured_scope["headers"] == scope["headers"]
