from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from fast_agent.mcp.experimental_session_client import (
    ExperimentalSessionClient,
    InMemorySessionCookieStore,
    JsonFileSessionCookieStore,
)
from fast_agent.mcp.mcp_aggregator import ServerStatus

if TYPE_CHECKING:
    from pathlib import Path


class _SessionStub:
    def __init__(self, cookie: dict[str, Any] | None = None) -> None:
        self.experimental_session_cookie = dict(cookie) if isinstance(cookie, dict) else None

    def set_experimental_session_cookie(self, cookie: dict[str, Any] | None) -> None:
        self.experimental_session_cookie = dict(cookie) if isinstance(cookie, dict) else None

    async def experimental_session_create(self, *, title: str | None = None, data=None):
        del data
        self.experimental_session_cookie = {
            "id": "sess-created",
            "data": {"title": title or "default"},
        }
        return dict(self.experimental_session_cookie)

    async def experimental_session_list(self):
        return [
            {"id": "sess-created", "data": {"title": "created"}},
            {"id": "sess-alt", "data": {"title": "alt"}},
        ]


class _ServerConnStub:
    def __init__(self, session: _SessionStub) -> None:
        self.session = session


class _ManagerStub:
    def __init__(self, sessions: dict[str, _SessionStub]) -> None:
        self._sessions = sessions

    async def get_server(self, server_name: str, client_session_factory=None):
        del client_session_factory
        return _ServerConnStub(self._sessions[server_name])


class _AggregatorStub:
    connection_persistence = True

    def __init__(self) -> None:
        self._statuses = {
            "alpha": ServerStatus(
                server_name="alpha",
                implementation_name="demo-alpha",
                session_cookie={"id": "sess-a"},
                experimental_session_supported=True,
                experimental_session_features=["create", "list"],
            ),
            "beta": ServerStatus(
                server_name="beta",
                implementation_name="demo-beta",
                session_cookie={"id": "sess-b"},
                experimental_session_supported=True,
                experimental_session_features=["create", "list"],
            ),
        }
        self._sessions = {
            "alpha": _SessionStub({"id": "sess-a"}),
            "beta": _SessionStub({"id": "sess-b"}),
        }
        self._manager = _ManagerStub(self._sessions)

    async def collect_server_status(self):
        return dict(self._statuses)

    def _require_connection_manager(self):
        return self._manager

    def _create_session_factory(self, server_name: str):
        del server_name
        return lambda *_args, **_kwargs: None


@pytest.mark.asyncio
async def test_resolve_server_name_supports_initialize_identity() -> None:
    client = ExperimentalSessionClient(_AggregatorStub(), cookie_store=InMemorySessionCookieStore())

    resolved = await client.resolve_server_name("demo-beta")

    assert resolved == "beta"


@pytest.mark.asyncio
async def test_resume_session_prefers_listed_cookie_payload() -> None:
    store = InMemorySessionCookieStore()
    client = ExperimentalSessionClient(_AggregatorStub(), cookie_store=store)

    server_name, cookie = await client.resume_session("alpha", session_id="sess-created")

    assert server_name == "alpha"
    assert cookie == {"id": "sess-created", "data": {"title": "created"}}


@pytest.mark.asyncio
async def test_clear_all_cookies_clears_each_server_entry() -> None:
    aggregator = _AggregatorStub()
    store = InMemorySessionCookieStore(
        {
            "demo-alpha": {"server_name": "alpha", "cookies": [{"id": "sess-a", "cookie": {"id": "sess-a"}}]},
            "demo-beta": {"server_name": "beta", "cookies": [{"id": "sess-b", "cookie": {"id": "sess-b"}}]},
            "stale-server": {"server_name": "stale", "cookies": [{"id": "sess-stale", "cookie": {"id": "sess-stale"}}]},
        }
    )
    client = ExperimentalSessionClient(aggregator, cookie_store=store)

    cleared = await client.clear_all_cookies()

    assert cleared == ["alpha", "beta"]
    assert aggregator._sessions["alpha"].experimental_session_cookie is None
    assert aggregator._sessions["beta"].experimental_session_cookie is None
    assert store.load() == {}


@pytest.mark.asyncio
async def test_get_cookie_hydrates_session_from_store_when_missing() -> None:
    aggregator = _AggregatorStub()
    aggregator._sessions["alpha"].set_experimental_session_cookie(None)
    store = InMemorySessionCookieStore(
        {
            "demo-alpha": {
                "server_name": "alpha",
                "last_used_id": "sess-stored",
                "cookies": [{"id": "sess-stored", "cookie": {"id": "sess-stored"}}],
            }
        }
    )
    client = ExperimentalSessionClient(aggregator, cookie_store=store)

    cookie = await client.get_cookie("alpha")

    assert cookie == {"id": "sess-stored"}
    assert aggregator._sessions["alpha"].experimental_session_cookie == {"id": "sess-stored"}


@pytest.mark.asyncio
async def test_create_session_persists_cookie_to_store() -> None:
    aggregator = _AggregatorStub()
    store = InMemorySessionCookieStore()
    client = ExperimentalSessionClient(aggregator, cookie_store=store)

    _server_name, cookie = await client.create_session("alpha", title="Demo")

    assert cookie == {"id": "sess-created", "data": {"title": "Demo"}}
    payload = store.load()["demo-alpha"]
    assert payload["last_used_id"] == "sess-created"
    assert payload["cookies"][0]["cookie"] == {"id": "sess-created", "data": {"title": "Demo"}}


@pytest.mark.asyncio
async def test_list_server_cookies_hydrates_from_status_session_id() -> None:
    aggregator = _AggregatorStub()
    aggregator._statuses["alpha"] = ServerStatus(
        server_name="alpha",
        implementation_name="demo-alpha",
        session_id="sess-live",
        session_cookie=None,
        session_title="live-title",
        experimental_session_supported=True,
    )
    store = InMemorySessionCookieStore()
    client = ExperimentalSessionClient(aggregator, cookie_store=store)

    server, identity, active_id, cookies = await client.list_server_cookies("alpha")

    assert server == "alpha"
    assert identity == "demo-alpha"
    assert active_id == "sess-live"
    assert cookies[0]["id"] == "sess-live"
    assert cookies[0]["title"] == "live-title"


def test_json_file_cookie_store_round_trip(tmp_path: Path) -> None:
    jar = tmp_path / "mcp-cookie.json"
    store = JsonFileSessionCookieStore(jar)

    store.save({"alpha": {"id": "sess-a"}})

    with open(jar, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["version"] == 2
    assert payload["cookies"] == {"alpha": {"id": "sess-a"}}
    assert store.load() == {"alpha": {"id": "sess-a"}}


def test_json_file_cookie_store_tolerates_invalid_json(tmp_path: Path) -> None:
    jar = tmp_path / "mcp-cookie.json"
    jar.write_text("not-json", encoding="utf-8")

    store = JsonFileSessionCookieStore(jar)

    assert store.load() == {}


def test_bootstrap_cookie_for_server_prefers_identity_record() -> None:
    aggregator = _AggregatorStub()
    store = InMemorySessionCookieStore(
        {
            "demo-alpha": {
                "server_name": "alpha",
                "last_used_id": "sess-new",
                "cookies": [
                    {
                        "id": "sess-old",
                        "cookie": {"id": "sess-old"},
                        "updatedAt": "2026-02-20T00:00:00Z",
                    },
                    {
                        "id": "sess-new",
                        "cookie": {"id": "sess-new", "data": {"title": "Latest"}},
                        "updatedAt": "2026-02-24T00:00:00Z",
                    },
                ],
            }
        }
    )
    client = ExperimentalSessionClient(aggregator, cookie_store=store)

    cookie = client.bootstrap_cookie_for_server("alpha")

    assert cookie == {"id": "sess-new", "data": {"title": "Latest"}}


def test_mark_cookie_invalidated_clears_last_used_and_skips_bootstrap() -> None:
    aggregator = _AggregatorStub()
    store = InMemorySessionCookieStore(
        {
            "demo-alpha": {
                "server_name": "alpha",
                "last_used_id": "sess-rejected",
                "cookies": [
                    {
                        "id": "sess-rejected",
                        "cookie": {"id": "sess-rejected"},
                        "updatedAt": "2026-02-24T00:00:00Z",
                    },
                    {
                        "id": "sess-fallback",
                        "cookie": {"id": "sess-fallback"},
                        "updatedAt": "2026-02-23T00:00:00Z",
                    },
                ],
            }
        }
    )
    client = ExperimentalSessionClient(aggregator, cookie_store=store)

    changed = client.mark_cookie_invalidated(
        "alpha",
        session_id="sess-rejected",
        reason="Session required",
    )

    assert changed is True

    payload = store.load()["demo-alpha"]
    assert payload["last_used_id"] is None

    rejected_entry = next(item for item in payload["cookies"] if item["id"] == "sess-rejected")
    assert isinstance(rejected_entry.get("invalidatedAt"), str)
    assert rejected_entry.get("invalidatedReason") == "Session required"

    assert client.bootstrap_cookie_for_server("alpha") == {"id": "sess-fallback"}


@pytest.mark.asyncio
async def test_list_server_cookies_includes_invalidation_flag() -> None:
    aggregator = _AggregatorStub()
    aggregator._statuses["alpha"] = ServerStatus(
        server_name="alpha",
        implementation_name="demo-alpha",
        session_cookie=None,
        experimental_session_supported=True,
    )
    store = InMemorySessionCookieStore(
        {
            "demo-alpha": {
                "server_name": "alpha",
                "last_used_id": None,
                "cookies": [
                    {
                        "id": "sess-invalid",
                        "cookie": {"id": "sess-invalid"},
                        "updatedAt": "2026-02-24T00:00:00Z",
                        "invalidatedAt": "2026-02-24T01:00:00Z",
                        "invalidatedReason": "Session required",
                    }
                ],
            }
        }
    )
    client = ExperimentalSessionClient(aggregator, cookie_store=store)

    _server, _identity, active_id, cookies = await client.list_server_cookies("alpha")

    assert active_id is None
    assert len(cookies) == 1
    assert cookies[0]["id"] == "sess-invalid"
    assert cookies[0]["invalidated"] is True
    assert cookies[0]["invalidatedReason"] == "Session required"
