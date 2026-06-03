from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from fast_agent.config import MCPServerSettings
from fast_agent.mcp.experimental_session_client import (
    ExperimentalSessionClient,
    InMemorySessionCookieStore,
    JsonFileSessionCookieStore,
    default_session_cookie_store,
)
from fast_agent.mcp.mcp_aggregator import ServerStatus

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class _SessionStub:
    def __init__(self, cookie: dict[str, Any] | None = None) -> None:
        self.experimental_session_cookie = dict(cookie) if isinstance(cookie, dict) else None

    def set_experimental_session_cookie(self, cookie: dict[str, Any] | None) -> None:
        self.experimental_session_cookie = dict(cookie) if isinstance(cookie, dict) else None

    async def experimental_session_create(self, *, title: str | None = None, data=None):
        del data
        self.experimental_session_cookie = {
            "sessionId": "sess-created",
            "data": {"title": title or "default"},
        }
        return dict(self.experimental_session_cookie)

    async def experimental_session_list(self):
        return [
            {"sessionId": "sess-created", "data": {"title": "created"}},
            {"sessionId": "sess-alt", "data": {"title": "alt"}},
        ]


class _ServerConnStub:
    def __init__(self, session: object, server_config: MCPServerSettings | None = None) -> None:
        self.session = session
        self.server_config = server_config


class _ManagerStub:
    def __init__(
        self,
        sessions: "Mapping[str, object]",
        configs: dict[str, MCPServerSettings] | None = None,
    ) -> None:
        self._sessions = dict(sessions)
        self.running_servers: dict[str, _ServerConnStub] = {
            name: _ServerConnStub(
                session,
                (configs or {}).get(name),
            )
            for name, session in sessions.items()
        }

    async def get_server(self, server_name: str, client_session_factory=None):
        del client_session_factory
        return self.running_servers[server_name]


class _AggregatorStub:
    connection_persistence = True

    def __init__(self) -> None:
        self._statuses = {
            "alpha": ServerStatus(
                server_name="alpha",
                implementation_name="demo-alpha",
                session_cookie={"sessionId": "sess-a"},
                experimental_session_supported=True,
                experimental_session_features=["create", "list"],
            ),
            "beta": ServerStatus(
                server_name="beta",
                implementation_name="demo-beta",
                session_cookie={"sessionId": "sess-b"},
                experimental_session_supported=True,
                experimental_session_features=["create", "list"],
            ),
        }
        self._sessions = {
            "alpha": _SessionStub({"sessionId": "sess-a"}),
            "beta": _SessionStub({"sessionId": "sess-b"}),
        }
        self._manager = _ManagerStub(self._sessions)

    async def collect_server_status(self):
        return dict(self._statuses)

    def _require_connection_manager(self):
        return self._manager

    def _create_session_factory(self, server_name: str):
        del server_name
        return lambda *_args, **_kwargs: None


class _FailingCookieStore:
    def load(self) -> dict[str, dict[str, Any]]:
        raise OSError("cannot read cookie jar")

    def save(self, cookies: dict[str, dict[str, Any]]) -> None:
        del cookies
        raise OSError("cannot write cookie jar")


class _SizedCookieStore:
    def __init__(self, size: object) -> None:
        self._size = size

    def load(self) -> dict[str, dict[str, Any]]:
        return {}

    def save(self, cookies: dict[str, dict[str, Any]]) -> None:
        del cookies

    def size_bytes(self) -> object:
        return self._size


class _LoggerStub:
    def __init__(self) -> None:
        self.warnings: list[dict[str, object]] = []

    def warning(self, message: str, **data: object) -> None:
        self.warnings.append({"message": message, **data})


class _InvalidSessionSurfaceStub:
    experimental_session_cookie = None
    set_experimental_session_cookie = "not-callable"
    experimental_session_create = "not-callable"
    experimental_session_list = "not-callable"


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, 0),
        (42, 42),
        (True, None),
        (False, None),
        (-1, None),
        ("42", None),
    ],
)
def test_store_size_bytes_normalizes_dynamic_store_size(size: object, expected: int | None) -> None:
    client = ExperimentalSessionClient(_AggregatorStub(), cookie_store=_SizedCookieStore(size))

    assert client.store_size_bytes() == expected


@pytest.mark.asyncio
async def test_resolve_server_name_supports_initialize_identity() -> None:
    client = ExperimentalSessionClient(_AggregatorStub(), cookie_store=InMemorySessionCookieStore())

    resolved = await client.resolve_server_name("demo-beta")

    assert resolved == "beta"


@pytest.mark.asyncio
async def test_create_session_rejects_non_callable_experimental_session_surface() -> None:
    aggregator = _AggregatorStub()
    aggregator._manager = _ManagerStub(
        {
            "alpha": _InvalidSessionSurfaceStub(),
            "beta": aggregator._sessions["beta"],
        }
    )
    client = ExperimentalSessionClient(aggregator, cookie_store=InMemorySessionCookieStore())

    with pytest.raises(RuntimeError, match="does not expose MCPAgentClientSession"):
        await client.create_session("alpha", title="Demo")


@pytest.mark.asyncio
async def test_resume_session_prefers_listed_cookie_payload() -> None:
    store = InMemorySessionCookieStore()
    client = ExperimentalSessionClient(_AggregatorStub(), cookie_store=store)

    server_name, cookie = await client.resume_session("alpha", session_id="sess-created")

    assert server_name == "alpha"
    assert cookie == {
        "sessionId": "sess-created",
        "data": {"title": "created"},
    }


@pytest.mark.asyncio
async def test_clear_all_cookies_clears_each_server_entry() -> None:
    aggregator = _AggregatorStub()
    store = InMemorySessionCookieStore(
        {
            "demo-alpha": {
                "server_name": "alpha",
                "cookies": [{"id": "sess-a", "cookie": {"sessionId": "sess-a"}}],
            },
            "demo-beta": {
                "server_name": "beta",
                "cookies": [{"id": "sess-b", "cookie": {"sessionId": "sess-b"}}],
            },
            "stale-server": {
                "server_name": "stale",
                "cookies": [{"id": "sess-stale", "cookie": {"sessionId": "sess-stale"}}],
            },
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
                "cookies": [
                    {"id": "sess-stored", "cookie": {"sessionId": "sess-stored"}}
                ],
            }
        }
    )
    client = ExperimentalSessionClient(aggregator, cookie_store=store)

    cookie = await client.get_cookie("alpha")

    assert cookie == {"sessionId": "sess-stored"}
    assert aggregator._sessions["alpha"].experimental_session_cookie == {
        "sessionId": "sess-stored",
    }


@pytest.mark.asyncio
async def test_create_session_persists_cookie_to_store() -> None:
    aggregator = _AggregatorStub()
    store = InMemorySessionCookieStore()
    client = ExperimentalSessionClient(aggregator, cookie_store=store)

    _server_name, cookie = await client.create_session("alpha", title="Demo")

    assert cookie == {
        "sessionId": "sess-created",
        "data": {"title": "Demo"},
    }
    payload = store.load()["demo-alpha"]
    assert payload["last_used_id"] == "sess-created"
    assert payload["cookies"][0]["cookie"] == {
        "sessionId": "sess-created",
        "data": {"title": "Demo"},
    }


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

    server_cookies = await client.list_server_cookies("alpha")

    assert server_cookies.server_name == "alpha"
    assert server_cookies.server_identity == "demo-alpha"
    assert server_cookies.target is None
    assert server_cookies.sessions_supported is True
    assert server_cookies.active_session_id == "sess-live"
    assert server_cookies.cookies[0]["id"] == "sess-live"
    assert server_cookies.cookies[0]["title"] == "live-title"


def test_json_file_cookie_store_round_trip(tmp_path: Path) -> None:
    jar = tmp_path / "mcp-cookie.json"
    store = JsonFileSessionCookieStore(jar)

    store.save({"alpha": {"sessionId": "sess-a"}})

    with open(jar, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["version"] == 3
    assert payload["cookies"] == {"alpha": {"sessionId": "sess-a"}}
    assert store.load() == {"alpha": {"sessionId": "sess-a"}}


def test_json_file_cookie_store_distinguishes_invalid_json(tmp_path: Path) -> None:
    jar = tmp_path / "mcp-cookie.json"
    jar.write_text("not-json", encoding="utf-8")

    store = JsonFileSessionCookieStore(jar)

    with pytest.raises(json.JSONDecodeError):
        store.load()


def test_experimental_session_client_warns_when_cookie_store_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _LoggerStub()
    monkeypatch.setattr(
        "fast_agent.mcp.experimental_session_client.logger",
        logger,
    )
    client = ExperimentalSessionClient(
        _AggregatorStub(),
        cookie_store=_FailingCookieStore(),
    )

    assert client._load_store() == {}
    assert logger.warnings
    assert logger.warnings[0]["name"] == "mcp_session_cookie_store_load_failed"


def test_experimental_session_client_warns_when_cookie_store_save_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _LoggerStub()
    monkeypatch.setattr(
        "fast_agent.mcp.experimental_session_client.logger",
        logger,
    )
    client = ExperimentalSessionClient(
        _AggregatorStub(),
        cookie_store=_FailingCookieStore(),
    )

    client._save_store({"alpha": {"sessionId": "sess-a"}})

    assert logger.warnings
    assert logger.warnings[0]["name"] == "mcp_session_cookie_store_save_failed"


def test_default_cookie_store_is_in_memory_when_noenv() -> None:
    from fast_agent.config import Settings, get_settings, update_global_settings

    settings = Settings()
    settings._fast_agent_noenv = True
    previous_settings = get_settings()

    try:
        update_global_settings(settings)
        store = default_session_cookie_store()
    finally:
        update_global_settings(previous_settings)

    assert isinstance(store, InMemorySessionCookieStore)


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
                        "cookie": {"sessionId": "sess-old"},
                        "updatedAt": "2026-02-20T00:00:00Z",
                    },
                    {
                        "id": "sess-new",
                        "cookie": {"sessionId": "sess-new", "data": {"title": "Latest"}},
                        "updatedAt": "2026-02-24T00:00:00Z",
                    },
                ],
            }
        }
    )
    client = ExperimentalSessionClient(aggregator, cookie_store=store)

    cookie = client.bootstrap_cookie_for_server("alpha")

    assert cookie == {
        "sessionId": "sess-new",
        "data": {"title": "Latest"},
    }


def test_extract_cookie_title_trims_direct_title_and_label_fallback() -> None:
    assert (
        ExperimentalSessionClient._extract_cookie_title(
            {"sessionId": "sess-a", "title": "  Direct title  "}
        )
        == "Direct title"
    )
    assert (
        ExperimentalSessionClient._extract_cookie_title(
            {
                "sessionId": "sess-a",
                "title": "   ",
                "data": {"label": "  Label title  "},
            }
        )
        == "Label title"
    )


def test_cookie_from_status_session_trims_title() -> None:
    cookie = ExperimentalSessionClient._cookie_from_status_session(
        ServerStatus(
            server_name="alpha",
            session_id="sess-live",
            session_title="  Live title  ",
            experimental_session_supported=True,
        )
    )

    assert cookie == {
        "sessionId": "sess-live",
        "data": {"title": "Live title"},
    }


@pytest.mark.asyncio
async def test_create_session_keys_store_by_target_before_identity() -> None:
    aggregator = _AggregatorStub()
    aggregator._manager = _ManagerStub(
        aggregator._sessions,
        {
            "alpha": MCPServerSettings(
                name="alpha",
                transport="stdio",
                command="python",
                args=["/tmp/session_server.py"],
                cwd="/workspace",
            )
        },
    )
    store = InMemorySessionCookieStore()
    client = ExperimentalSessionClient(aggregator, cookie_store=store)

    _server_name, _cookie = await client.create_session("alpha", title="Demo")

    payload = store.load()
    assert "cmd:python /tmp/session_server.py @ /workspace" in payload
    assert "demo-alpha" not in payload


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
                        "cookie": {"sessionId": "sess-rejected"},
                        "updatedAt": "2026-02-24T00:00:00Z",
                    },
                    {
                        "id": "sess-fallback",
                        "cookie": {"sessionId": "sess-fallback"},
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

    assert client.bootstrap_cookie_for_server("alpha") == {
        "sessionId": "sess-fallback",
    }


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
                        "cookie": {"sessionId": "sess-invalid"},
                        "updatedAt": "2026-02-24T00:00:00Z",
                        "invalidatedAt": "2026-02-24T01:00:00Z",
                        "invalidatedReason": "Session required",
                    }
                ],
            }
        }
    )
    client = ExperimentalSessionClient(aggregator, cookie_store=store)

    server_cookies = await client.list_server_cookies("alpha")

    assert server_cookies.active_session_id is None
    assert len(server_cookies.cookies) == 1
    assert server_cookies.cookies[0]["id"] == "sess-invalid"
    assert server_cookies.cookies[0]["invalidated"] is True
    assert server_cookies.cookies[0]["invalidatedReason"] == "Session required"
