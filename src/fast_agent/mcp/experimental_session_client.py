from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
from fast_agent.paths import resolve_environment_paths

if TYPE_CHECKING:
    from fast_agent.mcp.mcp_aggregator import ServerStatus


class SessionAggregatorProtocol(Protocol):
    connection_persistence: bool

    async def collect_server_status(self) -> dict[str, "ServerStatus"]: ...

    def _require_connection_manager(self): ...

    def _create_session_factory(self, server_name: str): ...


class SessionCookieStore(Protocol):
    """Persistence backend for experimental MCP session cookies."""

    def load(self) -> dict[str, dict[str, Any]]: ...

    def save(self, cookies: dict[str, dict[str, Any]]) -> None: ...


class InMemorySessionCookieStore:
    """Simple ephemeral cookie store for tests and in-process use."""

    def __init__(self, cookies: dict[str, dict[str, Any]] | None = None) -> None:
        self._cookies: dict[str, dict[str, Any]] = {
            key: dict(value)
            for key, value in (cookies or {}).items()
            if isinstance(key, str) and isinstance(value, dict)
        }

    def load(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self._cookies.items()}

    def save(self, cookies: dict[str, dict[str, Any]]) -> None:
        self._cookies = {
            key: dict(value)
            for key, value in cookies.items()
            if isinstance(key, str) and isinstance(value, dict)
        }


class JsonFileSessionCookieStore:
    """Very small JSON-backed cookie jar for experimental session state."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @classmethod
    def from_environment(cls) -> JsonFileSessionCookieStore:
        env_paths = resolve_environment_paths(override=os.getenv("ENVIRONMENT_DIR"))
        return cls(env_paths.root / "mcp-cookie.json")

    def load(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}

        try:
            with open(self._path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return {}

        if not isinstance(payload, dict):
            return {}

        version = payload.get("version")
        if version != 2:
            return {}

        raw_cookies: Any = payload.get("cookies")
        if not isinstance(raw_cookies, dict):
            return {}

        cookies: dict[str, dict[str, Any]] = {}
        for key, value in raw_cookies.items():
            if isinstance(key, str) and isinstance(value, dict):
                cookies[key] = dict(value)
        return cookies

    def save(self, cookies: dict[str, dict[str, Any]]) -> None:
        safe_cookies = {
            key: dict(value)
            for key, value in cookies.items()
            if isinstance(key, str) and isinstance(value, dict)
        }
        payload = {"version": 2, "cookies": safe_cookies}

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self._path.parent,
            prefix=f".{self._path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")

        os.replace(temp_path, self._path)


@dataclass(frozen=True, slots=True)
class SessionJarEntry:
    """Snapshot of one server's experimental session state."""

    server_name: str
    server_identity: str | None
    cookie: dict[str, Any] | None
    title: str | None
    supported: bool | None
    features: tuple[str, ...]
    connected: bool | None = None
    cookies: tuple[dict[str, Any], ...] = ()
    last_used_id: str | None = None


class ExperimentalSessionClient:
    """Client-side helper for experimental MCP session cookies ("the jar")."""

    def __init__(
        self,
        aggregator: SessionAggregatorProtocol,
        *,
        cookie_store: SessionCookieStore | None = None,
    ) -> None:
        self._aggregator = aggregator
        self._cookie_store = cookie_store or JsonFileSessionCookieStore.from_environment()

    async def list_jar(self) -> list[SessionJarEntry]:
        status_map = await self._aggregator.collect_server_status()
        stored = self._load_store()
        entries: list[SessionJarEntry] = []
        seen_store_keys: set[str] = set()
        for server_name, status in sorted(status_map.items()):
            store_key = self._store_key(server_name, status.implementation_name)
            seen_store_keys.add(store_key)
            record = self._normalize_store_record(
                stored.get(store_key),
                server_name=server_name,
                server_identity=status.implementation_name,
            )
            status_cookie = status.session_cookie if isinstance(status.session_cookie, dict) else None
            if status_cookie is None:
                status_cookie = self._cookie_from_status_session(status)
            if status_cookie is not None:
                self._upsert_cookie(record, status_cookie, mark_last_used=True)
            stored[store_key] = record
            entries.append(self._entry_from_status(server_name, status, record=record))

        # Include disconnected identities that still have cookies in the jar.
        for store_key in sorted(stored):
            if store_key in seen_store_keys:
                continue
            record = self._normalize_store_record(stored.get(store_key), server_name=store_key)
            entries.append(
                SessionJarEntry(
                    server_name=record.get("server_name", store_key),
                    server_identity=record.get("server_identity"),
                    cookie=self._select_active_cookie(record),
                    cookies=self._cookie_summaries(record),
                    last_used_id=self._last_used_id(record),
                    title=self._extract_cookie_title(self._select_active_cookie(record)),
                    supported=None,
                    features=(),
                    connected=False,
                )
            )

        self._save_store(stored)
        return entries

    async def resolve_server_name(self, server_identifier: str | None) -> str:
        status_map = await self._aggregator.collect_server_status()
        if not status_map:
            raise ValueError("No attached MCP servers available.")

        if server_identifier is None or not server_identifier.strip():
            if len(status_map) == 1:
                return next(iter(status_map.keys()))
            identities = ", ".join(
                self._format_server_identity(server_name, status)
                for server_name, status in sorted(status_map.items())
            )
            raise ValueError(
                "Multiple MCP servers are attached. Specify one by server name or identity "
                f"(initialize server name): {identities}"
            )

        candidate = server_identifier.strip()
        if candidate in status_map:
            return candidate

        matches = [
            server_name
            for server_name, status in status_map.items()
            if status.implementation_name == candidate
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                "Server identity is ambiguous; multiple attached servers share "
                f"'{candidate}': {', '.join(sorted(matches))}"
            )

        identities = ", ".join(
            self._format_server_identity(server_name, status)
            for server_name, status in sorted(status_map.items())
        )
        raise ValueError(
            f"Unknown MCP server '{candidate}'. Known servers/identities: {identities}"
        )

    async def get_cookie(self, server_identifier: str) -> dict[str, Any] | None:
        server_name = await self.resolve_server_name(server_identifier)
        status_map = await self._aggregator.collect_server_status()
        server_identity = self._status_identity(status_map, server_name)
        session = await self._get_live_session(server_name)
        self._hydrate_session_cookie_from_store(server_name, session, server_identity=server_identity)
        cookie = session.experimental_session_cookie
        if isinstance(cookie, dict):
            self._persist_server_cookie(
                server_name,
                cookie,
                mark_last_used=True,
                server_identity=server_identity,
            )
        return cookie

    async def clear_cookie(self, server_identifier: str | None) -> str:
        server_name = await self.resolve_server_name(server_identifier)
        status_map = await self._aggregator.collect_server_status()
        server_identity = self._status_identity(status_map, server_name)
        session = await self._get_live_session(server_name)
        session.set_experimental_session_cookie(None)
        self._persist_server_cookie(server_name, None, server_identity=server_identity)
        return server_name

    async def clear_all_cookies(self) -> list[str]:
        status_map = await self._aggregator.collect_server_status()
        cleared: list[str] = []
        for server_name in sorted(status_map):
            session = await self._get_live_session(server_name)
            session.set_experimental_session_cookie(None)
            cleared.append(server_name)
        self._save_store({})
        return cleared

    async def create_session(
        self,
        server_identifier: str | None,
        *,
        title: str | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        server_name = await self.resolve_server_name(server_identifier)
        status_map = await self._aggregator.collect_server_status()
        server_identity = self._status_identity(status_map, server_name)
        session = await self._get_live_session(server_name)
        self._hydrate_session_cookie_from_store(server_name, session, server_identity=server_identity)
        cookie = await session.experimental_session_create(title=title)
        self._persist_server_cookie(
            server_name,
            cookie,
            mark_last_used=True,
            server_identity=server_identity,
        )
        return server_name, cookie

    async def list_sessions(self, server_identifier: str | None) -> tuple[str, list[dict[str, Any]]]:
        server_name = await self.resolve_server_name(server_identifier)
        status_map = await self._aggregator.collect_server_status()
        server_identity = self._status_identity(status_map, server_name)
        session = await self._get_live_session(server_name)
        self._hydrate_session_cookie_from_store(server_name, session, server_identity=server_identity)
        sessions = await session.experimental_session_list()
        return server_name, sessions

    async def resume_session(
        self,
        server_identifier: str | None,
        *,
        session_id: str,
    ) -> tuple[str, dict[str, Any]]:
        server_name = await self.resolve_server_name(server_identifier)
        status_map = await self._aggregator.collect_server_status()
        server_identity = self._status_identity(status_map, server_name)
        session = await self._get_live_session(server_name)
        self._hydrate_session_cookie_from_store(server_name, session, server_identity=server_identity)

        selected_cookie: dict[str, Any] | None = None
        try:
            available_sessions = await session.experimental_session_list()
        except Exception:
            available_sessions = []

        for item in available_sessions:
            if not isinstance(item, dict):
                continue
            raw_id = item.get("id")
            if isinstance(raw_id, str) and raw_id == session_id:
                selected_cookie = dict(item)
                break

        if selected_cookie is None:
            selected_cookie = {"id": session_id}

        session.set_experimental_session_cookie(selected_cookie)
        self._persist_server_cookie(
            server_name,
            selected_cookie,
            mark_last_used=True,
            server_identity=server_identity,
        )
        return server_name, dict(selected_cookie)

    def _load_store(self) -> dict[str, dict[str, Any]]:
        try:
            return self._cookie_store.load()
        except Exception:
            return {}

    def _save_store(self, cookies: dict[str, dict[str, Any]]) -> None:
        try:
            self._cookie_store.save(cookies)
        except Exception:
            return

    def _persist_server_cookie(
        self,
        server_name: str,
        cookie: dict[str, Any] | None,
        *,
        mark_last_used: bool = False,
        server_identity: str | None = None,
    ) -> None:
        snapshot = self._load_store()
        identity = server_identity if isinstance(server_identity, str) else self._lookup_server_identity(server_name)
        store_key = self._store_key(server_name, identity)
        record = self._normalize_store_record(
            snapshot.get(store_key),
            server_name=server_name,
            server_identity=identity,
        )

        if isinstance(cookie, dict):
            self._upsert_cookie(record, cookie, mark_last_used=mark_last_used)
            snapshot[store_key] = record
        else:
            snapshot.pop(store_key, None)
        self._save_store(snapshot)

    def _hydrate_session_cookie_from_store(
        self,
        server_name: str,
        session: MCPAgentClientSession,
        *,
        server_identity: str | None = None,
    ) -> None:
        if session.experimental_session_cookie is not None:
            return
        snapshot = self._load_store()
        identity = server_identity if isinstance(server_identity, str) else self._lookup_server_identity(server_name)
        record = self._normalize_store_record(
            snapshot.get(self._store_key(server_name, identity)),
            server_name=server_name,
            server_identity=identity,
        )
        cookie = self._select_active_cookie(record)
        if isinstance(cookie, dict):
            session.set_experimental_session_cookie(cookie)

    async def _get_live_session(self, server_name: str) -> MCPAgentClientSession:
        if not self._aggregator.connection_persistence:
            raise RuntimeError(
                "Experimental session controls require connection_persistence=True."
            )

        manager = self._aggregator._require_connection_manager()  # noqa: SLF001
        server_conn = await manager.get_server(
            server_name,
            client_session_factory=self._aggregator._create_session_factory(server_name),  # noqa: SLF001
        )
        session = server_conn.session
        if isinstance(session, MCPAgentClientSession):
            return session

        required = (
            "experimental_session_cookie",
            "set_experimental_session_cookie",
            "experimental_session_create",
            "experimental_session_list",
        )
        if all(hasattr(session, attr) for attr in required):
            return cast("MCPAgentClientSession", session)

        raise RuntimeError(
            f"Server '{server_name}' does not expose MCPAgentClientSession."
        )

    @staticmethod
    def _entry_from_status(
        server_name: str,
        status: "ServerStatus",
        *,
        record: dict[str, Any],
    ) -> SessionJarEntry:
        features = tuple(status.experimental_session_features or [])
        status_cookie = dict(status.session_cookie) if isinstance(status.session_cookie, dict) else None
        cookie = status_cookie if status_cookie is not None else ExperimentalSessionClient._select_active_cookie(record)
        title = status.session_title
        if not title:
            title = ExperimentalSessionClient._extract_cookie_title(cookie)
        return SessionJarEntry(
            server_name=server_name,
            server_identity=status.implementation_name,
            cookie=dict(cookie) if isinstance(cookie, dict) else None,
            cookies=ExperimentalSessionClient._cookie_summaries(record),
            last_used_id=ExperimentalSessionClient._last_used_id(record),
            title=title,
            supported=status.experimental_session_supported,
            features=features,
            connected=status.is_connected,
        )

    @staticmethod
    def _format_server_identity(server_name: str, status: "ServerStatus") -> str:
        if status.implementation_name:
            return f"{server_name} ({status.implementation_name})"
        return server_name

    async def list_server_cookies(
        self, server_identifier: str | None
    ) -> tuple[str, str | None, str | None, list[dict[str, Any]]]:
        server_name = await self.resolve_server_name(server_identifier)
        status_map = await self._aggregator.collect_server_status()
        status = status_map.get(server_name)
        identity = status.implementation_name if status else None
        store = self._load_store()
        store_key = self._store_key(server_name, identity)
        record = self._normalize_store_record(
            store.get(store_key),
            server_name=server_name,
            server_identity=identity,
        )
        status_cookie = status.session_cookie if status and isinstance(status.session_cookie, dict) else None
        if status_cookie is None:
            status_cookie = self._cookie_from_status_session(status)
        if status_cookie is not None:
            self._upsert_cookie(record, status_cookie, mark_last_used=True)
            store[store_key] = record
            self._save_store(store)
        return server_name, identity, self._last_used_id(record), list(self._cookie_summaries(record))

    def mark_cookie_invalidated(
        self,
        server_name: str,
        *,
        session_id: str,
        reason: str | None = None,
    ) -> bool:
        """Mark a stored session cookie as invalidated after server rejection."""
        if not session_id:
            return False

        snapshot = self._load_store()
        identity = self._lookup_server_identity(server_name)
        store_key = self._store_key(server_name, identity)
        if store_key not in snapshot:
            for candidate_key, candidate_value in snapshot.items():
                if not isinstance(candidate_value, dict):
                    continue
                candidate_name = candidate_value.get("server_name")
                if isinstance(candidate_name, str) and candidate_name == server_name:
                    store_key = candidate_key
                    break
        record = self._normalize_store_record(
            snapshot.get(store_key),
            server_name=server_name,
            server_identity=identity,
        )

        cookies = record.get("cookies")
        if not isinstance(cookies, list):
            cookies = []
            record["cookies"] = cookies

        now = self._now_iso()
        changed = False
        for item in cookies:
            if not isinstance(item, dict):
                continue
            if item.get("id") != session_id:
                continue

            item["invalidatedAt"] = now
            if reason:
                item["invalidatedReason"] = reason
            else:
                item.pop("invalidatedReason", None)
            changed = True
            break

        if not changed:
            cookie = {"id": session_id}
            entry: dict[str, Any] = {
                "id": session_id,
                "cookie": cookie,
                "hash": self._cookie_hash(cookie),
                "updatedAt": now,
                "invalidatedAt": now,
            }
            if reason:
                entry["invalidatedReason"] = reason
            cookies.append(entry)
            changed = True

        if record.get("last_used_id") == session_id:
            record["last_used_id"] = None
            changed = True

        if not changed:
            return False

        snapshot[store_key] = record
        self._save_store(snapshot)
        return True

    def bootstrap_cookie_for_server(self, server_name: str) -> dict[str, Any] | None:
        """Return a best-effort last-used cookie for a server before connect/initialize.

        This is intentionally sync so the MCP session factory can use it to hydrate
        a new ``MCPAgentClientSession`` and avoid unnecessary ``session/create`` calls
        when a recent cookie is already available in the local jar.
        """
        snapshot = self._load_store()
        candidates: list[tuple[str, dict[str, Any]]] = []

        direct = snapshot.get(server_name)
        if isinstance(direct, dict):
            record = self._normalize_store_record(direct, server_name=server_name)
            cookie = self._select_active_cookie(record)
            if isinstance(cookie, dict):
                candidates.append((self._active_cookie_updated_at(record), cookie))

        for value in snapshot.values():
            if not isinstance(value, dict):
                continue
            record = self._normalize_store_record(value, server_name=server_name)
            if record.get("server_name") != server_name:
                continue
            cookie = self._select_active_cookie(record)
            if isinstance(cookie, dict):
                candidates.append((self._active_cookie_updated_at(record), cookie))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return dict(candidates[0][1])

    @staticmethod
    def _store_key(server_name: str, server_identity: str | None) -> str:
        identity = (server_identity or "").strip()
        if identity:
            return identity
        return server_name

    @staticmethod
    def _status_identity(status_map: dict[str, Any], server_name: str) -> str | None:
        status = status_map.get(server_name)
        identity = getattr(status, "implementation_name", None)
        if isinstance(identity, str) and identity.strip():
            return identity.strip()
        return None

    def _lookup_server_identity(self, server_name: str) -> str | None:
        try:
            manager = self._aggregator._require_connection_manager()  # noqa: SLF001
            conn = manager.running_servers.get(server_name)
            implementation = getattr(conn, "server_implementation", None) if conn else None
            identity = getattr(implementation, "name", None)
            if isinstance(identity, str) and identity.strip():
                return identity.strip()
        except Exception:
            return None
        return None

    @staticmethod
    def _normalize_store_record(
        value: dict[str, Any] | None,
        *,
        server_name: str,
        server_identity: str | None = None,
    ) -> dict[str, Any]:
        raw = dict(value) if isinstance(value, dict) else {}
        cookies = raw.get("cookies")
        normalized: list[dict[str, Any]] = []
        if isinstance(cookies, list):
            for item in cookies:
                if isinstance(item, dict):
                    normalized.append(dict(item))
        return {
            "server_name": raw.get("server_name") or server_name,
            "server_identity": raw.get("server_identity") or server_identity,
            "last_used_id": raw.get("last_used_id"),
            "cookies": normalized,
        }

    @staticmethod
    def _cookie_hash(cookie: dict[str, Any]) -> str:
        encoded = json.dumps(cookie, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _upsert_cookie(
        cls,
        record: dict[str, Any],
        cookie: dict[str, Any],
        *,
        mark_last_used: bool,
    ) -> None:
        session_id = cookie.get("id")
        if not isinstance(session_id, str) or not session_id:
            return
        cookies = record.get("cookies")
        if not isinstance(cookies, list):
            cookies = []
            record["cookies"] = cookies

        digest = cls._cookie_hash(cookie)
        updated = cls._now_iso()
        for index, item in enumerate(cookies):
            if not isinstance(item, dict):
                continue
            if item.get("id") != session_id:
                continue
            if item.get("hash") != digest:
                cookies[index] = {
                    "id": session_id,
                    "cookie": dict(cookie),
                    "hash": digest,
                    "updatedAt": updated,
                }
            if mark_last_used:
                record["last_used_id"] = session_id
            return

        cookies.append({"id": session_id, "cookie": dict(cookie), "hash": digest, "updatedAt": updated})
        if mark_last_used:
            record["last_used_id"] = session_id

    @staticmethod
    def _last_used_id(record: dict[str, Any]) -> str | None:
        value = record.get("last_used_id")
        if isinstance(value, str) and value:
            return value
        return None

    @classmethod
    def _select_active_cookie(cls, record: dict[str, Any]) -> dict[str, Any] | None:
        cookies = record.get("cookies")
        if not isinstance(cookies, list) or not cookies:
            return None

        last_used = cls._last_used_id(record)
        if last_used:
            for item in cookies:
                if isinstance(item, dict) and item.get("id") == last_used:
                    if cls._is_cookie_invalidated(item):
                        continue
                    payload = item.get("cookie")
                    if isinstance(payload, dict):
                        return dict(payload)

        latest: dict[str, Any] | None = None
        latest_ts = ""
        for item in cookies:
            if not isinstance(item, dict):
                continue
            if cls._is_cookie_invalidated(item):
                continue
            payload = item.get("cookie")
            if not isinstance(payload, dict):
                continue
            ts = item.get("updatedAt")
            if isinstance(ts, str) and ts >= latest_ts:
                latest_ts = ts
                latest = payload
            elif latest is None:
                latest = payload
        return dict(latest) if isinstance(latest, dict) else None

    @classmethod
    def _cookie_summaries(cls, record: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        cookies = record.get("cookies")
        if not isinstance(cookies, list):
            return ()
        active_id = cls._last_used_id(record)
        summaries: list[dict[str, Any]] = []
        for item in cookies:
            if not isinstance(item, dict):
                continue
            session_id = item.get("id")
            payload = item.get("cookie")
            if not isinstance(session_id, str) or not session_id or not isinstance(payload, dict):
                continue
            invalidated_at = item.get("invalidatedAt") if isinstance(item.get("invalidatedAt"), str) else None
            invalidated_reason = (
                item.get("invalidatedReason") if isinstance(item.get("invalidatedReason"), str) else None
            )
            summaries.append(
                {
                    "id": session_id,
                    "title": cls._extract_cookie_title(payload),
                    "expiry": payload.get("expiry") if isinstance(payload.get("expiry"), str) else None,
                    "updatedAt": item.get("updatedAt") if isinstance(item.get("updatedAt"), str) else None,
                    "active": session_id == active_id,
                    "invalidated": invalidated_at is not None,
                    "invalidatedAt": invalidated_at,
                    "invalidatedReason": invalidated_reason,
                }
            )
        summaries.sort(key=lambda value: str(value.get("updatedAt") or ""), reverse=True)
        return tuple(summaries)

    @staticmethod
    def _is_cookie_invalidated(item: dict[str, Any]) -> bool:
        invalidated_at = item.get("invalidatedAt")
        return isinstance(invalidated_at, str) and bool(invalidated_at)

    @classmethod
    def _active_cookie_updated_at(cls, record: dict[str, Any]) -> str:
        active_id = cls._last_used_id(record)
        cookies = record.get("cookies")
        if not isinstance(cookies, list):
            return ""
        if active_id:
            for item in cookies:
                if not isinstance(item, dict):
                    continue
                if item.get("id") != active_id:
                    continue
                updated = item.get("updatedAt")
                return updated if isinstance(updated, str) else ""
        latest = ""
        for item in cookies:
            if not isinstance(item, dict):
                continue
            updated = item.get("updatedAt")
            if isinstance(updated, str) and updated > latest:
                latest = updated
        return latest

    @staticmethod
    def _extract_cookie_title(cookie: dict[str, Any] | None) -> str | None:
        if not isinstance(cookie, dict):
            return None
        data = cookie.get("data")
        if isinstance(data, dict):
            title = data.get("title") or data.get("label")
            if isinstance(title, str) and title.strip():
                return title.strip()
        return None

    @classmethod
    def _cookie_from_status_session(cls, status: Any) -> dict[str, Any] | None:
        supported = getattr(status, "experimental_session_supported", None)
        if supported is not True:
            return None

        session_id = getattr(status, "session_id", None)
        if (
            not isinstance(session_id, str)
            or not session_id
            or session_id == "local"
            or not session_id.startswith("sess-")
        ):
            return None
        cookie: dict[str, Any] = {"id": session_id}
        title = getattr(status, "session_title", None)
        if isinstance(title, str) and title.strip():
            cookie["data"] = {"title": title.strip()}
        return cookie
